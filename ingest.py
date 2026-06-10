"""
KshanaVartha — RSS ingest pipeline + multilingual → Telugu AI summariser.

HOW IT RUNS
  GitHub Actions cron runs every 30 minutes (public repo — unlimited free minutes).
  Trigger manually from the Actions tab with workflow_dispatch.
  Local: python ingest.py

ARTICLE STORE (Decision 3)
  Source of truth is articles.json on Cloudflare R2 — NOT Firestore.
  The cron loads it once, works in memory, writes back at end-of-run.
  Firestore is only used for small client-write collections:
    community_reports, fcm_tokens, live_data (weather/mandi).

AI ENGINE PRIORITY (per article, in order)
  1. Ollama     — LOCAL model (zero quota). Only when OLLAMA_MODEL is set.
                  Used for offline prompt iteration; never in production.
  2. Cerebras   — PRIMARY production engine. gpt-oss-120b, generous free tier,
                  strong Telugu. Multi-key pool (CEREBRAS_API_KEYS).
  3. Gemini     — Secondary / overflow. Flash free tier ~200 RPD per key.
                  Multi-key pool (GEMINI_API_KEYS), rotates on 429.
  4. SambaNova  — Fallback of last resort. Llama 3.3 70B, free tier.
                  cloud.sambanova.ai → API Keys. (SAMBANOVA_API_KEYS).
  All engines: skip English-source articles on regular cron (flag
  ENGLISH_POLISH=1 or BACKFILL=1 to include them).

REQUIRED ENV VARS
  FIREBASE_SERVICE_ACCOUNT_JSON — full JSON string (Firebase service account)
  R2_ACCOUNT_ID / R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY / R2_PUBLIC_URL
    — Cloudflare R2 bucket credentials (articles.json + feed.json + audio)

AI ENGINE ENV VARS (at least one required)
  CEREBRAS_API_KEYS    — comma-separated Cerebras keys  (primary engine)
  GEMINI_API_KEYS      — comma-separated Gemini keys    (secondary engine)
  SAMBANOVA_API_KEYS   — comma-separated SambaNova keys (fallback engine)
  OLLAMA_MODEL         — local model name, e.g. "qwen2.5:3b" (testing only)

OPTIONAL TUNING
  LIMIT=2            — cap new articles per run (testing; saves AI quota)
  DRY_RUN=1          — fetch + summarise only; write to dry_run_output.csv
  BACKFILL=1         — re-polish existing ai=False articles; higher Gemini cap
  ENGLISH_POLISH=1   — include English-source feeds in AI polishing
  AUDIO_MAX_PER_RUN  — TTS articles per cron run (default 30)
  FEED_MAX           — public feed.json article cap (default 500)

Run locally:
  pip install -r requirements.txt
  # Copy run-local.env.example to run-local.env and fill in secrets, then:
  # Windows PowerShell:
  Get-Content run-local.env | ForEach-Object {
      if ($_ -match '^\s*([^#=]+?)\s*=\s*(.*)\s*$') { [System.Environment]::SetEnvironmentVariable($Matches[1], $Matches[2]) }
  }
  python ingest.py
"""

from __future__ import annotations

import hashlib
import html
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import feedparser
import requests
from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter
from google.oauth2 import service_account

import article_store
import tts_r2

# ─── Logging ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("kv-ingest")

# ─── Config ──────────────────────────────────────────────
# Firestore database name. Our project's DB is the named 'default' (not the
# primary '(default)') because it was created via Console with that name.
# Override with FIRESTORE_DB_ID GitHub Variable if your project uses a
# different database name.
FIRESTORE_DB_ID = (os.environ.get("FIRESTORE_DB_ID") or "default").strip()

GEMINI_MODEL = (os.environ.get("GEMINI_MODEL") or "gemini-2.0-flash").strip()
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
)
GEMINI_CALL_DELAY_S = float(os.environ.get("GEMINI_CALL_DELAY_S") or "4.5")
MAX_PER_FEED = int(os.environ.get("MAX_PER_FEED") or "8")
# Tighter cap for YouTube channel feeds. Video feeds are noisy — channels
# publish many short clips a day and a single channel can dominate the
# feed otherwise. 5 per cron × 48 crons/day = 240 video items/channel/day
# max in steady state; the dedup filter means only truly-new ones process.
MAX_PER_VIDEO_FEED = int(os.environ.get("MAX_PER_VIDEO_FEED") or "5")
# Google News links are unresolvable redirects with no image. We decode
# them to the real publisher URL + scrape og:image/description. Capped
# per run (3 HTTP calls each) so it can't blow the 15-min cron budget.
GNEWS_RESOLVE_MAX = int(os.environ.get("GNEWS_RESOLVE_MAX") or "30")
MAX_TOTAL_GEMINI = int(os.environ.get("MAX_TOTAL_GEMINI") or "15")
RETENTION_DAYS = int(os.environ.get("RETENTION_DAYS") or "10")
TRAINING_RETENTION_DAYS = 30          # separate, longer window for ML training data
_TRAINING_DATA_KEY = "training_data.jsonl"
DELETE_BATCH_SIZE = 400     # Firestore batch limit is 500 — keep headroom
# Audio (edge-tts → R2) is filled incrementally: a capped number of
# articles get narrated per run so a 250-article backlog spreads over a
# few cron cycles instead of one slow run.
AUDIO_MAX_PER_RUN = int(os.environ.get("AUDIO_MAX_PER_RUN") or "30")
FEED_MAX = int(os.environ.get("FEED_MAX") or "2000")   # public feed.json size cap — high ceiling, midnight window limits naturally
FEED_MIN = int(os.environ.get("FEED_MIN") or "50")     # minimum articles; if today < 50, pad with yesterday's overflow
# Cinema is capped so it never floods the feed regardless of volume.
# 0 = no cap. Other Tier 2 categories (health, sports) are uncapped by default.
CINEMA_MAX = int(os.environ.get("CINEMA_MAX") or "30")
# Per-run AI quota caps by geographic level. Village/mandal/district articles
# get full quota; state + national are capped so they don't crowd out local
# stories when the AI engines are limited. 0 = no cap.
NATIONAL_AI_MAX = int(os.environ.get("NATIONAL_AI_MAX") or "5")
STATE_AI_MAX    = int(os.environ.get("STATE_AI_MAX")    or "10")
# Time-bucket size for feed ordering. Within each bucket articles are ordered
# by tier (Tier 1 important before Tier 2 entertainment), but newer buckets
# always beat older ones so a fresh health article outranks a stale politics one.
FEED_BUCKET_HOURS = int(os.environ.get("FEED_BUCKET_HOURS") or "1")
# Number of daily archive snapshots (feed_vYYYYMMDD.json) to keep on R2.
# Oldest archives beyond this count are pruned each run.
ARCHIVE_KEEP_DAYS = int(os.environ.get("ARCHIVE_KEEP_DAYS") or "5")
# IST timezone — used for midnight feed cutoff and daily archival.
IST = timezone(timedelta(hours=5, minutes=30))
# Minimum hours between push notifications. Set via NOTIFICATION_GAP_HOURS
# GitHub Variable. Default 3 h → max 8 pushes/day even on a fast news day.
NOTIFICATION_GAP_HOURS = int(os.environ.get("NOTIFICATION_GAP_HOURS") or "3")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_dt(v) -> datetime | None:
    """Coerce a stored createdAt/publishedAt (ISO str or datetime) → aware dt."""
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    if isinstance(v, str) and v:
        try:
            d = datetime.fromisoformat(v.replace("Z", "+00:00"))
            return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None

# Backfill (BACKFILL=1) is a manual operator run to re-polish existing
# ai=False articles. It gets a higher Gemini cap since it's triggered when
# quota is fresh. Engine cascade is unchanged: Cerebras → Gemini → SambaNova.
BACKFILL_GEMINI_MAX = int(os.environ.get("BACKFILL_GEMINI_MAX") or "120")

# Test-mode cap: process at most LIMIT *new* articles per run (0 = no cap).
# Set LIMIT=2 during local testing so you don't burn the daily AI quota
# while iterating on prompt quality. e.g.  $env:LIMIT="2"
INGEST_LIMIT = int(os.environ.get("LIMIT") or "0")

# DRY_RUN=1 → fetch RSS + summarize ONLY. No Firestore reads/writes at all.
# Results are written to dry_run_output.csv for inspection. Combine with
# OLLAMA_MODEL for genuinely unlimited, zero-quota local prompt iteration.
DRY_RUN = os.environ.get("DRY_RUN", "") == "1"

# Key sanitizer: strips BOM / zero-width / whitespace that breaks latin-1
# header encoding (a copy-pasted secret can carry a stray ﻿ that makes
# every API call fail with "'latin-1' codec can't encode character").
def _sanitize_key(raw: str) -> str:
    """Strip BOM / zero-width / whitespace that breaks latin-1 header encoding."""
    return "".join(c for c in raw if c.isprintable() and not c.isspace()).strip()


# ─── Cerebras (free tier, OpenAI-compatible) — PRIMARY engine ──
# Generous free limits, strong Telugu, no card required.
# Keys: cloud.cerebras.ai → API Keys. Env: CEREBRAS_API_KEYS (comma-
# separated, preferred) or single CEREBRAS_API_KEY. Rotates on 429.
# Available models as of 2026-05: gpt-oss-120b, zai-glm-4.7, llama3.1-8b.
# gpt-oss-120b is the confirmed-working default; override with CEREBRAS_MODEL.
CEREBRAS_KEYS = [
    _sanitize_key(k)
    for k in os.environ.get(
        "CEREBRAS_API_KEYS", os.environ.get("CEREBRAS_API_KEY", "")
    ).split(",")
    if _sanitize_key(k)
]
CEREBRAS_API_KEY = CEREBRAS_KEYS[0] if CEREBRAS_KEYS else ""
CEREBRAS_MODEL = (os.environ.get("CEREBRAS_MODEL") or "gpt-oss-120b").strip()
CEREBRAS_URL = "https://api.cerebras.ai/v1/chat/completions"
CEREBRAS_CALL_DELAY_S = float(os.environ.get("CEREBRAS_CALL_DELAY_S") or "1.5")

# ─── SambaNova (free tier, OpenAI-compatible) — FALLBACK engine ──
# Free tier, no card required. Llama 3.3 70B, strong multilingual.
# Keys: cloud.sambanova.ai → API Keys. Env: SAMBANOVA_API_KEYS (comma-
# separated) or single SAMBANOVA_API_KEY. Rotates on 429.
# Used when Cerebras is exhausted and Gemini quota is gone.
SAMBANOVA_KEYS = [
    _sanitize_key(k)
    for k in os.environ.get(
        "SAMBANOVA_API_KEYS", os.environ.get("SAMBANOVA_API_KEY", "")
    ).split(",")
    if _sanitize_key(k)
]
SAMBANOVA_API_KEY = SAMBANOVA_KEYS[0] if SAMBANOVA_KEYS else ""
SAMBANOVA_MODEL = (os.environ.get("SAMBANOVA_MODEL") or "Meta-Llama-3.3-70B-Instruct").strip()
SAMBANOVA_URL = "https://api.sambanova.ai/v1/chat/completions"
SAMBANOVA_CALL_DELAY_S = float(os.environ.get("SAMBANOVA_CALL_DELAY_S") or "2.0")

# ─── Ollama (LOCAL model — testing only, zero API quota) ────
# Set OLLAMA_MODEL (e.g. "qwen2.5:3b", "gemma2:2b") to route ALL polishing
# through your local Ollama server instead of any API. Great for unlimited
# free prompt iteration. Production (GitHub Actions) leaves this unset.
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "").strip()
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")

# Gemini 2.0 Flash free tier is ~200 RPD per project. We have 4 keys across 4
# separate Google accounts (~800 RPD theoretical), but spikes burn it fast.
# To stay safe, regular cron skips English-source articles (they need fuller
# translation+rewrite). Set ENGLISH_POLISH=1 to opt back in (e.g. for backfill).
POLISH_ENGLISH = os.environ.get("ENGLISH_POLISH") == "1" or os.environ.get("BACKFILL") == "1"

# ─── Feed list (mirror of app/prototype/rss.js) ──────────────
# Telugu-native (lang=te) feeds are highest value: they show in the
# Telugu-readable feed even unpolished, no Gemini/Groq needed. English
# feeds only surface once polished. A broken feed fails gracefully
# (logged, 0 items) so candidate Telugu feeds are low-risk to include.
#
# DEFERRED (English noise + unverified channels, add when needed):
#   - Mainstream English: NDTV, Indian Express, The Hindu national
#   - YouTube per-channel RSS (needs verified channel_id per channel):
#       https://www.youtube.com/feeds/videos.xml?channel_id=<ID>
FEEDS: list[dict[str, str]] = [
    # Telugu-native (no translation needed)
    {"url": "https://www.sakshi.com/rss/andhra-pradesh.xml",                            "source": "సాక్షి · ఏపీ",         "lang": "te"},
    {"url": "https://www.sakshi.com/rss/telangana.xml",                                 "source": "సాక్షి · తెలంగాణ",     "lang": "te"},
    {"url": "https://www.sakshi.com/rss/national.xml",                                  "source": "సాక్షి · జాతీయం",      "lang": "te"},
    {"url": "https://www.sakshi.com/rss/business.xml",                                  "source": "సాక్షి · వాణిజ్యం",    "lang": "te"},
    {"url": "https://feeds.bbci.co.uk/telugu/rss.xml",                                  "source": "BBC తెలుగు",           "lang": "te"},
    # Telugu TV/news sites (WordPress-style feeds; fail gracefully if down)
    {"url": "https://ntvtelugu.com/feed",                                               "source": "NTV తెలుగు",           "lang": "te"},
    {"url": "https://tv9telugu.com/feed",                                               "source": "TV9 తెలుగు",           "lang": "te"},
    # andhrajyothy.com/feed REMOVED 2026-05-18 — confirmed 404 (was a dead
    # source the cron hit every run). See memory feed_audit_2026_05_18.
    # Probe-confirmed working Telugu sources (2026-05-18):
    {"url": "https://www.telugudesam.org/feed",                                         "source": "తెలుగుదేశం",          "lang": "te"},
    {"url": "https://tv9telugu.com/entertainment/feed",                                 "source": "TV9 సినిమా",          "lang": "te"},
    {"url": "https://www.mirchi9.com/feed",                                             "source": "మిర్చి9 సినిమా",      "lang": "te"},
    {"url": "https://www.123telugu.com/feed",                                           "source": "123తెలుగు సినిమా",    "lang": "te"},
    # Google News Telugu — very reliable aggregator (covers Eenadu, Andhra
    # Jyothy, Sakshi etc. even when their own RSS is broken).
    {"url": "https://news.google.com/rss?hl=te&gl=IN&ceid=IN:te",                        "source": "గూగుల్ వార్తలు",       "lang": "te"},
    {"url": "https://news.google.com/rss/search?q=%E0%B0%8E%E0%B0%A8%E0%B1%8D%E0%B0%9F%E0%B1%80%E0%B0%86%E0%B0%B0%E0%B1%8D+%E0%B0%9C%E0%B0%BF%E0%B0%B2%E0%B1%8D%E0%B0%B2%E0%B0%BE&hl=te&gl=IN&ceid=IN:te", "source": "గూగుల్ · NTR జిల్లా", "lang": "te"},
    # English (surface only after polish)
    {"url": "https://www.thehindu.com/news/cities/Vijayawada/feeder/default.rss",       "source": "The Hindu · విజయవాడ", "lang": "en"},
    {"url": "https://www.thehindu.com/news/national/andhra-pradesh/feeder/default.rss", "source": "The Hindu · ఏపీ",      "lang": "en"},
    {"url": "https://www.thehindu.com/news/national/telangana/feeder/default.rss",      "source": "The Hindu · తెలంగాణ", "lang": "en"},
    {"url": "https://www.thehindu.com/news/cities/Hyderabad/feeder/default.rss",        "source": "The Hindu · హైదరాబాద్", "lang": "en"},
    # BRK News — probe-verified 2026-05-26 (10 items, Telugu-native).
    # WordPress feed; published by brknews.in for AP/Telangana news.
    {"url": "https://www.brknews.in/feed",                                                    "source": "BRK న్యూస్",                  "lang": "te"},
    # YouTube video channels — added 2026-05-23, audited 2026-05-26.
    # Channel IDs must be the canonical UC… form (handle-based IDs don't
    # work as RSS). Verify new IDs with the probe script before adding.
    #
    # YouTube channels — channel_ids supplied by operator 2026-05-26.
    # YouTube rate-limits our dev IP during probe; IDs verified by operator
    # via browser. Ingest runs on GitHub Actions (different IP) so it works.
    {"url": "https://www.youtube.com/feeds/videos.xml?channel_id=UCW6dg32S3jvvrIUEYdrYMYA",  "source": "BRK న్యూస్ · వీడియో",           "lang": "te"},
    {"url": "https://www.youtube.com/feeds/videos.xml?channel_id=UCAR3h_9fLV82N2FH4cE4RKw",  "source": "TV5 తెలుగు · వీడియో",           "lang": "te"},
    {"url": "https://www.youtube.com/feeds/videos.xml?channel_id=UC986Hrkimq5gGR-Ikva6VfA",  "source": "స్టూడియో N · వీడియో",           "lang": "te"},
    {"url": "https://www.youtube.com/feeds/videos.xml?channel_id=UC_2irx_BQR7RsBKmUV9fePQ",  "source": "ఆంధ్రజ్యోతి · వీడియో",         "lang": "te"},
    # STATUS 2026-05-26:
    #   NTV UCumtYpCY26F6Jr3satUgMvA → 404 (channel migrated; removed)
    #   ETV UCJi8M0hRKjz8SLPvJKEVTOg → removed per operator request 2026-05-26
]

# ─── Category classifier ────────────────────────────────────────────────
# Trained sklearn model. Falls back gracefully if pkl not found yet.
try:
    from categorizer import (
        CategoryClassifier,
        CONFIDENCE_THRESHOLD as _CAT_THRESHOLD,
        MODEL_PATH as _CAT_MODEL_PATH,
        LevelClassifier,
        LEVEL_CONFIDENCE_THRESHOLD as _LVL_THRESHOLD,
        LEVEL_MODEL_PATH as _LVL_MODEL_PATH,
    )
    _cat_model = CategoryClassifier.load_or_none(_CAT_MODEL_PATH)
    if _cat_model:
        log.info("CategoryClassifier loaded (cv_acc=%.1f%%)",
                 _cat_model.training_stats.get("cv_accuracy", 0) * 100)
    else:
        log.info("category_model.pkl not found — using keyword detection")
    _lvl_model = LevelClassifier.load_or_none(_LVL_MODEL_PATH)
    if _lvl_model:
        log.info("LevelClassifier loaded (cv_acc=%.1f%%)",
                 _lvl_model.training_stats.get("cv_accuracy", 0) * 100)
    else:
        log.info("level_model.pkl not found — using source map + keywords")
except ImportError:
    _cat_model = None
    _CAT_THRESHOLD = 0.45
    _lvl_model = None
    _LVL_THRESHOLD = 0.50


def _classify_article(
    headline: str,
    summary: str,
    ai_cat: str | None = None,
    url: str | None = None,
) -> str:
    """Four-tier category classification.

    0. URL path/domain — unambiguous segment-level keyword → instant return
    1. sklearn model   — fast, local, no quota cost
    2. AI p_cat        — used when model confidence is low
    3. Keyword rules   — always-available fallback
    """
    # Tier 0: URL-based hint (strongest signal — publisher explicitly placed
    # the article under a category slug like /cinema/ or /sports/)
    url_cat = _cat_from_url(url)
    if url_cat:
        return url_cat

    text = f"{headline} {summary or ''}"
    if _cat_model:
        cat, conf = _cat_model.predict(text)
        if conf >= _CAT_THRESHOLD:
            return cat
        # Low confidence: try AI p_cat
        if ai_cat and ai_cat in _AI_VALID_CATEGORIES and ai_cat != 'general':
            return ai_cat
        # Return model's guess if it's specific (not general)
        if cat != 'general':
            return cat
    else:
        # No model: use AI p_cat if available
        if ai_cat and ai_cat in _AI_VALID_CATEGORIES and ai_cat != 'general':
            return ai_cat
    # Final fallback: keyword rules
    return detect_category(text)


# ─── Geographic level detection ────────────────────────────────────────
# SOURCE_LEVELS maps feed source label → default geographic level.
SOURCE_LEVELS: dict[str, str] = {
    "సాక్షి · ఏపీ":           "state",
    "సాక్షి · తెలంగాణ":       "state",
    "సాక్షి · జాతీయం":        "national",
    "సాక్షి · వాణిజ్యం":      "national",
    "BBC తెలుగు":             "national",
    "NTV తెలుగు":             "state",
    "TV9 తెలుగు":             "state",
    "తెలుగుదేశం":             "state",
    "TV9 సినిమా":             "national",
    "మిర్చి9 సినిమా":         "national",
    "123తెలుగు సినిమా":       "national",
    "గూగుల్ వార్తలు":         "state",
    "గూగుల్ · NTR జిల్లా":    "district",
    "The Hindu · విజయవాడ":   "district",
    "The Hindu · ఏపీ":        "state",
    "The Hindu · తెలంగాణ":   "state",
    "The Hindu · హైదరాబాద్": "district",
    "BRK న్యూస్":             "district",
    "BRK న్యూస్ · వీడియో":   "district",
    "TV5 తెలుగు · వీడియో":   "state",
    "స్టూడియో N · వీడియో":   "state",
    "ఆంధ్రజ్యోతి · వీడియో":   "state",
}
_LEVEL_KW: dict[str, list[str]] = {
    "national": [
        "కేంద్రం", "కేంద్ర ప్రభుత్వం", "ప్రధానమంత్రి", "పార్లమెంట్",
        "సుప్రీంకోర్టు", "లోక్‌సభ", "రాజ్యసభ",
        "parliament", "prime minister", "supreme court", "lok sabha",
    ],
    "state": [
        "ఆంధ్రప్రదేశ్", "తెలంగాణ", "రాష్ట్ర ప్రభుత్వం", "ముఖ్యమంత్రి",
        "హైదరాబాద్", "అమరావతి", "గవర్నర్", "విధాన సభ",
        "andhra pradesh", "telangana",
    ],
    "district": [
        "ntr జిల్లా", "ntr district", "కృష్ణా జిల్లా", "జిల్లా కలెక్టర్",
        "vijayawada", "విజయవాడ",
    ],
}
_VALID_LEVELS = frozenset({"village", "mandal", "district", "state", "national"})


def _detect_level_from_keywords(headline: str, summary: str) -> str | None:
    """Detect level from keyword scan; None if no match."""
    text = (headline + " " + (summary or "")).lower()
    for lv in ("national", "state", "district"):
        for kw in _LEVEL_KW[lv]:
            if kw.lower() in text:
                return lv
    return None


def _determine_level(
    mandal: str,
    village: str | None,
    source: str,
    headline: str,
    summary: str,
    ai_level: str | None,
) -> str:
    """Priority: village/mandal field > AI result > source map > keywords > district."""
    if village:
        return "village"
    if mandal and mandal != "all":
        return "mandal"
    if ai_level and ai_level in _VALID_LEVELS:
        return ai_level
    if source in SOURCE_LEVELS:
        return SOURCE_LEVELS[source]
    if _lvl_model:
        text = f"{headline} {summary or ''}"
        lvl, conf = _lvl_model.predict(text)
        if conf >= _LVL_THRESHOLD:
            return lvl
    kw = _detect_level_from_keywords(headline, summary)
    if kw:
        return kw
    return "district"


CATEGORY_RULES: list[tuple[str, list[str]]] = [
    # ── Rule ordering rationale ──────────────────────────────────────────
    # 1. Most-specific first: scheme names before generic farming keywords.
    # 2. "village" moved BEFORE "politics" — a local incident story (snake,
    #    accident, crime) that mentions "ప్రభుత్వం" or a minister in passing
    #    should be village/general, not politics.
    # 3. "politics" keywords tightened: removed "ప్రభుత్వం" (appears in every
    #    Telugu news story), removed generic English "minister" and " mp "
    #    (too many false positives). Only specific party names, leader names,
    #    and unambiguous political terms remain.
    # 4. Unmatched articles fall through to 'general' (no category rule for it
    #    — it is the default return value of detect_category).
    # ────────────────────────────────────────────────────────────────────
    ("schemes",  [
        # Govt scheme names — very high signal; must run before farming/politics
        "పథకం", "భరోసా", "రైతు భరోసా", "ఆరోగ్యశ్రీ", "ఆసరా", "కల్యాణ లక్ష్మి",
        "రైతు బంధు", "రైతు భీమా", "అమ్మ ఒడి", "విద్యా దీవెన", "వసతి దీవెన",
        "పెన్షన్", "సబ్సిడీ", "దరఖాస్తు", "లబ్ధిదారు",
        "scheme", "subsidy", "pension", "welfare", "benefici", "rythu bharosa",
        "rythu bandhu", "aarogyasri", "asara", "kalyana lakshmi", "amma vodi",
        "vidya deevena", "vasathi deevena", "pmkisan", "kcc",
    ]),
    ("weather",  [
        "వాతావరణ", "వర్షం", "ఉష్ణోగ్రత", "తుఫాన్", "వడగండ్ల", "ఈదురు",
        "వరదలు", "కరువు", "ఎండ", "మంచు", "హీట్‌వేవ్",
        "weather", "rain", "storm", "imd", "cyclone", "flood", "drought",
        "heatwave", "monsoon", "thunder",
    ]),
    ("farming",  [
        "రైతు", "వ్యవసాయ", "మామిడి", "వరి", "పత్తి", "మిర్చి", "మండి", "క్వింటాల్",
        "ధాన్యం", "పంట", "అరటి", "నిమ్మ", "చెఱకు", "మిల్లు", "డెయిరీ", "పాడి",
        "ఆక్వాకల్చర్", "రొయ్యలు", "చేప", "మత్స్య", "హార్టికల్చర్",
        "farmer", "mandi", "crop", "paddy", "cotton", "agri", "horticulture",
        "aquaculture", "shrimp", "dairy", "harvest", "yield", "fertili",
    ]),
    ("jobs",     [
        "ఉద్యోగ", "రిక్రూట్", "నోటిఫికేషన్", "ఖాళీలు", "పోస్టులు",
        "డీఎస్సీ", "గ్రూప్ 1", "గ్రూప్ 2", "గ్రూప్ 3", "ఏపీపీఎస్సీ", "టీఎస్‌పీఎస్సీ",
        "job", "recruit", "vacancy", "notification", "appsc", "tspsc", "ssc",
        "upsc", "constable", "ssb", "exam result", "interview",
    ]),
    # ── health: BEFORE village ────────────────────────────────────────────
    # Health articles mention hospitals, doctors, diseases. Placing before
    # village prevents "ఆసుపత్రి" (hospital) from landing in the village
    # bucket — health is a more precise classification.
    ("health",   [
        # Telugu — core medical terms
        "ఆరోగ్యం", "ఆసుపత్రి", "వైద్యం", "వైద్యుడు", "డాక్టర్",
        "వ్యాధి", "మందులు", "చికిత్స", "వ్యాక్సిన్", "టీకా",
        "డెంగ్యూ", "మలేరియా", "కోవిడ్", "వైరస్", "జ్వరం",
        "వైద్య కళాశాల", "సర్జరీ", "ఆపరేషన్", "రక్తదానం",
        "ఐసీయూ", "అంబులెన్స్", "మెడికల్",
        # English
        "health", "hospital", "doctor", "disease", "medicine", "vaccine",
        "dengue", "malaria", "covid", "virus", "surgery", "treatment",
        "AIIMS", "PHC", "ICU", "ambulance", "blood donation", "health camp",
    ]),
    # ── village: BEFORE politics ─────────────────────────────────────────
    # Local-life stories (incidents, wildlife, community events, local govt)
    # should be village even if a politician is mentioned in passing.
    # Added locative forms (ఊళ్లో, ఊర్లో) and community words (గ్రామస్తులు)
    # so stories like "ఆ ఊళ్లో ప్రతి ఇంట్లో నాగుపాము" reach this rule first.
    ("village",  [
        # Telugu — governance
        "గ్రామ", "మండల", "పంచాయతీ", "సర్పంచ్", "గ్రామ సచివాలయం",
        # Telugu — local-life locative forms (catches "ఆ ఊళ్లో", "మన ఊర్లో")
        "ఊళ్లో", "ఊర్లో", "ఊరు", "గ్రామస్తులు", "స్థానికులు", "ప్రజలు",
        # Telugu — local incidents / human-interest
        "నాగుపాము", "పాము", "అడవి పంది", "చిరుత", "భల్లూకం",   # wildlife
        "ప్రమాదం", "రోడ్డు ప్రమాదం", "మృతి", "గాయాలు",           # accidents
        "అగ్నిప్రమాదం", "మంటలు",                                   # fire
        "దొంగతనం", "దోపిడీ", "హత్య",                               # crime
        # English
        "village", "panchayat", "gram", "sarpanch", "ward sachivalayam",
        "accident", "fire", "snake", "theft",
    ]),
    # ── cinema: BEFORE politics ──────────────────────────────────────────
    # Cinema must be checked before politics. Many entertainment articles
    # mention political names (actor who is a politician's relative, a
    # film about politics, etc.) and would wrongly land in politics if
    # this rule came later.
    ("cinema",   [
        # Telugu — core cinema words
        "సినిమా", "చిత్రం", "సినీ", "హీరో", "హీరోయిన్", "దర్శకుడు",
        "నటుడు", "నటి", "నటులు", "నటన", "బాక్సాఫీస్", "ట్రైలర్", "టీజర్",
        "ఓటీటీ", "రివ్యూ", "షూటింగ్", "టాలీవుడ్", "ఫస్ట్ లుక్", "విడుదల",
        # Telugu — transliterations commonly used in headlines
        "డైరెక్టర్",           # director (Telugu script)
        "ఆడియో",               # audio launch / song release
        "బ్లాక్‌బస్టర్",      # blockbuster
        "మ్యూజిక్",            # music
        "ప్రి-రిలీజ్", "ప్రి రిలీజ్",  # pre-release event
        "అవార్డు",             # award (film awards)
        "వెబ్ సిరీస్",         # web series
        # English
        "movie", "cinema", "film", "tollywood", "box office", "trailer",
        "teaser", "ott", "review", "first look", "actor", "actress",
        "director", "song release", "pre-release", "audio launch",
        "blockbuster", "web series",
    ]),
    # ── politics: strong signals only ───────────────────────────────────
    # REMOVED: "ప్రభుత్వం" (govt — appears in every news story, too broad)
    # REMOVED: "minister" (English generic — fires on any minister visit/event)
    # REMOVED: " mp " (fires on English words like "camp", "stamp", "example")
    # KEPT: specific party names, leader names, election/legislative terms.
    ("politics", [
        # Telugu — specific leaders
        "చంద్రబాబు", "నారా లోకేష్", "పవన్ కల్యాణ్", "జగన్", "వైఎస్", "షర్మిల",
        "రేవంత్ రెడ్డి", "కేటీఆర్", "కేసీఆర్", "మోదీ", "రాహుల్", "షా",
        "భూమన", "పెద్దిరెడ్డి", "బొత్స", "విజయసాయి", "హరిరామ", "నారాయణ",
        # Telugu — parties and unambiguous political roles
        "పార్టీ", "ఎమ్మెల్యే", "ఎంపీ", "ఎంఎల్‌సీ", "సీఎం", "ముఖ్యమంత్రి",
        "ఎన్నికలు", "ఓటర్", "ఎన్నికల కమిషన్", "మంత్రి",
        "టీడీపీ", "వైసీపీ", "వైఎస్సార్సీపీ", "జనసేన", "బీఆర్‌ఎస్", "బీజేపీ", "కాంగ్రెస్",
        # English — specific names and parties only
        "tdp", "ysrcp", "ysr", "bjp", "congress", "brs", "jana sena", "janasena",
        "naidu", "lokesh", "jagan", "sharmila", "pawan kalyan", "revanth",
        "ktr", "kcr", "modi", "rahul", "amit shah",
        # English — unambiguous political terms (not generic)
        "cm ", " mla", "election", "eci",
        "parliament", "legisla", "cabinet", "governor", "assembly session",
    ]),
    ("sports",   [
        "క్రికెట్", "క్రీడ", "మ్యాచ్", "టోర్నీ", "ఐపీఎల్",
        "cricket", "sport", "match", "tournament", "ipl", "olympic",
        "kabaddi", "kho kho", "tennis", "football", "hockey", "athlet",
    ]),
]

COLOR_BY_CAT = {
    "village": "#EA580C", "farming": "#15803D", "schemes": "#0F172A",
    "weather": "#38BDF8", "jobs": "#B45309", "politics": "#7C2D12", "sports": "#15803D",
    "cinema": "#9333EA",
    "general": "#7C2D12",  # generic news — same colour as politics for now
}

# NTR District mandal aliases (Telugu short forms + English transliterations).
# Official source: ntr.ap.gov.in/mandal-wise-villages/ — 17 mandals.
# NOTE: Nuzvid is NOT in NTR District (it's in NTR District's predecessor Krishna District).
MANDAL_ALIASES: dict[str, dict[str, list[str]]] = {
    "vij-urban": {"en": ["vijayawada", "vijayawada urban", "vijayawada town"],
                  "te": ["విజయవాడ అర్బన్", "విజయవాడ"]},
    "vij-rural": {"en": ["vijayawada rural", "gollapudi", "nunna", "nidamanuru",
                          "jakkampudi", "enikepadu", "tadepalle"],
                  "te": ["విజయవాడ రూరల్"]},
    "mylavaram": {"en": ["mylavaram"],                    "te": ["మైలవరం"]},
    "ibrahim":   {"en": ["ibrahimpatnam", "kondapalli"],  "te": ["ఇబ్రహీంపట్నం"]},
    "akonduru":  {"en": ["a konduru", "a. konduru"],      "te": ["ఏ కొండూరు"]},
    "gkonduru":  {"en": ["g konduru", "g. konduru"],      "te": ["జి కొండూరు"]},
    "chandarla": {"en": ["chandarlapadu"],                "te": ["చందర్లపాడు"]},
    "gampalagu": {"en": ["gampalagudem"],                 "te": ["గంపలగూడెం"]},
    "jagga":     {"en": ["jaggayyapeta", "jaggaiahpeta"], "te": ["జగ్గయ్యపేట"]},
    "kanchika":  {"en": ["kanchikacherla"],               "te": ["కంచికచర్ల"]},
    "nandigama": {"en": ["nandigama"],                    "te": ["నందిగామ"]},
    "penuga":    {"en": ["penuganchiprolu"],              "te": ["పెనుగంచిప్రోలు"]},
    "reddi":     {"en": ["reddigudem"],                   "te": ["రెడ్డిగూడెం"]},
    "tiruvuru":  {"en": ["tiruvuru", "thiruvuru"],        "te": ["తిరువూరు"]},
    "vatsavai":  {"en": ["vatsavai"],                     "te": ["వత్సవాయి"]},
    "veerulla":  {"en": ["veerullapadu", "veerillapadu"], "te": ["వీరుళ్లపాడు"]},
    "vissanna":  {"en": ["vissannapet", "vissannapeta"],  "te": ["విస్సన్నపేట"]},
}
# Match order matters — specific before generic (vij-rural before vij-urban).
MANDAL_ORDER = [
    "vij-rural", "vij-urban",
    "mylavaram", "ibrahim", "akonduru", "gkonduru", "chandarla",
    "gampalagu", "jagga", "kanchika", "nandigama", "penuga",
    "reddi", "tiruvuru", "vatsavai", "veerulla", "vissanna",
]

VILLAGES_BY_MANDAL: dict[str, list[str]] = {
    "mylavaram": ["మైలవరం", "చంద్రల", "దాసుల్లపాలెం", "గణపవరం", "పొందుగుల", "పుల్లురు"],
    "ibrahim":   ["ఇబ్రహీంపట్నం", "కొండపల్లి", "ఎలప్రోలు", "గుంటుపల్లె", "మల్కాపురం"],
    "tiruvuru":  ["తిరువూరు", "అక్కాపాలెం", "ఎర్రమాడు", "మల్లెల", "పెద్దవరం"],
    "nandigama": ["నందిగామ", "గొల్లముడి", "లచ్చపాలెం", "సోమవరం", "చందాపురం"],
    "jagga":     ["జగ్గయ్యపేట", "అన్నవరం", "జయంతిపురం", "వేదాద్రి", "పోచంపల్లె"],
    "vissanna":  ["విస్సన్నపేట", "చంద్రుపట్ల", "కాలగర", "నరసాపురం"],
    "vatsavai":  ["వత్సవాయి", "గంగావల్లి", "భీమవరం", "మంగోల్లు", "తల్లూరు"],
    "veerulla":  ["వీరుళ్లపాడు", "అల్లూరు", "దాచవరం", "జమ్మవరం", "పొన్నవరం"],
    "penuga":    ["పెనుగంచిప్రోలు", "గుమ్మడిదుర్రు", "లింగగూడెం", "నవాబ్‌పేట"],
    "chandarla": ["చందర్లపాడు", "బొబ్బెల్లపాడు", "గుడిమెట్ల", "ముప్పల్ల", "పోపురు"],
    "gampalagu": ["గంపలగూడెం", "అనుమొల్లంక", "గోసవీడు", "నేమలి", "రాజావరం"],
    "kanchika":  ["కంచికచర్ల", "పారిటాల", "మొగులూరు", "గందేపల్లె", "బాధినపాడు"],
    "akonduru":  ["ఏ కొండూరు", "అట్లప్రగడ", "కంభంపాడు", "మధవరం", "రేపూడి"],
    "gkonduru":  ["జి కొండూరు", "అతుకూరు", "చేవుతూరు", "కవులూరు", "లోయ"],
    "reddi":     ["రెడ్డిగూడెం", "అన్నెరావుపేట", "కుదాప", "నాగులూరు", "రంగాపురం"],
}


# ─── Helpers ─────────────────────────────────────────────
def strip_html(s: str) -> str:
    if not s:
        return ""
    # Decode HTML entities FIRST (&nbsp; &amp; &#39; &quot; …). RSS feeds —
    # esp. Google News aggregated entries — leak these into descriptions;
    # without this they survive verbatim into summary/audioScript.
    s = html.unescape(s)
    s = re.sub(r"<[^>]+>", " ", s)          # drop any tags
    # &nbsp; decodes to U+00A0 (non-breaking space); normalize it and
    # other unicode separators to a plain space so \s+ can collapse them.
    s = s.replace("\xa0", " ").replace("​", "")
    return re.sub(r"\s+", " ", s).strip()


def hash_id(s: str) -> str:
    """Stable short ID — same algorithm style as JS rss.js (lower-cased base36)."""
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]
    return h


BLOCKLIST_R2_KEY = os.environ.get("R2_BLOCKLIST_KEY", "blocklist.json")

def load_blocklist() -> set[str]:
    """
    Load permanently blocked article IDs from R2 (blocklist.json in the
    same bucket as articles.json). Falls back to a local file when R2 is
    not configured (offline / test mode).

    To block an article:
      1. Download blocklist.json from Cloudflare R2 dashboard
      2. Add the 12-char ID to the JSON array and re-upload
      3. Trigger the workflow — no git commit needed
    """
    import article_store as _as

    # Try R2 first (production path)
    if _as._r2_enabled():
        try:
            obj = _as._r2_client().get_object(Bucket=_as.R2_BUCKET, Key=BLOCKLIST_R2_KEY)
            payload = json.loads(obj["Body"].read().decode("utf-8"))
            # Canonical shape is a plain JSON array of IDs:
            #     ["abc123def456", ...]
            # Older admin builds used to wrap this as {"blocked_ids": [...]},
            # which would silently disable the blocklist on every cron run.
            # Accept both shapes so a stale R2 file from the old admin still
            # blocks correctly; the next admin upload overwrites with the
            # canonical plain list. Drop this `dict` branch once we're
            # confident no stale wrapped blocklists exist anywhere.
            if isinstance(payload, list):
                ids = payload
            elif isinstance(payload, dict) and isinstance(payload.get("blocked_ids"), list):
                ids = payload["blocked_ids"]
                log.warning(
                    "blocklist.json on R2 uses legacy {blocked_ids: [...]} shape; "
                    "next admin 'Sync JSON Files' will rewrite as plain list."
                )
            else:
                log.warning("blocklist.json on R2 is not a list or {blocked_ids: [...]} — ignoring")
                return set()
            blocked = {str(i).strip() for i in ids if i}
            if blocked:
                log.info("blocklist loaded from R2 — %d blocked ID(s)", len(blocked))
            return blocked
        except Exception as e:
            # Missing key on first run is normal (no blocklist yet)
            log.info("blocklist: no R2 file yet (%s) — none blocked", type(e).__name__)
            return set()

    # Offline fallback: read local file if present
    local = os.path.join(os.path.dirname(__file__), "blocklist.json")
    try:
        with open(local, "r", encoding="utf-8") as f:
            ids = json.load(f)
        blocked = {str(i).strip() for i in ids if isinstance(ids, list) and i}
        if blocked:
            log.info("blocklist loaded from local file — %d blocked ID(s)", len(blocked))
        return blocked
    except FileNotFoundError:
        return set()
    except Exception as e:
        log.warning("blocklist local load failed: %s", e)
        return set()


def dedup_key(headline: str) -> str:
    """
    Normalized key for cross-feed deduplication. The SAME story often appears
    in multiple overlapping feeds (e.g. The Hindu · Vijayawada AND · AP) with
    identical headlines but slightly different publish timestamps. Keying the
    article ID off the normalized headline alone collapses those duplicates —
    no Firestore lookup needed (in-memory set against the loaded working set).
    Lowercase, strip punctuation, collapse whitespace.
    """
    s = (headline or "").lower()
    s = re.sub(r"[^\wఀ-౿]+", " ", s)  # keep alphanumerics + Telugu block
    return re.sub(r"\s+", " ", s).strip()


def detect_category(text: str) -> str:
    t = (text or "").lower()
    for cat_id, kws in CATEGORY_RULES:
        if any(k.lower() in t for k in kws):
            return cat_id
    return "general"  # explicit fallback (was "village" — but most don't match village keywords either)


# URL path/domain → category hints.
# Only strong, unambiguous segment-level matches are listed.
# Partial English words (e.g. "government" contains "govern") are handled by
# requiring the keyword to appear as a full path segment or subdomain token.
_URL_CAT_PATTERNS: list[tuple[str, list[str]]] = [
    ("cinema",  ["cinema", "movie", "movies", "entertainment", "film", "films",
                 "tollywood", "bollywood", "kollywood", "celeb", "celebrity",
                 "box-office", "boxoffice", "trailer", "teaser"]),
    ("health",  ["health", "lifestyle", "wellness", "fitness", "medical",
                 "medicine", "diet", "nutrition"]),
    ("sports",  ["sports", "sport", "cricket", "ipl", "football", "kabaddi",
                 "olympics", "tennis", "hockey"]),
    ("farming", ["farming", "agriculture", "agri", "krishi", "farmer",
                 "horticulture", "aquaculture", "mandi"]),
    ("weather", ["weather", "climate", "imd", "cyclone", "monsoon"]),
    ("jobs",    ["jobs", "recruitment", "vacancy", "employment", "career",
                 "sarkari-result", "sarkari-naukri"]),
    ("politics",["politics", "political", "election", "elections", "parliament", "assembly"]),
    ("schemes", ["scheme", "yojana", "welfare"]),
    ("village", ["village", "gram", "panchayat", "rural"]),
]


def _cat_from_url(url: str) -> str | None:
    """
    Return a category inferred from the article source URL, or None.

    Checks domain tokens and path segments for unambiguous category keywords.
    Only returns a result when confidence is high (full segment match), so
    partial matches like /government/ (politics) don't fire on /governmental-aid/.
    """
    if not url:
        return None
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url.lower())
        # Build a space-separated token list from subdomain+path segments
        domain = parsed.netloc.replace("www.", "")
        path = parsed.path
        # Collect individual tokens: domain parts split on '.' and path parts split on '/''-'
        tokens: set[str] = set()
        tokens.update(domain.split("."))
        for seg in path.split("/"):
            tokens.update(seg.split("-"))
        for cat, keywords in _URL_CAT_PATTERNS:
            for kw in keywords:
                if kw in tokens:
                    return cat
        return None
    except Exception:
        return None


def detect_mandal(text: str) -> tuple[str, str | None]:
    """Return (mandal_id, village_name | None). mandal_id can be 'all'."""
    if not text:
        return "all", None
    lower = text.lower()
    # Pass 1: aliases
    for mid in MANDAL_ORDER:
        aliases = MANDAL_ALIASES[mid]
        if any(t in text for t in aliases["te"]):
            village = _detect_village(text, mid)
            return mid, village
        if any(e in lower for e in aliases["en"]):
            village = _detect_village(text, mid)
            return mid, village
    # Pass 2: village → parent
    for mid, villages in VILLAGES_BY_MANDAL.items():
        for v in villages:
            if v in text:
                return mid, v
    return "all", None


def _detect_village(text: str, mandal_id: str) -> str | None:
    for v in VILLAGES_BY_MANDAL.get(mandal_id, []):
        if v in text:
            return v
    return None


def audio_seconds_estimate(text: str) -> int:
    n = len(text or "")
    return max(30, min(90, n // 12))


# ─── Gemini key pool (auto-rotation on 429) ───────────────
class GeminiKeyPool:
    """
    Holds N Gemini API keys. On 429 (quota hit), marks the current key
    as exhausted and rotates to the next. When all keys are exhausted,
    raises QuotaExhausted so the caller can stop trying.
    """
    def __init__(self, keys: list[str]):
        self.keys = [k.strip() for k in keys if k and k.strip()]
        self.exhausted: set[str] = set()
        self.idx = 0
        # Per-key usage stats accumulated during the run.
        self.key_stats: dict[str, dict] = {
            k: {"calls": 0, "success": 0, "fail": 0, "quota_hits": 0,
                "prompt_tokens": 0, "comp_tokens": 0}
            for k in self.keys
        }

    def record_call(self, key: str, *, success: bool, quota: bool = False,
                    prompt_tokens: int = 0, comp_tokens: int = 0) -> None:
        s = self.key_stats.get(key)
        if not s:
            return
        s["calls"] += 1
        if success:
            s["success"] += 1
        elif quota:
            s["quota_hits"] += 1
        else:
            s["fail"] += 1
        s["prompt_tokens"] += prompt_tokens
        s["comp_tokens"] += comp_tokens

    def current(self) -> str | None:
        for _ in range(len(self.keys)):
            k = self.keys[self.idx % len(self.keys)]
            if k not in self.exhausted:
                return k
            self.idx += 1
        return None

    def mark_exhausted(self, key: str) -> None:
        self.exhausted.add(key)
        log.warning(
            "key ...%s exhausted (%d/%d keys left)",
            key[-6:], len(self.keys) - len(self.exhausted), len(self.keys),
        )
        self.idx += 1

    def has_capacity(self) -> bool:
        return len(self.exhausted) < len(self.keys)


class QuotaExhausted(RuntimeError):
    pass


class CerebrasKeyPool(GeminiKeyPool):
    """Same rotation logic as GeminiKeyPool, for Cerebras API keys."""
    pass


class SambaNovaKeyPool(GeminiKeyPool):
    """Same rotation logic, for SambaNova API keys."""
    pass


# Process-wide pools. Rotation state lives for the whole ingest run so a
# 429'd key is skipped for the rest of the run.
cerebras_pool = CerebrasKeyPool(CEREBRAS_KEYS)
sambanova_pool = SambaNovaKeyPool(SAMBANOVA_KEYS)


# Inverted-Pyramid Telugu summary. Most critical info first, then details
# by decreasing importance. The audioScript is the same text as the summary,
# so a clean, readable summary also produces a clean narration.
SUMMARY_PROMPT = (
    "మీరు అనుభవజ్ఞుడైన తెలుగు వార్తా సంపాదకులు. 'విలోమ పిరమిడ్' "
    "(Inverted Pyramid) పద్ధతిలో వార్తను తిరిగి రాయండి — అత్యంత ముఖ్యమైన "
    "సమాచారం మొదట, తరువాత ప్రాముఖ్యత తగ్గుతూ వివరాలు. ఇంగ్లీష్ అయితే "
    "సహజమైన తెలుగులోకి అనువదించండి. ఖచ్చితంగా ఈ ఆకృతిలో జవాబివ్వండి:\n"
    "TITLE: <ముఖ్య సంఘటనను చెప్పే స్పష్టమైన శీర్షిక, ~8-12 పదాలు>\n"
    "BODY: <ఇది ఆడియోగా వినిపించబడుతుంది — తక్కువ చదువుకున్నవారికి కూడా "
    "మొదటిసారే అర్థమయ్యే సరళమైన తెలుగు వాడండి. మూలంలో నిజంగా ఉన్న "
    "సమాచారం మేరకే, మొత్తం ~90-130 పదాలు. మూలం తక్కువైతే తక్కువ "
    "వాక్యాలు — నిడివి కోసం సాగదీయవద్దు. ఈ క్రమంలో రాయండి:\n"
    " • ఏం జరిగింది: ముఖ్య విషయాన్ని సరళ భాషలో మొదటి వాక్యంలో చెప్పండి "
    "(ఏమి, ఎక్కడ, ఎప్పుడు).\n"
    " • ఎవరు ప్రమేయం: సంబంధిత వ్యక్తులు, సంస్థలు, ప్రాంతాలు/దేశాలు.\n"
    " • ఎందుకు ముఖ్యం: ప్రజలపై/రైతులపై ప్రభావం లేదా ప్రాముఖ్యత.\n"
    " • సంఘటనల వరుస: కథకు కాలక్రమం ఉంటే మాత్రమే క్లుప్తంగా; "
    "లేకపోతే ఈ భాగం వదిలేయండి.\n"
    " • చివరి సారాంశం: ఒక్క వాక్యంలో ముగింపు ముఖ్యాంశం.\n"
    " ప్రతి వాక్యం పూర్తి చేయండి — మధ్యలో ఆపవద్దు>\n"
    "నియమాలు: తటస్థ, వాస్తవిక భాష. వార్త సానుకూలమా/ప్రతికూలమా అని "
    "ముద్ర వేయవద్దు, మీ అభిప్రాయం రాయవద్దు. పునరావృతం, నింపుడు మాటలు, "
    "మూలంలో లేని నేపథ్యం వద్దు. TITLE, BODY రెండూ తెలుగులోనే; "
    "బుల్లెట్‌లు/నంబర్‌లు వద్దు (వచనం మాత్రమే). TITLE:, BODY:, CATEGORY: "
    "ట్యాగ్‌లు ఇంగ్లీష్‌లోనే ఉంచండి.\n"
    "CATEGORY: వార్త విషయాన్ని బట్టి ఒక్క వర్గం మాత్రమే ఎంచుకోండి: "
    "politics (రాజకీయాలు, నేతలు, పార్టీలు, ఎన్నికలు) | "
    "farming (వ్యవసాయం, రైతులు, పంటలు, మండీ ధరలు) | "
    "weather (వాతావరణం, వర్షం, తుఫాన్) | "
    "jobs (ఉద్యోగాలు, నోటిఫికేషన్లు, పరీక్షలు) | "
    "health (ఆరోగ్యం, ఆసుపత్రి, వైద్యం, వ్యాధులు, వ్యాక్సిన్లు) | "
    "village (గ్రామాలు, స్థానిక సంఘటనలు, ప్రమాదాలు, నేరాలు) | "
    "sports (క్రీడలు, క్రికెట్) | "
    "cinema (సినిమా, నటులు, చిత్రాలు) | "
    "schemes (ప్రభుత్వ పథకాలు, సబ్సిడీలు, పెన్షన్లు) | "
    "general (వేరే ఏ వర్గంలోనూ సరిగా సరిపోకపోతే)\n"
    "LEVEL: వార్త భౌగోళిక స్థాయి ఒక్క పదంలో: "
    "village (నిర్దిష్ట గ్రామం/పట్టణం) | "
    "mandal (నిర్దిష్ట మండలం) | "
    "district (జిల్లా స్థాయి, ఉదా. NTR జిల్లా) | "
    "state (రాష్ట్రం స్థాయి, ఉదా. ఆంధ్రప్రదేశ్/తెలంగాణ) | "
    "national (జాతీయ స్థాయి)\n\n"
    "శీర్షిక: {headline}\nమూల వచనం: {summary}"
)

# Valid category values the AI may return — must match CATEGORY_RULES keys + general.
_AI_VALID_CATEGORIES = frozenset({
    "politics", "farming", "weather", "jobs", "village",
    "health", "sports", "cinema", "schemes", "general",
})


def _split_polished(raw: str) -> tuple[str | None, str, str | None, str | None]:
    """
    Parse the TITLE:/BODY:/CATEGORY: structured polish output.
    Returns (telugu_headline_or_None, summary, ai_category_or_None).

    ai_category is only set when the model returns a recognised category
    token from _AI_VALID_CATEGORIES. Callers should use it to override the
    keyword-detected category but fall back to keyword rules when it is None
    or unrecognised.

    Falls back gracefully: if no tags found, the whole text is the summary,
    headline and category are None.
    """
    if not raw:
        return None, "", None
    title = None
    body_parts: list[str] = []
    ai_category: str | None = None
    ai_level: str | None = None
    mode = None
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        su = s.upper()
        if su.startswith("TITLE:"):
            title = s[6:].strip().strip("\"'").strip()
            mode = "t"
        elif su.startswith("BODY:"):
            body_parts.append(s[5:].strip())
            mode = "b"
        elif su.startswith("CATEGORY:"):
            cat = s[9:].strip().lower().strip("\"'").strip()
            if cat in _AI_VALID_CATEGORIES:
                ai_category = cat
            mode = "c"
        elif su.startswith("LEVEL:"):
            raw_lv = s[6:].strip().lower().strip("\"'").strip()
            lv = raw_lv.split()[0] if raw_lv else ""
            if lv in _VALID_LEVELS:
                ai_level = lv
            mode = "l"
        elif mode == "b":
            body_parts.append(s)
        elif mode == "t" and not title:
            title = s
    body = " ".join(p for p in body_parts if p).strip().strip("\"'").strip()
    if not body and not title:
        # Model ignored the format — treat the whole thing as the summary.
        return None, _sanitize_summary(raw.strip().strip("\"'").strip()), None
    if not body:
        body = raw.strip()
    # Truncation guard: if the model ran out of tokens, BODY ends mid-sentence.
    # Drop the dangling partial sentence so the summary always ends cleanly,
    # but only when a clean cut still leaves a substantial body.
    if body and body[-1] not in _SENTENCE_TERMS:
        cut = max(body.rfind(c) for c in _SENTENCE_TERMS)
        if cut >= 40:
            body = body[:cut + 1].strip()
    # Sanitize the model's output — even AI-generated summaries can echo
    # junk characters (►/🔥/etc.) that were in the source description.
    body = _sanitize_summary(body)
    title = _sanitize_summary(title) if title else title
    return (title or None), body, ai_category, ai_level


def _looks_english(s: str) -> bool:
    """True if the string has no Telugu characters (so it needs translation)."""
    if not s:
        return False
    return not any('ఀ' <= ch <= '౿' for ch in s)


def _te_word_count(text: str) -> int:
    """Count whitespace-separated tokens that contain at least one Telugu character."""
    if not text:
        return 0
    return sum(1 for w in text.split() if any('ఀ' <= ch <= '౿' for ch in w))


# Phrases that indicate a PDF/iframe embed failed to load — these appear
# verbatim as the article "content" when a WordPress PDF plugin times out.
_JUNK_PHRASES = [
    "reload document",
    "open in new tab",
    "taking too long?",
    "embed any document",
    "loading.svg",
]

def _is_junk_content(text: str, image: str = "") -> bool:
    """
    True when the article body is a document-viewer error message rather
    than real news. Catches WordPress PDF embeds (telugudesam.org et al.)
    that render "Loading... Taking too long? Reload document | Open in new
    tab Download" as the article text.
    """
    combined = (text + " " + (image or "")).lower()
    hits = sum(1 for p in _JUNK_PHRASES if p in combined)
    return hits >= 2  # require at least 2 signals to avoid false positives


def gemini_summarize(pool: GeminiKeyPool, headline: str, raw_summary: str) -> str | None:
    """
    Try each key in the pool until one works or all are exhausted.
    Returns the polished Telugu summary, or None on per-article failure.
    Raises QuotaExhausted only when the entire pool is dead.
    """
    body = {
        "contents": [{"role": "user", "parts": [{
            "text": SUMMARY_PROMPT.format(headline=headline, summary=raw_summary),
        }]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 450, "topP": 0.8},
    }
    while True:
        key = pool.current()
        if not key:
            raise QuotaExhausted()
        try:
            r = requests.post(f"{GEMINI_URL}?key={key}", json=body, timeout=30)
        except Exception as e:
            log.warning("gemini network error on key ...%s: %s", key[-6:], e)
            return None  # network blip — don't burn a key
        if r.status_code == 429:
            # 429 = quota/rate limit. Recovers at daily reset. Log the
            # reason so we can tell "daily quota gone" vs "per-minute throttle".
            log.warning("gemini key ...%s 429 QUOTA: %s",
                        key[-6:], r.text[:200].replace("\n", " "))
            pool.record_call(key, success=False, quota=True)
            pool.mark_exhausted(key)
            continue  # try next key
        if r.status_code == 403:
            # 403 = key invalid / API not enabled / billing. Does NOT recover
            # at reset — the key itself is broken and must be replaced.
            log.warning("gemini key ...%s 403 INVALID-KEY (needs replacing): %s",
                        key[-6:], r.text[:200].replace("\n", " "))
            pool.record_call(key, success=False, quota=True)
            pool.mark_exhausted(key)
            continue
        if not r.ok:
            log.warning("gemini http %d: %s", r.status_code, r.text[:120])
            pool.record_call(key, success=False)
            return None
        try:
            data = r.json()
            text = (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
            )
            usage = data.get("usageMetadata", {})
            ptok = int(usage.get("promptTokenCount") or 0)
            ctok = int(usage.get("candidatesTokenCount") or 0)
            pool.record_call(key, success=bool(text), prompt_tokens=ptok, comp_tokens=ctok)
            return text.strip().strip("\"'") if text else None
        except Exception as e:
            log.warning("gemini parse failed: %s", e)
            pool.record_call(key, success=False)
            return None


def sambanova_summarize(headline: str, raw_summary: str) -> str | None:
    """
    Polish via SambaNova (Llama 3.3 70B), OpenAI-compatible chat completions.
    Used as the fallback of last resort when Cerebras and Gemini are exhausted.
    Returns the Telugu summary or None on any failure (never raises).
    """
    body = {
        "model": SAMBANOVA_MODEL,
        "messages": [
            {"role": "user", "content": SUMMARY_PROMPT.format(headline=headline, summary=raw_summary)},
        ],
        "temperature": 0.3,
        "max_tokens": 600,
    }
    while True:
        key = sambanova_pool.current()
        if not key:
            log.warning("sambanova: all keys exhausted — skipping this article")
            return None
        try:
            r = requests.post(
                SAMBANOVA_URL,
                headers={"Authorization": f"Bearer {key}"},
                json=body, timeout=30,
            )
        except Exception as e:
            log.warning("sambanova network error: %s", e)
            return None
        if r.status_code == 429:
            log.warning("sambanova key ...%s 429: %s",
                        key[-6:], r.text[:220].replace("\n", " "))
            sambanova_pool.record_call(key, success=False, quota=True)
            sambanova_pool.mark_exhausted(key)
            continue  # try next key
        if not r.ok:
            log.warning("sambanova http %d: %s", r.status_code, r.text[:140])
            sambanova_pool.record_call(key, success=False)
            return None
        try:
            data = r.json()
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            ptok = int(usage.get("prompt_tokens") or 0)
            ctok = int(usage.get("completion_tokens") or 0)
            sambanova_pool.record_call(key, success=bool(text), prompt_tokens=ptok, comp_tokens=ctok)
            return text.strip().strip("\"'") if text else None
        except Exception as e:
            log.warning("sambanova parse failed: %s", e)
            sambanova_pool.record_call(key, success=False)
            return None


def cerebras_summarize(headline: str, raw_summary: str) -> str | None:
    """
    Polish via Cerebras (Llama 3.3 70B), OpenAI-compatible. Primary engine
    for production — generous free tier, strong Telugu. Multi-key rotation
    on 429. Returns Telugu summary or None on per-article failure.
    """
    body = {
        "model": CEREBRAS_MODEL,
        "messages": [
            {"role": "user", "content": SUMMARY_PROMPT.format(headline=headline, summary=raw_summary)},
        ],
        "temperature": 0.3,
        # gpt-oss-120b is a thinking model: internal reasoning tokens count
        # toward max_tokens BEFORE the visible answer is written.  450 was
        # enough to exhaust the budget on chain-of-thought → finish_reason=length
        # with no output.  1200 gives ~600 tokens of reasoning room + ~600 for
        # the actual Telugu headline + 2–3 sentence summary.
        "max_tokens": 1200,
    }
    while True:
        key = cerebras_pool.current()
        if not key:
            log.warning("cerebras: all keys exhausted")
            return None
        try:
            r = requests.post(
                CEREBRAS_URL,
                headers={"Authorization": f"Bearer {key}"},
                json=body, timeout=30,
            )
        except Exception as e:
            log.warning("cerebras network error: %s", e)
            return None
        if r.status_code == 429:
            log.warning("cerebras key ...%s 429: %s",
                        key[-6:], r.text[:220].replace("\n", " "))
            cerebras_pool.record_call(key, success=False, quota=True)
            cerebras_pool.mark_exhausted(key)
            continue
        if not r.ok:
            log.warning("cerebras http %d: %s", r.status_code, r.text[:160])
            cerebras_pool.record_call(key, success=False)
            return None
        data = r.json()
        choices = data.get("choices") or []
        if not choices:
            log.warning("cerebras no choices: %s", str(data)[:300])
            cerebras_pool.record_call(key, success=False)
            return None
        msg = choices[0].get("message") or {}
        # gpt-oss-120b is a thinking model — may return content=None with
        # reasoning_content populated. Try content first, fall back to
        # reasoning_content so we don't silently discard valid output.
        text = msg.get("content") or msg.get("reasoning_content") or ""
        usage = data.get("usage", {})
        ptok = int(usage.get("prompt_tokens") or 0)
        ctok = int(usage.get("completion_tokens") or 0)
        if not text:
            finish = choices[0].get("finish_reason", "?")
            log.warning("cerebras empty content (finish_reason=%s) — skipping article", finish)
            cerebras_pool.record_call(key, success=False, prompt_tokens=ptok, comp_tokens=ctok)
            return None
        cerebras_pool.record_call(key, success=True, prompt_tokens=ptok, comp_tokens=ctok)
        return text.strip().strip("\"'")


def ollama_summarize(headline: str, raw_summary: str) -> str | None:
    """
    Polish via a LOCAL Ollama model (zero API quota). Only used when
    OLLAMA_MODEL is set — for free unlimited local prompt iteration.
    CPU inference is slow; expect this to take a while per article.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "user", "content": SUMMARY_PROMPT.format(headline=headline, summary=raw_summary)},
        ],
        "stream": False,
        "options": {"temperature": 0.3},
    }
    try:
        r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=300)
    except Exception as e:
        log.warning("ollama network error (is `ollama serve` running?): %s", e)
        return None
    if not r.ok:
        log.warning("ollama http %d: %s", r.status_code, r.text[:160])
        return None
    try:
        text = r.json().get("message", {}).get("content", "")
        return text.strip().strip("\"'") if text else None
    except Exception as e:
        log.warning("ollama parse failed: %s", e)
        return None


def polish_one(
    pool: GeminiKeyPool,
    headline: str,
    body: str,
    counters: dict[str, int],
    gemini_cap: int,
) -> tuple[str | None, str]:
    """
    Polish one article. Engine priority:
      • OLLAMA_MODEL set  → local Ollama ONLY (zero API quota; for testing)
      • else              → Cerebras → Gemini (under cap) → SambaNova → none
    `counters` is mutated in place.
    Returns (summary_or_None, engine).
    """
    # Local testing mode: route everything through the local model, no API.
    if OLLAMA_MODEL:
        txt = ollama_summarize(headline, body)
        counters["ollama_calls"] = counters.get("ollama_calls", 0) + 1
        return (txt, "ollama") if txt else (None, "none")

    # Production: Cerebras first (primary free-tier engine + strong Telugu).
    if CEREBRAS_API_KEY and cerebras_pool.has_capacity():
        txt = cerebras_summarize(headline, body)
        counters["cerebras_calls"] = counters.get("cerebras_calls", 0) + 1
        time.sleep(CEREBRAS_CALL_DELAY_S)
        if txt:
            return txt, "cerebras"

    if counters["gemini_calls"] < gemini_cap and pool.has_capacity():
        try:
            txt = gemini_summarize(pool, headline, body)
            counters["gemini_calls"] += 1
            time.sleep(GEMINI_CALL_DELAY_S)
            if txt:
                return txt, "gemini"
        except QuotaExhausted:
            log.warning("gemini pool exhausted — switching to SambaNova for the rest")

    # SambaNova: fallback of last resort (Cerebras exhausted + Gemini quota gone).
    if SAMBANOVA_API_KEY and sambanova_pool.has_capacity():
        txt = sambanova_summarize(headline, body)
        counters["sambanova_calls"] = counters.get("sambanova_calls", 0) + 1
        time.sleep(SAMBANOVA_CALL_DELAY_S)
        return (txt, "sambanova") if txt else (None, "none")
    return None, "none"



# ─── Firestore init ──────────────────────────────────────
def init_firestore() -> firestore.Client:
    sa_raw = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON")
    if not sa_raw:
        log.error("FIREBASE_SERVICE_ACCOUNT_JSON env var missing")
        sys.exit(1)
    try:
        sa = json.loads(sa_raw)
    except json.JSONDecodeError as e:
        log.error("FIREBASE_SERVICE_ACCOUNT_JSON is not valid JSON: %s", e)
        sys.exit(1)
    creds = service_account.Credentials.from_service_account_info(sa)
    return firestore.Client(
        project=sa["project_id"],
        credentials=creds,
        database=FIRESTORE_DB_ID,
    )


# YouTube video URL patterns. Covers:
#   https://www.youtube.com/watch?v=VIDEO_ID
#   https://youtu.be/VIDEO_ID
#   https://www.youtube.com/embed/VIDEO_ID
# Video IDs are always exactly 11 chars from the [A-Za-z0-9_-] set.
_YOUTUBE_VIDEO_RE = re.compile(r'(?:youtube\.com/(?:watch\?v=|embed/)|youtu\.be/)([A-Za-z0-9_-]{11})')

def _extract_youtube_id(url: str) -> str | None:
    """Extract the 11-char YouTube video ID from any standard YouTube URL,
    or None if the URL isn't a YouTube watch/embed/short link."""
    if not url:
        return None
    m = _YOUTUBE_VIDEO_RE.search(url)
    return m.group(1) if m else None


# "LIVE" prefix on video headlines means it's a live-stream archive —
# usually hours-long unwatched recordings (KCR press meet LIVE, etc.).
# Low news value, high bytes. We skip these. Examples caught:
#   "LIVE : KTR Press Meet"
#   "🔴 LIVE - Andhra Pradesh Assembly"
#   "Live: Cabinet Meeting Update"
# Only applied to video items — text articles labeled "LIVE" are usually
# legitimate live-blog posts which DO have news value.
_LIVE_PREFIX_RE = re.compile(
    r'^\s*(?:🔴\s*|⭕\s*|🟢\s*)?'             # optional status emoji
    r'(?:LIVE|Live|live)\s*[:|\-—•·]',         # LIVE + separator
)

def _is_live_video_headline(title: str) -> bool:
    """True if the title begins with LIVE prefix (case-insensitive)."""
    return bool(_LIVE_PREFIX_RE.match(title or ''))


# Visual junk that publishers / channels embed in titles + summaries for
# engagement: bullet arrows, decorative shapes, "attention" emojis. These
# render as garbage in gTTS audio (the engine narrates "bullet" or
# stumbles) and look unprofessional in article cards. Strip aggressively
# from ALL article text — Telugu characters and standard punctuation
# (including Telugu danda ।) are preserved.
_JUNK_CHARS_RE = re.compile(
    # Right/left arrows and bullets used as visual separators in RSS headlines
    r'[►➤▶▸▷◄◀◁❮❯❱❰'
    # Box/shape decorations
    r'❑▪◆◾⬛◼◻☐▬▭▮▯'
    # Stars, checkmarks, ticks used for engagement bait
    r'★☆☞✓✔✅✗✘❌✦✨☑'
    # Status circles/dots used as bullet alternatives
    r'⚡🔴🟢🟡🔵🟠⚪⚫⏺🔘🔲🔳'
    # Common emoji noise in news/social media posts
    r'🔥💥💡📺📱📲📢📣📌📍🔔💬🗣🚨🆕🆓🎬🎥'
    # Warning / info symbols
    r'⚠️ℹ'
    # Circled digit decorations ①-⑩
    r'①②③④⑤⑥⑦⑧⑨⑩'
    r']'
)

def _sanitize_summary(text: str) -> str:
    """Strip visual junk + normalize whitespace + trim wrapping
    punctuation. Safe on Telugu text — the regex above whitelists
    by attacking specific junk codepoints rather than blacklisting
    Telugu/Latin alphabets."""
    if not text:
        return text
    s = _JUNK_CHARS_RE.sub('', text)
    s = re.sub(r'\s+', ' ', s).strip(' \t\n\r-|:·')
    return s


# Sentence terminators recognised by _smart_truncate. Includes ASCII
# (./!/?), Telugu danda (।), and common smart-quote variants. Order
# doesn't matter; we look for the LAST occurrence of ANY of these.
_SENTENCE_TERMS = '.।!?"”’'

def _smart_truncate(text: str, max_chars: int) -> str:
    """Truncate `text` to at most max_chars characters, ending at a clean
    sentence boundary (terminator: . ।  ? ! " ' "smart quotes"). When no
    terminator is found within the last 40% of the window, falls back to
    the last whitespace + ellipsis. Never produces a mid-word cut.

    Used in parse_feed so the per-item summary always ends cleanly even
    when the publisher's RSS description is char-capped mid-sentence."""
    if not text or len(text) <= max_chars:
        return text
    window = text[:max_chars]
    cut = max(window.rfind(c) for c in _SENTENCE_TERMS)
    # Accept a terminator only if it's in the last 40% of the window —
    # otherwise we'd throw away too much content for the sake of a clean
    # cut. (40% of 800 chars = 320 chars, plenty.)
    if cut >= int(max_chars * 0.6):
        return window[:cut + 1].rstrip()
    # Fall back to last whitespace + ellipsis so reader sees the content
    # was truncated rather than thinking the sentence is malformed.
    cut = window.rfind(' ')
    if cut > 0:
        return window[:cut].rstrip(' ,;-|:·\t') + '…'
    # Last resort: hard cut. Should be rare (text with no whitespace
    # at all in the first max_chars chars).
    return window


# Markers that signal the start of YouTube channel boilerplate in a video
# description (Subscribe, social links, hashtags, URLs). We cut the body
# at the FIRST marker found — everything before is real lead content,
# everything after is channel promo. Mixed Telugu + English because
# Telugu news channels almost always paste English boilerplate.
_YT_BOILERPLATE_MARKERS = [
    '►', '➤', '▶',                          # arrow bullets used for nav lists
    'For more', 'For More', 'For Latest',
    'Subscribe to', 'subscribe to',
    'Visit Our Website', 'Visit our website',
    'Like us on', 'Follow us on', 'Follow Us',
    'Watch NTV', 'Watch ETV', 'Watch TV5', 'Watch our',
    'Download our', 'Download the',
    'Whatsapp Channel', 'WhatsApp Channel',
    'Stay tuned', 'Stay Tuned',
    'http://', 'https://',                  # any URL
    '#ntv', '#NTV', '#etv', '#ETV', '#tv5', '#TV5', '#brk', '#BRK',  # channel hashtag spam
]

def _clean_youtube_description(headline: str, body: str) -> str:
    """For a YouTube video item, return a clean lead from the description:
    drop the leading repeated-headline prefix (NTV/ETV usually start the
    description with the title verbatim), cut at the first boilerplate
    marker, and cap at the first ~2 sentences. Returns '' if what's left
    isn't worth showing — caller can fall back to the headline alone."""
    if not body:
        return ''
    txt = body.strip()
    # Strip leading "<headline>" prefix and common separators.
    if headline and txt.lower().startswith(headline.lower()):
        txt = txt[len(headline):].lstrip(' -|:·\n\t')
    # Cut at first boilerplate marker (whichever appears earliest).
    cut_at = len(txt)
    for marker in _YT_BOILERPLATE_MARKERS:
        i = txt.find(marker)
        if i >= 0 and i < cut_at:
            cut_at = i
    txt = txt[:cut_at].strip(' -|:·\n\t')
    if len(txt) < 20:
        return ''  # nothing useful left; caller uses headline
    # Cap at first ~2 sentences. Telugu sentence terminators: . ।  ? !
    parts = re.split(r'(?<=[.।?!])\s+', txt, maxsplit=2)
    return ' '.join(parts[:2]).strip()


# Shorts detection — YouTube's channel RSS feed includes Shorts mixed in
# with regular videos and there's no flag in the RSS itself to tell them
# apart. The reliable signal: a HEAD on /shorts/<id> returns 200 for a
# Short and 303/redirect for a regular video. Shorts are dropped at
# ingest (not rendered) because they don't fit the news-first thesis —
# vertical promo clips, low signal, often embed-disabled.
_YT_SHORT_CHECK_TIMEOUT = 10  # seconds — generous because false-negatives
                              # let Shorts slip through and pollute the feed
_YT_SHORT_RETRIES = 1         # one retry on transient connection errors

def _is_youtube_short(video_id: str) -> bool:
    """True if the YouTube video ID is a Short. Stricter than 'best-effort'
    — we'd rather drop an occasional regular video on a flaky connection
    than let Shorts pollute the feed (which is what user feedback showed
    when we tried rendering them). On HEAD success: 200 = Short. On any
    redirect (3xx) = regular video. On transient errors: retry once, then
    assume Short (precautionary). On hard errors (timeout etc.): return
    False so the video still ships if the network is broken on that run."""
    last_err = None
    for attempt in range(_YT_SHORT_RETRIES + 1):
        try:
            r = requests.head(
                f"https://www.youtube.com/shorts/{video_id}",
                timeout=_YT_SHORT_CHECK_TIMEOUT,
                allow_redirects=False,
                headers={'User-Agent': 'curl/7.81.0'},
            )
            # 200 = Shorts page exists. 3xx (redirect to /watch?v=) = regular.
            # 404 = unknown video, treat as regular (probably a deleted Short
            # or odd state — let it through and the iframe will show YT's
            # standard error if it really doesn't exist).
            return r.status_code == 200
        except (requests.ConnectionError, requests.Timeout) as e:
            last_err = e
            log.debug('shorts check attempt %d failed for %s: %s',
                      attempt + 1, video_id, type(e).__name__)
            continue
        except Exception as e:
            log.debug('shorts check error for %s: %s', video_id, e)
            return False  # unknown error → assume regular video, ship it
    log.warning('shorts check gave up after %d tries for %s (%s) — assuming regular',
                _YT_SHORT_RETRIES + 1, video_id, type(last_err).__name__ if last_err else 'unknown')
    return False


# ─── RSS → Article ───────────────────────────────────────
@dataclass
class Article:
    id: str
    headline: str
    summary: str
    category: str
    mandal: str
    village: str | None
    source: str
    sources: list[str]
    link: str
    published_at: datetime
    audio_sec: int
    color: str
    lang: str
    image: str | None = None
    body: str = ""
    ai: bool = False
    needs_resolve: bool = False  # Google News: decode redirect + scrape og
    # When set, this article is a YouTube video — client renders an iframe
    # embed instead of gTTS audio. Thumbnail URL is derived in parse_feed.
    video_id: str | None = None
    # True if the video is a YouTube Short (9:16 vertical). Client uses
    # this to pick the iframe aspect ratio and show a "Short" badge
    # instead of the regular "video" badge.
    is_short: bool = False
    reporterId: str | None = None  # Reporter whitelist ID (e.g. rp_919154619599)
    reporter: dict | None = None  # Inlined reporter byline {name, village, mandal, title, avatar, phone_share}
    origin: str = "rss"  # "rss" | "whatsapp" | "telegram" (origin of the article)
    level: str = "district"  # village | mandal | district | state | national


def _best_body_text(entry) -> str:
    """
    Return the longest body text we can find in this RSS entry.
    Some feeds put the article body in <content:encoded> (feedparser: entry.content[0].value),
    others put it in <description> or <summary>. We take the longest.
    """
    candidates: list[str] = []
    # feedparser exposes <content:encoded> as entry.content (list of dicts)
    try:
        for c in (getattr(entry, "content", None) or []):
            v = c.get("value") if isinstance(c, dict) else getattr(c, "value", None)
            if v:
                candidates.append(strip_html(v))
    except Exception:
        pass
    for attr in ("summary", "description", "subtitle"):
        v = getattr(entry, attr, "") or ""
        if v:
            candidates.append(strip_html(v))
    if not candidates:
        return ""
    return max(candidates, key=len)


def _httpsify(url: str | None) -> str | None:
    """
    Force https. The app runs on an https WebView (androidScheme: https);
    http images are blocked as mixed content and silently fail to load.
    Most CDNs (Sakshi, The Hindu) serve the same path over https.
    """
    if not url:
        return None
    if url.startswith("http://"):
        return "https://" + url[len("http://"):]
    return url


def _best_image_url(entry) -> str | None:
    """
    Find a usable thumbnail/hero image for the article. RSS feeds expose
    images in several places; try them in order of reliability.
    """
    # 1) media:content (Sakshi, many Indian feeds)
    try:
        for m in (getattr(entry, "media_content", None) or []):
            url = m.get("url") if isinstance(m, dict) else None
            if url and url.startswith("http"):
                return url
    except Exception:
        pass
    # 2) media:thumbnail
    try:
        for m in (getattr(entry, "media_thumbnail", None) or []):
            url = m.get("url") if isinstance(m, dict) else None
            if url and url.startswith("http"):
                return url
    except Exception:
        pass
    # 3) enclosure links with an image type (The Hindu)
    try:
        for l in (getattr(entry, "links", None) or []):
            if l.get("rel") == "enclosure" and str(l.get("type", "")).startswith("image") and l.get("href"):
                return l["href"]
    except Exception:
        pass
    # 4) first <img> inside the HTML body (BBC Telugu etc.)
    try:
        for c in (getattr(entry, "content", None) or []):
            v = c.get("value") if isinstance(c, dict) else None
            if v:
                m = re.search(r'<img[^>]+src="([^"]+)"', v)
                if m and m.group(1).startswith("http"):
                    return m.group(1)
        for attr in ("summary", "description"):
            v = getattr(entry, attr, "") or ""
            m = re.search(r'<img[^>]+src="([^"]+)"', v)
            if m and m.group(1).startswith("http"):
                return m.group(1)
    except Exception:
        pass
    return None


# ─── Google News redirect decode + publisher scrape ──────
# Google News RSS gives a non-resolvable /rss/articles/CBMi… redirect,
# a run-on aggregated description, and NO image. We decode the redirect
# via Google's internal batchexecute endpoint to the real publisher URL,
# then scrape og:image + og:description so these articles get a real
# image and a real (distinct from the headline) summary. Best-effort:
# ANY failure leaves the article untouched (no regression).
_GN_SESSION = None


def _gn_session():
    global _GN_SESSION
    if _GN_SESSION is None:
        s = requests.Session()
        s.headers.update({"User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        )})
        _GN_SESSION = s
    return _GN_SESSION


def _resolve_gnews_url(url: str) -> str | None:
    """Decode a news.google.com/rss/articles/… link to the real URL."""
    m = re.search(r"/articles/([^?]+)", url or "")
    if not m:
        return None
    art = m.group(1)
    s = _gn_session()
    p = s.get(url, timeout=12)
    sg = re.search(r'data-n-a-sg="([^"]+)"', p.text)
    ts = re.search(r'data-n-a-ts="([^"]+)"', p.text)
    if not (sg and ts):
        return None
    req = ('["garturlreq",[["X","X",["X","X"],null,null,1,1,"US:en",'
           'null,1,null,null,null,null,null,0,1],"X","X",1,[1,1,1],1,1,'
           'null,0,0,null,0],"' + art + '",' + ts.group(1) + ',"'
           + sg.group(1) + '"]')
    r = s.post("https://news.google.com/_/DotsSplashUi/data/batchexecute",
               data={"f.req": json.dumps([[["Fbv4je", req]]])}, timeout=12)
    if "garturlres" not in r.text:
        return None
    u = re.search(r'(https?://[^\\"]+)', r.text.split("garturlres")[1])
    return u.group(1) if u else None


def _scrape_og(html_text: str) -> tuple[str | None, str | None]:
    og = re.search(
        r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)',
        html_text, re.I)
    de = re.search(
        r'<meta[^>]+(?:property=["\']og:description["\']|'
        r'name=["\']description["\'])[^>]+content=["\']([^"\']+)',
        html_text, re.I)
    img = html.unescape(og.group(1)).strip() if og else None
    desc = html.unescape(de.group(1)).strip() if de else None
    return img, desc


def enrich_gnews(a: "Article") -> bool:
    """Resolve a Google News article to its publisher page and fill in
    real link + image + summary. Returns True if it enriched anything.
    Never raises."""
    try:
        real = _resolve_gnews_url(a.link)
        if not real:
            return False
        a.link = real
        r = _gn_session().get(real, timeout=12, allow_redirects=True)
        if r.status_code != 200:
            return False
        # requests defaults to ISO-8859-1 when the page sends no charset
        # header → Telugu UTF-8 bytes become mojibake. Indian news pages
        # are UTF-8; decode the raw bytes as UTF-8 explicitly (fall back
        # to requests' detected encoding only if the header declared one).
        ctype = r.headers.get("Content-Type", "").lower()
        if "charset=" in ctype:
            html_text = r.text
        else:
            html_text = r.content.decode("utf-8", errors="ignore")
        img, desc = _scrape_og(html_text)
        changed = False
        if img and img.startswith("http"):
            a.image = _httpsify(img)
            changed = True
        if desc and len(desc) >= 40:
            a.summary = desc
            a.body = desc
            a.audio_sec = audio_seconds_estimate(desc)
            changed = True
        return changed
    except Exception as e:  # noqa: BLE001 — best-effort, never break ingest
        log.debug("gnews enrich skipped: %s", e)
        return False


def parse_feed(feed: dict[str, str]) -> list[Article]:
    parsed = feedparser.parse(feed["url"])
    items: list[Article] = []
    is_gnews = "news.google.com" in (feed.get("url", "") or "")
    # YouTube channel feeds get a tighter per-cron cap so a single channel
    # can't drown out text news on a busy publishing day. See
    # MAX_PER_VIDEO_FEED constant.
    is_video_feed = "youtube.com/feeds" in (feed.get("url", "") or "")
    cap = MAX_PER_VIDEO_FEED if is_video_feed else MAX_PER_FEED
    for entry in (parsed.entries or [])[:cap]:
        headline = strip_html(getattr(entry, "title", ""))
        if not headline:
            continue
        if is_gnews and " - " in headline:
            # Google News title = "Real Headline - Publisher". Drop the
            # publisher suffix (short tail) so the headline is clean; the
            # real summary comes from the scraped publisher page.
            base, _, publisher = headline.rpartition(" - ")
            if base.strip() and len(publisher) <= 40:
                headline = base.strip()
        # Use the richest body text available, cap at 800 chars (Gemini
        # handles plenty). Google News descriptions are kept as before —
        # strip_html now decodes the &nbsp;/&amp; entities that were the
        # only real problem; the content itself stays.
        # Sanitize once at the source — strip visual junk (►/▶/🔥/etc.)
        # and normalize whitespace. Every downstream slice + truncation
        # operates on already-clean text, so junk can't sneak through any
        # specific code path.
        body = _sanitize_summary(_best_body_text(entry))
        # Smart-truncate the lead so summaries always end at a sentence
        # boundary — was a hard body[:800] before, which routinely cut
        # mid-sentence and produced ugly dangling text. (User feedback.)
        raw_summary = _smart_truncate(body, 800)
        image = _httpsify(_best_image_url(entry))
        link = getattr(entry, "link", "") or ""
        # YouTube detection — if the entry link is a YouTube video URL,
        # extract the video ID and use the high-quality thumbnail as the
        # article image. The client renders these as a tap-to-embed iframe.
        video_id = _extract_youtube_id(link)
        is_short = False
        if video_id:
            # Skip live-stream archives. NTV/ETV stream press meets and
            # assembly sessions as "LIVE :" videos that archive forever —
            # hours-long recordings nobody watches end-to-end.
            if _is_live_video_headline(headline):
                log.info("skip LIVE video: %s", headline[:60])
                continue
            # Drop YouTube Shorts at ingest. Tried rendering them in 9:16
            # (commit 7bc847e) but user feedback was that embed-disabled
            # Shorts show as broken "Video unavailable" frames and the
            # vertical-format clutters a news feed. Reverted to drop.
            if _is_youtube_short(video_id):
                log.info("skip YouTube Short: %s", headline[:60])
                continue
            # Override image with the YouTube thumbnail. hqdefault is
            # 480x360 — small enough to be cheap on rural data plans but
            # crisp enough on phone screens.
            image = f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"
            # Clean the video description — strip channel boilerplate
            # (Subscribe/Visit/Follow/hashtags/URLs) and keep only the
            # first 1-2 sentences of real lead content. Fall back to the
            # headline alone if nothing useful is left (typical for NTV
            # videos whose descriptions are 100% promo links).
            cleaned = _clean_youtube_description(headline, body)
            raw_summary = cleaned or headline
            body = cleaned or headline
        text = f"{headline} {raw_summary}"
        category = _classify_article(headline, raw_summary or "", url=link)
        mandal, village = detect_mandal(text)
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
        else:
            published = datetime.now(timezone.utc)
        # Dedup on normalized headline only (not timestamp) so the same story
        # appearing across overlapping feeds collapses to one article.
        # Skip junk-content check for video items — YouTube RSS entries
        # don't have the PDF-iframe failure patterns that _is_junk_content
        # is hunting for, and the body is just a video description anyway.
        if not video_id and _is_junk_content(raw_summary, image):
            log.info("skip junk article: %s", headline[:60])
            continue
        aid = hash_id(dedup_key(headline))
        items.append(Article(
            id=aid,
            headline=headline,
            summary=raw_summary or headline,
            category=category,
            mandal=mandal,
            village=village,
            source=feed["source"],
            sources=[feed["source"]],
            link=link,
            published_at=published,
            audio_sec=audio_seconds_estimate(raw_summary or headline),
            color=COLOR_BY_CAT.get(category, "#7C2D12"),
            lang=feed.get("lang", "te"),
            image=image,
            body=_smart_truncate(body, 2500),
            needs_resolve=is_gnews,
            video_id=video_id,
            is_short=is_short,
            level=_determine_level(
                mandal, village, feed["source"],
                headline, raw_summary or "", None,
            ),
        ))
    return items


# ─── Main ────────────────────────────────────────────────
def load_gemini_keys() -> list[str]:
    """
    Accepts either GEMINI_API_KEYS (comma-separated, preferred) or
    GEMINI_API_KEY (single, backwards compatible).
    """
    plural = os.environ.get("GEMINI_API_KEYS", "")
    if plural.strip():
        return [k.strip() for k in plural.split(",") if k.strip()]
    single = os.environ.get("GEMINI_API_KEY", "").strip()
    return [single] if single else []


def main() -> int:
    keys = load_gemini_keys()
    # At least ONE engine must be configured (Ollama / Cerebras / Gemini / SambaNova).
    if not (keys or CEREBRAS_API_KEY or SAMBANOVA_API_KEY or OLLAMA_MODEL):
        log.error("No AI engine configured — set one of OLLAMA_MODEL, "
                  "CEREBRAS_API_KEYS, GEMINI_API_KEYS, or SAMBANOVA_API_KEYS")
        return 1
    pool = GeminiKeyPool(keys)
    if OLLAMA_MODEL:
        log.info("engine: LOCAL Ollama '%s' (no API quota used)", OLLAMA_MODEL)
    else:
        log.info("engines: cerebras=%d gemini=%d sambanova=%d key(s)",
                 len(CEREBRAS_KEYS), len(keys), len(SAMBANOVA_KEYS))

    # Decision 3: article store is R2 JSON (articles.json), NOT Firestore.
    # Load once at startup, mutate entirely in memory throughout the run,
    # write back as a single PUT at the end. Zero Firestore reads/writes on
    # the article path → free quota never throttles the cron.
    # Firestore is only used for client-write collections: fcm_tokens,
    # community_reports, live_data (weather/mandi) — all tiny.
    store: list[dict] = []
    seen_ids: set[str] = set()
    blocked_ids = load_blocklist()
    if DRY_RUN:
        db = None
        log.info("DRY_RUN=1 — no Firestore/R2 store. Output → dry_run_output.csv")
    else:
        db = init_firestore()
        store = article_store.load_articles()
        # Purge any blocked articles already in the working set so they're
        # removed from articles.json on this run's save.
        if blocked_ids:
            before = len(store)
            store = [a for a in store if a.get("id") not in blocked_ids]
            purged = before - len(store)
            if purged:
                log.info("blocklist purged %d article(s) from store", purged)
        # Backfill cleanup pass — sanitizes ALL article summaries (strip
        # junk chars like ►/🔥), drops Shorts and LIVE-stream videos that
        # slipped past earlier filters. Cheap on a clean store; the
        # video-side HEAD checks only fire for unclassified items.
        cleanup_existing_articles(store)
        seen_ids = {a.get("id") for a in store if a.get("id")}
        # Add blocked IDs to seen_ids so they are never re-fetched from RSS.
        seen_ids.update(blocked_ids)
        log.info("article store loaded — %d existing articles", len(store))

    dry_rows: list[dict] = []
    fetched = 0
    new_count = 0
    new_articles_list: list[dict] = []   # track new dicts for FCM (not just count)
    summarized = 0
    gnews_enriched = 0
    feeds_ok = 0
    feeds_failed = 0
    # Per-level AI quota tracking: village/mandal/district get unlimited;
    # state and national are capped per run so local news always gets quota.
    level_ai_counts: dict[str, int] = {"national": 0, "state": 0}
    counters = {
        "gemini_calls": 0, "sambanova_calls": 0,
        # AI engine success counts (tracked in article loop)
        "cerebras_ok": 0, "gemini_ok": 0, "sambanova_ok": 0,
        # Article skip reasons
        "skip_english": 0, "skip_te_rich": 0, "skip_video": 0,
        "skip_level_cap": 0, "ai_fail": 0,
    }

    for feed in FEEDS:
        try:
            items = parse_feed(feed)
            feeds_ok += 1
        except Exception as e:
            log.warning("feed failed %s: %s", feed["url"], e)
            feeds_failed += 1
            continue
        log.info("feed=%-30s items=%d", feed["source"], len(items))
        fetched += len(items)

        if INGEST_LIMIT and new_count >= INGEST_LIMIT:
            log.info("LIMIT=%d reached — stopping early (test mode)", INGEST_LIMIT)
            break

        for a in items:
            if INGEST_LIMIT and new_count >= INGEST_LIMIT:
                break
            if not DRY_RUN and a.id in seen_ids:
                continue  # in-memory dedup — 0 Firestore reads (Decision 3)
            # Google News: decode redirect → publisher page, scrape real
            # image + summary. Only for genuinely-new articles (after the
            # dedup skip above) and capped per run so the extra HTTP calls
            # stay within the cron budget. Best-effort: failure = no-op.
            if a.needs_resolve and gnews_enriched < GNEWS_RESOLVE_MAX:
                if enrich_gnews(a):
                    gnews_enriched += 1
            # Skip English-source articles on regular cron unless ENGLISH_POLISH=1
            # (or BACKFILL=1) — they cost more (translation + rewrite).
            polished: str | None = None
            headline_out = a.headline
            summary_out = a.summary
            skip_english = (a.lang == "en") and not POLISH_ENGLISH
            # Skip AI when the RSS feed already delivered a full Telugu summary
            # (≥60 Telugu words). Native feeds like Sakshi/BBC Telugu/NTV often
            # provide 80-120 word descriptions — polishing them wastes quota and
            # rarely improves quality for already-fluent Telugu content.
            # IMPORTANT: these articles are marked ai=True below even though
            # no AI ran — the content is already publication-quality Telugu,
            # so they deserve audio narration and full feed treatment.
            skip_te_rich = (a.lang == "te") and (_te_word_count(a.summary) >= 60)
            # YouTube items: headline + description are already the publisher's
            # own copy; no value in AI-rewriting it. Saves quota.
            skip_video = a.video_id is not None
            # Level-based AI cap: local news (village/mandal/district) gets
            # unlimited quota; state/national are capped per run so they
            # don't crowd out the hyper-local stories that matter most.
            # a.level is already set from source-map + keyword pre-analysis.
            skip_level_cap = False
            if NATIONAL_AI_MAX > 0 and a.level == "national":
                if level_ai_counts["national"] >= NATIONAL_AI_MAX:
                    skip_level_cap = True
                    counters["skip_level_cap"] = counters.get("skip_level_cap", 0) + 1
            elif STATE_AI_MAX > 0 and a.level == "state":
                if level_ai_counts["state"] >= STATE_AI_MAX:
                    skip_level_cap = True
                    counters["skip_level_cap"] = counters.get("skip_level_cap", 0) + 1
            if not skip_english and not skip_te_rich and not skip_video and not skip_level_cap:
                polished, _engine = polish_one(
                    pool, a.headline, a.summary, counters, MAX_TOTAL_GEMINI
                )
                if polished and len(polished) > 20:
                    p_title, p_body, p_cat, p_level = _split_polished(polished)
                    if p_body:
                        summary_out = p_body
                    # Use the translated headline when the original is English
                    # (or always, if we got a clean Telugu title).
                    if p_title and (a.lang == "en" or _looks_english(a.headline)):
                        headline_out = p_title
                    # Refine level using AI result (AI has full article context).
                    a.level = _determine_level(
                        a.mandal, a.village, a.source,
                        headline_out, summary_out, p_level,
                    )
                    # Track per-level AI usage against the caps
                    if a.level in level_ai_counts:
                        level_ai_counts[a.level] += 1
                    # Category — three-tier cascade on polished text.
                    # Polished headline+summary is clean Telugu — ideal for
                    # both the sklearn model and the keyword fallback.
                    category = _classify_article(
                        headline_out, summary_out, p_cat, url=a.link
                    )
                    summarized += 1
                    # Track which engine succeeded
                    if _engine == "cerebras":
                        counters["cerebras_ok"] += 1
                    elif _engine == "gemini":
                        counters["gemini_ok"] += 1
                    elif _engine == "sambanova":
                        counters["sambanova_ok"] += 1
                else:
                    counters["ai_fail"] += 1
                    polished = None
            else:
                if skip_english:
                    counters["skip_english"] += 1
                elif skip_te_rich:
                    counters["skip_te_rich"] += 1
                elif skip_video:
                    counters["skip_video"] += 1

            # For native-rich Telugu: no AI ran, but sanitize the raw RSS text
            # to strip junk chars (►, 🔥, ★ etc.) before storing.
            if skip_te_rich:
                headline_out = _sanitize_summary(headline_out)
                summary_out = _sanitize_summary(summary_out)

            if DRY_RUN:
                # No Firestore — collect for the local CSV instead.
                dry_rows.append({
                    "engine": _engine if (not skip_english and not skip_te_rich and not skip_video) else ("skipped-video" if skip_video else ("skipped-en" if skip_english else "skipped-te-rich")),
                    "source": a.source,
                    "orig_headline": a.headline,
                    "new_headline": headline_out,
                    "summary": summary_out,
                    "category": a.category,
                    "mandal": a.mandal,
                    "lang": a.lang,
                })
                new_count += 1
                summarized = summarized  # unchanged; counted above
                continue

            store.append({
                "id": a.id,
                "headline": headline_out,
                "summary": summary_out,
                "audioScript": summary_out,
                "category": a.category,
                "mandal": a.mandal,
                "village": a.village,
                "source": a.source,
                "sources": a.sources,
                "link": a.link,
                "publishedAt": (a.published_at.isoformat()
                                if a.published_at else _now_iso()),
                "createdAt": _now_iso(),
                "audioUrl": None,
                "audioSec": a.audio_sec,
                "color": a.color,
                "shares": 0,
                # ai=True if AI polished the text, OR if the source already
                # delivered publication-quality native Telugu (skip_te_rich).
                # Both cases earn audio narration + full feed treatment.
                # skip_video and skip_english stay ai=False — they need
                # translation or are just video embeds.
                "ai": bool(polished) or skip_te_rich,
                "lang": a.lang,
                "image": a.image,
                "body": a.body,
                # videoId is None for normal text articles, an 11-char
                # YouTube video ID for items from the YouTube RSS feeds.
                # The client renders these as a tap-to-embed iframe.
                "videoId": a.video_id,
                # True for YouTube Shorts — currently filtered out at
                # ingest, kept as metadata in case of future revival.
                "isShort": a.is_short,
                # Editorial / advertising fields. Cron NEVER sets these —
                # they're only flipped by the admin tool via R2 sync. Cron
                # preserves any existing values on re-ingest (see the
                # passthrough logic in main()'s store-merge step). Auto-
                # expiry happens in cleanup_existing_articles when the
                # corresponding *Until timestamp passes.
                "featured": False,
                "featuredUntil": None,
                "sponsored": False,
                "sponsoredUntil": None,
                "sponsoredAdvertiser": "",
                # Reporter fields: only set by WhatsApp ingest, never by cron.
                # reporterId = rp_<phone_digits>, reporter = inlined byline dict.
                # origin = "rss" for cron, "whatsapp" for manual ingest.
                "reporterId": a.reporterId,
                "reporter": a.reporter,
                "origin": a.origin,
                "level": a.level,
            })
            new_articles_list.append(store[-1])   # track for FCM
            seen_ids.add(a.id)
            new_count += 1

    log.info(
        "ingest done — fetched=%d new=%d summarized=%d gnews_enriched=%d "
        "gemini_calls=%d sambanova_calls=%d keys_dead=%d/%d sambanova=%s",
        fetched, new_count, summarized, gnews_enriched,
        counters["gemini_calls"], counters.get("sambanova_calls", 0),
        len(pool.exhausted), len(pool.keys),
        "on" if SAMBANOVA_API_KEY else "off",
    )

    if DRY_RUN:
        import csv as _csv
        out_path = "dry_run_output.csv"
        cols = ["engine", "source", "category", "mandal", "lang",
                "orig_headline", "new_headline", "summary"]
        with open(out_path, "w", newline="", encoding="utf-8-sig") as fh:
            w = _csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in dry_rows:
                w.writerow({k: r.get(k, "") for k in cols})
        log.info("DRY_RUN complete — %d article(s) → %s (no Firestore, no quota)",
                 len(dry_rows), out_path)
        return 0

    # ─── Backfill: re-polish existing articles that are ai=False ─
    # Run manually (BACKFILL=1) when Gemini quota is fresh. Uses a higher
    # Gemini cap then falls through to Groq, so it can finish even a large
    # backlog in one pass without dropping to raw RSS.
    if os.environ.get("BACKFILL") == "1":
        backfilled = backfill_unpolished(store, pool, counters)
        log.info("backfill done — polished=%d existing articles", backfilled)

    # ─── Village Reporter: approved community reports → feed ──
    try:
        vr = promote_community_reports(db, store, pool, counters)
        if vr:
            log.info("village reporter — promoted=%d report(s) to feed", vr)
    except Exception as e:
        log.warning("village reporter promote error: %s", e)

    # ─── Audio: narrate Telugu summaries → R2 (capped per run) ─
    audio_stats: dict = {"voiced": 0, "failed": 0, "skipped": 0, "candidates": 0}
    feed_visible_ids: set = set()
    try:
        # Compute the set of article IDs that will appear in feed.json so
        # audio synthesis focuses only on feed-visible content. Articles
        # excluded from the feed (old, over-cap, non-Telugu) are skipped.
        feed_visible_ids = _feed_article_ids(store)
        log.info("audio: %d feed-visible article(s) eligible for narration", len(feed_visible_ids))
        audio_stats = synthesize_pending_audio(store, AUDIO_MAX_PER_RUN, feed_ids=feed_visible_ids)
        if audio_stats["voiced"]:
            log.info("audio done — narrated=%d article(s) → R2", audio_stats["voiced"])
    except Exception as e:
        log.warning("audio step error: %s", e)

    # ─── Live data: weather + mandi prices ───────────────
    try:
        update_live_data(db)
    except Exception as e:
        log.warning("live_data update error: %s", e)

    # ─── Cleanup: drop articles older than RETENTION_DAYS ──
    store, deleted = prune_old_articles(store)
    log.info("cleanup done — pruned=%d articles + audio/images from R2 (older than %d days)", deleted, RETENTION_DAYS)

    # ─── Persist working set + publish public feed → R2 ─────
    # Decision 3: the article store IS R2 JSON (no Firestore in this path).
    # Save the full working set, then publish the Telugu-only public view.
    # Zero Firestore reads here → the free quota can never throttle the cron.
    try:
        article_store.save_articles(store)
    except Exception as e:
        log.warning("article store save error: %s", e)
    try:
        _update_training_data(store)
    except Exception as e:
        log.warning("training data update error: %s", e)
    try:
        archive_daily_feed_if_needed(store)
    except Exception as e:
        log.warning("archive error: %s", e)
    try:
        export_feed_to_r2(store)
    except Exception as e:
        log.warning("feed export error: %s", e)

    # ─── FCM push notifications (after R2 export so article is live) ─────
    # Must run after export_feed_to_r2 so the notified article is already in
    # feed.json when the user taps the notification. Sending before export
    # caused tryOpen to fail (article not in allArticles yet) → fell back to
    # opening article index 0 instead of the notified article.
    try:
        sa_info = json.loads(os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON", "{}"))
        notify_new_articles(db, store, new_articles_list, sa_info)
    except Exception as e:
        log.warning("fcm notify error: %s", e)

    # ─── Structured run report (captured in email) ────────
    feed_published = len([d for d in store if _is_telugu_feed_item(d)])
    audio_pending = sum(
        1 for d in store
        if not d.get("audioUrl")
        and d.get("ai") is True
        and d.get("id") in feed_visible_ids
        and not d.get("videoId")
        and not _looks_english((d.get("summary") or "").strip())
        and len((d.get("summary") or "").strip()) >= 10
    )
    _print_run_report(
        feeds_ok=feeds_ok, feeds_failed=feeds_failed,
        fetched=fetched, new_count=new_count,
        gnews_enriched=gnews_enriched, blocked_count=len(blocked_ids),
        counters=counters, summarized=summarized,
        audio_stats=audio_stats, audio_pending=audio_pending,
        feed_visible=len(feed_visible_ids),
        store_total=len(store), feed_published=feed_published,
        pruned=deleted, retention_days=RETENTION_DAYS,
        gemini_pool=pool, cerebras_pool=cerebras_pool,
        sambanova_pool=sambanova_pool,
    )
    return 0


def _is_telugu_feed_item(d: dict) -> bool:
    """
    Public feed is Telugu-ONLY (core product vision: a rural Telugu user
    must never see an English/Hindi article). Mirrors the client's
    news-store.js `keepTeluguReadable`: the article must be polished/Telugu
    (`ai is True` or `lang == "te"`) AND its headline must actually contain
    Telugu script (catches English stubs that slipped through unpolished).
    Also drops articles whose content is a PDF/iframe embed error.
    """
    if not (d.get("ai") is True or d.get("lang") == "te"):
        return False
    headline = d.get("headline") or ""
    if not any("ఀ" <= ch <= "౿" for ch in headline):
        return False
    summary = d.get("summary") or ""
    image = d.get("image") or ""
    if _is_junk_content(summary, image):
        return False
    return True


# ─── Feed category priority tiers ────────────────────────
# Tier 1 (important, FCM-eligible): all news categories that matter to readers.
# Tier 2 (entertainment / lifestyle): cinema, sports, health — no FCM push.
# Unknown categories default to Tier 1 (safer to surface unknown than suppress).
_FEED_TIER: dict[str, int] = {
    "politics": 1, "schemes": 1,
    "farming": 1, "weather": 1, "jobs": 1, "village": 1, "general": 1,
    "health": 2, "sports": 2, "cinema": 2,
}


def _pub_dt(d: dict):
    """Parse publishedAt (or createdAt fallback) → aware datetime, or None."""
    raw = d.get("publishedAt") or d.get("createdAt") or ""
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        return None


def _after_cutoff(d: dict, cutoff_dt) -> bool:
    """True if article was published at or after cutoff_dt."""
    dt = _pub_dt(d)
    return dt is not None and dt >= cutoff_dt


def _format_article_for_feed(d: dict) -> dict:
    """Produce the public-facing dict for one article (same shape as feed.json)."""
    return {
        "id": d.get("id", ""),
        "headline": d.get("headline", ""),
        "summary": d.get("summary", ""),
        "category": d.get("category", "general"),
        "mandal": d.get("mandal", "all"),
        "village": d.get("village", ""),
        "source": d.get("source", ""),
        "link": d.get("link", ""),
        "level": d.get("level") or _determine_level(
            d.get("mandal", "all"), d.get("village"), d.get("source", ""),
            d.get("headline", ""), d.get("summary", ""), None,
        ),
        "image": d.get("image", ""),
        "audioUrl": d.get("audioUrl"),
        "audioSec": d.get("audioSec", 60),
        "color": d.get("color", "#7C2D12"),
        "ai": bool(d.get("ai", False)),
        "publishedAt": d.get("publishedAt"),
        "videoId": d.get("videoId"),
        "videoUrl": d.get("videoUrl"),
        "isShort": bool(d.get("isShort")),
        "featured": bool(d.get("featured")),
        "featuredUntil": d.get("featuredUntil"),
        "sponsored": bool(d.get("sponsored")),
        "sponsoredUntil": d.get("sponsoredUntil"),
        "sponsoredAdvertiser": d.get("sponsoredAdvertiser") or "",
    }


def _build_feed_combined(store: list[dict]) -> tuple[list[dict], dict]:
    """Core feed-selection logic shared by export_feed_to_r2() and _feed_article_ids().

    Returns (combined_list, stats_dict).

    Algorithm:
      1. Filter to Telugu-readable articles only.
      2. Separate into Tier 1 (important) and Tier 2 (entertainment).
         Cinema within Tier 2 is capped at CINEMA_MAX to prevent flooding.
      3. If combined total is below FEED_MIN, pull overflow to pad up.
      4. Hard-cap at FEED_MAX.
      5. Sort by time-bucket + tier so within each 1-hour window Tier 1
         articles appear before Tier 2, but a fresh Tier 2 beats a stale Tier 1.
    """
    telugu = [d for d in store if _is_telugu_feed_item(d)]

    def _recency(d: dict) -> str:
        return d.get("publishedAt") or d.get("createdAt") or ""

    tier1_all = sorted(
        [d for d in telugu if _FEED_TIER.get(d.get("category") or "general", 1) == 1],
        key=_recency, reverse=True,
    )
    cinema_all = sorted(
        [d for d in telugu if d.get("category") == "cinema"],
        key=_recency, reverse=True,
    )
    tier2_other_all = sorted(
        [d for d in telugu if _FEED_TIER.get(d.get("category") or "general", 1) == 2
         and d.get("category") != "cinema"],
        key=_recency, reverse=True,
    )

    cinema = cinema_all[:CINEMA_MAX] if CINEMA_MAX > 0 else cinema_all
    combined = tier1_all + tier2_other_all + cinema

    # ── FEED_MIN fill ─────────────────────────────────────────────────────
    target = min(FEED_MIN, FEED_MAX)
    if FEED_MIN > 0 and len(combined) < target:
        combined_ids = {d["id"] for d in combined}
        overflow = [
            d for d in (tier1_all + tier2_other_all + cinema_all)
            if d["id"] not in combined_ids
        ]
        needed = target - len(combined)
        combined = combined + overflow[:needed]
        if overflow[:needed]:
            log.info(
                "feed: below FEED_MIN=%d — added %d overflow article(s) to reach %d",
                FEED_MIN, min(needed, len(overflow)), len(combined),
            )

    combined = combined[:FEED_MAX]

    # ── Time-bucket + tier sort ───────────────────────────────────────────
    # Key = (bucket, tier, -timestamp).
    # bucket 0 = most recent FEED_BUCKET_HOURS window.
    # Within a bucket: Tier 1 (important) before Tier 2 (entertainment).
    # Across buckets: fresh Tier 2 beats stale Tier 1.
    now_ts = datetime.now(timezone.utc).timestamp()

    def _bucket_key(d: dict) -> tuple:
        pub = d.get("publishedAt") or d.get("createdAt") or ""
        try:
            ts = datetime.fromisoformat(pub.replace("Z", "+00:00")).timestamp()
        except Exception:
            ts = 0.0
        hours_ago = max(0.0, (now_ts - ts) / 3600)
        bucket = int(hours_ago / FEED_BUCKET_HOURS)
        tier   = _FEED_TIER.get(d.get("category") or "general", 1)
        return (bucket, tier, -ts)

    combined.sort(key=_bucket_key)

    stats = {
        "tier1": len(tier1_all),
        "tier2_other": len(tier2_other_all),
        "cinema": len(cinema),
        "skipped_non_te": len(store) - len(telugu),
    }
    return combined, stats


def _midnight_ist_cutoff() -> datetime:
    """Return today's midnight IST as a UTC-aware datetime.

    The live feed shows articles published since midnight IST today. This gives
    a predictable, date-aligned window: users always see today's news, not a
    rolling 24-hour slice that shifts by the hour.
    """
    now_ist = datetime.now(timezone.utc).astimezone(IST)
    midnight_ist = now_ist.replace(hour=0, minute=0, second=0, microsecond=0)
    return midnight_ist.astimezone(timezone.utc)


def _feed_article_ids(store: list[dict]) -> set:
    """Return the set of article IDs that would appear in feed.json.

    Uses _build_feed_combined() so audio synthesis targets exactly the same
    articles that will be published. No R2 call.
    """
    cutoff = _midnight_ist_cutoff()
    recent = [d for d in store if _after_cutoff(d, cutoff)]
    combined, _ = _build_feed_combined(recent)
    return {d["id"] for d in combined if d.get("id")}


def export_feed_to_r2(store: list[dict]) -> None:
    """Publish the Telugu-only public feed.json to R2 from the in-memory
    working set (Decision 3 — NO Firestore reads).

    Window: articles published since midnight IST today (date-aligned).
    Ordering: time-bucket (1h) + tier — Tier 1 (important) before Tier 2
    (entertainment) within each bucket; fresher buckets beat older ones.
    Minimum FEED_MIN articles guaranteed (pads with older articles if needed).
    """
    import feed_r2
    if not feed_r2.r2_enabled():
        log.info("feed export skipped — R2 not configured")
        return

    # Midnight IST today: predictable date-aligned window.
    cutoff = _midnight_ist_cutoff()
    recent = [d for d in store if _after_cutoff(d, cutoff)]
    combined, stats = _build_feed_combined(recent)

    log.info(
        "feed: %d articles -> R2 (since midnight IST)  "
        "[tier1=%d  tier2_other=%d  cinema=%d(cap=%d)]  non-Telugu excluded=%d",
        len(combined),
        stats["tier1"], stats["tier2_other"], stats["cinema"], CINEMA_MAX,
        stats["skipped_non_te"],
    )

    out = [_format_article_for_feed(d) for d in combined]
    feed_r2.upload_feed(out)


def archive_daily_feed_if_needed(store: list[dict]) -> None:
    """Create a daily archive snapshot (feed_vYYYYMMDD.json) once per day.

    Archive window: yesterday midnight IST → today midnight IST (matches the live
    feed window so the app can load yesterday's articles seamlessly when the user
    swipes to the end of today's feed).

    Archive name: feed_v{YYYYMMDD}.json where YYYYMMDD = YESTERDAY's IST date
    (the date the articles actually belong to). This keeps naming intuitive:
    the app can request yesterday's archive by decrementing today's date by 1.

    No per-category caps are applied — the archive stores every Telugu article
    published that day so "load yesterday" shows the full picture.

    Guard:   HEAD-check on R2 so multiple runs on the same day only archive once.
    Prune:   Archives older than ARCHIVE_KEEP_DAYS are deleted from R2.
    """
    import feed_r2
    if not feed_r2.r2_enabled():
        return

    now_utc = datetime.now(timezone.utc)
    now_ist = now_utc.astimezone(IST)

    # Midnight boundaries in UTC.
    today_midnight_ist = now_ist.replace(hour=0, minute=0, second=0, microsecond=0)
    today_midnight_utc = today_midnight_ist.astimezone(timezone.utc)
    yesterday_midnight_utc = today_midnight_utc - timedelta(days=1)

    # Archive is named after yesterday (the date the articles belong to).
    yesterday_ist = now_ist - timedelta(days=1)
    date_str = yesterday_ist.strftime("%Y%m%d")

    if feed_r2.archive_exists(date_str):
        log.info("archive: feed_v%s.json already exists — skipping", date_str)
    else:
        def _in_window(d: dict) -> bool:
            dt = _pub_dt(d)
            return dt is not None and yesterday_midnight_utc <= dt < today_midnight_utc

        windowed = [d for d in store if _is_telugu_feed_item(d) and _in_window(d)]
        # Store ALL articles from yesterday — no caps.
        out = sorted(
            [_format_article_for_feed(d) for d in windowed],
            key=lambda d: d.get("publishedAt") or "",
            reverse=True,
        )
        if out:
            url = feed_r2.upload_archive(out, date_str)
            if url:
                log.info("archive: feed_v%s.json created — %d articles", date_str, len(out))
        else:
            log.info("archive: feed_v%s.json — 0 Telugu articles in window, skipping upload", date_str)

    # Prune archives older than ARCHIVE_KEEP_DAYS.
    for days_ago in range(ARCHIVE_KEEP_DAYS, ARCHIVE_KEEP_DAYS + 8):
        old_date = (yesterday_ist - timedelta(days=days_ago)).strftime("%Y%m%d")
        feed_r2.delete_archive(old_date)


def backfill_unpolished(store: list[dict], pool: GeminiKeyPool, counters: dict[str, int]) -> int:
    """
    Re-polish in-memory articles that are ai=False, or already-polished
    English-source articles whose headline is still English. Mutates the
    store dicts in place. Prefers Gemini up to BACKFILL_GEMINI_MAX then
    falls through to Groq. (Decision 3 — no Firestore.)
    """
    rows = [
        d for d in store
        if d.get("ai") is not True
        or (d.get("lang") == "en" and _looks_english(d.get("headline", "")))
    ]
    log.info("backfill: %d articles to (re)process", len(rows))
    polished_count = 0
    for d in rows:
        headline = d.get("headline", "")
        # Prefer the richer stored body as polish input; fall back to summary.
        src = d.get("body") or d.get("summary", "") or ""
        if not headline:
            continue
        polished, engine = polish_one(
            pool, headline, src, counters, BACKFILL_GEMINI_MAX
        )
        if engine == "none" and not SAMBANOVA_API_KEY and not pool.has_capacity():
            log.warning("backfill: all engines down, stopping at %d", polished_count)
            break
        if polished and len(polished) > 20:
            p_title, p_body, p_cat, p_level = _split_polished(polished)
            body = p_body or polished
            d["ai"] = True
            d["summary"] = body
            d["audioScript"] = body
            if p_title and _looks_english(headline):
                d["headline"] = p_title
            if p_cat:
                d["category"] = p_cat
                d["color"] = COLOR_BY_CAT.get(p_cat, d.get("color", "#7C2D12"))
            polished_count += 1
    return polished_count


def promote_community_reports(
    db: firestore.Client, store: list[dict], pool: GeminiKeyPool,
    counters: dict[str, int]
) -> int:
    """
    Village Reporter: turn admin-approved community_reports into feed
    articles. Reporters write in ANY language; if the submission isn't
    already Telugu it's run through the same translator the news
    pipeline uses (Gemini→Groq) so the reader always sees Telugu (and
    gets correct gTTS audio). If no translation engine is available the
    report is left for a later run rather than published in a language
    the feed would hide. Marked promoted so it publishes once.
    """
    # community_reports is a client-WRITE collection → stays on Firestore
    # (tiny, well under free tier). The promoted ARTICLE goes to the R2
    # store, not Firestore (Decision 3).
    reports_ref = db.collection("community_reports")
    promoted = 0
    deferred = 0
    for doc in reports_ref.where(filter=FieldFilter("status", "==", "approved")).limit(50).stream():
        r = doc.to_dict() or {}
        if r.get("promoted") is True:
            continue
        headline = (r.get("headline") or "").strip()
        detail = (r.get("detail") or "").strip()
        if not headline or not detail:
            doc.reference.update({"promoted": True, "promoteError": "empty"})
            continue

        # Convert to Telugu unless the reporter already wrote in Telugu.
        te_headline, te_body = headline, detail[:600]
        if _looks_english(headline) or _looks_english(detail):
            polished, _engine = polish_one(
                pool, headline, detail, counters, MAX_TOTAL_GEMINI
            )
            if polished and len(polished) > 20:
                p_title, p_body, p_cat, p_level = _split_polished(polished)
                if p_body:
                    te_body = p_body
                if p_title:
                    te_headline = p_title
            # Still not Telugu → no engine/failed. Leave it approved &
            # unpromoted; a later run (with Groq capacity) retries.
            if _looks_english(te_headline):
                deferred += 1
                continue

        art_id = f"vr_{doc.id[:18]}"
        if not any(s.get("id") == art_id for s in store):
            store.append({
                "id": art_id,
                "headline": te_headline,
                "summary": te_body[:600],
                "audioScript": te_body[:600],
                "category": "village",
                "mandal": "all",  # visible district-wide for the NTR pilot
                "village": r.get("village") or "",
                "level": "village" if r.get("village") else "mandal",
                "source": "గ్రామ విలేకరి",
                "sources": ["గ్రామ విలేకరి"],
                "link": "",
                "publishedAt": _now_iso(),
                "createdAt": _now_iso(),
                "audioUrl": None,
                "audioSec": 45,
                "color": "#B45309",
                "shares": 0,
                "ai": True,
                "lang": "te",
                "image": r.get("photo") or "",
                "body": te_body,
                "reporterId": None,  # Old village reporter flow, not using whitelist
                "reporter": None,
                "origin": "firestore_village",  # Legacy, different from WhatsApp
            })
        doc.reference.update({
            "promoted": True,
            "promotedAt": firestore.SERVER_TIMESTAMP,
            "articleId": art_id,
            "status": "published",
        })
        promoted += 1
    if deferred:
        log.info("village reporter — deferred=%d (no translator yet, retry next run)", deferred)
    return promoted


def synthesize_pending_audio(store: list[dict], limit: int,
                             feed_ids: "set | None" = None) -> int:
    """
    Narrate in-memory articles that have no audio yet (audioUrl falsy) via
    gTTS and upload the MP3 to R2, writing the public URL back onto the
    store dict. Only AI-polished articles (ai=True) are voiced — raw RSS
    text is low quality and wastes gTTS quota. Capped by `limit` so a large
    backlog fills over several cron runs. (Decision 3 — no Firestore.)

    feed_ids: if provided, only articles whose ID is in this set are eligible.
    Articles are processed newest-first so the most recent content gets audio
    priority within the per-run cap.
    """
    import time

    if not tts_r2.r2_enabled():
        log.info("audio skipped — R2 credentials not configured")
        return 0

    # Sort newest-first so the most recent feed articles get audio first.
    def _recency(d: dict) -> str:
        return d.get("publishedAt") or d.get("createdAt") or ""

    ordered = sorted(store, key=_recency, reverse=True)

    candidates = skipped = done = failed = 0
    for d in ordered:
        if done >= limit:
            break
        if d.get("audioUrl"):
            continue
        # YouTube video items don't get gTTS narration — the user watches
        # the video itself. The article's image is the video thumbnail
        # and the client renders a tap-to-embed iframe.
        if d.get("videoId"):
            skipped += 1
            continue
        # Only generate audio for articles currently visible in feed.json.
        # No point voicing content the user will never encounter.
        if feed_ids is not None and d.get("id") not in feed_ids:
            skipped += 1
            continue
        # Only narrate AI-polished articles (ai=True). Raw RSS text is often
        # incomplete, mid-sentence, or low quality — bad for audio. Voicing
        # unpolished text wastes precious gTTS quota on content that will
        # likely be re-polished (and re-voiced) on a future run anyway.
        # Native Telugu feeds that skipped AI (skip_te_rich) are still
        # good candidates: they have lang=te AND ai=True once polished.
        # If truly never polished, they'll get audio on a backfill run.
        is_ai_telugu = d.get("ai") is True
        text = (d.get("summary") or "").strip()
        # Bug fix: check Telugu content in the NARRATION TEXT, not the
        # headline. Many polished articles keep an English headline but have
        # a Telugu summary — the old headline check silently skipped them.
        if not is_ai_telugu or len(text) < 10 or _looks_english(text):
            skipped += 1
            continue
        candidates += 1
        url = tts_r2.synthesize_and_upload(d.get("id", ""), text)
        if url:
            d["audioUrl"] = url
            done += 1
            log.info("audio ok  [%d/%d] id=%s", done, limit, d.get("id", ""))
        else:
            failed += 1
            log.warning("audio fail id=%s headline=%s", d.get("id", ""),
                        (d.get("headline") or "")[:60])
        # Brief pause between gTTS calls — Google's unofficial endpoint
        # rate-limits rapid-fire requests; 1.5 s keeps us under the limit.
        time.sleep(1.5)

    log.info("audio summary — voiced=%d failed=%d skipped=%d candidates=%d",
             done, failed, skipped, candidates)
    return {"voiced": done, "failed": failed, "skipped": skipped, "candidates": candidates}


def cleanup_existing_articles(store: list[dict]) -> tuple[int, int]:
    """
    Backfill cleanup pass run once per cron. Responsibilities:
      1. Sanitize all summaries — strip visual junk + normalize whitespace.
      2. For VIDEO items only: drop Shorts + LIVE-streams that slipped
         past previous filters.
      3. Auto-expire editorial pins (featured) and sponsored placements
         whose *Until timestamps have passed. Admin sets the timestamps;
         cron is responsible for honoring them so stale BREAKING tags
         and expired ads don't sit at the top of the feed indefinitely.

    Returns (sanitized_count, dropped_count). featured/sponsored expiry
    is counted under sanitized_count for log brevity.
    """
    sanitized = 0
    dropped_ids: list[str] = []
    now_iso = datetime.now(timezone.utc).isoformat()
    for d in store:
        # Auto-expiry: turn off editorial / sponsored flags once their
        # *Until timestamp has passed. String ISO 8601 comparison works
        # lexicographically because both timestamps are zulu-formatted.
        if d.get("featured") and d.get("featuredUntil"):
            if d["featuredUntil"] < now_iso:
                d["featured"] = False
                d["featuredUntil"] = None
                sanitized += 1
        if d.get("sponsored") and d.get("sponsoredUntil"):
            if d["sponsoredUntil"] < now_iso:
                d["sponsored"] = False
                d["sponsoredUntil"] = None
                # Keep sponsoredAdvertiser on the record for audit/history.
                sanitized += 1
        # Pass 1: sanitize ALL articles (text + video).
        for field in ("summary", "audioScript", "body", "headline"):
            v = d.get(field)
            if not isinstance(v, str) or not v:
                continue
            new_v = _sanitize_summary(v)
            if new_v != v:
                d[field] = new_v
                sanitized += 1
        # Pass 2: video-only — drop Shorts and LIVE streams retroactively.
        vid = d.get("videoId")
        if not vid:
            continue
        headline = d.get("headline", "") or ""
        if _is_live_video_headline(headline):
            dropped_ids.append(d.get("id", ""))
            continue
        # Also re-clean video descriptions if boilerplate still present.
        summary = d.get("summary", "") or ""
        has_boilerplate = any(m in summary for m in _YT_BOILERPLATE_MARKERS)
        if has_boilerplate:
            new_summary = _clean_youtube_description(headline, summary) or headline
            if new_summary != summary:
                d["summary"] = new_summary
                d["audioScript"] = new_summary
                d["body"] = new_summary
                sanitized += 1
        # HEAD-check Shorts only for items not yet classified or marked True.
        if "isShort" not in d or d.get("isShort") is True:
            if _is_youtube_short(vid):
                dropped_ids.append(d.get("id", ""))
            else:
                d["isShort"] = False
    if dropped_ids:
        keep_ids = set(dropped_ids)
        store[:] = [d for d in store if d.get("id") not in keep_ids]
    if sanitized or dropped_ids:
        log.info("cleanup — sanitized_fields=%d dropped_videos=%d",
                 sanitized, len(dropped_ids))
    return sanitized, len(dropped_ids)


# Backwards-compatible alias for the old name. main() still calls this.
cleanup_existing_video_items = cleanup_existing_articles


def prune_old_articles(store: list[dict]) -> tuple[list[dict], int]:
    """
    Drop articles older than RETENTION_DAYS from the in-memory working set
    and delete their R2 audio + images (keeps R2 storage bounded). Returns
    (kept_articles, deleted_count). (Decision 3 — no Firestore.)
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)
    kept: list[dict] = []
    dropped_ids: list[str] = []
    for d in store:
        pub = _parse_dt(d.get("publishedAt")) or _parse_dt(d.get("createdAt"))
        if pub is not None and pub < cutoff:
            dropped_ids.append(d.get("id", ""))
        else:
            kept.append(d)
    # Bulk-delete audio + images in ONE R2 API call instead of N×12 individual
    # delete_object calls. delete_media_bulk handles chunking at 1000 keys.
    if dropped_ids:
        tts_r2.delete_media_bulk(dropped_ids)
    return kept, len(dropped_ids)


def _update_training_data(store: list[dict]) -> None:
    """
    Maintain a compact 30-day JSONL on R2 (training_data.jsonl) for ML training.
    Each line: {"id":…, "text":…, "cat":…, "lvl":…, "date":…}

    Called after save_articles() every run. Downloads the existing file, adds
    any store entries whose ID isn't already present, prunes entries older than
    TRAINING_RETENTION_DAYS, and uploads back. All categories (including
    'general') and all origins (RSS + WhatsApp) are stored — the classifier
    uses class_weight=balanced so the general class doesn't dominate.

    First run (file missing on R2): bootstraps from the entire current store.
    """
    if not tts_r2.r2_enabled():
        return

    from botocore.exceptions import ClientError  # noqa: PLC0415
    client = tts_r2._r2_client()
    bucket = tts_r2.R2_BUCKET
    cutoff_str = (
        datetime.now(timezone.utc) - timedelta(days=TRAINING_RETENTION_DAYS)
    ).strftime("%Y-%m-%d")

    # Download existing entries (id → compact dict)
    existing: dict[str, dict] = {}
    try:
        resp = client.get_object(Bucket=bucket, Key=_TRAINING_DATA_KEY)
        for raw in resp["Body"].read().decode("utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                entry = json.loads(raw)
                existing[entry["id"]] = entry
            except (json.JSONDecodeError, KeyError):
                pass
        log.info("training data: loaded %d existing entries", len(existing))
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("NoSuchKey", "404"):
            log.info("training data: not found on R2 — bootstrapping from store")
        else:
            log.warning("training data: download error — %s", exc)
            return
    except Exception as exc:
        log.warning("training data: download error — %s", exc)
        return

    # Only human-verified articles go into training data — no pipeline-guessed labels
    added = 0
    updated = 0
    for art in store:
        art_id = art.get("id") or ""
        if not art_id or not art.get("verified"):
            continue
        headline = (art.get("headline") or "").strip()
        summary  = (art.get("summary")  or "").strip()
        text     = f"{headline} {summary}".strip()
        if len(text) < 20:
            continue
        new_cat = (art.get("category") or "general").strip().lower()
        new_lvl = (art.get("level")    or "").strip().lower()
        pub = art.get("publishedAt") or art.get("createdAt") or ""
        date_str = pub[:10] if len(pub) >= 10 else datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if art_id in existing:
            # Active learning: propagate admin corrections back to training data
            old = existing[art_id]
            if old.get("cat") != new_cat or old.get("lvl") != new_lvl:
                existing[art_id] = {**old, "cat": new_cat, "lvl": new_lvl}
                updated += 1
            continue
        existing[art_id] = {
            "id":   art_id,
            "text": text,
            "cat":  new_cat,
            "lvl":  new_lvl,
            "date": date_str,
        }
        added += 1

    # Prune entries older than TRAINING_RETENTION_DAYS
    before_prune = len(existing)
    existing = {k: v for k, v in existing.items() if (v.get("date") or "9999") >= cutoff_str}
    pruned = before_prune - len(existing)

    # Upload back
    payload = "\n".join(
        json.dumps(e, ensure_ascii=False) for e in existing.values()
    ).encode("utf-8")
    client.put_object(
        Bucket=bucket,
        Key=_TRAINING_DATA_KEY,
        Body=payload,
        ContentType="application/jsonlines",
    )
    log.info(
        "training data: updated — total=%d added=%d updated=%d pruned=%d (%.1f KB)",
        len(existing), added, updated, pruned, len(payload) / 1024,
    )


# ─── Live Data (Alerts) ──────────────────────────────────
# Weather via Open-Meteo (free, no key). Mandi via AGMARKNET (data.gov.in, free key).
# Written to Firestore live_data/{weather,mandi}; AlertsScreen reads from there.

WEATHER_LAT = 16.5167   # Vijayawada, NTR District centre
WEATHER_LON = 80.6167

# Crop & market English→Telugu maps for AGMARKNET output
_CROP_TE: dict[str, str] = {
    "Mango": "మామిడి", "Paddy": "వరి", "Cotton": "పత్తి",
    "Chilli": "మిర్చి", "Banana": "అరటి", "Groundnut": "వేరుశనగ",
    "Maize": "మొక్కజొన్న", "Jowar": "జొన్న", "Onion": "ఉల్లిపాయ",
    "Tomato": "టమాటో", "Brinjal": "వంకాయ", "Coconut": "కొబ్బరి",
    "Turmeric": "పసుపు", "Tamarind": "చింతపండు",
}
_MARKET_TE: dict[str, str] = {
    "Jaggayyapeta": "జగ్గయ్యపేట",
    "Vijayawada": "విజయవాడ", "Nandigama": "నందిగామ",
    "Vatsavai": "వత్సవాయి", "Tiruvuru": "తిరువూరు",
    "Kanchikacherla": "కంచికచర్ల", "Vissannapet": "విస్సన్నపేట",
    "Gampalagudem": "గంపలగూడెం", "Mylavaram": "మైలవరం",
    "Ibrahimpatnam": "ఇబ్రహీంపట్నం", "Reddigudem": "రెడ్డిగూడెం",
    "Penuganchiprolu": "పెనుగంచిప్రోలు", "Chandarlapadu": "చందర్లపాడు",
    "Kanchikacherla": "కంచికచర్ల", "Tiruvuru": "తిరువూరు",
    "Mylavaram": "మైలవరం", "Reddigudem": "రెడ్డిగూడెం",
}
_NTR_MARKETS = set(k.lower() for k in _MARKET_TE)

# Initial scheme data — seeded to Firestore on first run, then admin-editable via console.
_SEED_SCHEMES = [
    {"title": "రైతు భరోసా — దరఖాస్తు", "meta": "₹7,500 · ఆధార్ లింక్ ఖాతా", "deadline": "జూన్ 30", "order": 1, "active": True},
    {"title": "ఆరోగ్యశ్రీ కార్డ్ రెన్యువల్", "meta": "మండల కార్యాలయం", "deadline": "జులై 31", "order": 2, "active": True},
    {"title": "పెన్షన్ లైఫ్ సర్టిఫికెట్", "meta": "గ్రామ సచివాలయం", "deadline": "జూన్ 15", "order": 3, "active": True},
    {"title": "పీఎం కిసాన్ e-KYC", "meta": "pm-kisan.gov.in లేదా CSC", "deadline": "నిరంతరం", "order": 4, "active": True},
]


def _wmo_to_te(code: int, temp: float, wind: float, rain: float) -> tuple[str, str]:
    """WMO weather code + current conditions → (Telugu condition, Telugu alert)."""
    wind_s = f" ఈదురుగాలులు {round(wind)} కి.మీ./గం." if wind > 30 else ""
    rain_s = f" వర్షపాతం {round(rain, 1)} మి.మీ." if rain > 0 else ""
    # temperature-based description for clear/partly-cloudy codes
    if temp >= 42:
        base_cond = "అత్యంత వేడి"
        base_alert = (f"ఉష్ణోగ్రత {round(temp)}°C. అత్యంత వేడి హెచ్చరిక. "
                      f"బయటకు వెళ్లకండి. నీరు ఎక్కువగా తాగండి.{wind_s}")
    elif temp >= 38:
        base_cond = "ఎండ తీవ్రం"
        base_alert = (f"మండే ఎండ {round(temp)}°C. మధ్యాహ్నం 12–4 గంటల మధ్య "
                      f"రైతులు జాగ్రత్తగా ఉండండి.{wind_s}")
    else:
        base_cond = "వేడి వాతావరణం"
        base_alert = f"ఉష్ణోగ్రత {round(temp)}°C.{wind_s} వాతావరణం సాధారణంగా ఉంది."

    if code == 0:
        return base_cond, base_alert
    if code in (1, 2):
        return base_cond, base_alert
    if code == 3:
        return "మేఘావృతం", f"మబ్బుతో వాతావరణం. {round(temp)}°C.{wind_s}"
    if code in (45, 48):
        return "పొగమంచు", "పొగమంచు కారణంగా దృశ్యమానత తక్కువగా ఉంది. వాహనదారులు జాగ్రత్తగా ఉండండి."
    if code in (51, 53, 55):
        return "తుంపర వర్షం", f"తుంపర వర్షం పడే అవకాశం.{rain_s}"
    if code in (61, 63):
        return "వర్షం", f"వర్షం పడే అవకాశం.{rain_s} పంటలకు చర్యలు తీసుకోండి."
    if code == 65:
        return "భారీ వర్షం", f"భారీ వర్షం హెచ్చరిక.{rain_s} తక్కువ ఎత్తు ప్రాంతాల వారు జాగ్రత్తగా ఉండండి."
    if code in (80, 81):
        return "వర్షపు జల్లులు", f"వర్షపు జల్లులు.{rain_s}"
    if code == 82:
        return "తీవ్ర వర్షపు జల్లులు", f"తీవ్రమైన వర్షం హెచ్చరిక.{rain_s} జాగ్రత్తగా ఉండండి."
    if code == 95:
        return "ఉరుములు మెరుపులతో వర్షం", f"ఉరుముల వర్షం హెచ్చరిక.{rain_s} చెట్ల కింద నిలబడకండి."
    if code in (96, 99):
        return "వడగండ్లతో తుఫాన్", f"వడగండ్లతో తుఫాన్ హెచ్చరిక!{rain_s} పంటలకు రక్షణ చర్యలు తక్షణమే తీసుకోండి."
    return base_cond, base_alert


def fetch_weather_ntr() -> dict | None:
    """Fetch current weather for NTR District from Open-Meteo (free, no key)."""
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={WEATHER_LAT}&longitude={WEATHER_LON}"
        "&current=temperature_2m,weathercode,windspeed_10m,relative_humidity_2m"
        "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
        "&timezone=Asia%2FKolkata&forecast_days=1"
    )
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        d = r.json()
    except Exception as e:
        log.warning("weather fetch failed: %s", e)
        return None
    cur = d.get("current", {})
    daily = d.get("daily", {})
    temp = float(cur.get("temperature_2m") or 32)
    code = int(cur.get("weathercode") or 0)
    wind = float(cur.get("windspeed_10m") or 0)
    humidity = float(cur.get("relative_humidity_2m") or 60)
    rain = float((daily.get("precipitation_sum") or [0])[0] or 0)
    cond, alert = _wmo_to_te(code, temp, wind, rain)
    return {
        "place": "విజయవాడ · NTR జిల్లా",
        "temp": round(temp),
        "condition": cond,
        "alert": alert,
        "windspeed": round(wind),
        "humidity": round(humidity),
        "rainToday": round(rain, 1),
        "wmoCode": code,
        "tempMin": round(float((daily.get("temperature_2m_min") or [0])[0] or 0)),
        "tempMax": round(float((daily.get("temperature_2m_max") or [0])[0] or 0)),
    }


def fetch_mandi_ntr(api_key: str) -> list[dict] | None:
    """
    Fetch today's mandi prices for NTR District from AGMARKNET via data.gov.in.
    Returns a list of price dicts, or None on failure.
    Crops with multiple varieties → keep the one with highest modal price.
    Free API key: data.gov.in → Register → My Profile → API Key.
    """
    if not api_key:
        return None
    # AGMARKNET still lists NTR markets under "Krishna" district (pre-2022 bifurcation).
    # We filter by market name rather than district.
    base = (
        "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
        f"?api-key={api_key}&format=json&limit=100"
    )
    # A whole-state pull hangs the govt server (read timeouts). NTR markets
    # are under "Krishna" district in this dataset (pre-2022 bifurcation),
    # so filter server-side → tiny, fast response. Try Krishna first, then
    # NTR (in case the dataset later splits), then the old broad query.
    candidate_urls = [
        base + "&filters%5BState%5D=Andhra+Pradesh&filters%5BDistrict%5D=Krishna",
        base + "&filters%5BState%5D=Andhra+Pradesh&filters%5BDistrict%5D=NTR",
        base + "&filters%5BState%5D=Andhra+Pradesh",
    ]
    data = None
    for url in candidate_urls:
        for attempt in range(1, 3):  # 2 tries per query
            try:
                r = requests.get(url, timeout=(10, 30))
                r.raise_for_status()
                j = r.json()
                if j.get("records"):
                    data = j
                    break
                # Empty but responded — query worked, just no rows; stop here.
                data = j
                break
            except Exception as e:
                log.warning("mandi attempt %d failed (%s): %s",
                            attempt, url.split("filters")[-1][:40], e)
                if attempt < 2:
                    time.sleep(2)
        if data and data.get("records"):
            break
    if data is None:
        log.warning("mandi: data.gov.in unreachable on all queries")
        return None

    records = data.get("records") or []

    def _parse_date(s: str):
        # data.gov.in arrival_date is usually "DD/MM/YYYY"
        for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%d-%b-%Y"):
            try:
                return datetime.strptime(s.strip(), fmt).date()
            except (ValueError, AttributeError):
                continue
        return None

    # Group by crop+market. Keep the record with the LATEST arrival_date
    # (so we always show the most recent available price, even if it's not
    # today's). Tie-break on higher modal price.
    best: dict[str, dict] = {}
    for rec in records:
        market_en = (rec.get("market") or rec.get("Market") or "").strip()
        if market_en.lower() not in _NTR_MARKETS:
            continue
        crop_en = (rec.get("commodity") or rec.get("Commodity") or "").strip()
        variety_en = (rec.get("variety") or rec.get("Variety") or "").strip()
        try:
            price = float(rec.get("modal_price") or rec.get("Modal_Price") or 0)
        except (ValueError, TypeError):
            price = 0.0
        if price <= 0:
            continue
        date_raw = (rec.get("arrival_date") or rec.get("Arrival_Date") or "").strip()
        d = _parse_date(date_raw)
        d_ord = d.toordinal() if d else 0

        key = f"{crop_en}|{market_en}"
        prev = best.get(key)
        # newer date wins; same date → higher modal wins
        if (prev is None or d_ord > prev["_dord"]
                or (d_ord == prev["_dord"] and price > prev["_price"])):
            best[key] = {
                "_price": price,
                "_dord": d_ord,
                "crop": _CROP_TE.get(crop_en, crop_en),
                "variety": variety_en,
                "price": round(price),
                "unit": "₹/క్వింటాల్",
                "market": _MARKET_TE.get(market_en, market_en),
                "date": date_raw,          # show "as of this date" in the app
                "change": 0,               # diff computed below vs previous
            }

    if not best:
        return None
    # Newest first, then by price
    out = sorted(best.values(), key=lambda x: (-x["_dord"], -x["_price"]))[:8]
    for p in out:
        p.pop("_dord", None)
    return out


def fetch_mandi_agmarknet_scrape() -> list[dict] | None:
    """
    Fallback mandi price scraper using AGMARKNET search form (no API key needed).
    AGMARKNET is an ASP.NET WebForms site; it uses __VIEWSTATE. We do a two-step
    GET (capture hidden fields) then POST (submit search for AP/NTR markets).

    This is best-effort: if AGMARKNET changes their form layout, it will silently
    return None and the previous Firestore values remain.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        log.warning("beautifulsoup4 not installed — AGMARKNET scraper skipped")
        return None

    BASE_URL = "https://agmarknet.gov.in"
    SEARCH_URL = f"{BASE_URL}/SearchCmmMkt.aspx"
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; KshanaVarthaBot/1.0; +https://kshanavartha.github.io)",
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-IN,te;q=0.9",
        "Referer": SEARCH_URL,
    }
    try:
        # Step 1: GET the search form to capture hidden ASP.NET fields
        r1 = requests.get(SEARCH_URL, headers=headers, timeout=20)
        r1.raise_for_status()
        soup1 = BeautifulSoup(r1.text, "html.parser")

        def _val(name: str) -> str:
            tag = soup1.find("input", {"name": name})
            return tag["value"] if tag and tag.get("value") else ""

        viewstate = _val("__VIEWSTATE")
        vsg = _val("__VIEWSTATEGENERATOR")
        ev = _val("__EVENTVALIDATION")
        if not viewstate:
            log.warning("AGMARKNET: no __VIEWSTATE found — site layout may have changed")
            return None

        # Step 2: POST with State=Andhra Pradesh, commodity=0 (All), market=0 (All)
        # Date defaults to today on the server side.
        from datetime import date as _date
        today_str = _date.today().strftime("%d-%b-%Y")  # e.g. 16-May-2026
        payload = {
            "__VIEWSTATE": viewstate,
            "__VIEWSTATEGENERATOR": vsg,
            "__EVENTVALIDATION": ev,
            "ddlState": "2",          # Andhra Pradesh state code in AGMARKNET
            "ddlDistrict": "0",       # All districts
            "ddlCommodity": "0",      # All commodities
            "ddlArrivalDate": today_str,
            "btnGo": "Go",
        }
        r2 = requests.post(SEARCH_URL, data=payload, headers=headers, timeout=25)
        r2.raise_for_status()
        soup2 = BeautifulSoup(r2.text, "html.parser")

        # Result table: look for the data grid table
        table = soup2.find("table", {"id": "cphBody_GridPriceData"})
        if not table:
            # Try any table with expected headers
            for t in soup2.find_all("table"):
                if t.find(string=lambda s: s and "Commodity" in s):
                    table = t
                    break
        if not table:
            log.warning("AGMARKNET scrape: results table not found")
            return None

        rows = table.find_all("tr")
        if len(rows) < 2:
            return None

        # Parse header row to find column indices
        headers_row = [th.get_text(strip=True) for th in rows[0].find_all(["th", "td"])]
        col = {}
        for i, h in enumerate(headers_row):
            h_lower = h.lower()
            if "commodity" in h_lower:   col["crop"] = i
            elif "market" in h_lower:    col["market"] = i
            elif "modal" in h_lower:     col["modal"] = i
            elif "variety" in h_lower:   col["variety"] = i

        if "crop" not in col or "market" not in col or "modal" not in col:
            log.warning("AGMARKNET scrape: unexpected column layout: %s", headers_row)
            return None

        best: dict[str, dict] = {}
        for row in rows[1:]:
            cells = [td.get_text(strip=True) for td in row.find_all("td")]
            if len(cells) <= max(col.values()):
                continue
            market_en = cells[col["market"]].strip().title()
            if market_en.lower() not in _NTR_MARKETS:
                continue
            crop_en = cells[col["crop"]].strip().title()
            variety_en = cells[col.get("variety", col["crop"])].strip() if "variety" in col else ""
            try:
                price = float(cells[col["modal"]].replace(",", "") or 0)
            except (ValueError, TypeError):
                price = 0.0
            if price <= 0:
                continue
            key = f"{crop_en}|{market_en}"
            if key not in best or price > best[key]["_price"]:
                best[key] = {
                    "_price": price,
                    "crop": _CROP_TE.get(crop_en, crop_en),
                    "variety": variety_en,
                    "price": round(price),
                    "unit": "₹/క్వింటాల్",
                    "market": _MARKET_TE.get(market_en, market_en),
                    "change": 0,
                }

        if not best:
            log.info("AGMARKNET scrape: no NTR markets found in today's results")
            return None
        result = sorted(best.values(), key=lambda x: -x["_price"])[:8]
        log.info("AGMARKNET scrape: %d prices for NTR markets", len(result))
        return result

    except Exception as e:
        log.warning("AGMARKNET scrape failed: %s", e)
        return None


def _compute_price_changes(new_prices: list[dict], old_prices: list[dict]) -> None:
    """Mutate `new_prices` to fill `change` vs matching entries in `old_prices`."""
    old_map: dict[str, int] = {}
    for p in (old_prices or []):
        k = f"{p.get('crop', '')}|{p.get('market', '')}"
        if k not in old_map:
            old_map[k] = p.get("price", 0)
    for p in new_prices:
        k = f"{p.get('crop', '')}|{p.get('market', '')}"
        if k in old_map and old_map[k]:
            p["change"] = p["price"] - old_map[k]


def _seed_schemes(db: firestore.Client) -> None:
    """Seed alerts_schemes if empty. Runs only once; admin can edit via console."""
    col = db.collection("alerts_schemes")
    # Quick check: if any doc exists, skip seed (don't overwrite admin edits)
    if list(col.limit(1).stream()):
        return
    log.info("seeding alerts_schemes (%d entries)", len(_SEED_SCHEMES))
    for entry in _SEED_SCHEMES:
        col.document().set(entry)


def update_live_data(db: firestore.Client) -> None:
    """
    Refresh live_data/weather and live_data/mandi in Firestore.
    Seeds alerts_schemes on first run (subsequently admin-managed via console).
    Called each cron run; skips gracefully on any failure so ingest continues.
    """
    live_ref = db.collection("live_data")

    # ─── Weather ───────────────────────────────────────
    weather = fetch_weather_ntr()
    if weather:
        weather["updatedAt"] = firestore.SERVER_TIMESTAMP
        live_ref.document("weather").set(weather)
        log.info("weather updated — %s %d°C code=%s", weather["condition"], weather["temp"], weather["wmoCode"])
    else:
        log.warning("weather update skipped — fetch failed")

    # ─── Mandi ─────────────────────────────────────────
    # Gated behind MANDI_ENABLED=1. Both data.gov.in API and AGMARKNET
    # scraper were confirmed dead from all networks on 2026-05-18 — see
    # memory mandi_source_dead_2026_05_18. Calling them every run wastes
    # ~30s on timeouts and floods the log with WARNINGs. When a working
    # source is found (or one of the dead endpoints revives), flip
    # MANDI_ENABLED=1 in the workflow env / run-local.ps1 to re-enable.
    if os.environ.get("MANDI_ENABLED") != "1":
        log.info("mandi: disabled (MANDI_ENABLED!=1) — sources verified dead 2026-05-18")
        new_prices = None
    else:
        # Priority: data.gov.in API key → AGMARKNET scraper (no key) → keep old values.
        api_key = os.environ.get("DATA_GOV_API_KEY", "").strip()
        new_prices = fetch_mandi_ntr(api_key)
        if not new_prices:
            if api_key:
                log.warning("data.gov.in mandi fetch failed; trying AGMARKNET scraper")
            else:
                log.info("DATA_GOV_API_KEY not set; trying AGMARKNET scraper (no key)")
            new_prices = fetch_mandi_agmarknet_scrape()

    if new_prices:
        # Read previous prices to compute change direction
        try:
            old_doc = live_ref.document("mandi").get()
            old_prices = (old_doc.to_dict() or {}).get("prices") or []
        except Exception:
            old_prices = []
        _compute_price_changes(new_prices, old_prices)
        # Drop internal _price helper key before writing
        for p in new_prices:
            p.pop("_price", None)

        # Merge with previous: a crop missing from today's pull keeps its
        # last-known price+date instead of vanishing. Latest date wins.
        def _dord(s: str) -> int:
            for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%d-%b-%Y"):
                try:
                    return datetime.strptime((s or "").strip(), fmt).toordinal()
                except ValueError:
                    continue
            return 0

        merged: dict[str, dict] = {}
        for p in old_prices + new_prices:   # new listed last → wins ties
            k = f"{p.get('crop','')}|{p.get('market','')}"
            if k not in merged or _dord(p.get("date", "")) >= _dord(merged[k].get("date", "")):
                merged[k] = p
        final = sorted(
            merged.values(),
            key=lambda x: (_dord(x.get("date", "")), x.get("price", 0)),
            reverse=True,
        )[:8]

        live_ref.document("mandi").set({
            "prices": final,
            "updatedAt": firestore.SERVER_TIMESTAMP,
        })
        log.info("mandi updated — %d price(s) (merged latest-known)", len(final))
    else:
        log.warning("mandi update skipped — both data.gov.in and AGMARKNET scraper failed")

    # ─── Schemes seed (one-time) ────────────────────────
    try:
        _seed_schemes(db)
    except Exception as e:
        log.warning("schemes seed failed: %s", e)


# ─── FCM Push Notifications (Phase B) ────────────────────
# Sends push notifications to FCM topics when breaking news or sharp
# mandi moves occur. Uses the existing Firebase service account —
# no extra secret needed. Set FCM_ENABLED=1 to enable.
# Topics: kv-all (all users), kv-weather (weather alerts).

def _fcm_access_token(sa_info: dict) -> str | None:
    """Get a short-lived OAuth2 token scoped to Firebase Messaging."""
    try:
        from google.oauth2 import service_account as _sa
        import google.auth.transport.requests as _req
        creds = _sa.Credentials.from_service_account_info(
            sa_info,
            scopes=["https://www.googleapis.com/auth/firebase.messaging"],
        )
        creds.refresh(_req.Request())
        return creds.token
    except Exception as e:
        log.warning("fcm token error: %s", e)
        return None


def _fcm_send(project_id: str, sa_info: dict, title: str, body: str,
              topic: str, data: dict | None = None) -> bool:
    """Send one FCM topic notification. Returns True on success."""
    token = _fcm_access_token(sa_info)
    if not token:
        return False
    payload: dict = {
        "message": {
            "topic": topic,
            "notification": {"title": title, "body": body},
            "android": {
                "priority": "high",
                "notification": {
                    "channel_id": "kv-news",
                    "sound": "default",
                },
            },
            "data": {k: str(v) for k, v in (data or {}).items()},
        },
    }
    url = f"https://fcm.googleapis.com/v1/projects/{project_id}/messages:send"
    try:
        r = requests.post(url, headers={"Authorization": f"Bearer {token}"}, json=payload, timeout=15)
        if r.ok:
            return True
        log.warning("fcm send http %d: %s", r.status_code, r.text[:120])
        return False
    except Exception as e:
        log.warning("fcm send error: %s", e)
        return False
        
def _fcm_send_to_token(project_id: str, sa_info: dict, title: str, body: str,
                       token: str, data: dict | None = None) -> tuple[bool, bool]:
    """Send FCM notification directly to a specific device token.

    Returns (sent_ok, token_is_dead). `token_is_dead` is True when FCM v1
    reports the token is no longer valid (UNREGISTERED) so the caller can
    delete it from Firestore. Triggered by HTTP 404 or 400-UNREGISTERED —
    the two canonical "this device reinstalled / token rotated" signals.
    """
    access_token = _fcm_access_token(sa_info)
    if not access_token:
        return False, False
    payload = {
        "message": {
            "token": token,
            "notification": {"title": title, "body": body},
            "android": {
                "priority": "high",
            },
            "data": {k: str(v) for k, v in (data or {}).items()},
        }
    }
    url = f"https://fcm.googleapis.com/v1/projects/{project_id}/messages:send"
    try:
        r = requests.post(url, headers={"Authorization": f"Bearer {access_token}"},
                          json=payload, timeout=15)
        if r.ok:
            return True, False
        # Detect dead tokens — FCM v1 reports these as 404 or 400 with
        # UNREGISTERED in the error body. Either way the token will never
        # work again (app uninstalled, reinstalled, or FCM rotated the
        # registration). Caller deletes the row so future runs stay clean.
        body_l = (r.text or "").lower()
        is_dead = (r.status_code == 404) or (
            r.status_code == 400 and "unregistered" in body_l
        )
        # Don't log a noisy warning for known-dead tokens — they're expected
        # after every reinstall and the cleanup is automatic.
        if is_dead:
            log.info("fcm token dead (http %d) — will be removed from store", r.status_code)
        else:
            log.warning("fcm token send http %d: %s", r.status_code, r.text[:200])
        return False, is_dead
    except Exception as e:
        log.warning("fcm token send error: %s", e)
        return False, False

def notify_new_articles(db: firestore.Client, store: list[dict],
                        new_articles: list[dict], sa_info: dict) -> None:
    """Send ONE push notification per run, Tier 0/1 articles only, with a
    NOTIFICATION_GAP_HOURS cooldown enforced via Firestore.

    Design (confirmed 2026-05):
    - One notification per run — avoids notification fatigue, free-tier safe.
    - Only Tier 0 (politics, schemes) + Tier 1 (farming, weather, jobs, village, health)
      qualify. Cinema/sports/general are not urgent enough.
    - 3-hour gap (tunable via NOTIFICATION_GAP_HOURS) prevents spam even when
      many articles land at once.
    - Notification title = article headline (≤65 chars). Body is empty so the
      headline has full width on the device lock screen.
    - data payload: type="article", articleId=<id> — the app's existing
      pushNotificationActionPerformed handler opens SwipeReader at that article.
    - Dead tokens are pruned on the spot (FCM v1 UNREGISTERED signal).

    Only fires if FCM_ENABLED=1 env var is set.
    """
    if os.environ.get("FCM_ENABLED") != "1":
        return
    if not new_articles or not sa_info:
        return

    # ── Pick the best article to feature in the notification ─────────────
    # Tier 1 categories — matches _FEED_TIER in the feed exporter.
    # Tier 2 (cinema, sports, health) never triggers a push notification.
    NOTIFY_TIERS = {"politics", "schemes", "farming", "weather", "jobs", "village", "general"}
    # Only consider newly-added ai=True Telugu articles in qualifying tiers.
    candidates = [
        a for a in new_articles
        if a.get("ai") is True
        and a.get("lang") == "te"
        and a.get("category", "general") in NOTIFY_TIERS
    ]
    if not candidates:
        log.info("fcm: no Tier 1 Telugu articles in this run — skip notification")
        return

    # Newest qualifying article is the lead.
    candidates.sort(key=lambda a: a.get("createdAt") or "", reverse=True)
    lead = candidates[0]
    article_id = (lead.get("id") or "").strip()
    headline = (lead.get("headline") or "").strip()
    if not headline or not article_id:
        log.info("fcm: lead article has no headline/id — skip")
        return

    project_id = sa_info.get("project_id", "kshanavartha-bdeab")

    # ── Enforce cooldown gap using Firestore ──────────────────────────────
    notif_ref = db.collection("live_data").document("notification_state")
    try:
        snap = notif_ref.get()
        if snap.exists:
            last_raw = (snap.to_dict() or {}).get("lastNotifiedAt", "")
            if last_raw:
                last_dt = _parse_dt(last_raw)
                if last_dt:
                    gap = datetime.now(timezone.utc) - last_dt
                    if gap.total_seconds() < NOTIFICATION_GAP_HOURS * 3600:
                        log.info(
                            "fcm: gap %.1f h < %d h — skipping this run",
                            gap.total_seconds() / 3600, NOTIFICATION_GAP_HOURS,
                        )
                        return
    except Exception as e:
        log.warning("fcm: could not check notification_state (%s); proceeding anyway", e)

    # ── Fetch registered device tokens ────────────────────────────────────
    try:
        token_docs = list(db.collection("fcm_tokens").stream())
    except Exception as e:
        log.warning("fcm: failed to fetch tokens: %s", e)
        return

    if not token_docs:
        log.info("fcm: no tokens registered, skipping")
        return

    # ── Send to every token ───────────────────────────────────────────────
    title = headline[:65]
    sent = 0
    dead_refs: list = []
    for doc in token_docs:
        token = (doc.to_dict() or {}).get("token")
        if not token:
            continue
        ok, is_dead = _fcm_send_to_token(
            project_id, sa_info, title, "",   # body="" so headline takes full width
            token,
            data={"type": "article", "articleId": article_id},
        )
        if ok:
            sent += 1
        elif is_dead:
            dead_refs.append(doc.reference)

    # ── Prune dead tokens ─────────────────────────────────────────────────
    if dead_refs:
        for ref in dead_refs:
            try:
                ref.delete()
            except Exception as e:
                log.warning("fcm: failed to prune dead token %s: %s", ref.id[:12], e)
        log.info("fcm: pruned %d dead token(s) from store", len(dead_refs))

    # ── Record last-notified time so next run respects the gap ───────────
    if sent > 0:
        try:
            notif_ref.set({"lastNotifiedAt": _now_iso()}, merge=True)
        except Exception as e:
            log.warning("fcm: could not write lastNotifiedAt: %s", e)

    log.info(
        "fcm sent — article=%s category=%s sent_to=%d/%d tokens",
        article_id[:12], lead.get("category", "?"), sent, len(token_docs),
    )


def _print_run_report(
    *,
    feeds_ok: int, feeds_failed: int,
    fetched: int, new_count: int,
    gnews_enriched: int, blocked_count: int,
    counters: dict, summarized: int,
    audio_stats: dict, audio_pending: int, feed_visible: int,
    store_total: int, feed_published: int, pruned: int, retention_days: int,
    gemini_pool, cerebras_pool, sambanova_pool,
) -> None:
    """Print a structured run summary to stdout. Appears in the last 300
    log lines so the email always captures it."""
    sep = "=" * 62
    print(f"\n{sep}")
    print("  KshanaVartha Ingest — Run Summary")
    print(sep)

    # Feeds
    print(f"\nFEEDS  ({feeds_ok + feeds_failed} sources)")
    print(f"  OK: {feeds_ok}  |  Failed: {feeds_failed}")
    if gnews_enriched:
        print(f"  Google News resolved: {gnews_enriched} links → real publisher URLs")

    # Articles
    dup_skipped = fetched - new_count
    print(f"\nARTICLES")
    print(f"  Fetched        : {fetched}")
    print(f"  New (stored)   : {new_count}")
    print(f"  Duplicates     : {dup_skipped}")
    if blocked_count:
        print(f"  Blocked IDs    : {blocked_count} (in blocklist.json)")

    # AI processing
    skip_eng = counters.get("skip_english", 0)
    skip_te  = counters.get("skip_te_rich", 0)
    skip_vid = counters.get("skip_video", 0)
    skip_lvl = counters.get("skip_level_cap", 0)
    ai_fail  = counters.get("ai_fail", 0)
    print(f"\nAI PROCESSING  (of {new_count} new articles)")
    print(f"  AI polished    : {summarized}  "
          f"(cerebras={counters.get('cerebras_ok',0)}  "
          f"gemini={counters.get('gemini_ok',0)}  "
          f"sambanova={counters.get('sambanova_ok',0)})")
    print(f"  Native Telugu  : {skip_te}   (≥60 Te words — no AI needed, audio-eligible)")
    print(f"  English held   : {skip_eng}   (ENGLISH_POLISH not set)")
    print(f"  YouTube videos : {skip_vid}   (no AI for video items)")
    print(f"  Level-capped   : {skip_lvl}   (national≤{NATIONAL_AI_MAX}, state≤{STATE_AI_MAX} per run)")
    print(f"  AI called, failed : {ai_fail}")

    def _pool_section(title: str, model: str, pool) -> None:
        n = len(pool.keys)
        exhausted = len(pool.exhausted)
        print(f"\n{title} ({model}) — {n} key(s)  [exhausted this run: {exhausted}]")
        if not pool.keys:
            print("  (no keys configured)")
            return
        total_calls = total_ok = total_fail = total_quota = 0
        total_ptok = total_ctok = 0
        for k in pool.keys:
            s = pool.key_stats.get(k, {})
            calls = s.get("calls", 0)
            ok    = s.get("success", 0)
            fail  = s.get("fail", 0)
            quota = s.get("quota_hits", 0)
            ptok  = s.get("prompt_tokens", 0)
            ctok  = s.get("comp_tokens", 0)
            tag   = " *** EXHAUSTED ***" if k in pool.exhausted else ""
            tok_s = f"  in={ptok:,} out={ctok:,} tok" if (ptok or ctok) else "  (tokens: n/a)"
            print(f"  ...{k[-6:]} : {calls:3d} calls | {ok:3d} OK | "
                  f"{fail:2d} fail | {quota:2d} quota{tok_s}{tag}")
            total_calls += calls; total_ok += ok; total_fail += fail
            total_quota += quota; total_ptok += ptok; total_ctok += ctok
        if n > 1:
            tok_total = f"  in={total_ptok:,} out={total_ctok:,} tok" if (total_ptok or total_ctok) else ""
            print(f"  {'TOTAL':8s} : {total_calls:3d} calls | {total_ok:3d} OK | "
                  f"{total_fail:2d} fail | {total_quota:2d} quota{tok_total}")

    _pool_section("CEREBRAS", CEREBRAS_MODEL, cerebras_pool)
    _pool_section("GEMINI", GEMINI_MODEL, gemini_pool)
    _pool_section("SAMBANOVA", SAMBANOVA_MODEL, sambanova_pool)

    # Audio
    voiced     = audio_stats.get("voiced", 0)
    aud_fail   = audio_stats.get("failed", 0)
    candidates = audio_stats.get("candidates", 0)
    print(f"\nAUDIO (gTTS → R2)  [cap={AUDIO_MAX_PER_RUN}/run]")
    print(f"  Feed-visible articles : {feed_visible}")
    print(f"  Eligible (ai=True, no audio yet) : {candidates}")
    print(f"  Voiced this run  : {voiced}  |  Failed: {aud_fail}")
    print(f"  Still pending    : {audio_pending}  (will voice in later runs)")

    # Store & feed
    print(f"\nSTORE → R2")
    print(f"  Total articles saved : {store_total}")
    print(f"  Feed published       : {feed_published}  (Telugu-readable)")
    print(f"  Pruned (old)         : {pruned}  (older than {retention_days} days)")

    print(f"\n{sep}\n")


if __name__ == "__main__":
    sys.exit(main())
