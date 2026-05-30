"""
Offline test for the Decision-3 in-memory article pipeline.
NO network, NO R2, NO Firestore, NO quota — tts_r2/feed_r2 are monkeypatched.

    python ingest/test_pipeline_store.py

Covers: prune-by-date, audio selection (Telugu-only, cap, skip English &
already-voiced), and the Telugu-only newest-first feed export.
"""
import os
import sys
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.join(os.getcwd(), "ingest"))
import ingest
import tts_r2
import feed_r2

failed = 0


def check(desc, cond):
    global failed
    if not cond:
        failed += 1
    print(f"  [{'PASS' if cond else 'FAIL'}] {desc}")


now = datetime.now(timezone.utc)
iso = lambda d: d.isoformat()

# ── prune_old_articles ────────────────────────────────────
# delete_media_bulk receives the full list of IDs at once (bulk R2 call).
deleted_media_bulk: list[list[str]] = []
tts_r2.delete_media_bulk = lambda ids: deleted_media_bulk.append(list(ids))

store = [
    {"id": "fresh", "publishedAt": iso(now - timedelta(days=2)),
     "ai": True, "lang": "te", "headline": "తాజా వార్త", "summary": "x" * 30},
    {"id": "old", "publishedAt": iso(now - timedelta(days=20)),
     "ai": True, "lang": "te", "headline": "పాత వార్త"},
    {"id": "nodate", "createdAt": iso(now), "ai": True, "lang": "te",
     "headline": "తేదీ లేని వార్త", "summary": "y" * 30},
]
kept, ndel = ingest.prune_old_articles(store)
check("prune drops the 20-day-old article", ndel == 1)
check("prune keeps fresh + nodate", {a["id"] for a in kept} == {"fresh", "nodate"})
check("prune calls delete_media_bulk once with all dropped IDs",
      deleted_media_bulk == [["old"]])

# ── synthesize_pending_audio ──────────────────────────────
tts_r2.r2_enabled = lambda: True
tts_r2.synthesize_and_upload = lambda aid, text: f"https://r2/{aid}.mp3"

audio_store = [
    {"id": "te1", "ai": True, "lang": "te", "headline": "తెలుగు ఒకటి",
     "summary": "ఇది తెలుగు సారాంశం, పది అక్షరాల కంటే ఎక్కువ."},
    {"id": "en1", "ai": False, "lang": "en",
     "headline": "English stub", "summary": "English summary text here"},
    {"id": "te2", "ai": True, "lang": "te", "headline": "తెలుగు రెండు",
     "summary": "మరో తెలుగు సారాంశం ఇక్కడ ఉంది సరిపడా పొడవు."},
    {"id": "already", "ai": True, "lang": "te", "headline": "ఇప్పటికే ఆడియో",
     "summary": "ఆడియో ఉన్న వార్త సారాంశం ఇక్కడ.",
     "audioUrl": "https://r2/already.mp3"},
]
n = ingest.synthesize_pending_audio(audio_store, limit=1)
check("audio respects limit=1", n == 1)
check("audio voiced the first Telugu pending", audio_store[0]["audioUrl"] == "https://r2/te1.mp3")
check("audio skipped English stub", "audioUrl" not in audio_store[1])
check("audio left already-voiced untouched",
      audio_store[3]["audioUrl"] == "https://r2/already.mp3")

# ── export_feed_to_r2 (Telugu-only, tier-priority + newest-first, cap) ────
feed_r2.r2_enabled = lambda: True
captured = {}
feed_r2.upload_feed = lambda arts: captured.setdefault("arts", arts) or "url"

ingest.FEED_MAX = 2
ingest.CINEMA_MAX = 5  # no cinema in this test so cap is irrelevant
feed_store = [
    {"id": "old_te", "ai": True, "lang": "te", "headline": "పాత తెలుగు",
     "category": "general", "createdAt": iso(now - timedelta(hours=5))},
    {"id": "new_te", "ai": True, "lang": "te", "headline": "కొత్త తెలుగు",
     "category": "general", "createdAt": iso(now)},
    {"id": "eng", "ai": False, "lang": "en", "headline": "English news",
     "category": "general", "createdAt": iso(now)},
    {"id": "mid_te", "ai": True, "lang": "te", "headline": "మధ్య తెలుగు",
     "category": "general", "createdAt": iso(now - timedelta(hours=1))},
]
ingest.export_feed_to_r2(feed_store)
arts = captured.get("arts", [])
check("feed excludes English", all(a["id"] != "eng" for a in arts))
check("feed capped at FEED_MAX=2", len(arts) == 2)
check("feed newest-first within same tier (new_te, mid_te)",
      [a["id"] for a in arts] == ["new_te", "mid_te"])

# ── feed tier ordering: politics before cinema even when cinema is newer ──
captured2 = {}
feed_r2.upload_feed = lambda arts: captured2.setdefault("arts", arts) or "url"

ingest.FEED_MAX = 10
ingest.CINEMA_MAX = 5
tier_store = [
    {"id": "cin_new", "ai": True, "lang": "te", "headline": "సినిమా కొత్త",
     "category": "cinema",   "publishedAt": iso(now)},
    {"id": "pol_old", "ai": True, "lang": "te", "headline": "రాజకీయాలు పాత",
     "category": "politics", "publishedAt": iso(now - timedelta(hours=3))},
    {"id": "farm",    "ai": True, "lang": "te", "headline": "వ్యవసాయం",
     "category": "farming",  "publishedAt": iso(now - timedelta(hours=1))},
    {"id": "cin_old", "ai": True, "lang": "te", "headline": "సినిమా పాత",
     "category": "cinema",   "publishedAt": iso(now - timedelta(hours=5))},
]
ingest.export_feed_to_r2(tier_store)
arts2 = captured2.get("arts", [])
ids2 = [a["id"] for a in arts2]
check("politics before farming (even though farming is newer)",
      ids2.index("pol_old") < ids2.index("farm"))
check("farming before cinema (even though cinema is newest)",
      ids2.index("farm") < ids2.index("cin_new"))
check("cinema newest before cinema oldest",
      ids2.index("cin_new") < ids2.index("cin_old"))

if failed:
    print(f"\n{failed} FAILED")
    sys.exit(1)
print(f"\nAll checks passed — Decision-3 pipeline works offline (no Firestore/quota).")
