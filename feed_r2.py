"""
KshanaVartha — Telugu-only public feed → Cloudflare R2 (CDN) exporter.

WHY THIS EXISTS (cost architecture — Decision 3):
  All users read the same ~500 news articles. Letting clients read Firestore
  directly would cost ~$900-4500/month at scale. Instead:
    • The ingest pipeline maintains a private article working set (articles.json)
      on R2 — that's the source of truth, never Firestore.
    • At end of each run, this module writes a FILTERED public view (feed.json)
      to R2 — Telugu-only, newest-first, capped at FEED_MAX articles.
    • Every app client fetches feed.json from the R2 CDN (zero egress fees →
      ~$0 at any user scale, no Firestore reads on the article hot path).

  Firestore is used only for client-write collections that are tiny and
  server-side: fcm_tokens, community_reports, live_data.

FILTERING applied before publishing:
  • Non-Telugu articles excluded (lang != "te" and ai != True)
  • Headlines with no Telugu script excluded (English stubs not yet polished)
  • Junk-content articles excluded (PDF/iframe embed failures)
  • YouTube Shorts excluded (isShort == True)
  • Sorted by publishedAt desc (real publish time, not cron-discovery time)

Reuses the same R2 bucket/creds as tts_r2.py (audio).
Env: R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY,
     R2_PUBLIC_URL, R2_BUCKET (default kv-audio), R2_FEED_KEY (default feed.json).
"""
from __future__ import annotations

import gzip
import json
import logging
import os

log = logging.getLogger("kv-ingest")

R2_ACCOUNT_ID = os.environ.get("R2_ACCOUNT_ID", "").strip()
R2_ACCESS_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID", "").strip()
R2_SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY", "").strip()
R2_PUBLIC_URL = os.environ.get("R2_PUBLIC_URL", "").strip().rstrip("/")
R2_BUCKET = os.environ.get("R2_BUCKET", "kv-audio").strip()
FEED_KEY = os.environ.get("R2_FEED_KEY", "feed.json").strip()


def r2_enabled() -> bool:
    return all([
        R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_PUBLIC_URL,
    ])


def _r2_client():
    import boto3
    from botocore.config import Config
    return boto3.client(
        "s3",
        endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )


def _upload_json_gzip(key: str, articles: list[dict], cache_control: str = "no-cache") -> str | None:
    """Internal: gzip-encode articles JSON and PUT to R2 under `key`. Returns public URL or None."""
    try:
        payload = json.dumps(
            {"articles": articles, "count": len(articles)},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        compressed = gzip.compress(payload, compresslevel=9)
        _r2_client().put_object(
            Bucket=R2_BUCKET,
            Key=key,
            Body=compressed,
            ContentType="application/json; charset=utf-8",
            ContentEncoding="gzip",
            CacheControl=cache_control,
        )
        url = f"{R2_PUBLIC_URL}/{key}"
        log.info(
            "uploaded → %s — %d articles (%d KB raw → %d KB gzip)",
            url, len(articles), len(payload) // 1024, len(compressed) // 1024,
        )
        return url
    except Exception as e:
        log.warning("upload failed for key=%s: %s", key, e)
        return None


def upload_feed(articles: list[dict]) -> str | None:
    """
    Serialize `articles` to JSON and upload to R2 as feed.json.
    Returns the public URL or None on failure (never raises — best effort).
    Cache-Control is short (5 min) since the cron refreshes every ~2h and
    the CDN edge revalidates cheaply; clients still get instant cached reads.
    """
    if not r2_enabled():
        log.info("feed export skipped — R2 not configured")
        return None
    return _upload_json_gzip(FEED_KEY, articles, cache_control="no-cache")


def upload_archive(articles: list[dict], date_str: str) -> str | None:
    """
    Upload a daily archive snapshot to R2 as feed_v{date_str}.json (e.g. feed_v20260604.json).
    Archives are immutable historical snapshots so Cache-Control is long (24h).
    Returns the public URL or None on failure.
    """
    if not r2_enabled():
        log.info("archive upload skipped — R2 not configured")
        return None
    key = f"feed_v{date_str}.json"
    return _upload_json_gzip(key, articles, cache_control="max-age=86400")


def archive_exists(date_str: str) -> bool:
    """
    Returns True if feed_v{date_str}.json already exists in R2 (HEAD check).
    Used to skip re-archiving if ingest runs multiple times after 6 AM.
    """
    if not r2_enabled():
        return False
    try:
        _r2_client().head_object(Bucket=R2_BUCKET, Key=f"feed_v{date_str}.json")
        return True
    except Exception:
        return False


def delete_archive(date_str: str) -> None:
    """Delete feed_v{date_str}.json from R2. Best-effort, never raises."""
    if not r2_enabled():
        return
    try:
        _r2_client().delete_object(Bucket=R2_BUCKET, Key=f"feed_v{date_str}.json")
        log.info("archive pruned: feed_v%s.json", date_str)
    except Exception as e:
        log.warning("archive prune failed for %s: %s", date_str, e)
