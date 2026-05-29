"""
KshanaVartha — Article working-set store on Cloudflare R2 (Decision 3).

SOURCE OF TRUTH
  All articles live as a single JSON list in R2 (articles.json), NOT Firestore.
  The cron loads it once at startup, works entirely in memory (dedup / AI polish /
  audio narration / prune), then writes it back at end-of-run. A separate
  Telugu-only public view (feed.json) is published to the same R2 bucket.

  Why not Firestore?
    Firestore free tier is ~50k reads/day. A 500-article store × 4 cron runs/day
    would burn that quota on article fetches alone, leaving nothing for real
    client reads. R2 JSON is a single GET/PUT per cron run regardless of size.
    Firestore is only used for collections that need client-write access:
      • fcm_tokens        — push subscription tokens (written by app)
      • community_reports — village reporter submissions (written by app)
      • live_data         — weather + mandi prices (written by cron, read by app)

DATA FLOW
  ingest.py main()
    load_articles()     ← R2 GET articles.json  (one call)
    … mutate in memory …
    save_articles()     → R2 PUT articles.json  (one call)
    export_feed_to_r2() → R2 PUT feed.json      (one call, Telugu-only view)

OFFLINE TESTING
  Set env KV_LOCAL_STORE=/path/to/articles.json. Reads/writes that local file
  instead of R2 — pipeline runs with NO network, NO R2 creds, NO quota.

Env (R2 mode): R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY,
  R2_BUCKET (default kv-audio), R2_ARTICLES_KEY (default articles.json).
"""
from __future__ import annotations

import json
import logging
import os

log = logging.getLogger("kv-ingest")

R2_ACCOUNT_ID = os.environ.get("R2_ACCOUNT_ID", "").strip()
R2_ACCESS_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID", "").strip()
R2_SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY", "").strip()
R2_BUCKET = os.environ.get("R2_BUCKET", "kv-audio").strip()
ARTICLES_KEY = os.environ.get("R2_ARTICLES_KEY", "articles.json").strip()


def _local_path() -> str:
    """If set, the store operates on this local file (offline test mode)."""
    return os.environ.get("KV_LOCAL_STORE", "").strip()


def _r2_enabled() -> bool:
    return all([R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY])


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


def load_articles() -> list[dict]:
    """
    Return the current article working set (list of dicts). Empty list if
    the store doesn't exist yet (fresh start — Decision 3: no migration).
    Never raises: a load failure must not wipe the set, so callers get []
    and should treat that as "couldn't load" (see save_articles guard).
    """
    local = _local_path()
    if local:
        if not os.path.isfile(local):
            return []
        try:
            with open(local, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return _coerce(data)
        except Exception as e:
            log.warning("article_store local load failed: %s", e)
            return []

    if not _r2_enabled():
        log.warning("article_store: R2 not configured and no KV_LOCAL_STORE")
        return []
    try:
        obj = _r2_client().get_object(Bucket=R2_BUCKET, Key=ARTICLES_KEY)
        data = json.loads(obj["Body"].read().decode("utf-8"))
        return _coerce(data)
    except Exception as e:
        # Missing key on first ever run is normal → fresh empty set.
        log.info("article_store: no existing %s (%s) — starting fresh",
                 ARTICLES_KEY, type(e).__name__)
        return []


def save_articles(articles: list[dict]) -> bool:
    """
    Persist the full working set with merge-on-drift protection.

    Before writing, re-fetches the current R2 state and merges any articles
    that were added by a concurrent run (e.g. GitHub Actions cron firing
    while a local manual run was in progress). This prevents the classic
    read-modify-write race condition where two processes load the same
    snapshot, work independently, and the last writer silently discards
    the other's new articles.

    Merge strategy (by article ID):
      • Articles in OUR store    → use our version (preserves audioUrl
                                   updates, AI polish, etc.)
      • Articles in R2 but NOT   → preserve as-is (added by the concurrent
        in our store               run while we were processing)

    If R2 has >50 more articles than us, that means our load was stale
    (not just a small concurrent race) — we log a WARNING so the operator
    knows something unusual happened.

    Only applies in R2 mode. Local KV_LOCAL_STORE test mode skips the
    merge (no concurrency concern in offline tests).
    """
    if not articles and os.environ.get("KV_ALLOW_EMPTY_STORE") != "1":
        log.warning("article_store: refusing to save empty set "
                    "(set KV_ALLOW_EMPTY_STORE=1 to override)")
        return False

    local = _local_path()   # resolved once; used by merge block + write block

    # ── Merge-on-drift safeguard (R2 mode only) ───────────────────────────
    if not local and _r2_enabled():
        try:
            r2_current = load_articles()
            r2_count   = len(r2_current)
            our_count  = len(articles)
            our_ids    = {a["id"] for a in articles}
            # Articles added to R2 while we were running (concurrent write)
            r2_only    = [a for a in r2_current if a["id"] not in our_ids]
            if r2_only:
                articles = articles + r2_only   # our updates win; R2-only preserved
                if r2_count > our_count + 50:
                    # Big drift → stale load, not just a small race
                    log.warning(
                        "article_store: STALE LOAD detected — "
                        "R2 had %d articles, we loaded %d. "
                        "Merged +%d R2-only article(s) → saving %d total. "
                        "Tip: avoid running local ingest while GitHub Actions cron is active.",
                        r2_count, our_count, len(r2_only), len(articles),
                    )
                else:
                    log.info(
                        "article_store: merged %d article(s) from concurrent run "
                        "(R2=%d our=%d) → %d total",
                        len(r2_only), r2_count, our_count, len(articles),
                    )
        except Exception as e:
            log.warning("article_store: merge-check failed (%s) — saving as-is", e)
    # ─────────────────────────────────────────────────────────────────────

    payload = json.dumps(
        {"articles": articles, "count": len(articles)},
        ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")

    if local:  # set above in the merge block (or "" in R2 mode)
        try:
            with open(local, "w", encoding="utf-8") as fh:
                fh.write(payload.decode("utf-8"))
            return True
        except Exception as e:
            log.warning("article_store local save failed: %s", e)
            return False

    if not _r2_enabled():
        log.warning("article_store: R2 not configured — save skipped")
        return False
    try:
        _r2_client().put_object(
            Bucket=R2_BUCKET, Key=ARTICLES_KEY, Body=payload,
            ContentType="application/json; charset=utf-8",
            CacheControl="no-store",   # working set, not public-cached
        )
        log.info("article_store: saved %d articles → R2/%s",
                 len(articles), ARTICLES_KEY)
        return True
    except Exception as e:
        log.warning("article_store save failed: %s", e)
        return False


def _coerce(data) -> list[dict]:
    """Accept either {"articles":[...]} or a bare [...]; ignore junk."""
    if isinstance(data, dict):
        data = data.get("articles", [])
    if not isinstance(data, list):
        return []
    return [a for a in data if isinstance(a, dict)]
