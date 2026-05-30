"""
KshanaVartha — Telugu TTS (gTTS) + Cloudflare R2 audio upload.

Free, no API key, no card: gTTS uses Google Translate's TTS endpoint
(lang='te'), outputs MP3 directly, and works reliably from CI runners.
(edge-tts was abandoned — Microsoft 403s GitHub Actions datacenter IPs.)

Audio is uploaded to R2 as audio/{article_id}.mp3. The article record in
articles.json stores the public CDN URL in the `audioUrl` field. Audio is
narrated incrementally — up to AUDIO_MAX_PER_RUN articles per cron run —
so a large backlog spreads across multiple runs instead of blowing the
cron budget in one go.

Storage bounds: the ingest prune step calls delete_audio() for every
article older than RETENTION_DAYS (14 days), deleting the R2 MP3 at the
same time as the article record. R2 bucket stays lean automatically.

TTS ENGINES (controlled by TTS_ENGINE env var)
  gtts    — default. Google Translate TTS, free, no billing needed.
  gcloud  — Google Cloud TTS (news-reader tone, speaking-rate control).
            Requires GCP billing account even for the free tier. Reuses
            the Firebase service account (same GCP project) for auth.
            Override voice: TTS_VOICE_NAME=te-IN-Standard-B (male).

R2 LAYOUT
  audio/{id}.mp3   — narrated articles (canonical path since 2026-05-23)
  {id}.mp3         — legacy root-level path (pre-2026-05-23); orphans
                     age out naturally via RETENTION_DAYS prune.

Env (all optional — missing R2 creds → audio silently skipped):
  R2_ACCOUNT_ID         Cloudflare account id
  R2_ACCESS_KEY_ID      R2 API token key id
  R2_SECRET_ACCESS_KEY  R2 API token secret
  R2_PUBLIC_URL         https://pub-xxxx.r2.dev  (bucket public base URL)
  R2_BUCKET             bucket name (default: kv-audio)
  TTS_ENGINE            "gtts" (default) or "gcloud"
  TTS_VOICE_NAME        Cloud TTS voice (default: te-IN-Standard-A, female)
  TTS_RATE              Cloud TTS speaking rate (default: 1.05; gTTS ignores)
"""

from __future__ import annotations

import logging
import os
import tempfile

log = logging.getLogger("kv-ingest")

R2_ACCOUNT_ID = os.environ.get("R2_ACCOUNT_ID", "").strip()
R2_ACCESS_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID", "").strip()
R2_SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY", "").strip()
R2_PUBLIC_URL = os.environ.get("R2_PUBLIC_URL", "").strip().rstrip("/")
R2_BUCKET = os.environ.get("R2_BUCKET", "kv-audio").strip()


def r2_enabled() -> bool:
    """True only if every R2 credential is present."""
    return all([
        R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_PUBLIC_URL,
    ])


def _r2_client():
    import boto3  # imported lazily so a missing dep never breaks ingest
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )


# TTS engine config. Default: FREE gTTS (no API key, no GCP billing —
# matches the project's free-only / no-card constraint). Google Cloud TTS
# (news-reader tone, rate control) stays available but OFF by default —
# it needs a GCP billing account even for its free tier. Enable later
# with TTS_ENGINE=gcloud. Override per-run with env:
#   TTS_ENGINE=gcloud          → use Cloud TTS (requires GCP billing)
#   TTS_VOICE_NAME=te-IN-Standard-B  → male voice (default Standard-A female)
#   TTS_RATE=1.05              → Cloud TTS speaking rate (gTTS ignores this)
TTS_ENGINE = os.environ.get("TTS_ENGINE", "gtts").strip().lower()
TTS_VOICE_NAME = os.environ.get("TTS_VOICE_NAME", "te-IN-Standard-A").strip()
TTS_RATE = float(os.environ.get("TTS_RATE", "1.05"))


def _gtts_synth(text: str, path: str) -> None:
    from gtts import gTTS  # lazy import — missing dep never breaks ingest
    gTTS(text=text, lang="te").save(path)


def _gcloud_creds():
    """Reuse the Firebase service account (same GCP project) for Cloud TTS."""
    import json
    from google.oauth2 import service_account
    raw = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON")
    if raw:
        return service_account.Credentials.from_service_account_info(
            json.loads(raw))
    for p in ("service-account.json",
              os.path.join(os.path.dirname(__file__), "..", "service-account.json")):
        if os.path.isfile(p):
            return service_account.Credentials.from_service_account_file(p)
    return None  # let the client try ADC; else _synth falls back to gTTS


def _gcloud_synth(text: str, path: str) -> None:
    from google.cloud import texttospeech  # lazy import
    creds = _gcloud_creds()
    client = (texttospeech.TextToSpeechClient(credentials=creds)
              if creds else texttospeech.TextToSpeechClient())
    resp = client.synthesize_speech(
        input=texttospeech.SynthesisInput(text=text),
        voice=texttospeech.VoiceSelectionParams(
            language_code="te-IN", name=TTS_VOICE_NAME),
        audio_config=texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=TTS_RATE),
    )
    with open(path, "wb") as fh:
        fh.write(resp.audio_content)


def _synth(text: str, path: str) -> None:
    """Cloud TTS (news-reader voice, rate-controlled) → gTTS fallback."""
    if TTS_ENGINE != "gtts":
        try:
            _gcloud_synth(text, path)
            return
        except Exception as e:
            log.warning("Cloud TTS failed (%s) — falling back to gTTS",
                        type(e).__name__)
    _gtts_synth(text, path)


def _recompress_mp3_file(path: str) -> None:
    """Re-encode an MP3 file in-place at 64 kbps / 22 050 Hz mono via ffmpeg.

    gTTS outputs 128 kbps MP3 (~128 KB/min of speech). At 64 kbps mono
    22 kHz, speech is fully intelligible and the file is ~50% smaller —
    a meaningful saving when storing hundreds of narrated articles on R2.

    Best-effort: no-op if ffmpeg is absent or if re-encoding fails.
    GitHub Actions ubuntu-latest has ffmpeg pre-installed.
    """
    import shutil
    import subprocess

    if not shutil.which("ffmpeg"):
        log.debug("_recompress_mp3_file: ffmpeg not found — skipping")
        return
    tmp = path + ".reenc.mp3"
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", path,
                "-ar", "22050",  # 22 kHz — adequate for speech
                "-ac", "1",      # mono
                "-b:a", "64k",   # 64 kbps CBR
                tmp,
            ],
            capture_output=True,
            timeout=60,
        )
        if result.returncode != 0:
            log.warning(
                "_recompress_mp3_file: ffmpeg exited %d for %s",
                result.returncode, path,
            )
            return
        new_size = os.path.getsize(tmp)
        if new_size == 0:
            return
        orig_size = os.path.getsize(path)
        os.replace(tmp, path)   # atomic in-place replace
        log.info(
            "_recompress_mp3_file: %s %d KB → %d KB (saved %.0f%%)",
            os.path.basename(path),
            orig_size // 1024, new_size // 1024,
            100 * (1 - new_size / max(orig_size, 1)),
        )
    except Exception as e:
        log.warning("_recompress_mp3_file: exception for %s: %s", path, e)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def synthesize_and_upload(article_id: str, text: str) -> str | None:
    """
    Narrate `text` (the Telugu summary) and upload as
    {article_id}.mp3 to R2. Returns the public URL, or None on any
    failure (never raises — audio is best-effort, must not break ingest).
    Retries once after 5 s to handle transient gTTS / network errors.
    """
    import time

    if not r2_enabled():
        return None
    text = (text or "").strip()
    if len(text) < 10:
        return None

    for attempt in range(2):
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tmp = f.name
            _synth(text, tmp)
            if not os.path.getsize(tmp):
                log.warning("tts produced empty audio for %s (attempt %d)",
                            article_id, attempt + 1)
                raise ValueError("empty audio")

            # Re-encode at 64 kbps mono — ~50% smaller than raw gTTS output.
            # No-op if ffmpeg not installed; never raises.
            _recompress_mp3_file(tmp)

            # Canonical R2 layout: audio under audio/<id>.mp3, matching
            # what the admin tool uses (admin/main.py uploads to
            # audio/<filename>). Bucket root stays for JSON manifests
            # only (articles.json, feed.json, blocklist.json).
            # Pre-2026-05-23 ingest wrote {id}.mp3 at root; those
            # orphans age out via RETENTION_DAYS so no migration.
            key = f"audio/{article_id}.mp3"
            with open(tmp, "rb") as fh:
                _r2_client().put_object(
                    Bucket=R2_BUCKET,
                    Key=key,
                    Body=fh,
                    ContentType="audio/mpeg",
                    CacheControl="public, max-age=31536000, immutable",
                )
            return f"{R2_PUBLIC_URL}/{key}"
        except Exception as e:
            log.warning("audio synth/upload failed for %s (attempt %d): %s",
                        article_id, attempt + 1, e)
            if attempt == 0:
                time.sleep(5)  # wait before retry
        finally:
            if tmp and os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
    return None


def delete_audio(article_id: str) -> None:
    """Single-article delete — kept for ad-hoc / admin use.
    For prune (many articles at once) use delete_audio_bulk() instead.
    """
    delete_audio_bulk([article_id])


def delete_media_bulk(article_ids: list[str]) -> None:
    """Delete R2 audio + image files for multiple articles in ONE API call.

    WHY: the old per-article loop called delete_object() twice per article
    (canonical path + legacy root path). Pruning 30 articles = 60 separate
    R2 API requests. R2's delete_objects API accepts up to 1,000 keys per
    request, so this collapses the entire prune batch into a single call
    regardless of how many articles are dropped.

    Keys built per article:
      audio/{id}.mp3        — canonical audio path (since 2026-05-23)
      {id}.mp3              — legacy root path (pre-2026-05-23 orphans)
      images/{id}_0..4.webp — admin-uploaded images (WebP, up to 5 per article)
      images/{id}_0..4.jpg  — legacy JPG format (pre-WebP conversion)

    Missing keys are silently ignored by R2 (idempotent). Trying indices
    0-4 for images is safe — most articles have 0-2 images and R2 treats
    a delete of a non-existent key as a no-op.
    """
    if not r2_enabled() or not article_ids:
        return
    keys = []
    for aid in article_ids:
        if not aid:
            continue
        # Audio — canonical + legacy root path
        keys.append({"Key": f"audio/{aid}.mp3"})
        keys.append({"Key": f"{aid}.mp3"})
        # Images — try indices 0-4 in both WebP (current) and JPG (legacy)
        for idx in range(5):
            keys.append({"Key": f"images/{aid}_{idx}.webp"})
            keys.append({"Key": f"images/{aid}_{idx}.jpg"})
    if not keys:
        return
    # Chunk at 1000 — S3/R2 hard limit per delete_objects request.
    # Per article: 2 audio + 10 image keys = 12 keys. A prune batch of
    # ~80 articles → ~960 keys, fits in a single API call.
    _CHUNK = 1000
    for i in range(0, len(keys), _CHUNK):
        batch = keys[i : i + _CHUNK]
        try:
            resp = _r2_client().delete_objects(
                Bucket=R2_BUCKET,
                Delete={"Objects": batch, "Quiet": True},
            )
            # Quiet=True suppresses per-key confirmations; only errors returned.
            for err in resp.get("Errors", []):
                log.debug("delete_objects partial error — key=%s msg=%s",
                          err.get("Key"), err.get("Message"))
            log.info("delete_media_bulk: %d keys attempted in 1 R2 call", len(batch))
        except Exception as e:
            log.warning("delete_media_bulk failed (batch %d keys): %s", len(batch), e)


# Keep old name as alias so any external callers aren't broken.
delete_audio_bulk = delete_media_bulk
