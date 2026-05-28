"""
KshanaVartha — AI engine key tester.

Tests every Cerebras, Gemini, and SambaNova key individually and reports
its exact status. Use this before a run to confirm which engines are live.

Usage (PowerShell):
    $env:CEREBRAS_API_KEYS   = "csk-key1,csk-key2"
    $env:GEMINI_API_KEYS     = "AIza-key1,AIza-key2"
    $env:SAMBANOVA_API_KEYS  = "snk-key1,snk-key2"
    python ingest/test_keys.py

    # Or single-key vars still work:
    $env:CEREBRAS_API_KEY  = "csk-..."
    $env:GEMINI_API_KEY    = "AIza-..."
    $env:SAMBANOVA_API_KEY = "..."

What each status means:
    ✅ OK           — key works, engine returned a Telugu response
    💰 QUOTA (429)  — key valid but daily/minute quota exhausted
    🚫 BANNED (400) — account restricted by provider
    ❌ NOT FOUND (404) — model name doesn't exist on this account
    🔑 INVALID (403/401) — key itself is wrong or API not enabled
    ⚠️  ERROR       — unexpected HTTP or network error
"""

from __future__ import annotations

import os
import sys
import time

import requests

# ── Sanitise keys (same logic as ingest.py) ──────────────────────────────────
def _clean(raw: str) -> str:
    return "".join(c for c in raw if c.isprintable() and not c.isspace()).strip()

def _keys(plural_var: str, single_var: str) -> list[str]:
    raw = os.environ.get(plural_var, "") or os.environ.get(single_var, "")
    return [_clean(k) for k in raw.split(",") if _clean(k)]

CEREBRAS_KEYS   = _keys("CEREBRAS_API_KEYS",  "CEREBRAS_API_KEY")
GEMINI_KEYS     = _keys("GEMINI_API_KEYS",    "GEMINI_API_KEY")
SAMBANOVA_KEYS  = _keys("SAMBANOVA_API_KEYS", "SAMBANOVA_API_KEY")

# ── Model names (same as ingest.py) ──────────────────────────────────────────
CEREBRAS_MODEL  = os.environ.get("CEREBRAS_MODEL",  "gpt-oss-120b").strip()
SAMBANOVA_MODEL = os.environ.get("SAMBANOVA_MODEL", "Meta-Llama-3.3-70B-Instruct").strip()
GEMINI_MODEL    = "gemini-2.0-flash"

CEREBRAS_URL  = "https://api.cerebras.ai/v1/chat/completions"
SAMBANOVA_URL = "https://api.sambanova.ai/v1/chat/completions"
GEMINI_URL    = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)

# Short test prompt — just enough to prove the engine can reply in Telugu.
# Does NOT use the full SUMMARY_PROMPT so it won't burn meaningful quota.
TEST_PROMPT = (
    "ఒక్క వాక్యంలో తెలుగులో చెప్పండి: ఈరోజు వాతావరణం ఎలా ఉంది?"
)

TIMEOUT = 20  # seconds per request

# ── Colour helpers ────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def _status(code: int, body: str) -> tuple[str, str]:
    """Return (icon+label, detail) from an HTTP response."""
    low = body.lower()
    if code == 200:
        return f"{GREEN}✅ OK{RESET}", ""
    if code == 429:
        return f"{YELLOW}💰 QUOTA (429){RESET}", "daily/minute quota exhausted — will recover at reset"
    if code in (401, 403):
        return f"{RED}🔑 INVALID KEY ({code}){RESET}", "key itself is wrong or API/billing not enabled"
    if code == 404:
        return f"{RED}❌ NOT FOUND (404){RESET}", "model name doesn't exist on this account"
    if code == 400:
        if "restricted" in low or "organization" in low:
            return f"{RED}🚫 BANNED (400){RESET}", "account/org restricted — contact provider support"
        return f"{RED}❌ BAD REQUEST (400){RESET}", body[:120]
    return f"{RED}⚠️  HTTP {code}{RESET}", body[:120]

def _mask(key: str) -> str:
    """Show only last 6 chars so logs are safe to share."""
    return f"...{key[-6:]}" if len(key) > 6 else key

# ── Cerebras ─────────────────────────────────────────────────────────────────
def test_cerebras() -> None:
    print(f"\n{BOLD}{'─'*60}{RESET}")
    print(f"{BOLD}{CYAN}CEREBRAS  (model: {CEREBRAS_MODEL}){RESET}")
    print(f"{'─'*60}")
    if not CEREBRAS_KEYS:
        print(f"  {YELLOW}⚠️  No keys found (CEREBRAS_API_KEYS / CEREBRAS_API_KEY){RESET}")
        return

    for i, key in enumerate(CEREBRAS_KEYS, 1):
        print(f"  Key {i}/{len(CEREBRAS_KEYS)}  {_mask(key)}  ", end="", flush=True)
        body = {
            "model": CEREBRAS_MODEL,
            "messages": [{"role": "user", "content": TEST_PROMPT}],
            "max_tokens": 80,
            "temperature": 0.3,
        }
        try:
            r = requests.post(
                CEREBRAS_URL,
                headers={"Authorization": f"Bearer {key}"},
                json=body, timeout=TIMEOUT,
            )
        except Exception as e:
            print(f"{RED}⚠️  NETWORK ERROR: {e}{RESET}")
            continue

        label, detail = _status(r.status_code, r.text)
        print(label)
        if detail:
            print(f"             {detail}")
        if r.status_code == 200:
            try:
                reply = r.json()["choices"][0]["message"]["content"].strip()
                print(f"             reply: {reply[:100]}")
            except Exception:
                pass
        time.sleep(0.5)

    # Also list available models so wrong model names are easy to spot
    print(f"\n  {CYAN}Available Cerebras models on key 1:{RESET}")
    try:
        r = requests.get(
            "https://api.cerebras.ai/v1/models",
            headers={"Authorization": f"Bearer {CEREBRAS_KEYS[0]}"},
            timeout=TIMEOUT,
        )
        if r.status_code == 200:
            models = r.json().get("data", [])
            for m in models:
                mid = m.get("id", "?")
                marker = f"  {GREEN}← CURRENT{RESET}" if mid == CEREBRAS_MODEL else ""
                print(f"    • {mid}{marker}")
            if not models:
                print("    (no models returned)")
        else:
            print(f"    {RED}Could not list models: HTTP {r.status_code}{RESET}")
    except Exception as e:
        print(f"    {RED}Network error listing models: {e}{RESET}")


# ── Gemini ────────────────────────────────────────────────────────────────────
def test_gemini() -> None:
    print(f"\n{BOLD}{'─'*60}{RESET}")
    print(f"{BOLD}{CYAN}GEMINI FLASH  (model: {GEMINI_MODEL}){RESET}")
    print(f"{'─'*60}")
    if not GEMINI_KEYS:
        print(f"  {YELLOW}⚠️  No keys found (GEMINI_API_KEYS / GEMINI_API_KEY){RESET}")
        return

    body = {
        "contents": [{"role": "user", "parts": [{"text": TEST_PROMPT}]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 80},
    }
    for i, key in enumerate(GEMINI_KEYS, 1):
        print(f"  Key {i}/{len(GEMINI_KEYS)}  {_mask(key)}  ", end="", flush=True)
        try:
            r = requests.post(f"{GEMINI_URL}?key={key}", json=body, timeout=TIMEOUT)
        except Exception as e:
            print(f"{RED}⚠️  NETWORK ERROR: {e}{RESET}")
            continue

        label, detail = _status(r.status_code, r.text)
        print(label)
        if detail:
            print(f"             {detail}")
        if r.status_code == 200:
            try:
                reply = (r.json()["candidates"][0]["content"]["parts"][0]["text"]).strip()
                print(f"             reply: {reply[:100]}")
            except Exception:
                pass
        time.sleep(0.3)


# ── SambaNova ─────────────────────────────────────────────────────────────────
def test_sambanova() -> None:
    print(f"\n{BOLD}{'─'*60}{RESET}")
    print(f"{BOLD}{CYAN}SAMBANOVA  (model: {SAMBANOVA_MODEL}){RESET}")
    print(f"{'─'*60}")
    if not SAMBANOVA_KEYS:
        print(f"  {YELLOW}⚠️  No keys found (SAMBANOVA_API_KEYS / SAMBANOVA_API_KEY){RESET}")
        return

    body = {
        "model": SAMBANOVA_MODEL,
        "messages": [{"role": "user", "content": TEST_PROMPT}],
        "max_tokens": 80,
        "temperature": 0.3,
    }
    for i, key in enumerate(SAMBANOVA_KEYS, 1):
        print(f"  Key {i}/{len(SAMBANOVA_KEYS)}  {_mask(key)}  ", end="", flush=True)
        try:
            r = requests.post(
                SAMBANOVA_URL,
                headers={"Authorization": f"Bearer {key}"},
                json=body, timeout=TIMEOUT,
            )
        except Exception as e:
            print(f"{RED}⚠️  NETWORK ERROR: {e}{RESET}")
            continue

        label, detail = _status(r.status_code, r.text)
        print(label)
        if detail:
            print(f"             {detail}")
        if r.status_code == 200:
            try:
                reply = r.json()["choices"][0]["message"]["content"].strip()
                print(f"             reply: {reply[:100]}")
            except Exception:
                pass
        time.sleep(0.5)


# ── Summary ───────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"\n{BOLD}KshanaVartha — AI Engine Key Tester{RESET}")
    print(f"Test prompt: \"{TEST_PROMPT}\"")
    print(f"Keys loaded:  cerebras={len(CEREBRAS_KEYS)}  gemini={len(GEMINI_KEYS)}  sambanova={len(SAMBANOVA_KEYS)}")

    if not any([CEREBRAS_KEYS, GEMINI_KEYS, SAMBANOVA_KEYS]):
        print(f"\n{RED}No keys found in environment. Set at least one of:{RESET}")
        print("  CEREBRAS_API_KEYS, GEMINI_API_KEYS, SAMBANOVA_API_KEYS")
        sys.exit(1)

    test_cerebras()
    test_gemini()
    test_sambanova()

    print(f"\n{BOLD}{'─'*60}{RESET}")
    print(f"{BOLD}Done.{RESET}  Fix any 🚫/❌ engines before running the ingest pipeline.")
    print(f"Production engine priority: Cerebras → Gemini → SambaNova\n")


if __name__ == "__main__":
    main()
