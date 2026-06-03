"""
One-time script to train the KshanaVartha category classifier.

Usage
-----
    # From within kshanavartha-ingest/:
    python train_categorizer.py

    # Point to a different articles.json:
    python train_categorizer.py --articles /path/to/articles.json

    # Retrain and show per-class report:
    python train_categorizer.py --verbose

What it does
------------
1. Reads articles.json (from admin/data/ by default).
2. Selects high-confidence training examples:
   - ai=True  (AI-polished = clean Telugu text)
   - category != 'general' (keyword-matched = reliable label)
   - sufficient text length (avoids empty/stub articles)
3. Balances classes (max MAX_PER_CLASS examples per category).
4. Trains TF-IDF + LogisticRegression via 5-fold cross-validation.
5. Saves the model to category_model.pkl.
6. Prints a per-class accuracy report.

Rerun whenever you have more articles (monthly is plenty).
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import random
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("train")

# ── Configuration ──────────────────────────────────────────────────────────
DEFAULT_ARTICLES = os.path.join(
    os.path.dirname(__file__), "..", "KshnaVartha", "admin", "data", "articles.json"
)
# Fallback: try sibling admin folder (common local layout)
_ALT = os.path.join(os.path.dirname(__file__), "..", "admin", "data", "articles.json")
if not os.path.isfile(DEFAULT_ARTICLES) and os.path.isfile(_ALT):
    DEFAULT_ARTICLES = _ALT

MAX_PER_CLASS  = 250   # cap per category to keep classes balanced
MIN_TEXT_LEN   = 40    # drop very short stubs
CATEGORIES     = ["politics", "farming", "weather", "jobs",
                  "village", "sports", "cinema", "schemes"]
CATEGORIES_ALL = CATEGORIES + ["general"]   # JSONL training includes 'general'


def load_examples_jsonl(jsonl_path: str) -> tuple[list[str], list[str]]:
    """
    Read training_data.jsonl and return (texts, labels) for training.

    Includes ALL categories (including 'general') and all origins (RSS +
    WhatsApp). The classifier uses class_weight=balanced so 'general' does
    not dominate. No ai/non-ai split needed — quality is already good.
    """
    log.info("Loading JSONL training data from %s", os.path.abspath(jsonl_path))
    by_cat: dict[str, list[str]] = {c: [] for c in CATEGORIES_ALL}
    total = 0
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            cat  = (entry.get("cat") or "general").strip().lower()
            text = (entry.get("text") or "").strip()
            if cat not in by_cat or len(text) < MIN_TEXT_LEN:
                continue
            by_cat[cat].append(text)
            total += 1
    log.info("Total JSONL entries: %d", total)

    texts: list[str] = []
    labels: list[str] = []
    counts: dict[str, int] = {}
    for cat, examples in by_cat.items():
        random.seed(42)
        random.shuffle(examples)
        chosen = examples[:MAX_PER_CLASS]
        texts.extend(chosen)
        labels.extend([cat] * len(chosen))
        counts[cat] = len(chosen)

    log.info("Training examples per category:")
    for cat, n in sorted(counts.items(), key=lambda x: -x[1]):
        if n > 0:
            log.info("  %-12s %d", cat, n)
    log.info("Total training examples: %d", len(texts))
    return texts, labels


def load_level_examples_jsonl(jsonl_path: str) -> tuple[list[str], list[str]]:
    """
    Read training_data.jsonl and return (texts, labels) for level training.

    Only includes entries with a known, non-empty `lvl` field. Entries
    whose level was never resolved (empty string) are skipped — we'd rather
    have fewer but reliable labels than noisy ones.
    """
    VALID = {"village", "mandal", "district", "state", "national"}
    log.info("Loading JSONL level training data from %s", os.path.abspath(jsonl_path))
    by_lvl: dict[str, list[str]] = {l: [] for l in VALID}
    total = 0
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            lvl  = (entry.get("lvl") or "").strip().lower()
            text = (entry.get("text") or "").strip()
            if lvl not in VALID or len(text) < MIN_TEXT_LEN:
                continue
            by_lvl[lvl].append(text)
            total += 1
    log.info("Total JSONL entries with level: %d", total)

    texts: list[str] = []
    labels: list[str] = []
    counts: dict[str, int] = {}
    for lvl, examples in by_lvl.items():
        random.seed(42)
        random.shuffle(examples)
        chosen = examples[:MAX_PER_CLASS]
        texts.extend(chosen)
        labels.extend([lvl] * len(chosen))
        counts[lvl] = len(chosen)

    log.info("Level training examples:")
    for lvl, n in sorted(counts.items(), key=lambda x: -x[1]):
        if n > 0:
            log.info("  %-12s %d", lvl, n)
    log.info("Total level training examples: %d", len(texts))
    return texts, labels


def load_examples(articles_path: str) -> tuple[list[str], list[str]]:
    """
    Read articles.json and return (texts, labels) for training.

    Selection criteria (in order of confidence):
      Tier 1 — ai=True, category known, not 'general'  → most reliable
      Tier 2 — ai=False, category known, not 'general' → acceptable backup
    """
    log.info("Loading articles from %s", os.path.abspath(articles_path))
    with open(articles_path, encoding="utf-8") as f:
        data = json.load(f)

    arts = data.get("articles") or data  # handle both wrapped and raw list
    log.info("Total articles: %d", len(arts))

    by_cat: dict[str, list[str]] = {c: [] for c in CATEGORIES}

    for a in arts:
        cat = (a.get("category") or "").strip().lower()
        if cat not in CATEGORIES:
            continue
        headline = (a.get("headline") or "").strip()
        summary  = (a.get("summary")  or "").strip()
        text = f"{headline} {summary}".strip()
        if len(text) < MIN_TEXT_LEN:
            continue
        by_cat[cat].append((text, bool(a.get("ai"))))

    # Sort: ai=True examples first (higher quality), then cap
    texts:  list[str] = []
    labels: list[str] = []
    counts: dict[str, int] = {}
    for cat, examples in by_cat.items():
        # Tier-1 first, then Tier-2, shuffle within each tier
        tier1 = [(t, c) for t, c in examples if c]
        tier2 = [(t, c) for t, c in examples if not c]
        random.seed(42)
        random.shuffle(tier1); random.shuffle(tier2)
        chosen = (tier1 + tier2)[:MAX_PER_CLASS]
        for text, _ in chosen:
            texts.append(text)
            labels.append(cat)
        counts[cat] = len(chosen)

    log.info("Training examples per category:")
    for cat, n in sorted(counts.items(), key=lambda x: -x[1]):
        log.info("  %-12s %d", cat, n)
    log.info("Total training examples: %d", len(texts))
    return texts, labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the KshanaVartha category classifier")
    parser.add_argument("--training-data", default=None,
                        help="Path to training_data.jsonl (preferred; 30-day compact store)")
    parser.add_argument("--articles", default=DEFAULT_ARTICLES,
                        help="Path to articles.json (fallback when --training-data not given)")
    parser.add_argument("--model", default="category_model.pkl",
                        help="Output category model path (default: category_model.pkl)")
    parser.add_argument("--level-model", default="level_model.pkl",
                        help="Output level model path (default: level_model.pkl); "
                             "only trained when --training-data is provided")
    parser.add_argument("--verbose", action="store_true",
                        help="Print full per-class metrics")
    args = parser.parse_args()

    use_jsonl = args.training_data and os.path.isfile(args.training_data)
    if not use_jsonl and not os.path.isfile(args.articles):
        log.error("No training data found.")
        log.error("  --training-data %s", args.training_data or "(not set)")
        log.error("  --articles      %s", args.articles)
        sys.exit(1)

    # ── Check scikit-learn ────────────────────────────────────────────────
    try:
        import sklearn  # noqa: F401
    except ImportError:
        log.error("scikit-learn not installed. Run: pip install scikit-learn")
        sys.exit(1)

    from categorizer import (
        CategoryClassifier, CONFIDENCE_THRESHOLD,
        LevelClassifier, LEVEL_CONFIDENCE_THRESHOLD, LEVELS,
    )

    if use_jsonl:
        log.info("Using JSONL training data (30-day compact store)")
        texts, labels = load_examples_jsonl(args.training_data)
    else:
        log.info("Using articles.json (JSONL not available — fallback path)")
        texts, labels = load_examples(args.articles)

    if len(texts) < 50:
        log.error("Not enough training data (found %d, need >= 50). "
                  "Run the ingest pipeline first to build up articles.", len(texts))
        sys.exit(1)

    log.info("Training classifier…")
    model = CategoryClassifier()
    stats = model.train(texts, labels)

    # ── Report ────────────────────────────────────────────────────────────
    print()
    print("=" * 55)
    print(f"  Cross-val accuracy: {stats['cv_accuracy']*100:.1f}% ± {stats['cv_std']*100:.1f}%")
    print(f"  Training examples:  {stats['n_train']}")
    print(f"  Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print()
    if args.verbose or True:  # always show per-class
        print("  Per-class (on held-out CV folds):")
        print(f"  {'Category':<12}  {'Precision':>9}  {'Recall':>7}  {'F1':>5}  {'Support':>7}")
        print("  " + "-" * 47)
        for cat, m in sorted(stats["per_class"].items(), key=lambda x: -x[1]["f1"]):
            print(f"  {cat:<12}  {m['precision']:>9.3f}  {m['recall']:>7.3f}  "
                  f"{m['f1']:>5.3f}  {m['support']:>7}")
    print("=" * 55)
    print()

    model.save(args.model)
    print(f"OK category_model saved -> {os.path.abspath(args.model)}")

    # ── Level model (only when JSONL available — needs lvl field) ─────────
    if use_jsonl:
        print()
        lvl_texts, lvl_labels = load_level_examples_jsonl(args.training_data)
        if len(lvl_texts) >= 50:
            log.info("Training level classifier…")
            lvl_model = LevelClassifier()
            lvl_stats = lvl_model.train(lvl_texts, lvl_labels)
            print()
            print("=" * 55)
            print(f"  LEVEL — Cross-val accuracy: {lvl_stats['cv_accuracy']*100:.1f}%"
                  f" ± {lvl_stats['cv_std']*100:.1f}%")
            print(f"  Training examples:  {lvl_stats['n_train']}")
            print(f"  Confidence threshold: {LEVEL_CONFIDENCE_THRESHOLD}")
            print()
            print(f"  {'Level':<12}  {'Precision':>9}  {'Recall':>7}  {'F1':>5}  {'Support':>7}")
            print("  " + "-" * 47)
            for lvl, m in sorted(lvl_stats["per_class"].items(), key=lambda x: -x[1]["f1"]):
                print(f"  {lvl:<12}  {m['precision']:>9.3f}  {m['recall']:>7.3f}  "
                      f"{m['f1']:>5.3f}  {m['support']:>7}")
            print("=" * 55)
            print()
            lvl_model.save(args.level_model)
            print(f"OK level_model saved -> {os.path.abspath(args.level_model)}")
        else:
            log.warning("Not enough level-labelled entries (%d) — skipping level model", len(lvl_texts))

    print()
    print("Next steps:")
    print("  Both models are picked up automatically by ingest.py on next run.")


if __name__ == "__main__":
    main()
