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

MAX_PER_CLASS = 250   # cap per category to keep classes balanced
MIN_TEXT_LEN  = 40    # drop very short stubs
CATEGORIES    = ["politics", "farming", "weather", "jobs",
                 "village", "sports", "cinema", "schemes"]


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
    parser.add_argument("--articles", default=DEFAULT_ARTICLES,
                        help="Path to articles.json")
    parser.add_argument("--model", default="category_model.pkl",
                        help="Output model path (default: category_model.pkl)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print full per-class metrics")
    args = parser.parse_args()

    if not os.path.isfile(args.articles):
        log.error("articles.json not found at: %s", args.articles)
        log.error("Pass --articles /path/to/articles.json")
        sys.exit(1)

    # ── Check scikit-learn ────────────────────────────────────────────────
    try:
        import sklearn  # noqa: F401
    except ImportError:
        log.error("scikit-learn not installed. Run: pip install scikit-learn")
        sys.exit(1)

    from categorizer import CategoryClassifier, CONFIDENCE_THRESHOLD

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
    print(f"OK Model saved -> {os.path.abspath(args.model)}")
    print()
    print("Next steps:")
    print("  1. Copy category_model.pkl to the kshanavartha-ingest folder.")
    print("  2. The ingest pipeline picks it up automatically on next run.")
    print("  3. Retrain monthly as more articles accumulate.")


if __name__ == "__main__":
    main()
