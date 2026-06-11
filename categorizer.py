"""
KshanaVartha article category classifier.

Uses a TF-IDF character n-gram vectorizer + Logistic Regression.
Character n-grams (3-6 chars) work well for Telugu without needing
a language-specific tokenizer — they naturally capture Telugu morpheme
boundaries and handle English words in the mix.

Usage
-----
Training (run once via train_categorizer.py):
    from categorizer import CategoryClassifier
    model = CategoryClassifier()
    metrics = model.train(texts, labels)
    model.save("category_model.pkl")

Inference (ingest.py):
    model = CategoryClassifier.load_or_none("category_model.pkl")
    if model:
        cat, conf = model.predict(headline + " " + summary)
"""
from __future__ import annotations

import logging
import os
import pickle
from typing import Optional

log = logging.getLogger("kv.categorizer")

# Confidence below this threshold → caller should try a better signal
# (AI p_cat or keyword fallback). Chosen empirically: LR usually
# returns 0.7-0.95 for clear cases and 0.2-0.4 for ambiguous ones.
CONFIDENCE_THRESHOLD = 0.45

CATEGORIES = [
    "politics", "farming", "weather", "jobs",
    "health", "village", "sports", "cinema", "schemes", "spiritual", "general",
]

MODEL_PATH = os.path.join(os.path.dirname(__file__), "category_model.pkl")


class CategoryClassifier:
    """Wraps a sklearn TF-IDF + LogisticRegression pipeline."""

    def __init__(self) -> None:
        self._pipeline = None
        self.trained = False
        self.training_stats: dict = {}

    # ── Training ──────────────────────────────────────────────────────────

    def train(self, texts: list[str], labels: list[str]) -> dict:
        """
        Fit on (texts, labels). Returns cross-val accuracy metrics.

        texts  — list of strings (headline + " " + summary)
        labels — list of category strings (must be in CATEGORIES)
        """
        from sklearn.pipeline import Pipeline
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        import numpy as np

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                # char_wb: character n-grams respecting word boundaries.
                # Handles Telugu morphology without a dedicated tokenizer.
                analyzer="char_wb",
                ngram_range=(3, 6),
                max_features=30000,
                sublinear_tf=True,     # log(1+tf) — reduces impact of freq words
                min_df=2,              # drop features appearing in only 1 doc
                strip_accents=None,    # keep all Unicode (Telugu script)
            )),
            ("clf", LogisticRegression(
                C=5.0,                  # regularisation: 5 = moderate, empirically good
                max_iter=1000,
                solver="lbfgs",
                class_weight="balanced",  # compensate for unequal category sizes
            )),
        ])

        # 5-fold stratified cross-validation before final fit
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(pipeline, texts, labels, cv=cv, scoring="accuracy")
        cv_acc = float(np.mean(scores))
        cv_std = float(np.std(scores))

        # Full fit on all data for production use
        pipeline.fit(texts, labels)
        self._pipeline = pipeline
        self.trained = True

        # Per-class accuracy via classification report
        from sklearn.model_selection import cross_val_predict
        from sklearn.metrics import classification_report
        y_pred = cross_val_predict(pipeline, texts, labels, cv=cv)
        report = classification_report(labels, y_pred, output_dict=True, zero_division=0)

        self.training_stats = {
            "cv_accuracy": round(cv_acc, 4),
            "cv_std": round(cv_std, 4),
            "n_train": len(texts),
            "per_class": {
                cat: {
                    "precision": round(report.get(cat, {}).get("precision", 0), 3),
                    "recall":    round(report.get(cat, {}).get("recall", 0), 3),
                    "f1":        round(report.get(cat, {}).get("f1-score", 0), 3),
                    "support":   int(report.get(cat, {}).get("support", 0)),
                }
                for cat in CATEGORIES if cat in report
            },
        }
        log.info(
            "%s trained: n=%d  cv_acc=%.1f%%±%.1f%%",
            type(self).__name__, len(texts), cv_acc * 100, cv_std * 100,
        )
        return self.training_stats

    # ── Inference ─────────────────────────────────────────────────────────

    def predict(self, text: str) -> tuple[str, float]:
        """
        Predict category for a text (headline + summary recommended).

        Returns (category, confidence) where confidence ∈ [0, 1].
        If confidence < CONFIDENCE_THRESHOLD the caller should fall back
        to AI classification or keyword detection.

        Returns ("general", 0.0) if the model is not trained.
        """
        if not self.trained or self._pipeline is None:
            return "general", 0.0
        try:
            probs = self._pipeline.predict_proba([text])[0]
            idx = int(probs.argmax())
            return str(self._pipeline.classes_[idx]), float(probs[idx])
        except Exception as e:
            log.warning("CategoryClassifier.predict failed: %s", e)
            return "general", 0.0

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str = MODEL_PATH) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=4)
        log.info("%s saved → %s", type(self).__name__, path)

    @classmethod
    def load(cls, path: str = MODEL_PATH) -> "CategoryClassifier":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise ValueError(f"Pickle at {path} is not a {cls.__name__}")
        log.info("%s loaded ← %s (cv_acc=%.1f%%)",
                 cls.__name__, path, obj.training_stats.get("cv_accuracy", 0) * 100)
        return obj

    @classmethod
    def load_or_none(cls, path: str = MODEL_PATH) -> Optional["CategoryClassifier"]:
        """Load the model if the file exists, else return None (graceful degradation)."""
        if not os.path.isfile(path):
            return None
        try:
            return cls.load(path)
        except Exception as e:
            log.warning("%s: could not load %s: %s", cls.__name__, path, e)
            return None


# ─── Level classifier ─────────────────────────────────────────────────────────

LEVELS = ["village", "mandal", "district", "state", "national"]
LEVEL_CONFIDENCE_THRESHOLD = 0.50   # 5-class problem; require higher confidence
LEVEL_MODEL_PATH = os.path.join(os.path.dirname(__file__), "level_model.pkl")


class LevelClassifier(CategoryClassifier):
    """
    Predicts geographic level (village/mandal/district/state/national).

    Inherits the same TF-IDF char n-gram + LogisticRegression pipeline.
    Training data quality is high: national/state articles come from reliably
    labelled source feeds; village/mandal from WhatsApp reporter profiles.

    Usage (identical to CategoryClassifier):
        model = LevelClassifier()
        model.train(texts, labels)   # labels ∈ LEVELS
        model.save("level_model.pkl")

        model = LevelClassifier.load_or_none("level_model.pkl")
        level, conf = model.predict(headline + " " + summary)
    """

    def predict(self, text: str) -> tuple[str, float]:
        """Returns (level, confidence). Falls back to ("district", 0.0) if untrained."""
        if not self.trained or self._pipeline is None:
            return "district", 0.0
        try:
            probs = self._pipeline.predict_proba([text])[0]
            idx = int(probs.argmax())
            return str(self._pipeline.classes_[idx]), float(probs[idx])
        except Exception as e:
            log.warning("LevelClassifier.predict failed: %s", e)
            return "district", 0.0

    def save(self, path: str = LEVEL_MODEL_PATH) -> None:
        super().save(path)

    @classmethod
    def load(cls, path: str = LEVEL_MODEL_PATH) -> "LevelClassifier":
        return super().load(path)   # type: ignore[return-value]

    @classmethod
    def load_or_none(cls, path: str = LEVEL_MODEL_PATH) -> Optional["LevelClassifier"]:
        return super().load_or_none(path)   # type: ignore[return-value]
