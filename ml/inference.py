"""
Inference Module for ToxicBuddy 2.0
Provides fast, lightweight prediction for toxicity categories, confidence scores,
overall toxicity index, and conversation tone.
"""

import os
from typing import List, Optional
import joblib
import numpy as np

from ml.text_features import enrich_text_features

class ToxicityAnalyzer:
    """
    Lightweight Toxicity & Tone Classifier Service.
    Loads saved scikit-learn models for fast inference.
    """
    _instance = None

    def __init__(self, models_dir: str = "ml/models"):
        self.models_dir = models_dir
        self.toxic_artifact_path = os.path.join(models_dir, "toxic_classifier.pkl")
        self.tone_artifact_path = os.path.join(models_dir, "tone_classifier.pkl")
        
        self.vectorizer = None
        self.toxic_classifier = None
        self.tone_classifier = None
        self.target_labels = ["toxic", "insult", "harassment", "threat", "obscene", "identity_attack"]
        
        self.load_models()

    def load_models(self):
        """Loads model artifacts from disk, training them if not present."""
        if not os.path.exists(self.toxic_artifact_path) or not os.path.exists(self.tone_artifact_path):
            print("Model artifacts missing. Training model pipeline...")
            from ml.train import train_and_evaluate
            train_and_evaluate()

        toxic_artifact = joblib.load(self.toxic_artifact_path)
        tone_artifact = joblib.load(self.tone_artifact_path)

        self.vectorizer = toxic_artifact["vectorizer"]
        self.toxic_classifier = toxic_artifact["toxic_classifier"]
        self.target_labels = toxic_artifact.get("target_labels", self.target_labels)
        
        self.tone_classifier = tone_artifact["tone_classifier"]

    def analyze_message(self, text: str, context: Optional[List[str]] = None) -> dict:
        """
        Analyzes a single text string and returns detailed toxicity categories,
        confidence scores, overall toxicity score, and detected tone.
        Supports optional conversational context to disambiguate multi-turn sarcasm.
        """
        if not text or not text.strip():
            return {
                "is_toxic": False,
                "overall_score": 0.0,
                "severity_level": "Clean",
                "detected_tone": "Neutral",
                "categories": {label: 0.0 for label in self.target_labels},
                "flagged_categories": []
            }

        text_clean = enrich_text_features(text.strip(), context=context)
        vec = self.vectorizer.transform([text_clean])

        # Multi-label probabilities
        proba_list = self.toxic_classifier.predict_proba(vec)
        categories = {}
        for idx, label in enumerate(self.target_labels):
            p = proba_list[idx]
            est = self.toxic_classifier.estimators_[idx]
            classes = list(est.classes_)
            if 1 in classes:
                prob = float(p[0, classes.index(1)])
            else:
                prob = 0.0
            categories[label] = round(prob, 4)

        # Tone prediction
        tone_pred = self.tone_classifier.predict(vec)[0]

        # Calculate overall toxicity score
        # Weighted combination of toxicity labels
        weights = {
            "toxic": 0.35,
            "insult": 0.20,
            "harassment": 0.20,
            "threat": 0.30,
            "obscene": 0.15,
            "identity_attack": 0.30
        }

        # Check if explicitly toxic by max score or weighted average
        weighted_score = sum(categories[k] * weights.get(k, 0.2) for k in self.target_labels)
        max_score = max(categories.values())
        overall_score = round(min(1.0, max(weighted_score, max_score)), 4)

        # Flagged categories (> 0.40 confidence threshold)
        flagged_categories = [cat for cat, score in categories.items() if score >= 0.40]
        
        is_toxic = overall_score >= 0.45 or len(flagged_categories) > 0 or categories.get("toxic", 0) >= 0.45

        # Determine severity level
        if overall_score < 0.25:
            severity = "Clean"
        elif overall_score < 0.50:
            severity = "Low"
        elif overall_score < 0.75:
            severity = "Medium"
        elif overall_score < 0.90:
            severity = "High"
        else:
            severity = "Critical"

        return {
            "is_toxic": is_toxic,
            "overall_score": overall_score,
            "severity_level": severity,
            "detected_tone": str(tone_pred),
            "categories": categories,
            "flagged_categories": flagged_categories
        }

# Global singleton accessor
_analyzer_instance = None

def get_analyzer() -> ToxicityAnalyzer:
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = ToxicityAnalyzer()
    return _analyzer_instance
