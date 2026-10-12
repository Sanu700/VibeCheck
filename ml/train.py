"""
Model Training and Evaluation Pipeline for ToxicBuddy 2.0
Trains lightweight multi-label toxicity and tone classification models,
evaluates baseline alternatives, and exports model artifacts + performance metrics.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
)

from ml.dataset_generator import build_dataset

TARGET_LABELS = ["toxic", "insult", "harassment", "threat", "obscene", "identity_attack"]

from sklearn.pipeline import FeatureUnion
from ml.text_features import CUSTOM_STOP_WORDS, enrich_text_features

def train_and_evaluate():
    dataset_path = "data/toxic_dataset_multilabel.csv"
    # Always regenerate dataset during retraining to incorporate phrase updates
    print(f"Generating dataset at {dataset_path}...")
    df = build_dataset(dataset_path)

    X = df["text"].apply(lambda t: enrich_text_features(str(t)))
    Y = df[TARGET_LABELS]
    y_tone = df["tone"]

    # Split dataset
    X_train_raw, X_test_raw, Y_train, Y_test, y_tone_train, y_tone_test = train_test_split(
        X, Y, y_tone, test_size=0.2, random_state=42, stratify=df["toxic"]
    )

    # Word + Character N-Gram Feature Union
    word_vec = TfidfVectorizer(
        ngram_range=(1, 3),
        stop_words=CUSTOM_STOP_WORDS,
        max_features=12000,
        sublinear_tf=True
    )
    char_vec = TfidfVectorizer(
        ngram_range=(3, 5),
        analyzer="char_wb",
        max_features=15000,
        sublinear_tf=True
    )
    vectorizer = FeatureUnion([
        ("word", word_vec),
        ("char", char_vec)
    ])

    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)

    print("=== 1. Evaluating Multi-Label Model Alternatives ===")

    models_to_evaluate = {
        "LogisticRegression": MultiOutputClassifier(LogisticRegression(C=1.5, max_iter=500, random_state=42)),
        "MultinomialNB": MultiOutputClassifier(MultinomialNB(alpha=0.5)),
        "CalibratedLinearSVC": MultiOutputClassifier(CalibratedClassifierCV(LinearSVC(C=1.0, random_state=42), cv=3)),
        "RandomForest": MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    }

    comparison_results = {}

    for model_name, model_inst in models_to_evaluate.items():
        print(f"\nTraining & evaluating model: {model_name}...")
        model_inst.fit(X_train, Y_train)

        # Predict
        Y_pred = model_inst.predict(X_test)

        # Probabilities if available
        Y_proba = np.zeros(Y_test.shape)
        try:
            proba_list = model_inst.predict_proba(X_test)
            for i, p in enumerate(proba_list):
                est = model_inst.estimators_[i]
                classes = list(est.classes_)
                if 1 in classes:
                    Y_proba[:, i] = p[:, classes.index(1)]
                else:
                    Y_proba[:, i] = 0.0
        except Exception as e:
            Y_proba = Y_pred.astype(float)

        label_metrics = {}
        for idx, label in enumerate(TARGET_LABELS):
            y_true_col = Y_test[label]
            y_pred_col = Y_pred[:, idx]
            y_prob_col = Y_proba[:, idx]

            p = precision_score(y_true_col, y_pred_col, zero_division=0)
            r = recall_score(y_true_col, y_pred_col, zero_division=0)
            f1 = f1_score(y_true_col, y_pred_col, zero_division=0)
            try:
                auc = roc_auc_score(y_true_col, y_prob_col) if len(np.unique(y_true_col)) > 1 else 1.0
            except Exception:
                auc = 0.5

            label_metrics[label] = {
                "precision": round(float(p), 4),
                "recall": round(float(r), 4),
                "f1": round(float(f1), 4),
                "roc_auc": round(float(auc), 4)
            }

        macro_p = float(np.mean([m["precision"] for m in label_metrics.values()]))
        macro_r = float(np.mean([m["recall"] for m in label_metrics.values()]))
        macro_f1 = float(np.mean([m["f1"] for m in label_metrics.values()]))
        macro_auc = float(np.mean([m["roc_auc"] for m in label_metrics.values()]))

        comparison_results[model_name] = {
            "macro_precision": round(macro_p, 4),
            "macro_recall": round(macro_r, 4),
            "macro_f1": round(macro_f1, 4),
            "macro_roc_auc": round(macro_auc, 4),
            "per_label": label_metrics
        }
        print(f"  [{model_name}] Macro F1: {macro_f1:.4f} | Macro ROC-AUC: {macro_auc:.4f}")

    # Select best model (LogisticRegression / CalibratedLinearSVC)
    best_model_name = "LogisticRegression"
    print(f"\nSelecting primary model: {best_model_name}")

    best_toxic_model = models_to_evaluate[best_model_name]

    # === 2. Train Tone Classification Model ===
    print("\n=== 2. Training Tone Classification Model ===")
    tone_model = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    tone_model.fit(X_train, y_tone_train)
    y_tone_pred = tone_model.predict(X_test)

    tone_acc = float(accuracy_score(y_tone_test, y_tone_pred))
    tone_f1 = float(f1_score(y_tone_test, y_tone_pred, average="weighted"))

    print(f"Tone Classifier Accuracy: {tone_acc:.4f} | Weighted F1: {tone_f1:.4f}")

    # Save models
    os.makedirs("ml/models", exist_ok=True)
    
    pipeline_artifact = {
        "vectorizer": vectorizer,
        "toxic_classifier": best_toxic_model,
        "target_labels": TARGET_LABELS
    }
    joblib.dump(pipeline_artifact, "ml/models/toxic_classifier.pkl")

    tone_artifact = {
        "vectorizer": vectorizer,
        "tone_classifier": tone_model,
        "classes": list(tone_model.classes_)
    }
    joblib.dump(tone_artifact, "ml/models/tone_classifier.pkl")

    metrics_export = {
        "selected_model": best_model_name,
        "comparison": comparison_results,
        "tone_classifier": {
            "accuracy": round(tone_acc, 4),
            "weighted_f1": round(tone_f1, 4)
        },
        "target_labels": TARGET_LABELS
    }

    with open("ml/models/metrics.json", "w") as f:
        json.dump(metrics_export, f, indent=2)

    print("\nModel training & evaluation successfully completed!")
    print("Artifacts saved in ml/models/: toxic_classifier.pkl, tone_classifier.pkl, metrics.json")

    return metrics_export

if __name__ == "__main__":
    train_and_evaluate()
