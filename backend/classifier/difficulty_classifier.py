"""
Classifieur de difficulté — Random Forest
Entraîné sur des QCM annotés (facile / moyen / difficile)
Features : longueur, complexité lexicale, lisibilité Flesch
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

try:
    import textstat
except ImportError:
    textstat = None

MODEL_PATH = "models/difficulty_classifier.pkl"
ENCODER_PATH = "models/label_encoder.pkl"
DATA_PATH = "data/annotated_qcm/sample_dataset.json"


def extract_features(question: str, correct_answer: str,
                     distractors: list) -> np.ndarray:
    """
    Extrait les features numériques d'une question QCM.

    Features :
    - Nombre de mots dans la question
    - Longueur moyenne des mots
    - Nombre de mots dans la bonne réponse
    - Score de lisibilité Flesch (si textstat disponible)
    - Longueur moyenne des distracteurs
    - Écart de longueur entre réponse et distracteurs
    """
    words_q = question.split()
    words_a = correct_answer.split()

    num_words_question = len(words_q)
    avg_word_len = np.mean([len(w) for w in words_q]) if words_q else 0
    num_words_answer = len(words_a)

    if textstat:
        flesch = textstat.flesch_reading_ease(question + " " + correct_answer)
    else:
        # Approximation simple si textstat indisponible
        total_chars = sum(len(w) for w in words_q)
        flesch = max(0, 100 - total_chars / max(len(words_q), 1) * 5)

    dist_lengths = [len(d.split()) for d in distractors] if distractors else [0]
    avg_distractor_len = np.mean(dist_lengths)
    len_diff = abs(num_words_answer - avg_distractor_len)

    return np.array([
        num_words_question,
        avg_word_len,
        num_words_answer,
        flesch,
        avg_distractor_len,
        len_diff
    ])


def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    """Charge le dataset annoté depuis le JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)


def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Construit la matrice de features depuis le DataFrame."""
    features = []
    for _, row in df.iterrows():
        distractors = [
            row.get("distractor1", ""),
            row.get("distractor2", ""),
            row.get("distractor3", "")
        ]
        feat = extract_features(
            row["question"],
            row["correct_answer"],
            distractors
        )
        features.append(feat)
    return np.array(features)


def train_classifier(data_path: str = DATA_PATH) -> dict:
    """
    Entraîne le classifieur Random Forest.

    Returns:
        dict avec accuracy et rapport de classification
    """
    print("[RF] Chargement des données...")
    df = load_dataset(data_path)

    print(f"[RF] {len(df)} exemples chargés. Distribution des classes :")
    print(df["difficulty"].value_counts().to_string())

    X = build_feature_matrix(df)
    y = df["difficulty"].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    print("[RF] Entraînement du Random Forest...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        class_weight="balanced"
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Cross-validation
    cv_scores = cross_val_score(clf, X, y_enc, cv=3)

    report = classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        output_dict=True
    )

    # Sauvegarde du modèle
    Path("models").mkdir(exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)
    print(f"[RF] Modèle sauvegardé : {MODEL_PATH}")

    result = {
        "accuracy": round(acc, 3),
        "cv_mean": round(cv_scores.mean(), 3),
        "cv_std": round(cv_scores.std(), 3),
        "report": report,
        "classes": list(le.classes_)
    }

    print(f"[RF] Accuracy : {acc:.1%} | CV: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")
    return result


def predict_difficulty(question: str, correct_answer: str,
                       distractors: list) -> dict:
    """
    Prédit la difficulté d'une question QCM.

    Returns:
        dict avec 'label' (facile/moyen/difficile) et 'probabilities'
    """
    if not os.path.exists(MODEL_PATH):
        print("[RF] Modèle introuvable, entraînement automatique...")
        train_classifier()

    clf = joblib.load(MODEL_PATH)
    le = joblib.load(ENCODER_PATH)

    features = extract_features(question, correct_answer, distractors).reshape(1, -1)
    pred_enc = clf.predict(features)[0]
    proba = clf.predict_proba(features)[0]

    label = le.inverse_transform([pred_enc])[0]
    proba_dict = {
        cls: round(float(p), 3)
        for cls, p in zip(le.classes_, proba)
    }

    return {
        "label": label,
        "probabilities": proba_dict,
        "confidence": round(float(proba.max()), 3)
    }


def predict_batch(questions: list) -> list:
    """
    Prédit la difficulté pour une liste de questions QCM.

    Args:
        questions : liste de dicts avec 'question', 'correct_answer', 'options'

    Returns:
        liste de questions enrichies avec 'predicted_difficulty'
    """
    enriched = []
    for q in questions:
        distractors = [
            v for k, v in q.get("options", {}).items()
            if k != q.get("correct_answer")
        ]
        pred = predict_difficulty(
            q.get("question", ""),
            q.get("options", {}).get(q.get("correct_answer", "A"), ""),
            distractors
        )
        q["predicted_difficulty"] = pred["label"]
        q["difficulty_confidence"] = pred["confidence"]
        q["difficulty_probabilities"] = pred["probabilities"]
        enriched.append(q)
    return enriched


if __name__ == "__main__":
    print("=== ENTRAÎNEMENT DU CLASSIFIEUR ===")
    result = train_classifier()
    print(f"\nAccuracy : {result['accuracy']}")
    print(f"CV Score : {result['cv_mean']} ± {result['cv_std']}")

    print("\n=== TEST DE PRÉDICTION ===")
    pred = predict_difficulty(
        question="Quelle est la formule chimique du glucose ?",
        correct_answer="C6H12O6",
        distractors=["C12H22O11", "H2O", "CO2"]
    )
    print(f"Difficulté prédite : {pred['label']} (confiance : {pred['confidence']})")
