"""
Module de classification de la difficulté des questions QCM.
Utilise un Random Forest entraîné sur des QCM historiques annotés.
"""

import json
import os
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder


# Chemin du modèle sauvegardé
CHEMIN_MODELE = "models/random_forest_difficulte.pkl"
CHEMIN_ENCODER = "models/label_encoder.pkl"


def extraire_features(question: dict) -> list:
    """
    Extrait les features numériques d'une question QCM pour le Random Forest.

    Features extraites :
    - Longueur de la question (nb mots)
    - Longueur moyenne des distracteurs
    - Longueur de la bonne réponse
    - Présence de négation dans la question
    - Nombre de distracteurs
    - Longueur de l'explication
    """
    texte_question = question.get("question", "")
    bonne_reponse = question.get("bonne_reponse", "")
    distracteurs = question.get("distracteurs", [])
    explication = question.get("explication", "")

    longueur_question = len(texte_question.split())
    longueur_reponse = len(bonne_reponse.split())
    nb_distracteurs = len(distracteurs)
    longueur_moy_distracteurs = (
        np.mean([len(d.split()) for d in distracteurs]) if distracteurs else 0
    )
    a_negation = int(any(mot in texte_question.lower() for mot in ["ne", "pas", "jamais", "non", "aucun", "sans"]))
    longueur_explication = len(explication.split())

    return [
        longueur_question,
        longueur_reponse,
        nb_distracteurs,
        longueur_moy_distracteurs,
        a_negation,
        longueur_explication
    ]


def generer_donnees_historiques_exemple():
    """
    Génère des données d'entraînement d'exemple si aucune donnée historique n'existe.
    En production, ces données viennent de vrais QCM annotés par des experts.
    """
    import random
    random.seed(42)
    np.random.seed(42)

    questions_exemple = []
    for i in range(200):
        difficulte = random.choice(["facile", "moyen", "difficile"])

        if difficulte == "facile":
            q = {
                "question": " ".join(["mot"] * random.randint(5, 10)),
                "bonne_reponse": " ".join(["rep"] * random.randint(2, 4)),
                "distracteurs": [" ".join(["dis"] * random.randint(2, 3)) for _ in range(3)],
                "explication": " ".join(["exp"] * random.randint(5, 10)),
                "difficulte_annotee": "facile"
            }
        elif difficulte == "moyen":
            q = {
                "question": " ".join(["mot"] * random.randint(10, 18)),
                "bonne_reponse": " ".join(["rep"] * random.randint(4, 8)),
                "distracteurs": [" ".join(["dis"] * random.randint(4, 6)) for _ in range(3)],
                "explication": " ".join(["exp"] * random.randint(15, 25)),
                "difficulte_annotee": "moyen"
            }
        else:
            q = {
                "question": " ".join(["mot"] * random.randint(18, 30)),
                "bonne_reponse": " ".join(["rep"] * random.randint(8, 15)),
                "distracteurs": [" ".join(["dis"] * random.randint(7, 12)) for _ in range(3)],
                "explication": " ".join(["exp"] * random.randint(25, 50)),
                "difficulte_annotee": "difficile"
            }
        questions_exemple.append(q)

    return questions_exemple


def entrainer_modele(chemin_historique: str = "data/historical", forcer: bool = False) -> RandomForestClassifier:
    """
    Entraîne le Random Forest sur les QCM historiques annotés.

    Args:
        chemin_historique: Dossier contenant les QCM annotés (JSON)
        forcer: Forcer le ré-entraînement même si le modèle existe déjà

    Returns:
        Le modèle entraîné
    """
    # Si modèle existe déjà et pas de forçage
    if os.path.exists(CHEMIN_MODELE) and not forcer:
        print("[RF] Modèle existant chargé.")
        return joblib.load(CHEMIN_MODELE)

    print("[RF] Chargement des données historiques...")
    questions = []

    # Charger les fichiers JSON du dossier historique
    for fichier in Path(chemin_historique).glob("*.json"):
        with open(fichier, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                questions.extend(data)
            elif isinstance(data, dict) and "questions" in data:
                questions.extend(data["questions"])

    # Si pas de données historiques → utiliser les données d'exemple
    if len(questions) < 10:
        print("[RF] Pas assez de données historiques, utilisation de données d'exemple...")
        questions = generer_donnees_historiques_exemple()

    print(f"[RF] {len(questions)} questions d'entraînement")

    # Extraire features et labels
    X = []
    y = []
    for q in questions:
        features = extraire_features(q)
        label = q.get("difficulte_annotee") or q.get("difficulte_estimee", "moyen")
        X.append(features)
        y.append(label)

    X = np.array(X)

    # Encoder les labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Entraîner le Random Forest
    print("[RF] Entraînement du Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight="balanced"
    )
    rf.fit(X_train, y_train)

    # Évaluation
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[RF] Accuracy sur test : {acc:.2%}")
    print(f"[RF] Rapport :\n{classification_report(y_test, y_pred, target_names=encoder.classes_)}")

    # Sauvegarder
    Path("models").mkdir(exist_ok=True)
    joblib.dump(rf, CHEMIN_MODELE)
    joblib.dump(encoder, CHEMIN_ENCODER)
    print(f"[RF] Modèle sauvegardé : {CHEMIN_MODELE}")

    return rf


def classifier_difficulte(chemin_qcm: str, dossier_sortie: str = "data/qcm_filtered") -> str:
    """
    Classifie la difficulté de chaque question d'un QCM et filtre les incohérences.

    Args:
        chemin_qcm: Chemin vers le fichier QCM brut (JSON)
        dossier_sortie: Dossier de sortie

    Returns:
        Chemin vers le QCM filtré et enrichi
    """
    # Charger le modèle
    if not os.path.exists(CHEMIN_MODELE):
        print("[RF] Modèle non trouvé, entraînement en cours...")
        entrainer_modele()

    rf = joblib.load(CHEMIN_MODELE)
    encoder = joblib.load(CHEMIN_ENCODER)

    # Charger le QCM
    print(f"[RF] Classification de : {chemin_qcm}")
    with open(chemin_qcm, "r", encoding="utf-8") as f:
        qcm = json.load(f)

    questions = qcm.get("questions", [])
    questions_classifiees = []

    for q in questions:
        features = np.array([extraire_features(q)])
        idx_pred = rf.predict(features)[0]
        probas = rf.predict_proba(features)[0]

        difficulte_predite = encoder.inverse_transform([idx_pred])[0]
        confiance = float(max(probas))

        q_enrichie = q.copy()
        q_enrichie["difficulte_predite_rf"] = difficulte_predite
        q_enrichie["confiance_rf"] = round(confiance, 3)
        q_enrichie["distribution_difficulte"] = {
            classe: round(float(prob), 3)
            for classe, prob in zip(encoder.classes_, probas)
        }

        # Filtrage : garder uniquement les questions avec confiance > 0.5
        if confiance >= 0.5:
            questions_classifiees.append(q_enrichie)
        else:
            print(f"[RF] Question rejetée (confiance trop faible : {confiance:.2%}) : {q.get('question', '')[:50]}...")

    qcm_filtre = {
        "metadata": qcm.get("metadata", {}),
        "statistiques": {
            "total_initial": len(questions),
            "total_apres_filtrage": len(questions_classifiees),
            "taux_conservation": round(len(questions_classifiees) / max(len(questions), 1), 2),
            "repartition_difficulte": {
                niveau: sum(1 for q in questions_classifiees if q.get("difficulte_predite_rf") == niveau)
                for niveau in ["facile", "moyen", "difficile"]
            }
        },
        "questions": questions_classifiees
    }

    # Sauvegarder
    Path(dossier_sortie).mkdir(parents=True, exist_ok=True)
    nom = Path(chemin_qcm).stem
    chemin_sortie = os.path.join(dossier_sortie, f"{nom}_filtre.json")

    with open(chemin_sortie, "w", encoding="utf-8") as f:
        json.dump(qcm_filtre, f, ensure_ascii=False, indent=2)

    print(f"[RF] {len(questions_classifiees)}/{len(questions)} questions conservées")
    print(f"[RF] Résultat sauvegardé : {chemin_sortie}")

    return chemin_sortie


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Classification de difficulté Random Forest")
    parser.add_argument("--qcm", help="Chemin vers le QCM brut JSON")
    parser.add_argument("--entrainer", action="store_true", help="Entraîner le modèle")
    parser.add_argument("--forcer", action="store_true", help="Forcer le ré-entraînement")
    args = parser.parse_args()

    if args.entrainer or args.forcer:
        entrainer_modele(forcer=args.forcer)

    if args.qcm:
        chemin = classifier_difficulte(args.qcm)
        print(f"\nClassification terminée : {chemin}")
