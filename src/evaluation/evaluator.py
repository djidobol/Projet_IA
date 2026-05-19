"""
Module d'évaluation : taux de rejet des distracteurs et corrélation difficulté/réussite.
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path


def calculer_taux_rejet(chemin_qcm_annote: str) -> dict:
    """
    Calcule le taux de rejet des distracteurs par un expert.
    
    Le fichier doit contenir un champ 'rejete_par_expert' (bool) pour chaque distracteur.

    Args:
        chemin_qcm_annote: Chemin vers le QCM avec annotations d'expert

    Returns:
        Dictionnaire avec les métriques de rejet
    """
    with open(chemin_qcm_annote, "r", encoding="utf-8") as f:
        qcm = json.load(f)

    questions = qcm.get("questions", [])
    total_distracteurs = 0
    distracteurs_rejetes = 0

    for q in questions:
        for d in q.get("distracteurs_annotes", []):
            total_distracteurs += 1
            if d.get("rejete_par_expert", False):
                distracteurs_rejetes += 1

    taux_rejet = distracteurs_rejetes / max(total_distracteurs, 1)

    return {
        "total_distracteurs": total_distracteurs,
        "distracteurs_rejetes": distracteurs_rejetes,
        "taux_rejet": round(taux_rejet, 4),
        "taux_rejet_pct": f"{taux_rejet:.1%}"
    }


def calculer_correlation_difficulte(chemin_qcm_avec_resultats: str) -> dict:
    """
    Calcule la corrélation de Pearson entre la difficulté prédite par le RF
    et le taux de réussite observé chez les étudiants.

    Le fichier doit contenir pour chaque question :
    - 'difficulte_predite_rf' : "facile", "moyen", "difficile"
    - 'taux_reussite_observe' : float entre 0 et 1

    Args:
        chemin_qcm_avec_resultats: Chemin vers le QCM avec résultats étudiants

    Returns:
        Dictionnaire avec corrélation et p-value
    """
    with open(chemin_qcm_avec_resultats, "r", encoding="utf-8") as f:
        qcm = json.load(f)

    questions = qcm.get("questions", [])

    # Convertir difficulté en valeur numérique
    mapping = {"facile": 1, "moyen": 2, "difficile": 3}

    difficultes = []
    taux_reussites = []

    for q in questions:
        diff = q.get("difficulte_predite_rf", "")
        taux = q.get("taux_reussite_observe")

        if diff in mapping and taux is not None:
            difficultes.append(mapping[diff])
            taux_reussites.append(float(taux))

    if len(difficultes) < 3:
        return {
            "erreur": "Pas assez de données pour calculer la corrélation (minimum 3 questions)",
            "nb_questions": len(difficultes)
        }

    # Pearson (difficulté haute → taux réussite bas → corrélation négative attendue)
    correlation, p_value = stats.pearsonr(difficultes, taux_reussites)

    interpretation = ""
    if abs(correlation) >= 0.7:
        interpretation = "Forte corrélation"
    elif abs(correlation) >= 0.4:
        interpretation = "Corrélation modérée"
    else:
        interpretation = "Faible corrélation"

    return {
        "nb_questions_analysees": len(difficultes),
        "correlation_pearson": round(correlation, 4),
        "p_value": round(p_value, 4),
        "significatif": p_value < 0.05,
        "interpretation": interpretation,
        "note": "Une corrélation négative est attendue (plus difficile = moins de réussite)"
    }


def rapport_complet(chemin_qcm_filtre: str) -> dict:
    """
    Génère un rapport complet d'évaluation du QCM généré.

    Args:
        chemin_qcm_filtre: Chemin vers le QCM filtré par le RF

    Returns:
        Rapport complet en dictionnaire
    """
    with open(chemin_qcm_filtre, "r", encoding="utf-8") as f:
        qcm = json.load(f)

    questions = qcm.get("questions", [])
    stats_difficulte = qcm.get("statistiques", {}).get("repartition_difficulte", {})

    rapport = {
        "resume_qcm": {
            "total_questions": len(questions),
            "repartition_difficulte": stats_difficulte,
            "taux_conservation_filtrage": qcm.get("statistiques", {}).get("taux_conservation", 1.0)
        },
        "qualite_distracteurs": {
            "info": "Annoter les distracteurs avec 'rejete_par_expert' pour calculer le taux de rejet"
        },
        "correlation_difficulte": {
            "info": "Ajouter 'taux_reussite_observe' après passage en classe pour calculer la corrélation"
        },
        "confiance_moyenne_rf": round(
            np.mean([q.get("confiance_rf", 0) for q in questions]), 3
        ) if questions else 0
    }

    return rapport


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Évaluation du QCM généré")
    parser.add_argument("--qcm", required=True, help="Chemin vers le QCM filtré")
    args = parser.parse_args()

    rapport = rapport_complet(args.qcm)
    print(json.dumps(rapport, ensure_ascii=False, indent=2))
