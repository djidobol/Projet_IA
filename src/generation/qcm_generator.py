"""
Module de génération de QCM avec distracteurs plausibles via Ollama (LLaMA3).
"""

import ollama
import json
import os
from pathlib import Path


PROMPT_QCM = """Tu es un expert en création de QCM pédagogiques. 
À partir des concepts suivants extraits d'un cours, génère des questions QCM de qualité.

Concepts du cours :
{concepts}

Résumé du cours :
{resume}

Génère exactement {nb_questions} questions QCM. 
Retourne UNIQUEMENT un JSON valide avec cette structure :
{{
  "questions": [
    {{
      "id": 1,
      "question": "Texte de la question ?",
      "bonne_reponse": "La réponse correcte",
      "distracteurs": [
        "Fausse réponse plausible 1",
        "Fausse réponse plausible 2", 
        "Fausse réponse plausible 3"
      ],
      "concept_cible": "Le concept testé",
      "difficulte_estimee": "facile|moyen|difficile",
      "explication": "Pourquoi cette réponse est correcte"
    }}
  ]
}}

Règles importantes :
- Les distracteurs doivent être plausibles mais clairement faux
- Varie les niveaux de difficulté
- Les questions doivent tester la compréhension, pas la mémorisation
- Réponds UNIQUEMENT avec le JSON"""


def generer_qcm(chemin_concepts: str, nb_questions: int = 10, dossier_sortie: str = "data/qcm_raw") -> str:
    """
    Génère un QCM complet avec distracteurs à partir des concepts extraits.

    Args:
        chemin_concepts: Chemin vers le fichier JSON de concepts
        nb_questions: Nombre de questions à générer
        dossier_sortie: Dossier de sortie

    Returns:
        Chemin vers le fichier QCM généré
    """
    print(f"[QCM] Lecture des concepts : {chemin_concepts}")
    with open(chemin_concepts, "r", encoding="utf-8") as f:
        concepts = json.load(f)

    # Préparer le texte des concepts
    concepts_texte = "\n".join([
        f"- {c['concept']} : {c['definition']} (importance: {c['importance']})"
        for c in concepts.get("concepts_cles", [])
    ])

    if not concepts_texte:
        concepts_texte = "Concepts non disponibles, basez-vous sur le résumé."

    resume = concepts.get("resume", "")

    print(f"[QCM] Génération de {nb_questions} questions avec LLaMA3...")
    prompt = PROMPT_QCM.format(
        concepts=concepts_texte,
        resume=resume,
        nb_questions=nb_questions
    )

    reponse = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )

    texte_reponse = reponse["message"]["content"].strip()

    # Nettoyer
    if "```json" in texte_reponse:
        texte_reponse = texte_reponse.split("```json")[1].split("```")[0].strip()
    elif "```" in texte_reponse:
        texte_reponse = texte_reponse.split("```")[1].strip()

    try:
        qcm = json.loads(texte_reponse)
    except json.JSONDecodeError:
        print("[QCM] Erreur JSON, création d'un QCM vide")
        qcm = {"questions": []}

    # Ajouter métadonnées
    qcm["metadata"] = {
        "source_concepts": chemin_concepts,
        "nb_questions": len(qcm.get("questions", [])),
        "niveau_cours": concepts.get("niveau", "intermédiaire"),
        "themes": concepts.get("themes_principaux", [])
    }

    # Sauvegarder
    Path(dossier_sortie).mkdir(parents=True, exist_ok=True)
    nom = Path(chemin_concepts).stem
    chemin_sortie = os.path.join(dossier_sortie, f"{nom}_qcm.json")

    with open(chemin_sortie, "w", encoding="utf-8") as f:
        json.dump(qcm, f, ensure_ascii=False, indent=2)

    print(f"[QCM] {len(qcm.get('questions', []))} questions générées")
    print(f"[QCM] Résultat sauvegardé : {chemin_sortie}")

    return chemin_sortie


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Génération de QCM")
    parser.add_argument("--concepts", required=True, help="Chemin vers le fichier JSON de concepts")
    parser.add_argument("--nb", type=int, default=10, help="Nombre de questions")
    parser.add_argument("--sortie", default="data/qcm_raw", help="Dossier de sortie")
    args = parser.parse_args()

    chemin = generer_qcm(args.concepts, args.nb, args.sortie)
    print(f"\nGénération terminée : {chemin}")
