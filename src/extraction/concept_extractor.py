"""
Module d'extraction des concepts clés et résumé via Ollama (LLaMA3 - gratuit).
"""

import ollama
import json
import os
from pathlib import Path


PROMPT_EXTRACTION = """Tu es un assistant pédagogique expert. Analyse le texte suivant qui est la transcription d'un cours.

Retourne UNIQUEMENT un JSON valide avec cette structure exacte :
{{
  "resume": "Résumé du cours en 3-5 phrases",
  "concepts_cles": [
    {{
      "concept": "Nom du concept",
      "definition": "Définition courte",
      "importance": "haute|moyenne|faible"
    }}
  ],
  "themes_principaux": ["theme1", "theme2"],
  "niveau": "débutant|intermédiaire|avancé"
}}

Transcription du cours :
{transcription}

Réponds UNIQUEMENT avec le JSON, sans texte avant ou après."""


def extraire_concepts(chemin_transcription: str, dossier_sortie: str = "data/concepts") -> str:
    """
    Extrait les concepts clés d'une transcription via Ollama LLaMA3.

    Args:
        chemin_transcription: Chemin vers le fichier .txt de transcription
        dossier_sortie: Dossier où sauvegarder les concepts extraits

    Returns:
        Chemin vers le fichier JSON de concepts
    """
    print(f"[LLM] Lecture de la transcription : {chemin_transcription}")
    with open(chemin_transcription, "r", encoding="utf-8") as f:
        transcription = f.read()

    # Limiter à 4000 mots pour éviter de dépasser le contexte
    mots = transcription.split()
    if len(mots) > 4000:
        transcription = " ".join(mots[:4000])
        print(f"[LLM] Transcription tronquée à 4000 mots")

    print(f"[LLM] Extraction des concepts avec LLaMA3...")
    prompt = PROMPT_EXTRACTION.format(transcription=transcription)

    reponse = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )

    texte_reponse = reponse["message"]["content"].strip()

    # Nettoyer la réponse si elle contient des backticks
    if "```json" in texte_reponse:
        texte_reponse = texte_reponse.split("```json")[1].split("```")[0].strip()
    elif "```" in texte_reponse:
        texte_reponse = texte_reponse.split("```")[1].strip()

    try:
        concepts = json.loads(texte_reponse)
    except json.JSONDecodeError:
        print("[LLM] Erreur de parsing JSON, utilisation d'un résultat par défaut")
        concepts = {
            "resume": texte_reponse[:500],
            "concepts_cles": [],
            "themes_principaux": [],
            "niveau": "intermédiaire"
        }

    # Sauvegarder
    Path(dossier_sortie).mkdir(parents=True, exist_ok=True)
    nom = Path(chemin_transcription).stem
    chemin_sortie = os.path.join(dossier_sortie, f"{nom}.json")

    with open(chemin_sortie, "w", encoding="utf-8") as f:
        json.dump(concepts, f, ensure_ascii=False, indent=2)

    print(f"[LLM] {len(concepts.get('concepts_cles', []))} concepts extraits")
    print(f"[LLM] Résultat sauvegardé : {chemin_sortie}")

    return chemin_sortie


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extraction de concepts avec Ollama")
    parser.add_argument("--transcription", required=True, help="Chemin vers la transcription .txt")
    parser.add_argument("--sortie", default="data/concepts", help="Dossier de sortie")
    args = parser.parse_args()

    chemin = extraire_concepts(args.transcription, args.sortie)
    print(f"\nExtraction terminée : {chemin}")
