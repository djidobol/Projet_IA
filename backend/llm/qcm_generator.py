"""
Module LLM — Extraction de concepts clés + Génération de QCM
Utilise l'API Groq (gratuite) avec le modèle llama3-8b-8192
"""

import os
import json
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL = "llama-3.1-8b-instant"


def extract_concepts(transcript_text: str) -> dict:
    """
    Extrait les concepts clés d'une transcription.

    Returns:
        dict avec 'summary' et 'concepts' (liste de concepts importants)
    """
    prompt = f"""Tu es un assistant pédagogique expert. Voici la transcription d'un cours :

---
{transcript_text[:6000]}
---

Ta tâche :
1. Rédige un résumé du cours en 5 phrases maximum.
2. Liste les 8 concepts clés les plus importants abordés dans ce cours.

Réponds UNIQUEMENT en JSON valide, sans aucun texte avant ou après, au format suivant :
{{
  "summary": "résumé en 5 phrases",
  "concepts": [
    {{"concept": "nom du concept", "definition": "explication courte en 1 phrase"}},
    ...
  ]
}}"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1500
    )

    raw = response.choices[0].message.content.strip()

    # Nettoyer le JSON si le modèle ajoute des backticks
    raw = re.sub(r"```json|```", "", raw).strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {"summary": raw, "concepts": []}

    print(f"[LLM] {len(result.get('concepts', []))} concepts extraits.")
    return result


def generate_qcm(concepts: list, transcript_text: str, num_questions: int = 10) -> list:
    """
    Génère des questions QCM avec distracteurs plausibles.

    Args:
        concepts       : liste de concepts extraits
        transcript_text: texte source pour le contexte
        num_questions  : nombre de questions à générer

    Returns:
        liste de questions QCM au format JSON
    """
    concepts_str = "\n".join(
        f"- {c['concept']}: {c['definition']}" for c in concepts
    )

    prompt = f"""Tu es un expert en création de QCM pédagogiques. 
Voici les concepts clés d'un cours :

{concepts_str}

Contexte du cours (extrait) :
{transcript_text[:3000]}

Génère exactement {num_questions} questions QCM à choix unique.
Chaque question doit :
- Tester la compréhension (pas juste la mémorisation)
- Avoir 1 bonne réponse et 3 distracteurs PLAUSIBLES (pas absurdes)
- Inclure un niveau de difficulté estimé : "facile", "moyen" ou "difficile"

Réponds UNIQUEMENT en JSON valide, sans texte avant ou après :
[
  {{
    "id": 1,
    "question": "texte de la question",
    "options": {{
      "A": "option A",
      "B": "option B",
      "C": "option C",
      "D": "option D"
    }},
    "correct_answer": "A",
    "explanation": "explication courte de la bonne réponse",
    "concept": "nom du concept testé",
    "difficulty_label": "facile"
  }},
  ...
]"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=4000
    )

    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"```json|```", "", raw).strip()

    try:
        questions = json.loads(raw)
        if not isinstance(questions, list):
            questions = []
    except json.JSONDecodeError:
        questions = []

    print(f"[LLM] {len(questions)} questions QCM générées.")
    return questions


def run_pipeline(transcript_text: str, num_questions: int = 10) -> dict:
    """
    Pipeline complet : transcript → concepts → QCM

    Returns:
        dict avec summary, concepts et questions
    """
    print("[LLM] Extraction des concepts...")
    concepts_data = extract_concepts(transcript_text)

    print("[LLM] Génération des QCM...")
    questions = generate_qcm(
        concepts_data.get("concepts", []),
        transcript_text,
        num_questions
    )

    return {
        "summary":   concepts_data.get("summary", ""),
        "concepts":  concepts_data.get("concepts", []),
        "questions": questions
    }


if __name__ == "__main__":
    # Test avec un texte fictif
    test_text = """
    La photosynthèse est le processus par lequel les plantes convertissent la lumière solaire
    en énergie chimique. Ce processus se déroule dans les chloroplastes, organites contenant
    la chlorophylle. La réaction globale est : 6CO2 + 6H2O + lumière → C6H12O6 + 6O2.
    La phase claire nécessite de la lumière et produit de l'ATP et du NADPH.
    La phase sombre (cycle de Calvin) fixe le CO2 pour produire du glucose.
    """
    result = run_pipeline(test_text, num_questions=3)
    print(json.dumps(result, ensure_ascii=False, indent=2))
