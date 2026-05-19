"""
Module de transcription automatique avec OpenAI Whisper.
Utilise le modèle 'base' adapté à une machine avec 8Go RAM.
"""

import whisper
import os
import json
from pathlib import Path


def transcrire_video(chemin_video: str, dossier_sortie: str = "data/transcripts") -> str:
    """
    Transcrit une vidéo en texte avec Whisper.

    Args:
        chemin_video: Chemin vers le fichier vidéo (mp4, mkv, avi...)
        dossier_sortie: Dossier où sauvegarder la transcription

    Returns:
        Chemin vers le fichier de transcription généré
    """
    print(f"[Whisper] Chargement du modèle 'base'...")
    modele = whisper.load_model("base")

    print(f"[Whisper] Transcription de : {chemin_video}")
    resultat = modele.transcribe(chemin_video, language="fr", verbose=False)

    texte = resultat["text"].strip()

    # Créer le dossier de sortie si nécessaire
    Path(dossier_sortie).mkdir(parents=True, exist_ok=True)

    # Nom du fichier de sortie basé sur le nom de la vidéo
    nom_video = Path(chemin_video).stem
    chemin_sortie = os.path.join(dossier_sortie, f"{nom_video}.txt")

    with open(chemin_sortie, "w", encoding="utf-8") as f:
        f.write(texte)

    # Sauvegarder aussi les segments avec timestamps
    chemin_json = os.path.join(dossier_sortie, f"{nom_video}_segments.json")
    segments = [
        {
            "debut": seg["start"],
            "fin": seg["end"],
            "texte": seg["text"].strip()
        }
        for seg in resultat["segments"]
    ]
    with open(chemin_json, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)

    print(f"[Whisper] Transcription sauvegardée : {chemin_sortie}")
    print(f"[Whisper] Segments sauvegardés     : {chemin_json}")
    print(f"[Whisper] Nombre de mots transcrits : {len(texte.split())}")

    return chemin_sortie


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Transcription vidéo avec Whisper")
    parser.add_argument("--video", required=True, help="Chemin vers la vidéo")
    parser.add_argument("--sortie", default="data/transcripts", help="Dossier de sortie")
    args = parser.parse_args()

    chemin = transcrire_video(args.video, args.sortie)
    print(f"\nTranscription terminée : {chemin}")
