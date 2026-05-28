"""
Pipeline principal : exécute toutes les étapes de A à Z.
Usage : python src/pipeline.py --video data/videos/mon_cours.mp4
"""

import argparse
import os
import sys
from pathlib import Path

# Ajouter le dossier racine au path Python
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transcription.whisper_transcriber import transcrire_video
from src.extraction.concept_extractor import extraire_concepts
from src.generation.qcm_generator import generer_qcm
from src.classification.difficulty_classifier import entrainer_modele, classifier_difficulte
from src.evaluation.evaluator import rapport_complet


def pipeline_complet(chemin_video: str, nb_questions: int = 10):
    """
    Lance le pipeline complet de génération de QCM.

    Args:
        chemin_video: Chemin vers la vidéo de cours
        nb_questions: Nombre de questions à générer
    """
    print("\n" + "="*60)
    print("  PIPELINE QCM AUTO - DÉMARRAGE")
    print("="*60)

    # Étape 1 : Transcription
    print("\n[ÉTAPE 1/4] Transcription avec Whisper...")
    chemin_transcription = transcrire_video(chemin_video)

    # Étape 2 : Extraction des concepts
    print("\n[ÉTAPE 2/4] Extraction des concepts avec LLaMA3...")
    chemin_concepts = extraire_concepts(chemin_transcription)

    # Étape 3 : Génération du QCM
    print(f"\n[ÉTAPE 3/4] Génération de {nb_questions} questions QCM...")
    chemin_qcm_brut = generer_qcm(chemin_concepts, nb_questions=nb_questions)

    # Étape 4 : Classification + filtrage
    print("\n[ÉTAPE 4/4] Classification de la difficulté (Random Forest)...")
    entrainer_modele()
    chemin_qcm_final = classifier_difficulte(chemin_qcm_brut)

    # Rapport final
    print("\n" + "="*60)
    print("  PIPELINE TERMINÉ")
    print("="*60)
    rapport = rapport_complet(chemin_qcm_final)
    print(f"\nRésumé :")
    print(f"  - Questions générées : {rapport['resume_qcm']['total_questions']}")
    print(f"  - Répartition : {rapport['resume_qcm']['repartition_difficulte']}")
    print(f"  - Confiance RF moyenne : {rapport['confiance_moyenne_rf']:.1%}")
    print(f"\nFichier QCM final : {chemin_qcm_final}")
    print("="*60)

    return chemin_qcm_final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline complet de génération de QCM")
    parser.add_argument("--video", required=True, help="Chemin vers la vidéo de cours")
    parser.add_argument("--nb", type=int, default=10, help="Nombre de questions à générer")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Erreur : vidéo non trouvée : {args.video}")
        sys.exit(1)

    pipeline_complet(args.video, args.nb)
