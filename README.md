# QCM Auto — Génération automatique de QCM sur cours vidéo

> Projet IA — Note de TP  
> Pipeline complet : Transcription → Extraction → Génération → Classification → Interface

---

## Présentation

Ce projet génère automatiquement des QCM (Questions à Choix Multiples) à partir de vidéos de cours, avec validation automatique de la difficulté par un classifieur Random Forest.

### Pipeline

```
Vidéo cours
    │
    ▼
Transcription automatique (Whisper)
    │
    ▼
Résumé + extraction des concepts clés (LLM Claude)
    │
    ▼
Génération de questions + distracteurs plausibles (LLM)
    │
    ▼
Filtrage par difficulté (Random Forest entraîné sur QCM historiques)
    │
    ▼
QCM éditable avec indication de difficulté
```

---

## Structure du projet

```
qcm_auto/
├── data/
│   ├── videos/          # Vidéos de cours (non versionné)
│   ├── transcripts/     # Transcriptions générées
│   ├── concepts/        # Concepts extraits par le LLM
│   ├── qcm_raw/         # QCM avant filtrage
│   ├── qcm_filtered/    # QCM après filtrage par difficulté
│   └── historical/      # QCM annotés pour entraîner le RF
├── models/              # Modèles entraînés (.pkl)
├── src/
│   ├── transcription/   # Module Whisper
│   ├── extraction/      # Module LLM extraction
│   ├── generation/      # Module génération QCM
│   ├── classification/  # Module Random Forest
│   └── evaluation/      # Module métriques
├── interface/           # Application Flask (portail professeur)
│   ├── static/
│   └── templates/
├── tests/               # Tests unitaires
├── docs/                # Documentation
├── notebooks/           # Expérimentations
├── .env.example         # Template variables d'environnement
├── requirements.txt     # Dépendances Python
└── README.md
```

---

## Installation

### Prérequis
- Python 3.10+
- ffmpeg installé sur le système ([télécharger ici](https://ffmpeg.org/download.html))
- Clé API Anthropic ([obtenir ici](https://console.anthropic.com))

### Étapes

```bash
# 1. Cloner le dépôt
git clone https://github.com/VOTRE_USERNAME/qcm_auto.git
cd qcm_auto

# 2. Créer un environnement virtuel
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Configurer les variables d'environnement
copy .env.example .env   # Windows
# puis éditer .env avec votre clé API Anthropic
```

---

## Utilisation

### Lancer l'interface web (portail professeur)

```bash
python interface/app.py
```
Ouvrir dans le navigateur : `http://localhost:5000`

### Utilisation en ligne de commande

```bash
# Pipeline complet sur une vidéo
python src/pipeline.py --video data/videos/mon_cours.mp4

# Étapes séparées
python src/transcription/whisper_transcriber.py --video data/videos/mon_cours.mp4
python src/extraction/concept_extractor.py --transcript data/transcripts/mon_cours.txt
python src/generation/qcm_generator.py --concepts data/concepts/mon_cours.json
python src/classification/difficulty_classifier.py --qcm data/qcm_raw/mon_cours.json
```

---

## Évaluation

| Métrique | Description |
|---|---|
| Taux de rejet des distracteurs | % de distracteurs jugés non pertinents par un expert |
| Corrélation difficulté prédite/observée | Pearson entre difficulté RF et taux de réussite réel |

---

## Technologies utilisées

| Composant | Technologie |
|---|---|
| Transcription | OpenAI Whisper |
| LLM | Anthropic Claude (claude-sonnet-4-20250514) |
| Classifieur | Scikit-learn Random Forest |
| Interface | Flask |
| Déploiement | GitHub |

---

## Auteur

Projet réalisé dans le cadre du cours d'Intelligence Artificielle.
