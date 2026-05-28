"""
QCM-AUTO — Point d'entrée principal
Lance le serveur FastAPI sur http://localhost:8000
"""

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os, json, shutil, tempfile
from pathlib import Path

load_dotenv()

# ── Imports des modules du projet ──────────────────────────────────────────
from backend.transcription.transcriber import transcribe_audio
from backend.llm.qcm_generator import run_pipeline
from backend.classifier.difficulty_classifier import predict_batch, train_classifier

app = FastAPI(title="QCM-Auto", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dossiers
UPLOAD_DIR = Path("data/raw_transcripts")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ── Entraîner le modèle RF au démarrage ────────────────────────────────────
print("🤖 Entraînement du classifieur de difficulté...")
try:
    train_classifier()
    print("✅ Classifieur prêt !")
except Exception as e:
    print(f"⚠️  Classifieur ignoré : {e}")

# ── Routes API ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    """Sert la page principale du portail professeur"""
    html_path = Path("frontend/index.html")
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>QCM-Auto</h1><p>Frontend introuvable.</p>")


@app.post("/api/generate")
async def generate_qcm(
    video: UploadFile = File(...),
    num_questions: int = 10,
    language: str = "fr"
):
    """
    Pipeline complet :
    1. Reçoit la vidéo
    2. Transcrit avec Whisper
    3. Génère les QCM avec Groq LLM
    4. Prédit la difficulté avec Random Forest
    """
    # Vérification du type de fichier
    allowed = [".mp4", ".mp3", ".wav", ".m4a", ".webm", ".ogg"]
    ext = Path(video.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(400, f"Format non supporté. Utilisez : {', '.join(allowed)}")

    # Sauvegarde temporaire
    tmp_path = Path(tempfile.mktemp(suffix=ext))
    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(video.file, f)

        # Étape 1 : Transcription
        print(f"🎙️  Transcription de '{video.filename}'...")
        transcript = transcribe_audio(
            str(tmp_path),
            model_size=os.getenv("WHISPER_MODEL", "tiny"),
            language=language
        )
        text = transcript["text"]
        if not text.strip():
            raise HTTPException(422, "Transcription vide — vérifiez que la vidéo contient de l'audio.")

        # Étape 2 : Génération QCM via LLM
        print("🧠 Génération des QCM...")
        result = run_pipeline(text, num_questions=num_questions)

        # Étape 3 : Prédiction de difficulté
        print("📊 Prédiction de la difficulté...")
        result["questions"] = predict_batch(result["questions"])

        # Ajout de la transcription dans la réponse
        result["transcript"] = text[:1000] + ("..." if len(text) > 1000 else "")
        result["filename"] = video.filename
        result["total_questions"] = len(result["questions"])

        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Erreur pipeline : {str(e)}")
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


@app.post("/api/train")
async def retrain():
    """Ré-entraîne le classifieur Random Forest"""
    try:
        result = train_classifier()
        return {"status": "ok", "accuracy": result["accuracy"], "cv_mean": result["cv_mean"]}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/health")
async def health():
    groq_key = os.getenv("GROQ_API_KEY", "")
    return {
        "status": "ok",
        "groq_configured": bool(groq_key and groq_key != "votre_cle_groq_ici"),
        "whisper_model": os.getenv("WHISPER_MODEL", "tiny")
    }


# ── Lancement ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("🚀 QCM-Auto démarre sur http://localhost:8000")
    print("=" * 50)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
