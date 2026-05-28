"""
Transcription automatique via faster-whisper (léger, CPU, 8GB RAM)
"""
import os, json
from pathlib import Path
from faster_whisper import WhisperModel

_model = None

def load_model(model_size: str = "tiny") -> WhisperModel:
    global _model
    if _model is None:
        print(f"[Whisper] Chargement modèle '{model_size}'...")
        _model = WhisperModel(model_size, device="cpu", compute_type="int8")
        print("[Whisper] Modèle prêt.")
    return _model

def transcribe_audio(audio_path: str, model_size: str = "tiny", language: str = "fr") -> dict:
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Fichier introuvable : {audio_path}")
    model = load_model(model_size)
    print(f"[Whisper] Transcription de '{audio_path}'...")
    segments, info = model.transcribe(audio_path, language=language, beam_size=1)
    full_text = ""
    seg_list = []
    for seg in segments:
        full_text += seg.text + " "
        seg_list.append({"start": round(seg.start,2), "end": round(seg.end,2), "text": seg.text.strip()})
    result = {"text": full_text.strip(), "segments": seg_list, "language": language, "source_file": str(audio_path)}
    print(f"[Whisper] Terminé : {len(result['text'])} caractères.")
    return result

def save_transcript(transcript: dict, output_dir: str = "data/raw_transcripts") -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    source = Path(transcript["source_file"]).stem
    out = os.path.join(output_dir, f"{source}_transcript.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(transcript, f, ensure_ascii=False, indent=2)
    return out
