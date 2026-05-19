"""
Interface web Flask - Portail professeur pour la génération de QCM.
"""

from flask import Flask, render_template, request, jsonify, send_file
import os
import sys
import json
import threading
from pathlib import Path
from werkzeug.utils import secure_filename

# Ajouter le dossier racine au path
sys.path.insert(0, str(Path(__file__).parent.parent))

app = Flask(__name__)
app.config["SECRET_KEY"] = "qcm_auto_secret_2024"
app.config["UPLOAD_FOLDER"] = "data/videos"
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB max

EXTENSIONS_AUTORISEES = {"mp4", "mkv", "avi", "mov", "webm"}

# Stockage de l'état des tâches en cours
taches = {}


def extension_autorisee(nom_fichier):
    return "." in nom_fichier and nom_fichier.rsplit(".", 1)[1].lower() in EXTENSIONS_AUTORISEES


def executer_pipeline(id_tache: str, chemin_video: str, nb_questions: int):
    """Exécute le pipeline en arrière-plan."""
    from src.transcription.whisper_transcriber import transcrire_video
    from src.extraction.concept_extractor import extraire_concepts
    from src.generation.qcm_generator import generer_qcm
    from src.classification.difficulty_classifier import entrainer_modele, classifier_difficulte

    try:
        taches[id_tache] = {"statut": "transcription", "progression": 10, "erreur": None}
        chemin_transcription = transcrire_video(chemin_video)

        taches[id_tache] = {"statut": "extraction", "progression": 35, "erreur": None}
        chemin_concepts = extraire_concepts(chemin_transcription)

        taches[id_tache] = {"statut": "generation", "progression": 60, "erreur": None}
        chemin_qcm_brut = generer_qcm(chemin_concepts, nb_questions=nb_questions)

        taches[id_tache] = {"statut": "classification", "progression": 85, "erreur": None}
        entrainer_modele()
        chemin_qcm_final = classifier_difficulte(chemin_qcm_brut)

        with open(chemin_qcm_final, "r", encoding="utf-8") as f:
            qcm_data = json.load(f)

        taches[id_tache] = {
            "statut": "termine",
            "progression": 100,
            "erreur": None,
            "qcm": qcm_data,
            "chemin_fichier": chemin_qcm_final
        }

    except Exception as e:
        taches[id_tache] = {
            "statut": "erreur",
            "progression": 0,
            "erreur": str(e)
        }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"erreur": "Aucun fichier vidéo fourni"}), 400

    fichier = request.files["video"]
    nb_questions = int(request.form.get("nb_questions", 10))

    if fichier.filename == "":
        return jsonify({"erreur": "Nom de fichier vide"}), 400

    if not extension_autorisee(fichier.filename):
        return jsonify({"erreur": "Format vidéo non supporté. Utilisez mp4, mkv, avi, mov ou webm"}), 400

    nom_securise = secure_filename(fichier.filename)
    Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)
    chemin_video = os.path.join(app.config["UPLOAD_FOLDER"], nom_securise)
    fichier.save(chemin_video)

    # Créer un ID de tâche unique
    import time
    id_tache = f"tache_{int(time.time())}"
    taches[id_tache] = {"statut": "demarrage", "progression": 0, "erreur": None}

    # Lancer le pipeline en arrière-plan
    thread = threading.Thread(
        target=executer_pipeline,
        args=(id_tache, chemin_video, nb_questions)
    )
    thread.daemon = True
    thread.start()

    return jsonify({"id_tache": id_tache, "message": "Pipeline démarré"})


@app.route("/statut/<id_tache>")
def statut_tache(id_tache):
    if id_tache not in taches:
        return jsonify({"erreur": "Tâche non trouvée"}), 404
    return jsonify(taches[id_tache])


@app.route("/qcm/<id_tache>")
def obtenir_qcm(id_tache):
    if id_tache not in taches:
        return jsonify({"erreur": "Tâche non trouvée"}), 404

    tache = taches[id_tache]
    if tache["statut"] != "termine":
        return jsonify({"erreur": "QCM pas encore prêt"}), 202

    return jsonify(tache.get("qcm", {}))


@app.route("/exporter/<id_tache>")
def exporter_qcm(id_tache):
    if id_tache not in taches or taches[id_tache]["statut"] != "termine":
        return jsonify({"erreur": "QCM non disponible"}), 404

    chemin = taches[id_tache]["chemin_fichier"]
    return send_file(chemin, as_attachment=True, download_name="qcm_genere.json")


if __name__ == "__main__":
    # Créer les dossiers nécessaires
    for dossier in ["data/videos", "data/transcripts", "data/concepts",
                    "data/qcm_raw", "data/qcm_filtered", "models"]:
        Path(dossier).mkdir(parents=True, exist_ok=True)

    print("\n" + "="*50)
    print("  PORTAIL PROFESSEUR - QCM AUTO")
    print("  Ouvrir : http://localhost:5000")
    print("="*50 + "\n")

    app.run(debug=True, host="0.0.0.0", port=5000)
