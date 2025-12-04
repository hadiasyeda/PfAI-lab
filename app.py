
import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from detector import detect_animals, cluster_herds, annotate_image
from werkzeug.utils import secure_filename
from geopy.geocoders import Nominatim
import requests
import uuid

UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXT = {"png","jpg","jpeg"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

def allowed_file(filename):
    return "." in filename and filename.rsplit(".",1)[1].lower() in ALLOWED_EXT

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return redirect(url_for("index"))
    file = request.files["image"]
    lat = request.form.get("lat", type=float)
    lon = request.form.get("lon", type=float)
    if file and allowed_file(file.filename):
        fname = secure_filename(file.filename)
        unique = f"{uuid.uuid4().hex}_{fname}"
        in_path = os.path.join(app.config["UPLOAD_FOLDER"], unique)
        file.save(in_path)

        
        detections = detect_animals(in_path, conf_thresh=0.25)
        clusters = cluster_herds(detections, eps_px=120, min_samples=3)

        out_name = f"annot_{unique}"
        out_path = os.path.join(app.config["UPLOAD_FOLDER"], out_name)
        annotate_image(in_path, detections, clusters, out_path)

        
        location = None
        if lat is not None and lon is not None:
            try:
                geolocator = Nominatim(user_agent="herd_detector_app")
                loc = geolocator.reverse((lat, lon), exactly_one=True, timeout=10)
                location = loc.address if loc else None
            except Exception as e:
                location = None

        resp = {
            "input": url_for("static", filename=f"uploads/{unique}"),
            "output": url_for("static", filename=f"uploads/{out_name}"),
            "detections": detections,
            "clusters": clusters,
            "location": location,
            "coords": (lat, lon) if lat is not None and lon is not None else None
        }
        return render_template("index.html", result=resp)
    else:
        return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
