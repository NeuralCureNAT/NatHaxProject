# Backend/app.py
from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime
import json, os, glob, csv, itertools

app = Flask(__name__)
CORS(app)

# ---------- helpers ----------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))  # Backend/
ROOT_PARENT  = os.path.dirname(PROJECT_ROOT)                # project root

def read_latest_json():
    """
    Reads latest.json written by muse2_tracker.py.
    We try both Backend/ and project root, depending on where you run the tracker.
    """
    candidate_paths = [
        os.path.join(PROJECT_ROOT, "latest.json"),
        os.path.join(ROOT_PARENT, "latest.json")
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
    return None

def latest_processed_csv_path():
    """
    Finds the most recent processed CSV produced by muse2_tracker.py.
    Pattern is *processed_YYYYMMDD_HHMMSS.csv.
    Checks both Backend/ and project root.
    """
    patterns = [
        os.path.join(PROJECT_ROOT, "muse2_data_processed_*.csv"),
        os.path.join(ROOT_PARENT,  "muse2_data_processed_*.csv"),
    ]
    matches = []
    for pat in patterns:
        matches.extend(glob.glob(pat))
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)

# ---------- endpoints ----------
@app.get("/api/health")
def health():
    return jsonify(status="ok", time=datetime.utcnow().isoformat() + "Z")

@app.get("/api/eeg/current")
def eeg_current():
    data = read_latest_json()
    if data is None:
        # Safe fallback (your frontend already handles mock too)
        return jsonify(focus=78, attention=72, meditation=58, timestamp=datetime.utcnow().isoformat() + "Z")
    return jsonify(data)

@app.get("/api/eeg/history")
def eeg_history():
    """
    Returns last N rows from the newest processed CSV.
    Works with your existing CSVs that have headers (columns + values).
    """
    N = int(request.args.get("limit", 200))
    csv_path = latest_processed_csv_path()
    if not csv_path or not os.path.exists(csv_path):
        return jsonify([])

    rows = []
    # Efficient tail using csv + reversed iterator
    with open(csv_path, newline="") as f:
        reader = list(csv.DictReader(f))
        rows = reader[-N:] if len(reader) > N else reader
    return jsonify(rows)

@app.get("/api/environment/current")
def env_current():
    # Stub until Arduino is wired in
    return jsonify(light=45, temperature=22, humidity=65, timestamp=datetime.utcnow().isoformat()+"Z")

@app.post("/api/arduino/light")
def set_light():
    # Stub: accept brightness and pretend to set it
    body = request.get_json(silent=True) or {}
    brightness = int(body.get("brightness", 50))
    # TODO: write to serial here later
    return jsonify(success=True, brightness=brightness)

@app.post("/api/user/profile")
def user_profile():
    # Accept onboarding profile; you can persist later (Firestore/SQLite)
    body = request.get_json(silent=True) or {}
    # TODO: save to DB if desired
    return jsonify(success=True, received=body)

if __name__ == "__main__":
    # Run from project root:  python Backend/app.py
    app.run(host="0.0.0.0", port=5000, debug=True)
