# Backend/app.py
from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime
import json, os, glob, csv, itertools
from pathlib import Path

app = Flask(__name__)
CORS(app)

# ---------- paths & helpers ----------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))  # Backend/
ROOT_PARENT  = os.path.dirname(PROJECT_ROOT)                # project root

# Rolling CSV written by muse2_tracker.py (e.g., data/muse2_data_processed_latest.csv)
DATA_ROLLING = os.path.join(ROOT_PARENT, "data", "muse2_data_processed_latest.csv")


def latest_processed_csv_path():
    """
    Finds the most recent timestamped processed CSV produced by muse2_tracker.py.
    Pattern: muse2_data_processed_YYYYMMDD_HHMMSS.csv
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


def _to_float_or_none(val):
    """Convert val to float if possible, else None."""
    try:
        if val is None or val == "":
            return None
        return float(val)
    except Exception:
        return None


def read_latest_row_from_rolling_csv():
    """
    Read the last row from the rolling CSV at data/muse2_data_processed_latest.csv.
    Returns a dict or None if file/rows are unavailable.
    """
    p = DATA_ROLLING
    if not os.path.exists(p):
        return None
    try:
        with open(p, newline="") as f:
            rows = list(csv.DictReader(f))
            return rows[-1] if rows else None
    except Exception:
        return None


# ---------- endpoints ----------
@app.get("/api/health")
def health():
    return jsonify(status="ok", time=datetime.utcnow().isoformat() + "Z")


@app.get("/api/eeg/current")
def eeg_current():
    """
    Returns the latest processed snapshot.
    Prefers the rolling CSV row; falls back to a safe mock if unavailable.
    """
    row = read_latest_row_from_rolling_csv()
    if row:
        return jsonify(
            timestamp=row.get("timestamp"),
            delta=_to_float_or_none(row.get("delta")),
            theta=_to_float_or_none(row.get("theta")),
            alpha=_to_float_or_none(row.get("alpha")),
            beta=_to_float_or_none(row.get("beta")),
            gamma=_to_float_or_none(row.get("gamma")),
            heart_rate_bpm=_to_float_or_none(row.get("heart_rate_bpm")),
            breathing_rate_bpm=_to_float_or_none(row.get("breathing_rate_bpm")),
            head_pitch=_to_float_or_none(row.get("head_pitch")),
            head_roll=_to_float_or_none(row.get("head_roll")),
            head_movement=_to_float_or_none(row.get("head_movement")),
        )
    # Safe fallback (your frontend already has a mock too)
    return jsonify(
        focus=78, attention=72, meditation=58, timestamp=datetime.utcnow().isoformat() + "Z"
    )


@app.get("/api/eeg/history")
def eeg_history():
    """
    Returns last N rows of processed data.
    Prefers the rolling CSV (data/muse2_data_processed_latest.csv) if present,
    else falls back to the newest timestamped processed CSV.
    """
    N = int(request.args.get("limit", 200))

    # Prefer rolling file
    if os.path.exists(DATA_ROLLING):
        try:
            with open(DATA_ROLLING, newline="") as f:
                reader = list(csv.DictReader(f))
                return jsonify(reader[-N:] if len(reader) > N else reader)
        except Exception:
            pass  # fall through to timestamped fallback

    # Fallback: newest timestamped processed CSV
    csv_path = latest_processed_csv_path()
    if not csv_path or not os.path.exists(csv_path):
        return jsonify([])

    with open(csv_path, newline="") as f:
        reader = list(csv.DictReader(f))
        return jsonify(reader[-N:] if len(reader) > N else reader)


@app.get("/api/environment/current")
def env_current():
    # Stub until Arduino is wired in
    return jsonify(
        light=45,
        temperature=22,
        humidity=65,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


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
    import os
    port = int(os.environ.get("PORT", "5050"))  # default to 5050
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)