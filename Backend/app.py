# Backend/app.py
from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime
import json, os, glob, csv, itertools
from pathlib import Path
import sys
import numpy as np
import pandas as pd

# Add model directory to path
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model')
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

app = Flask(__name__)
CORS(app)

# Load model and processor (lazy loading)
_model = None
_processor = None

def load_model():
    """Load the trained model and processor"""
    global _model, _processor
    if _model is not None and _processor is not None:
        return _model, _processor
    
    try:
        import pickle
        model_path = os.path.join(MODEL_DIR, 'eeg_migraine_augmented_model.pkl')
        processor_path = os.path.join(MODEL_DIR, 'eeg_migraine_processor.pkl')
        
        # Check if model files exist
        if not os.path.exists(model_path) or not os.path.exists(processor_path):
            print(f"Warning: Model files not found. Expected:")
            print(f"  - {model_path}")
            print(f"  - {processor_path}")
            print("Please train the model first by running the pipeline in the model/ directory.")
            return None, None
        
        with open(model_path, 'rb') as f:
            _model = pickle.load(f)
        
        with open(processor_path, 'rb') as f:
            _processor = pickle.load(f)
        
        print("✓ Model and processor loaded successfully")
        return _model, _processor
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

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


@app.get("/api/muse/status")
def muse_status():
    """
    Check if Muse 2 is connected and streaming data.
    Returns connection status and whether real data is available.
    """
    row = read_latest_row_from_rolling_csv()
    if row:
        # Check if we have actual EEG data (not all zeros)
        delta = _to_float_or_none(row.get("delta")) or 0.0
        theta = _to_float_or_none(row.get("theta")) or 0.0
        alpha = _to_float_or_none(row.get("alpha")) or 0.0
        beta = _to_float_or_none(row.get("beta")) or 0.0
        gamma = _to_float_or_none(row.get("gamma")) or 0.0
        
        # Check if we have meaningful data (at least one band has non-zero value)
        has_data = any([delta > 0, theta > 0, alpha > 0, beta > 0, gamma > 0])
        
        return jsonify({
            'connected': has_data,
            'has_data': has_data,
            'timestamp': row.get("timestamp"),
            'message': 'Muse 2 connected and streaming' if has_data else 'Muse 2 file found but no data yet'
        })
    
    return jsonify({
        'connected': False,
        'has_data': False,
        'timestamp': None,
        'message': 'Muse 2 not connected. Please start the Muse 2 stream.'
    })


@app.get("/api/eeg/current")
def eeg_current():
    """
    Returns the latest processed snapshot with migraine prediction.
    Only returns real data if Muse 2 is connected, otherwise returns null values.
    """
    row = read_latest_row_from_rolling_csv()
    if row:
        # Extract EEG data
        delta = _to_float_or_none(row.get("delta")) or 0.0
        theta = _to_float_or_none(row.get("theta")) or 0.0
        alpha = _to_float_or_none(row.get("alpha")) or 0.0
        beta = _to_float_or_none(row.get("beta")) or 0.0
        gamma = _to_float_or_none(row.get("gamma")) or 0.0
        
        # Check if we have actual data (not all zeros)
        has_data = any([delta > 0, theta > 0, alpha > 0, beta > 0, gamma > 0])
        
        if has_data:
            eeg_data = {
                'timestamp': row.get("timestamp"),
                'delta': delta,
                'theta': theta,
                'alpha': alpha,
                'beta': beta,
                'gamma': gamma,
                'heart_rate_bpm': _to_float_or_none(row.get("heart_rate_bpm")),
                'breathing_rate_bpm': _to_float_or_none(row.get("breathing_rate_bpm")),
                'head_pitch': _to_float_or_none(row.get("head_pitch")),
                'head_roll': _to_float_or_none(row.get("head_roll")),
                'head_movement': _to_float_or_none(row.get("head_movement")),
                'connected': True
            }
            
            # Add prediction if model is available
            prediction = predict_migraine_severity(eeg_data)
            if prediction:
                eeg_data.update(prediction)
            
            return jsonify(eeg_data)
    
    # Return null/empty data when Muse 2 is not connected
    return jsonify({
        'connected': False,
        'timestamp': None,
        'delta': None,
        'theta': None,
        'alpha': None,
        'beta': None,
        'gamma': None,
        'heart_rate_bpm': None,
        'breathing_rate_bpm': None,
        'head_pitch': None,
        'head_roll': None,
        'head_movement': None,
        'migraine_severity': None,
        'migraine_stage': None,
        'migraine_interpretation': 'Connect Muse 2 to see predictions',
        'migraine_risk_level': None
    })


def predict_migraine_severity(eeg_data):
    """
    Predict migraine severity from EEG data using the trained model.
    Returns prediction dict or None if model unavailable.
    """
    model, processor = load_model()
    if model is None or processor is None:
        return None
    
    try:
        # Extract frequency band values
        delta = eeg_data.get('delta', 0.0) or 0.0
        theta = eeg_data.get('theta', 0.0) or 0.0
        alpha = eeg_data.get('alpha', 0.0) or 0.0
        beta = eeg_data.get('beta', 0.0) or 0.0
        gamma = eeg_data.get('gamma', 0.0) or 0.0
        
        # Check if we have valid EEG data
        if all(v == 0.0 for v in [delta, theta, alpha, beta, gamma]):
            return None
        
        # Map Muse 2 channels (TP9, AF7, AF8, TP10) to model features
        # The model expects specific channel names. We'll use average values across channels
        # or map to available features in the processor
        if not hasattr(processor, 'feature_columns'):
            return None
        
        # Create a DataFrame with the expected feature columns
        # Use average values for all channels (simplified approach)
        feature_dict = {}
        for col in processor.feature_columns:
            # Try to match column names or use average values
            if 'delta' in col.lower():
                feature_dict[col] = delta
            elif 'theta' in col.lower():
                feature_dict[col] = theta
            elif 'alpha' in col.lower():
                feature_dict[col] = alpha
            elif 'beta' in col.lower():
                feature_dict[col] = beta
            elif 'gamma' in col.lower():
                feature_dict[col] = gamma
            else:
                # Use average of all bands for unknown features
                feature_dict[col] = (delta + theta + alpha + beta + gamma) / 5.0
        
        # Create DataFrame
        eeg_df = pd.DataFrame([feature_dict])
        
        # Prepare features using processor
        X = eeg_df.values
        
        # Handle missing values
        X_imputed = processor.imputer.transform(X)
        
        # Scale features
        X_scaled = processor.scaler.transform(X_imputed)
        
        # Make prediction
        severity_score = model.predict(X_scaled)[0]
        severity_score = np.clip(severity_score, 0.0, 1.0)
        
        # Convert to label
        predicted_label = processor.severity_to_label(severity_score)
        
        # Determine interpretation
        if severity_score < 0.25:
            interpretation = "Non-Ictal (baseline/healthy)"
            risk_level = "low"
        elif severity_score < 0.75:
            interpretation = "Preictal (pre-migraine)"
            risk_level = "medium"
        else:
            interpretation = "Ictal (active migraine)"
            risk_level = "high"
        
        return {
            'migraine_severity': float(severity_score),
            'migraine_stage': predicted_label,
            'migraine_interpretation': interpretation,
            'migraine_risk_level': risk_level
        }
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None


@app.get("/api/eeg/history")
def eeg_history():
    """
    Returns last N rows of processed data with predictions.
    Prefers the rolling CSV (data/muse2_data_processed_latest.csv) if present,
    else falls back to the newest timestamped processed CSV.
    """
    N = int(request.args.get("limit", 200))

    # Prefer rolling file
    rows = []
    if os.path.exists(DATA_ROLLING):
        try:
            with open(DATA_ROLLING, newline="") as f:
                reader = list(csv.DictReader(f))
                rows = reader[-N:] if len(reader) > N else reader
        except Exception:
            pass  # fall through to timestamped fallback

    # Fallback: newest timestamped processed CSV
    if not rows:
        csv_path = latest_processed_csv_path()
        if csv_path and os.path.exists(csv_path):
            with open(csv_path, newline="") as f:
                reader = list(csv.DictReader(f))
                rows = reader[-N:] if len(reader) > N else reader
    
    # Add predictions to each row
    model, processor = load_model()
    if model is not None and processor is not None:
        for row in rows:
            eeg_data = {
                'delta': _to_float_or_none(row.get("delta")),
                'theta': _to_float_or_none(row.get("theta")),
                'alpha': _to_float_or_none(row.get("alpha")),
                'beta': _to_float_or_none(row.get("beta")),
                'gamma': _to_float_or_none(row.get("gamma")),
            }
            prediction = predict_migraine_severity(eeg_data)
            if prediction:
                row.update(prediction)
    
    return jsonify(rows)


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


@app.get("/api/prediction/current")
def prediction_current():
    """
    Get current migraine prediction based on latest EEG data.
    Only returns predictions if Muse 2 is connected and has data.
    """
    row = read_latest_row_from_rolling_csv()
    if row:
        delta = _to_float_or_none(row.get("delta")) or 0.0
        theta = _to_float_or_none(row.get("theta")) or 0.0
        alpha = _to_float_or_none(row.get("alpha")) or 0.0
        beta = _to_float_or_none(row.get("beta")) or 0.0
        gamma = _to_float_or_none(row.get("gamma")) or 0.0
        
        # Check if we have actual data
        has_data = any([delta > 0, theta > 0, alpha > 0, beta > 0, gamma > 0])
        
        if has_data:
            eeg_data = {
                'delta': delta,
                'theta': theta,
                'alpha': alpha,
                'beta': beta,
                'gamma': gamma,
            }
            prediction = predict_migraine_severity(eeg_data)
            if prediction:
                prediction['connected'] = True
                return jsonify(prediction)
    
    return jsonify({
        'connected': False,
        'migraine_severity': None,
        'migraine_stage': None,
        'migraine_interpretation': 'Connect Muse 2 to see predictions',
        'migraine_risk_level': None
    })


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", "5050"))  # default to 5050
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)