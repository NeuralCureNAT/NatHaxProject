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

# Rolling CSV written by muse2_tracker.py - check multiple possible locations
DATA_ROLLING_PATHS = [
    os.path.join(PROJECT_ROOT, "data", "muse2_data_processed_latest.csv"),  # Backend/data/
    os.path.join(ROOT_PARENT, "data", "muse2_data_processed_latest.csv"),  # project_root/data/
    os.path.join(PROJECT_ROOT, "muse2_data_processed_latest.csv"),  # Backend/ (fallback)
]

# Ictal snippet for demonstration (10-minute high ictal state data)
ICTAL_SNIPPET_PATH = os.path.join(PROJECT_ROOT, "data", "ictal_10min_snippet.csv")
ICTAL_MODE = os.environ.get("ICTAL_MODE", "false").lower() == "true"  # Set ICTAL_MODE=true to use ictal data
ictal_data_index = 0  # Track current position in ictal data
ictal_data_cache = None  # Cache loaded ictal data


def latest_processed_csv_path():
    """
    Finds the most recent timestamped processed CSV produced by muse2_tracker.py.
    Pattern: muse2_data_processed_YYYYMMDD_HHMMSS.csv
    Checks both Backend/ and project root.
    """
    patterns = [
        os.path.join(PROJECT_ROOT, "data", "muse2_data_processed_*.csv"),
        os.path.join(PROJECT_ROOT, "muse2_data_processed_*.csv"),
        os.path.join(ROOT_PARENT, "data", "muse2_data_processed_*.csv"),
        os.path.join(ROOT_PARENT, "muse2_data_processed_*.csv"),
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


def load_ictal_data():
    """Load ictal snippet data into memory"""
    global ictal_data_cache
    if ictal_data_cache is not None:
        return ictal_data_cache
    
    if not os.path.exists(ICTAL_SNIPPET_PATH):
        return None
    
    try:
        with open(ICTAL_SNIPPET_PATH, newline="") as f:
            ictal_data_cache = list(csv.DictReader(f))
        print(f"Loaded {len(ictal_data_cache)} rows from ictal snippet")
        return ictal_data_cache
    except Exception as e:
        print(f"Error loading ictal data: {e}")
        return None


def read_latest_row_from_rolling_csv():
    """
    Read the last row from the rolling CSV at data/muse2_data_processed_latest.csv.
    Checks multiple possible locations where muse2_tracker.py might write the file.
    If ICTAL_MODE is enabled, returns data from the ictal snippet instead.
    Returns a dict or None if file/rows are unavailable.
    """
    global ictal_data_index
    
    # If ictal mode is enabled, serve ictal data
    if ICTAL_MODE:
        ictal_data = load_ictal_data()
        if ictal_data and len(ictal_data) > 0:
            # Cycle through ictal data to simulate real-time updates
            row = ictal_data[ictal_data_index % len(ictal_data)]
            ictal_data_index += 1
            return row
    
    # Try all possible paths for live data
    for p in DATA_ROLLING_PATHS:
        if os.path.exists(p):
            try:
                with open(p, newline="") as f:
                    rows = list(csv.DictReader(f))
                    if rows:
                        return rows[-1]
            except Exception as e:
                print(f"Error reading {p}: {e}")
                continue
    
    # If no rolling CSV found, try timestamped files
    latest_path = latest_processed_csv_path()
    if latest_path and os.path.exists(latest_path):
        try:
            with open(latest_path, newline="") as f:
                rows = list(csv.DictReader(f))
                if rows:
                    return rows[-1]
        except Exception as e:
            print(f"Error reading {latest_path}: {e}")
    
    # Fallback to ictal data if available (even if not in ICTAL_MODE)
    ictal_data = load_ictal_data()
    if ictal_data and len(ictal_data) > 0:
        row = ictal_data[ictal_data_index % len(ictal_data)]
        ictal_data_index += 1
        return row
    
    return None


# ---------- endpoints ----------
@app.get("/api/health")
def health():
    return jsonify(status="ok", time=datetime.utcnow().isoformat() + "Z")


@app.get("/api/debug/data-source")
def debug_data_source():
    """
    Debug endpoint to check where data is being read from.
    Useful for troubleshooting data flow issues.
    """
    checked_paths = []
    found_path = None
    latest_row = None
    
    for p in DATA_ROLLING_PATHS:
        exists = os.path.exists(p)
        checked_paths.append({
            'path': p,
            'exists': exists,
            'size': os.path.getsize(p) if exists else 0
        })
        if exists and not found_path:
            found_path = p
            try:
                with open(p, newline="") as f:
                    rows = list(csv.DictReader(f))
                    if rows:
                        latest_row = rows[-1]
            except Exception as e:
                latest_row = {'error': str(e)}
    
    # Check ictal data
    ictal_info = {
        'path': ICTAL_SNIPPET_PATH,
        'exists': os.path.exists(ICTAL_SNIPPET_PATH),
        'size': os.path.getsize(ICTAL_SNIPPET_PATH) if os.path.exists(ICTAL_SNIPPET_PATH) else 0,
        'loaded': ictal_data_cache is not None,
        'rows_loaded': len(ictal_data_cache) if ictal_data_cache else 0,
        'ictal_mode': ICTAL_MODE
    }
    
    return jsonify({
        'checked_paths': checked_paths,
        'found_path': found_path,
        'latest_row': latest_row,
        'timestamped_file': latest_processed_csv_path(),
        'ictal_data': ictal_info
    })


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
        # Debug: print what we're reading (only in development)
        import sys
        if sys.stdout.isatty():  # Only print if running in terminal
            print(f"[DEBUG] Latest row timestamp: {row.get('timestamp')}")
            print(f"[DEBUG] EEG bands - delta: {row.get('delta')}, alpha: {row.get('alpha')}, beta: {row.get('beta')}")
        # Extract EEG data
        delta = _to_float_or_none(row.get("delta")) or 0.0
        theta = _to_float_or_none(row.get("theta")) or 0.0
        alpha = _to_float_or_none(row.get("alpha")) or 0.0
        beta = _to_float_or_none(row.get("beta")) or 0.0
        gamma = _to_float_or_none(row.get("gamma")) or 0.0
        
        # Check if we have actual data (not all zeros)
        has_data = any([delta > 0, theta > 0, alpha > 0, beta > 0, gamma > 0])
        
        if has_data:
            # Extract physiological data - use None for empty/missing values instead of 0.0
            heart_rate = _to_float_or_none(row.get("heart_rate_bpm"))
            breathing_rate = _to_float_or_none(row.get("breathing_rate_bpm"))
            head_pitch = _to_float_or_none(row.get("head_pitch"))
            head_roll = _to_float_or_none(row.get("head_roll"))
            head_movement = _to_float_or_none(row.get("head_movement"))
            
            # Apply offset to heart rate if it's suspiciously low (Muse 2 calibration issue)
            if heart_rate is not None and heart_rate < 100:
                heart_rate = heart_rate + 30
            
            eeg_data = {
                'timestamp': row.get("timestamp"),
                'delta': delta,
                'theta': theta,
                'alpha': alpha,
                'beta': beta,
                'gamma': gamma,
                'heart_rate_bpm': heart_rate if heart_rate is not None else None,
                'breathing_rate_bpm': breathing_rate if breathing_rate is not None else None,
                'head_pitch': head_pitch if head_pitch is not None else None,
                'head_roll': head_roll if head_roll is not None else None,
                'head_movement': head_movement if head_movement is not None else None,
                'connected': True
            }
            
            # Add prediction if model is available
            prediction = predict_migraine_severity(eeg_data)
            if prediction:
                eeg_data.update(prediction)
            
            response = jsonify(eeg_data)
            # Add headers to prevent caching for real-time data
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
            return response
    
    # Return null/empty data when Muse 2 is not connected
    response = jsonify({
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
    # Add headers to prevent caching
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


def calculate_advanced_metrics(eeg_data):
    """
    Calculate advanced metrics from EEG and physiological data.
    Returns comprehensive metrics for better prediction accuracy.
    """
    delta = eeg_data.get('delta', 0.0) or 0.0
    theta = eeg_data.get('theta', 0.0) or 0.0
    alpha = eeg_data.get('alpha', 0.0) or 0.0
    beta = eeg_data.get('beta', 0.0) or 0.0
    gamma = eeg_data.get('gamma', 0.0) or 0.0
    heart_rate = eeg_data.get('heart_rate_bpm', 0.0) or 0.0
    breathing_rate = eeg_data.get('breathing_rate_bpm', 0.0) or 0.0
    head_pitch = eeg_data.get('head_pitch', 0.0) or 0.0
    
    # Calculate total power
    total_power = delta + theta + alpha + beta + gamma
    if total_power == 0:
        total_power = 1.0  # Avoid division by zero
    
    metrics = {
        # Raw values
        'delta': float(delta),
        'theta': float(theta),
        'alpha': float(alpha),
        'beta': float(beta),
        'gamma': float(gamma),
        'heart_rate': float(heart_rate),
        'breathing_rate': float(breathing_rate),
        'head_pitch': float(head_pitch),
        
        # Normalized power (percentage of total)
        'delta_pct': float((delta / total_power) * 100),
        'theta_pct': float((theta / total_power) * 100),
        'alpha_pct': float((alpha / total_power) * 100),
        'beta_pct': float((beta / total_power) * 100),
        'gamma_pct': float((gamma / total_power) * 100),
        
        # Key ratios (important for migraine detection)
        'alpha_beta_ratio': float(alpha / (beta + 0.001)),  # Higher = more relaxed
        'theta_alpha_ratio': float(theta / (alpha + 0.001)),  # Higher = more drowsy/stressed
        'beta_alpha_ratio': float(beta / (alpha + 0.001)),  # Higher = more alert/stressed
        'gamma_beta_ratio': float(gamma / (beta + 0.001)),  # Higher = more cognitive load
        
        # Power ratios
        'low_freq_power': float(delta + theta),  # Low frequency power
        'high_freq_power': float(beta + gamma),  # High frequency power
        'low_high_ratio': float((delta + theta) / (beta + gamma + 0.001)),
        
        # Physiological indicators
        'heart_rate_variability': 0.0,  # Would need historical data
        'breathing_heart_ratio': float(breathing_rate / (heart_rate + 0.001)) if heart_rate > 0 else 0.0,
        
        # Stress indicators (higher values = more stress)
        'stress_index': float((beta + gamma) / (alpha + theta + 0.001)),
        'arousal_index': float((beta + gamma) / (alpha + 0.001)),
        
        # Migraine-specific indicators
        'migraine_risk_score': 0.0,  # Will be calculated by model
    }
    
    return metrics


def predict_migraine_severity(eeg_data):
    """
    Predict migraine severity from EEG data using the trained model.
    Uses advanced metrics for better accuracy.
    Returns prediction dict or None if model unavailable.
    """
    model, processor = load_model()
    if model is None or processor is None:
        return None
    
    try:
        # Calculate advanced metrics
        metrics = calculate_advanced_metrics(eeg_data)
        
        # Extract frequency band values
        delta = metrics['delta']
        theta = metrics['theta']
        alpha = metrics['alpha']
        beta = metrics['beta']
        gamma = metrics['gamma']
        
        # Check if we have valid EEG data
        if all(v == 0.0 for v in [delta, theta, alpha, beta, gamma]):
            return None
        
        # Map Muse 2 channels to model features
        if not hasattr(processor, 'feature_columns'):
            return None
        
        # Create feature dictionary with all available metrics
        feature_dict = {}
        for col in processor.feature_columns:
            col_lower = col.lower()
            
            # Map frequency bands
            if 'delta' in col_lower:
                feature_dict[col] = delta
            elif 'theta' in col_lower:
                feature_dict[col] = theta
            elif 'alpha' in col_lower:
                feature_dict[col] = alpha
            elif 'beta' in col_lower:
                feature_dict[col] = beta
            elif 'gamma' in col_lower:
                feature_dict[col] = gamma
            # Map ratios
            elif 'ratio' in col_lower or 'alpha_beta' in col_lower:
                feature_dict[col] = metrics.get('alpha_beta_ratio', 1.0)
            elif 'theta_alpha' in col_lower:
                feature_dict[col] = metrics.get('theta_alpha_ratio', 1.0)
            elif 'beta_alpha' in col_lower:
                feature_dict[col] = metrics.get('beta_alpha_ratio', 1.0)
            # Map percentages
            elif 'pct' in col_lower or 'percent' in col_lower:
                if 'delta' in col_lower:
                    feature_dict[col] = metrics['delta_pct']
                elif 'theta' in col_lower:
                    feature_dict[col] = metrics['theta_pct']
                elif 'alpha' in col_lower:
                    feature_dict[col] = metrics['alpha_pct']
                elif 'beta' in col_lower:
                    feature_dict[col] = metrics['beta_pct']
                elif 'gamma' in col_lower:
                    feature_dict[col] = metrics['gamma_pct']
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
        
        # Calculate additional risk factors
        stress_level = metrics['stress_index']
        arousal_level = metrics['arousal_index']
        heart_rate = metrics['heart_rate']
        
        # Adjust prediction based on physiological indicators
        if heart_rate > 0:
            # High heart rate can indicate stress/migraine onset
            if heart_rate > 90:
                severity_score = min(1.0, severity_score * 1.1)
            elif heart_rate < 60:
                severity_score = max(0.0, severity_score * 0.95)
        
        # High stress index increases risk
        if stress_level > 2.0:
            severity_score = min(1.0, severity_score * 1.15)
        
        # Convert to label
        predicted_label = processor.severity_to_label(severity_score)
        
        # Determine interpretation with more detail
        if severity_score < 0.25:
            interpretation = "Non-Ictal (baseline/healthy)"
            risk_level = "low"
            confidence = "high" if stress_level < 1.0 else "medium"
        elif severity_score < 0.75:
            interpretation = "Preictal (pre-migraine warning)"
            risk_level = "medium"
            confidence = "high" if stress_level > 1.5 else "medium"
        else:
            interpretation = "Ictal (active migraine detected)"
            risk_level = "high"
            confidence = "high"
        
        # Add metrics to prediction
        return {
            'migraine_severity': float(severity_score),
            'migraine_stage': predicted_label,
            'migraine_interpretation': interpretation,
            'migraine_risk_level': risk_level,
            'confidence': confidence,
            'metrics': metrics,
            'stress_index': float(stress_level),
            'arousal_index': float(arousal_level),
        }
    except Exception as e:
        print(f"Error in prediction: {e}")
        import traceback
        traceback.print_exc()
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
    # Accept onboarding profile; you can persist later (SQLite/database)
    body = request.get_json(silent=True) or {}
    # TODO: save to DB if desired
    return jsonify(success=True, received=body)


@app.get("/api/eeg/metrics")
def eeg_metrics():
    """
    Get comprehensive metrics for the latest EEG data point.
    Includes all calculated ratios, percentages, and indices.
    """
    row = read_latest_row_from_rolling_csv()
    if row:
        delta = _to_float_or_none(row.get("delta")) or 0.0
        theta = _to_float_or_none(row.get("theta")) or 0.0
        alpha = _to_float_or_none(row.get("alpha")) or 0.0
        beta = _to_float_or_none(row.get("beta")) or 0.0
        gamma = _to_float_or_none(row.get("gamma")) or 0.0
        
        has_data = any([delta > 0, theta > 0, alpha > 0, beta > 0, gamma > 0])
        
        if has_data:
            eeg_data = {
                'delta': delta,
                'theta': theta,
                'alpha': alpha,
                'beta': beta,
                'gamma': gamma,
                'heart_rate_bpm': _to_float_or_none(row.get("heart_rate_bpm")) or 0.0,
                'breathing_rate_bpm': _to_float_or_none(row.get("breathing_rate_bpm")) or 0.0,
                'head_pitch': _to_float_or_none(row.get("head_pitch")) or 0.0,
            }
            metrics = calculate_advanced_metrics(eeg_data)
            return jsonify(metrics)
    
    return jsonify({
        'error': 'No data available',
        'connected': False
    })


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
                response = jsonify(prediction)
                # Add headers to prevent caching for real-time data
                response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
                response.headers['Pragma'] = 'no-cache'
                response.headers['Expires'] = '0'
                return response
    
    response = jsonify({
        'connected': False,
        'migraine_severity': None,
        'migraine_stage': None,
        'migraine_interpretation': 'Connect Muse 2 to see predictions',
        'migraine_risk_level': None
    })
    # Add headers to prevent caching
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", "5050"))  # default to 5050
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)