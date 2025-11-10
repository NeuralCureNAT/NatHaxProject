# MigraineMinder Backend

This backend handles **real-time signal tracking** from your **Muse 2 (MUSE-F7DD)** device — streaming EEG, PPG, motion, and breathing data.  
It operates alongside `muse2_tracker.py` and integrates with the Flask backend (`app.py`) to serve live EEG data to the frontend.

---

## 🚀 Quick Start

### Setup Instructions

**1) Create & activate a fresh venv HERE**

```bash
python3 -m venv .venv
source .venv/bin/activate          # your prompt should show (.venv)
```

**2) Install deps (correct path since you're in Backend/)**

```bash
python3 -m pip install -r requirements.txt
```

**3) Run backend**

```bash
PORT=5050 python3 app.py
```

**Keep this terminal open.** In a second terminal (same venv):

```bash
cd Backend
python3 muse2_tracker.py        # or your start script
```

In a third terminal for the frontend:

```bash
cd Frontend
python3 -m http.server 5500
```

Then visit the UI at **http://localhost:5500**

### Testing the Backend

If `/api/health` shows JSON and `/api/eeg/current` returns data (or 204 if none yet), you're good.

---

## Overview

- **EEG (Electroencephalography):** Brainwave tracking — Delta, Theta, Alpha, Beta, Gamma bands  
- **PPG (Photoplethysmography):** Heart rate detection  
- **Motion/Posture:** Accelerometer and gyroscope  
- **Breathing:** Estimated from accelerometer signals  

Processed data is automatically written to `data/muse2_data_processed_latest.csv`, which the Flask backend reads to power `/api/eeg/current` and `/api/eeg/history` endpoints.

---

## Setup (Detailed)

### 1. Virtual Environment & Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
