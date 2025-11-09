# Muse 2 EEG, PPG, Motion, and Breathing Tracker

This script handles **real-time signal tracking** from your **Muse 2 (MUSE-F7DD)** device — streaming EEG, PPG, motion, and breathing data into the Neuralcare backend.  
It operates alongside `start_muse_stream.py` and integrates with the Flask backend (`app.py`) to serve live EEG data to the frontend.

---

## Overview

- **EEG (Electroencephalography):** Brainwave tracking — Delta, Theta, Alpha, Beta, Gamma bands  
- **PPG (Photoplethysmography):** Heart rate detection  
- **Motion/Posture:** Accelerometer and gyroscope  
- **Breathing:** Estimated from accelerometer signals  

Processed data is automatically written to `data/muse2_data_processed_latest.csv`, which the Flask backend reads to power `/api/eeg/latest` and `/api/eeg/history` endpoints.

---

## Setup

### 1. Virtual Environment & Dependencies

```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
