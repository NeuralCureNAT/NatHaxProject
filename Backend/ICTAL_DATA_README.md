# Ictal Data Integration

## Overview

A 10-minute snippet has been extracted from the `chbmit_ictal_cleaned.csv` file, focusing on the highest ictal state. This data has been converted to frequency bands (delta, theta, alpha, beta, gamma) and formatted to match the live website data format.

## Files

- **Source**: `chbmit_ictal_cleaned.csv` (original ictal data file)
- **Extracted Snippet**: `data/ictal_10min_snippet.csv` (600 rows, 10 minutes at 1 sample/second)
- **Extraction Script**: `extract_ictal_snippet.py`

## Data Format

The ictal snippet contains:
- `timestamp`: ISO format timestamp
- `delta`: Delta band power (0.5-4 Hz)
- `theta`: Theta band power (4-8 Hz)
- `alpha`: Alpha band power (8-13 Hz)
- `beta`: Beta band power (13-30 Hz)
- `gamma`: Gamma band power (30-100 Hz)
- `heart_rate_bpm`: Empty (not available in ictal data)
- `breathing_rate_bpm`: Empty (not available in ictal data)
- `head_pitch`: Empty (not available in ictal data)
- `head_roll`: Empty (not available in ictal data)
- `head_movement`: Empty (not available in ictal data)

## Usage

### Option 1: Automatic Fallback
The backend will automatically use ictal data as a fallback if no live Muse 2 data is available. No configuration needed!

### Option 2: Force Ictal Mode
To force the backend to use ictal data (even if live data is available):

```bash
cd Backend
ICTAL_MODE=true python3 app.py
```

Or set it as an environment variable:
```bash
export ICTAL_MODE=true
python3 app.py
```

### Option 3: Re-extract Snippet
To extract a new snippet from the ictal file:

```bash
cd Backend
python3 extract_ictal_snippet.py
```

This will:
1. Scan the entire ictal file for the highest ictal state
2. Extract a 10-minute snippet centered around that state
3. Convert channel data to frequency bands
4. Save to `data/ictal_10min_snippet.csv`

## How It Works

1. **Ictal Intensity Calculation**: The script calculates ictal intensity based on high-frequency activity (beta/gamma bands), as ictal states typically show increased high-frequency activity.

2. **Frequency Band Conversion**: Channel data is aggregated and converted to frequency bands using a simplified model that distributes power across bands based on signal characteristics.

3. **Real-time Simulation**: The backend cycles through the ictal data, serving one row per API call to simulate real-time updates.

4. **Seamless Integration**: The data format matches exactly what the frontend expects, so it displays automatically on the live website.

## Display on Website

The ictal data will appear on the website just like live data:
- **EEG Frequency Bands**: Delta, Theta, Alpha, Beta, Gamma values
- **Real-time Graphs**: All frequency bands plotted over time
- **Focus Meter**: Calculated from alpha/beta ratio
- **Migraine Predictions**: Generated from the ictal EEG patterns

## Notes

- The ictal snippet represents a 10-minute period of high ictal activity
- Data cycles continuously (loops back to start after 600 rows)
- Physiological metrics (heart rate, breathing, head position) are not available in ictal data
- The data is cached in memory for performance

