# Muse 2 EEG, PPG, Motion, and Breathing Tracker

This Python script connects to your **MUSE-F7DD** device and tracks:
- **EEG (Electroencephalography)**: Brain activity with frequency bands (Delta, Theta, Alpha, Beta, Gamma)
- **PPG (Photoplethysmography)**: Heart rate monitoring
- **Head Movement/Posture**: Accelerometer and gyroscope data
- **Breathing Activity**: Estimated from accelerometer data

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Start Streaming (Terminal 1)

```bash
source venv/bin/activate
python start_muse_stream.py
```

This will automatically find and connect to **MUSE-F7DD**. The script will:
- Scan for your Muse device
- Connect to MUSE-F7DD automatically
- Start streaming with all sensors (EEG, PPG, accelerometer, gyroscope)
- **Run continuously** until you stop it

**Important:** 
- **Keep this terminal open** - the stream runs continuously
- **Press `Ctrl+C`** in this terminal when you want to stop streaming
- The stream will keep running until you manually stop it
- You can leave it running for as long as you need to collect data

### 3. Run Tracker (Terminal 2)

In a **new terminal**:

```bash
source venv/bin/activate
python muse2_tracker.py
```

When prompted, type `start` and press Enter.

### 4. Stop Tracking

**To stop data collection:**
- Press `Ctrl+C` in the **tracker terminal** (Terminal 2)
- Data will be automatically saved to CSV files

**To stop streaming:**
- Press `Ctrl+C` in the **stream terminal** (Terminal 1)
- This will disconnect from your Muse device

**Note:** You can stop the tracker anytime while keeping the stream running, or stop both. The stream runs continuously until you stop it with `Ctrl+C`.

## Installation Details

### Python Version

**Recommended: Python 3.11 or 3.12**

Python 3.14 has known compatibility issues with muselsl. If you encounter errors:

```bash
# Install Python 3.11
brew install python@3.11

# Create new virtual environment
python3.11 -m venv venv311
source venv311/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

All required packages are in `requirements.txt`:
- `pylsl` - Lab Streaming Layer for data streaming
- `numpy` - Numerical computations
- `pandas` - Data processing and CSV export
- `scipy` - Signal processing and filtering
- `muselsl` - Muse device connection
- `nest_asyncio` - Python 3.14 compatibility fix

### Optional: Install LSL via Homebrew

If you encounter linking errors with pylsl:

```bash
brew install labstreaminglayer/tap/lsl
```

**Note:** This is usually not needed as `pylsl` includes necessary binaries.

## Usage

### Connecting to Your Device

The script automatically connects to **MUSE-F7DD**. To connect to a different device:

```bash
python start_muse_stream.py DEVICE-NAME
```

### What Gets Tracked

The tracker collects and processes:

1. **EEG Data** (4 channels: TP9, AF7, AF8, TP10)
   - Delta waves (0.5-4 Hz)
   - Theta waves (4-8 Hz)
   - Alpha waves (8-13 Hz)
   - Beta waves (13-30 Hz)
   - Gamma waves (30-100 Hz)

2. **Heart Rate** (from PPG sensor)
   - Calculated in beats per minute (BPM)

3. **Head Posture** (from accelerometer)
   - Pitch angle
   - Roll angle
   - Movement magnitude

4. **Breathing Rate** (estimated from accelerometer)
   - Calculated in breaths per minute

### Output Files

When you stop tracking, two CSV files are created:

1. **`muse2_data_raw_TIMESTAMP.csv`**
   - All raw sensor data with timestamps
   - EEG channels, PPG values, accelerometer, gyroscope

2. **`muse2_data_processed_TIMESTAMP.csv`**
   - Processed features extracted from raw data
   - Frequency band powers, heart rate, posture, breathing rate

## Noise Reduction

The script automatically applies noise reduction:

- ✅ **Notch filtering** - Removes 50/60 Hz power line noise
- ✅ **Artifact rejection** - Filters extreme values (z-score thresholding)
- ✅ **Moving average** - Smooths data to reduce high-frequency noise
- ✅ **Bandpass filtering** - Isolates frequency bands of interest
- ✅ **Peak detection** - Uses prominence thresholds for accurate heart/breath rate

### Before Starting - Device Setup

1. **Electrode Contact**
   - Ensure headband is properly fitted
   - Clean electrodes
   - Wet electrodes slightly for better contact (use water or electrode gel)
   - Check contact quality (should show good signal in Muse app)

2. **Environment**
   - Minimize electrical interference
   - Avoid fluorescent lights (they create 50/60 Hz noise)
   - Stay away from power lines and electrical equipment
   - Use in a quiet, stable environment

3. **Physical Preparation**
   - Remove metal jewelry or accessories
   - Stay still during data collection
   - Keep body relaxed
   - Close eyes or maintain steady gaze to reduce eye movement artifacts

## Troubleshooting

### "No EEG stream found"
- Make sure `python start_muse_stream.py` is running in another terminal
- Check that your Muse 2 is turned on and connected
- Wait a few seconds after starting the stream before running the tracker

### "MUSE-F7DD not found"
- Make sure your Muse 2 is turned on
- Check Bluetooth is enabled on your Mac
- Try turning the Muse 2 off and on again
- Move closer to your computer
- The script will show all available devices if MUSE-F7DD isn't found

### Python 3.14 "Timeout should be used inside a task" error
- **Recommended:** Use Python 3.11 or 3.12 (see Installation section)
- The workaround script should work, but Python 3.11/3.12 is more reliable

### Poor signal quality
- Check electrode contact (should be green in Muse app)
- Wet electrodes slightly for better contact
- Minimize movement and electrical interference
- Ensure headband is properly fitted

### Import errors
- Make sure virtual environment is activated: `source venv/bin/activate`
- Reinstall dependencies: `pip install -r requirements.txt`

## Data Analysis

You can analyze the CSV files using:

- **Python**: pandas, matplotlib, seaborn
- **Excel** or **Google Sheets**
- **MATLAB** or **R**
- **Jupyter Notebooks**

Example Python analysis:
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load processed data
df = pd.read_csv('muse2_data_processed_20240101_120000.csv')

# Plot alpha power over time
plt.plot(df['timestamp'], df['AF7_alpha'])
plt.xlabel('Time')
plt.ylabel('Alpha Power')
plt.title('Alpha Wave Activity')
plt.show()
```

## Technical Notes

- Data processing happens in real-time
- Processing occurs every second to balance accuracy and performance
- Frequency band extraction requires at least 0.25 seconds of data
- Heart rate calculation requires at least 1 second of PPG data
- Breathing rate estimation uses accelerometer z-axis (chest movement)
- All noise reduction is applied automatically

## Files

- `muse2_tracker.py` - Main tracking script
- `start_muse_stream.py` - Stream starter (connects to MUSE-F7DD)
- `requirements.txt` - Python dependencies
- `README.md` - This file

## License

This script is provided as-is for educational and research purposes.
