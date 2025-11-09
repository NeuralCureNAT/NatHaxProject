# EEG Migraine Classification Project

All project files are in the `model/` directory.

## Quick Start

### Linux/macOS

1. **Navigate to the model directory:**
```bash
cd model
```

2. **Activate virtual environment (from parent directory):**
```bash
cd ..
source venv/bin/activate
cd model
```

3. **Run the pipeline:**
```bash
python run_pipeline.py
```

### Windows

1. **Navigate to the model directory:**
```cmd
cd model
```

2. **Activate virtual environment (from parent directory):**
```cmd
cd ..
venv\Scripts\activate
cd model
```

3. **Run the pipeline:**
```cmd
python run_pipeline.py
```
   **Note:** If `python` doesn't work, try `py` or `python3`

## Project Structure

```
nathax/
├── venv/                    # Virtual environment (keep in parent)
├── model/                   # All project files are here
│   ├── *.py                # Python scripts
│   ├── *.xlsx              # Data files
│   ├── *.pkl               # Trained models
│   ├── *.md                # Documentation
│   └── requirements.txt    # Dependencies
└── README.md               # This file
```

## Platform Support

✅ **Cross-platform compatible:**
- ✅ Windows 10/11
- ✅ Linux (Ubuntu, Debian, etc.)
- ✅ macOS

## For Detailed Instructions

See `model/README.md` for complete documentation, installation instructions, and troubleshooting guide for all platforms.
