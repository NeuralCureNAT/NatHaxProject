# Quick Start Guide

## For All Platforms

### 1. Setup (One-time)

**Linux/macOS:**
```bash
cd nathax
cd ..
python3 -m venv venv
source venv/bin/activate
cd model
pip install -r requirements.txt
```

**Windows:**
```cmd
cd nathax
cd ..
python -m venv venv
venv\Scripts\activate
cd model
pip install -r requirements.txt
```

### 2. Run Pipeline

**All Platforms:**
```bash
# Make sure you're in the model/ directory
# And virtual environment is activated

python run_pipeline.py
```

### 3. Make Predictions

```bash
python predict.py your_data.xlsx
```

## Platform Differences

| Task | Linux/macOS | Windows |
|------|-------------|---------|
| Activate venv | `source venv/bin/activate` | `venv\Scripts\activate` |
| Python command | `python3` or `python` | `python` or `py` |
| Path separator | `/` | `\` (but `/` also works) |
| Everything else | Same | Same |

## Need Help?

- **Windows users:** See `INSTALL_WINDOWS.md`
- **All users:** See `README.md` for full documentation
- **Troubleshooting:** See Troubleshooting section in `README.md`

