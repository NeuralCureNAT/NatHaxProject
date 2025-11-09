# Windows Installation Guide

This guide provides step-by-step instructions for setting up and running the EEG Migraine Classification project on Windows.

## Prerequisites

1. **Python 3.7 or higher**
   - Download from [python.org](https://www.python.org/downloads/)
   - **Important:** Check "Add Python to PATH" during installation
   - Verify installation: Open Command Prompt and run `python --version`

2. **Git (optional, for cloning)**
   - Download from [git-scm.com](https://git-scm.com/download/win)

## Step-by-Step Installation

### Step 1: Get the Project

**Option A: Clone with Git**
```cmd
git clone <repository-url>
cd nathax
```

**Option B: Download ZIP**
1. Download the project as a ZIP file
2. Extract it to a folder (e.g., `C:\Users\YourName\Documents\nathax`)
3. Open Command Prompt and navigate to the folder:
```cmd
cd C:\Users\YourName\Documents\nathax
```

### Step 2: Create Virtual Environment

1. **Navigate to parent directory:**
```cmd
cd ..
```

2. **Create virtual environment:**
```cmd
python -m venv venv
```

   **Alternative (if python doesn't work):**
```cmd
py -3 -m venv venv
```

3. **Activate virtual environment:**

   **In Command Prompt:**
```cmd
venv\Scripts\activate
```

   **In PowerShell:**
```powershell
.\venv\Scripts\Activate.ps1
```

   **If you get an execution policy error in PowerShell:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
   Then try activating again.

   **You should see `(venv)` at the beginning of your command prompt when activated.**

### Step 3: Install Dependencies

1. **Navigate to model directory:**
```cmd
cd model
```

2. **Upgrade pip (recommended):**
```cmd
python -m pip install --upgrade pip
```

3. **Install dependencies:**
```cmd
pip install -r requirements.txt
```

   This may take several minutes. You should see packages being installed.

### Step 4: Verify Installation

Run this command to verify everything is installed:
```cmd
python -c "import tensorflow; import sklearn; import pandas; print('All packages installed successfully!')"
```

## Running the Project

### Basic Usage

1. **Make sure virtual environment is activated** (you should see `(venv)` in your prompt)

2. **Navigate to model directory:**
```cmd
cd model
```

3. **Run the pipeline:**
```cmd
python run_pipeline.py
```

### Other Commands

**Run with better quality settings:**
```cmd
python improve_augmentation.py
```

**Make predictions:**
```cmd
python predict.py path\to\your\data.xlsx
```

## Common Windows Issues and Solutions

### Issue 1: "python is not recognized"

**Solution:**
- Use `py` instead: `py run_pipeline.py`
- Or add Python to PATH:
  1. Search "Environment Variables" in Windows
  2. Edit "Path" variable
  3. Add Python installation directory (e.g., `C:\Python39`)
  4. Add Scripts directory (e.g., `C:\Python39\Scripts`)
  5. Restart Command Prompt

### Issue 2: PowerShell Execution Policy Error

**Error:** `cannot be loaded because running scripts is disabled on this system`

**Solution:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Issue 3: Virtual Environment Activation Fails

**Solutions:**
- Make sure you're in the correct directory
- Try: `.\venv\Scripts\activate` (with backslash and dot)
- In Command Prompt, use: `venv\Scripts\activate.bat`
- Check that venv folder exists and contains Scripts folder

### Issue 4: TensorFlow Installation Fails

**Solutions:**
1. **Install Visual C++ Redistributable:**
   - Download from [Microsoft](https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads)
   - Install both x64 and x86 versions

2. **Update pip first:**
```cmd
python -m pip install --upgrade pip
```

3. **Install TensorFlow:**
```cmd
pip install tensorflow
```

### Issue 5: ModuleNotFoundError

**Solutions:**
- Make sure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt --upgrade`
- Check that you're in the `model/` directory

### Issue 6: Out of Memory Errors

**Solutions:**
- Close other applications
- Reduce `target_samples` in `run_pipeline.py` (e.g., change to 50)
- Reduce `augmentation_epochs` (e.g., change to 30)

### Issue 7: Excel File Reading Errors

**Solutions:**
- Install openpyxl: `pip install openpyxl`
- Check that Excel files are not corrupted
- Verify file paths are correct (use forward slashes or raw strings)

## File Paths on Windows

The code uses relative paths that work on Windows:
- Forward slashes `/` work on Windows (Python handles this)
- All files should be in the `model/` directory
- Scripts expect to be run from the `model/` directory

**Example:**
```python
# This works on Windows:
df = pd.read_excel('raw_ictal.xlsx')

# This also works:
df = pd.read_excel('model/raw_ictal.xlsx')
```

## Quick Reference

### Activating Virtual Environment
```cmd
# Command Prompt
venv\Scripts\activate

# PowerShell
.\venv\Scripts\Activate.ps1
```

### Running Scripts
```cmd
# From model/ directory
python run_pipeline.py
python predict.py data.xlsx
```

### Deactivating Virtual Environment
```cmd
deactivate
```

### Checking Python Version
```cmd
python --version
```

### Checking Installed Packages
```cmd
pip list
```

## Getting Help

If you encounter issues not covered here:

1. Check the main README.md for general troubleshooting
2. Verify Python version: `python --version` (should be 3.7+)
3. Make sure virtual environment is activated
4. Check that you're in the `model/` directory
5. Review error messages carefully

## Notes

- The project works the same on Windows, Linux, and macOS
- File paths are handled automatically by Python
- All commands are the same across platforms (except virtual environment activation)
- Virtual environment keeps dependencies isolated from system Python

