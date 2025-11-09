# Neuralcare Backend

## Setup (macOS Apple Silicon)

1. Install LSL runtime:

/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
brew install labstreaminglayer/tap/lsl

2. Python env:

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

3. Export for current shell (or put these in your venv/bin/activate):

export DYLD_LIBRARY_PATH="/opt/homebrew/opt/lsl/lib:$DYLD_LIBRARY_PATH"
export PYLSL_LIB="/opt/homebrew/opt/lsl/lib/liblsl.dylib"

## Run

source venv/bin/activate
python start_muse_stream.py

