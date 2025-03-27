# B2G-24-015: calculating efficiencies with Delphes

## Setup Instructions

### 1. Create a Virtual Environment

```bash
# Create a new virtual environment
python3.9 -m venv jet_analysis_env

# Activate the virtual environment
source jet_analysis_env/bin/activate
```

### 2. Install Required Packages

```bash
# Upgrade pip to the latest version
pip install --upgrade pip

# Install packages
pip install numpy==1.21.2
pip install tensorflow==2.6.0
pip install h5py==3.1.0
pip install uproot==4.1.5
```

### 3. Run example script
```
python3 delphesProcessor.py
```
