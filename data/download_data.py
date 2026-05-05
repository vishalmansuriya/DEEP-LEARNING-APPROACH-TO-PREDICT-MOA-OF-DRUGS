"""
download_data.py — Kaggle Dataset Downloader
=============================================
Downloads the Kaggle "Mechanisms of Action Prediction" competition dataset.

PREREQUISITES:
  1. Create a Kaggle account at https://www.kaggle.com
  2. Go to: Account → API → Create New Token
  3. This downloads a kaggle.json file with your credentials
  4. On Windows, place kaggle.json at: C:\\Users\\<username>\\.kaggle\\kaggle.json
  5. Install kaggle CLI: pip install kaggle

USAGE:
    python data/download_data.py

ALTERNATIVE (Manual Download):
  If you cannot use the Kaggle API, manually download from:
  https://www.kaggle.com/c/lish-moa/data
  And place these files in: data/raw/
    - train_features.csv
    - train_targets_scored.csv
    - train_targets_nonscored.csv
    - test_features.csv
    - sample_submission.csv
"""

import os, sys, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import RAW_DATA_DIR  # The folder where raw data should be stored


def download_kaggle_dataset():
    """
    Download the MoA competition dataset using the Kaggle CLI.
    
    The Kaggle CLI is a command-line tool that can download competition data
    automatically if you have valid API credentials (kaggle.json).
    
    Args:
        None
    
    Returns:
        bool: True if download succeeded, False if it failed
    """
    
    # Create the raw data directory if it doesn't exist
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    # Check if files already exist (skip download if they do)
    required_files = [
        "train_features.csv",
        "train_targets_scored.csv",
        "test_features.csv",
    ]
    
    existing = [f for f in required_files if os.path.exists(os.path.join(RAW_DATA_DIR, f))]
    
    if len(existing) == len(required_files):
        print("[Download] All required files already exist. Skipping download.")
        return True
    
    print("[Download] Attempting to download Kaggle MoA dataset...")
    print(f"[Download] Target directory: {RAW_DATA_DIR}")
    
    # Kaggle CLI command to download the competition data
    # -c = competition name
    # -p = path to save files
    # --unzip = automatically unzip the downloaded archive
    cmd = [
        "kaggle", "competitions", "download",
        "-c", "lish-moa",       # The Kaggle competition identifier
        "-p", RAW_DATA_DIR,     # Where to save the files
        "--unzip",              # Automatically unzip (saves a manual step)
    ]
    
    try:
        # subprocess.run() executes the command and waits for it to finish
        result = subprocess.run(
            cmd,
            check=True,         # Raise an exception if the command fails
            capture_output=True,  # Capture stdout and stderr
            text=True,          # Return output as strings, not bytes
        )
        print(result.stdout)    # Print the Kaggle CLI output
        print("[Download] Dataset downloaded successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"[Error] Kaggle download failed: {e.stderr}")
        print("\n[Solution] Manual download instructions:")
        print("1. Go to: https://www.kaggle.com/c/lish-moa/data")
        print("2. Download all files")
        print(f"3. Extract them to: {RAW_DATA_DIR}")
        print("\nRequired files:")
        for f in required_files:
            print(f"  - {f}")
        return False
    
    except FileNotFoundError:
        print("[Error] 'kaggle' command not found. Install it with: pip install kaggle")
        print("Then set up your API credentials (kaggle.json)")
        return False


if __name__ == "__main__":
    success = download_kaggle_dataset()
    if success:
        print("\n[Next] Run: python data/preprocess.py")
    else:
        print("\n[Next] Please manually download the data and place it in data/raw/")
