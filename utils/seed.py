"""
seed.py — Global Random Seed Setter
=====================================
Every neural network training run involves randomness:
  - How weights are initialized at the start
  - Which samples appear in each mini-batch
  - Which neurons are dropped by Dropout
  - PyTorch's internal random operations

By fixing a SEED, we make all of this randomness DETERMINISTIC — meaning
running the same code twice will produce EXACTLY the same results.
This is crucial for academic reproducibility.

USAGE:
    from utils.seed import set_seed
    set_seed()          # Uses the default seed from config.py (42)
    set_seed(123)       # Uses a custom seed
"""

import random          # Python's built-in random number generator
import numpy as np     # NumPy's random number generator (used in data processing)
import torch           # PyTorch's random number generator (used in model training)
import os              # For setting environment variables

from utils.config import RANDOM_SEED  # Import the project-wide seed value (42)


def set_seed(seed: int = RANDOM_SEED):
    """
    Set the random seed for ALL sources of randomness in the project.
    
    This function must be called at the start of EVERY script that involves
    training or data splitting to ensure reproducible results.
    
    Args:
        seed (int): The seed value to use. Defaults to RANDOM_SEED from config.py
                    (which is 42). You can override this for experiments.
    
    Returns:
        None — modifies the global state of all random number generators.
    
    Example:
        >>> set_seed(42)
        >>> # Now every random operation is deterministic
    """
    
    # 1. Set Python's built-in random module seed
    # This affects things like Python's random.shuffle() and random.choice()
    random.seed(seed)
    
    # 2. Set NumPy's random seed
    # This affects all numpy random operations: np.random.rand(), np.random.choice(), etc.
    # NumPy is used heavily in data preprocessing and evaluation
    np.random.seed(seed)
    
    # 3. Set PyTorch's CPU random seed
    # This affects weight initialization, Dropout, and data loader shuffling on CPU
    torch.manual_seed(seed)
    
    # 4. Set PyTorch's GPU random seed (CUDA)
    # If training on a GPU, this ensures GPU operations are also deterministic
    if torch.cuda.is_available():  # Only set if a GPU is present (won't crash on CPU-only machines)
        torch.cuda.manual_seed(seed)           # Seed for the current GPU
        torch.cuda.manual_seed_all(seed)       # Seed for ALL GPUs (if using multi-GPU training)
    
    # 5. Make cuDNN (GPU math library) deterministic
    # cuDNN sometimes uses non-deterministic algorithms for speed; we disable that
    # Note: This may slow down GPU training slightly — worth it for reproducibility
    torch.backends.cudnn.deterministic = True   # Force deterministic algorithms
    torch.backends.cudnn.benchmark     = False  # Disable auto-tuning (non-deterministic)
    
    # 6. Set a Python hash seed via environment variable
    # Python's built-in hash() function is random by default since Python 3.3
    # Setting PYTHONHASHSEED makes it deterministic (important for set/dict ordering)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    print(f"[Seed] All random seeds set to {seed} — results are now reproducible.")
