"""
dataset.py — PyTorch Dataset for MoA Data
==========================================
Defines a custom PyTorch Dataset class that wraps our preprocessed
numpy arrays and serves individual samples to the DataLoader.

USAGE:
    from data.dataset import MoADataset
    train_dataset = MoADataset(X_train, y_train)
    train_loader  = DataLoader(train_dataset, batch_size=256, shuffle=True)
"""

import torch                    # Core PyTorch library
from torch.utils.data import Dataset  # Base class for all PyTorch datasets
import numpy as np             # For type checking numpy arrays


class MoADataset(Dataset):
    """
    A PyTorch Dataset that wraps feature arrays and label arrays for the MoA task.
    
    PyTorch's DataLoader requires data to be wrapped in a Dataset object.
    This class provides __len__ (how many samples?) and __getitem__ (give me sample N)
    methods that the DataLoader calls automatically during training.
    
    Args:
        X (np.ndarray or torch.Tensor): Feature matrix, shape (N_samples, N_features)
        y (np.ndarray or torch.Tensor or None): Label matrix, shape (N_samples, 206)
                                                 None for test data (no labels available)
    """
    
    def __init__(self, X, y=None):
        """
        Initialize the dataset by converting arrays to PyTorch tensors.
        
        Args:
            X: Feature matrix — gene expression + cell viability + encoded categoricals
            y: Label matrix — binary MoA labels (0 or 1 for each of 206 classes)
               None when building a dataset for test/inference (no ground truth labels)
        """
        # Convert numpy arrays to PyTorch float tensors
        # torch.FloatTensor = 32-bit floating point, which matches our model's weights
        if isinstance(X, np.ndarray):
            self.X = torch.FloatTensor(X)  # Convert numpy → tensor; shape: (N, 184)
        else:
            self.X = X.float()             # Already a tensor, just ensure float32
        
        # Handle labels (y can be None for test sets where no labels exist)
        if y is not None:
            if isinstance(y, np.ndarray):
                self.y = torch.FloatTensor(y)  # Shape: (N, 206) — 206 binary MoA labels
            else:
                self.y = y.float()
        else:
            self.y = None  # Test set has no labels
    
    def __len__(self):
        """
        Return the total number of samples in the dataset.
        
        The DataLoader calls this to know how many samples exist,
        so it can figure out how many batches to create.
        
        Returns:
            int: Number of samples (= number of rows in X)
        """
        return len(self.X)  # Number of drug compounds in this dataset
    
    def __getitem__(self, idx):
        """
        Return a single sample (feature vector + label vector) by index.
        
        The DataLoader calls this repeatedly with different indices to
        assemble batches. For example, for batch_size=256, it calls
        __getitem__ 256 times and stacks the results.
        
        Args:
            idx (int): The index of the sample to return (0 to N-1)
        
        Returns:
            tuple: (features, labels) where:
                   features = 1D tensor of shape (184,)
                   labels   = 1D tensor of shape (206,) or None if test set
        """
        features = self.X[idx]  # Get the feature vector for compound at index idx
        
        if self.y is not None:
            labels = self.y[idx]    # Get the 206 MoA labels for this compound
            return features, labels  # Return both for training/validation
        else:
            return features          # Return only features for inference/test
