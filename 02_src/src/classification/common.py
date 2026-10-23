"""Shared helpers for the classification strategies."""
import os
import random

import numpy as np


def seed_everything(seed: int) -> int:
    """random, numpy, torch (CPU + CUDA) and cuDNN determinism."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    return seed


def balanced_class_weights(y, n_classes: int):
    """sklearn-style 'balanced' weights n / (k · count_c); classes absent from y get weight 1."""
    counts = np.bincount(np.asarray(y, dtype=int), minlength=n_classes).astype(float)
    w = np.ones(n_classes)
    present = counts > 0
    w[present] = counts.sum() / (present.sum() * counts[present])
    return w
