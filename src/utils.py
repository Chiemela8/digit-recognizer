
"""
utils.py — helper functions for Kaggle Digit Recognizer
Author: You (Dempsey) & teammate(s)
"""
from __future__ import annotations

import os, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy, and (if available) TensorFlow for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass

def load_train_test(data_dir: str = "data"):
    """
    Load train.csv and test.csv and return normalized arrays.
    Returns:
        X (np.ndarray): (n_train, 784) float32 in [0,1]
        y (np.ndarray): (n_train,) int labels 0..9
        X_test (np.ndarray): (n_test, 784) float32 in [0,1]
    """
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    X = train.drop(columns=["label"]).values.astype("float32") / 255.0
    y = train["label"].values.astype("int64")
    test = pd.read_csv(os.path.join(data_dir, "test.csv")).values.astype("float32") / 255.0
    return X, y, test

def as_images(X: np.ndarray) -> np.ndarray:
    """Reshape flat pixel vectors to (N, 28, 28, 1)."""
    return X.reshape(-1, 28, 28, 1)

def plot_samples(X: np.ndarray, y=None, n: int = 25, title: str = "Samples") -> None:
    """Plot n sample images in a grid (uses first n rows)."""
    imgs = as_images(X[:n])
    cols = int(np.sqrt(n))
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(cols*2, rows*2))
    for i in range(n):
        plt.subplot(rows, cols, i+1)
        plt.imshow(imgs[i].squeeze(), cmap="gray")
        if y is not None:
            plt.title(int(y[i]))
        plt.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def save_submission(pred: np.ndarray, path: str) -> None:
    """Save predictions to Kaggle submission CSV format."""
    df = pd.DataFrame({"ImageId": np.arange(1, len(pred)+1), "Label": pred})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

def confusion_matrix_plot(y_true, y_pred, normalize: bool = True, title: str = "Confusion Matrix"):
    """Plot a confusion matrix (requires scikit-learn)."""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=False, cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

def show_misclassified(X, y_true, y_pred, n: int = 16, title: str = "Misclassified examples"):
    """Display n misclassified digits."""
    wrong_idx = np.where(y_true != y_pred)[0]
    if len(wrong_idx) == 0:
        print("No misclassifications to show ✨")
        return
    sel = wrong_idx[:n]
    imgs = as_images(X[sel])
    rows = 4
    cols = int(np.ceil(n/rows))
    import matplotlib.pyplot as plt
    plt.figure(figsize=(cols*2.2, rows*2.2))
    for i, idx in enumerate(sel):
        plt.subplot(rows, cols, i+1)
        plt.imshow(imgs[i].squeeze(), cmap="gray")
        plt.title(f"T:{y_true[idx]} / P:{y_pred[idx]}")
        plt.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
