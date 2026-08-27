import numpy as np


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    # 1 - cosine_similarity  →  0 means identical, 2 means opposite
    return 1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def dot_product_distance(a: np.ndarray, b: np.ndarray) -> float:
    # Negated so that higher similarity = lower "distance" (same convention as above)
    return -float(np.dot(a, b))
