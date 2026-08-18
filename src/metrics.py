"""
Similarity Measurement Using Distance Metrics
Module: src/metrics.py

This module provides complete, from-scratch implementations of fundamental
distance and similarity metrics used in Pattern Recognition and Machine Learning:
1. Euclidean Distance (L2 Norm)
2. Manhattan Distance (L1 Norm / Taxicab Distance)
3. Cosine Similarity & Cosine Distance
4. Minkowski Distance (Generalized Lp Norm)
5. Chebyshev Distance (L_infinity Norm)
6. Jaccard Distance (Binary / Set-based)

Both single-vector and vectorized matrix-matrix pairwise distance functions
are implemented for computational efficiency and educational clarity.
"""

import numpy as np


# =====================================================================
# 1. Single Vector Pair Distance Metrics
# =====================================================================

def euclidean_distance(u: np.ndarray, v: np.ndarray) -> float:
    r"""
    Compute Euclidean distance (L2 norm) between two vectors u and v.
    Formula:
        d(u, v) = \sqrt{ \sum_{i=1}^{n} (u_i - v_i)^2 } = ||u - v||_2

    Properties:
    - Metric axioms satisfied: non-negativity, identity of indiscernibles,
      symmetry, and triangle inequality.
    - Represents the ordinary straight-line distance in Euclidean space.
    - Sensitive to magnitudes and scale differences across dimensions.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if u.shape != v.shape:
        raise ValueError(f"Vector dimensions must match. Got {u.shape} and {v.shape}")
    return float(np.sqrt(np.sum((u - v) ** 2)))


def manhattan_distance(u: np.ndarray, v: np.ndarray) -> float:
    r"""
    Compute Manhattan distance (L1 norm / City Block / Taxicab distance).
    Formula:
        d(u, v) = \sum_{i=1}^{n} |u_i - v_i| = ||u - v||_1

    Properties:
    - Measures grid-based distance along coordinate axes.
    - More robust to outliers and extreme values compared to Euclidean distance.
    - Often performs better in higher-dimensional sparse data than Euclidean metric.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if u.shape != v.shape:
        raise ValueError(f"Vector dimensions must match. Got {u.shape} and {v.shape}")
    return float(np.sum(np.abs(u - v)))


def cosine_similarity(u: np.ndarray, v: np.ndarray, eps: float = 1e-12) -> float:
    r"""
    Compute Cosine Similarity between two vectors u and v.
    Formula:
        cos(\theta) = \frac{u \cdot v}{||u||_2 ||v||_2} = \frac{\sum u_i v_i}{\sqrt{\sum u_i^2}\sqrt{\sum v_i^2}}

    Properties:
    - Measures the cosine of the angle between two non-zero vectors.
    - Range: [-1.0, 1.0] (or [0.0, 1.0] for non-negative feature spaces like TF-IDF / BoW).
    - Invariant to vector scaling (magnitude-independent).
    - Perfect for text documents, high-dimensional word vectors, and directional pattern recognition.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if u.shape != v.shape:
        raise ValueError(f"Vector dimensions must match. Got {u.shape} and {v.shape}")

    dot_product = np.dot(u, v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)

    if norm_u < eps or norm_v < eps:
        # Zero-vector edge case
        return 0.0

    cos_sim = dot_product / (norm_u * norm_v)
    # Clip numerical floating-point inaccuracies outside [-1.0, 1.0]
    return float(np.clip(cos_sim, -1.0, 1.0))


def cosine_distance(u: np.ndarray, v: np.ndarray, eps: float = 1e-12) -> float:
    r"""
    Compute Cosine Distance between two vectors u and v.
    Formula:
        d_{cos}(u, v) = 1.0 - \text{cosine\_similarity}(u, v)

    Properties:
    - Measures dissimilarity based on orientation rather than magnitude.
    - Range: [0.0, 2.0] in general spaces, [0.0, 1.0] for non-negative spaces.
    """
    return float(1.0 - cosine_similarity(u, v, eps=eps))


def minkowski_distance(u: np.ndarray, v: np.ndarray, p: float = 3.0) -> float:
    r"""
    Compute Minkowski distance (generalized Lp norm).
    Formula:
        d_p(u, v) = \left( \sum_{i=1}^{n} |u_i - v_i|^p \right)^{1/p}

    Special cases:
    - p = 1: Manhattan Distance
    - p = 2: Euclidean Distance
    - p -> \infty: Chebyshev Distance
    """
    if p < 1:
        raise ValueError("Minkowski parameter p must be >= 1 for metric triangle inequality.")
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if u.shape != v.shape:
        raise ValueError(f"Vector dimensions must match. Got {u.shape} and {v.shape}")
    return float(np.sum(np.abs(u - v) ** p) ** (1.0 / p))


def chebyshev_distance(u: np.ndarray, v: np.ndarray) -> float:
    r"""
    Compute Chebyshev distance (L_infinity norm / Chessboard distance).
    Formula:
        d_\infty(u, v) = \max_{i=1..n} |u_i - v_i|
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if u.shape != v.shape:
        raise ValueError(f"Vector dimensions must match. Got {u.shape} and {v.shape}")
    return float(np.max(np.abs(u - v)))


# =====================================================================
# 2. Vectorized Pairwise Distance Matrices (X: N x D, Y: M x D -> N x M)
# =====================================================================

def pairwise_euclidean_distances(X: np.ndarray, Y: np.ndarray = None) -> np.ndarray:
    """
    Compute vectorized pairwise Euclidean distance matrix between rows of X and Y.
    Returns matrix D of shape (N, M), where D[i, j] = ||X[i] - Y[j]||_2.
    """
    X = np.asarray(X, dtype=np.float64)
    if Y is None:
        Y = X
    else:
        Y = np.asarray(Y, dtype=np.float64)

    # Use binomial expansion: ||x - y||^2 = ||x||^2 + ||y||^2 - 2*(x . y)
    X_sq = np.sum(X ** 2, axis=1, keepdims=True)  # Shape (N, 1)
    Y_sq = np.sum(Y ** 2, axis=1, keepdims=True)  # Shape (M, 1)
    dot = np.dot(X, Y.T)                          # Shape (N, M)

    dist_sq = X_sq + Y_sq.T - 2.0 * dot
    # Prevent negative values from precision rounding
    dist_sq = np.maximum(dist_sq, 0.0)
    return np.sqrt(dist_sq)


def pairwise_manhattan_distances(X: np.ndarray, Y: np.ndarray = None) -> np.ndarray:
    """
    Compute vectorized pairwise Manhattan distance matrix between rows of X and Y.
    Returns matrix D of shape (N, M), where D[i, j] = ||X[i] - Y[j]||_1.
    """
    X = np.asarray(X, dtype=np.float64)
    if Y is None:
        Y = X
    else:
        Y = np.asarray(Y, dtype=np.float64)

    # Broadcasting over feature dimension: (N, 1, D) - (1, M, D) -> sum over D
    # For large memory safety, chunking can be used if needed, or simple broadcast
    return np.sum(np.abs(X[:, np.newaxis, :] - Y[np.newaxis, :, :]), axis=2)


def pairwise_cosine_distances(X: np.ndarray, Y: np.ndarray = None, eps: float = 1e-12) -> np.ndarray:
    """
    Compute vectorized pairwise Cosine distance matrix between rows of X and Y.
    Returns matrix D of shape (N, M), where D[i, j] = 1 - (X[i] . Y[j]) / (||X[i]|| * ||Y[j]||).
    """
    X = np.asarray(X, dtype=np.float64)
    if Y is None:
        Y = X
    else:
        Y = np.asarray(Y, dtype=np.float64)

    norm_X = np.linalg.norm(X, axis=1, keepdims=True)
    norm_Y = np.linalg.norm(Y, axis=1, keepdims=True)

    norm_X = np.maximum(norm_X, eps)
    norm_Y = np.maximum(norm_Y, eps)

    X_normalized = X / norm_X
    Y_normalized = Y / norm_Y

    cosine_sim_matrix = np.dot(X_normalized, Y_normalized.T)
    cosine_sim_matrix = np.clip(cosine_sim_matrix, -1.0, 1.0)
    return 1.0 - cosine_sim_matrix


# =====================================================================
# 3. Metric Registry & Factory
# =====================================================================

METRIC_MAP = {
    'euclidean': {
        'single': euclidean_distance,
        'pairwise': pairwise_euclidean_distances,
        'name': 'Euclidean Distance (L2)',
        'symbol': r'$L_2$',
        'formula': r'\sqrt{\sum (u_i - v_i)^2}'
    },
    'manhattan': {
        'single': manhattan_distance,
        'pairwise': pairwise_manhattan_distances,
        'name': 'Manhattan Distance (L1)',
        'symbol': r'$L_1$',
        'formula': r'\sum |u_i - v_i|'
    },
    'cosine': {
        'single': cosine_distance,
        'pairwise': pairwise_cosine_distances,
        'name': 'Cosine Distance',
        'symbol': r'$D_{cos}$',
        'formula': r'1 - \frac{u \cdot v}{\|u\|_2 \|v\|_2}'
    }
}


def compute_distance(u: np.ndarray, v: np.ndarray, metric: str = 'euclidean') -> float:
    """Compute distance between two 1D vectors given a metric name."""
    metric = metric.lower()
    if metric not in METRIC_MAP:
        raise ValueError(f"Unknown metric '{metric}'. Choose from {list(METRIC_MAP.keys())}")
    return METRIC_MAP[metric]['single'](u, v)


def compute_pairwise_distances(X: np.ndarray, Y: np.ndarray = None, metric: str = 'euclidean') -> np.ndarray:
    """Compute pairwise distance matrix between X and Y given a metric name."""
    metric = metric.lower()
    if metric not in METRIC_MAP:
        raise ValueError(f"Unknown metric '{metric}'. Choose from {list(METRIC_MAP.keys())}")
    return METRIC_MAP[metric]['pairwise'](X, Y)
