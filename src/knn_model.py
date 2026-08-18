"""
Similarity Measurement Using Distance Metrics
Module: src/knn_model.py

From-scratch implementation of K-Nearest Neighbors (KNN) Classifier and
Similarity Retrieval Engine supporting arbitrary distance metrics (Euclidean,
Manhattan, Cosine).
"""

import numpy as np
from typing import Optional, Union, Tuple, List, Dict
from src.metrics import compute_pairwise_distances, METRIC_MAP


class CustomKNNClassifier:
    """
    K-Nearest Neighbors Classifier built from scratch with configurable distance metrics.

    Parameters:
    -----------
    n_neighbors : int, default=5
        Number of neighbors to use for classification queries.
    metric : str, default='euclidean'
        Distance metric to use. Options: 'euclidean', 'manhattan', 'cosine'.
    weights : str, default='uniform'
        Weight function used in prediction.
        - 'uniform': all points in each neighborhood are weighted equally.
        - 'distance': weight points by inverse of distance (1 / (d + eps)).
    """
    def __init__(self, n_neighbors: int = 5, metric: str = 'euclidean', weights: str = 'uniform'):
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be a positive integer >= 1")
        if metric.lower() not in METRIC_MAP:
            raise ValueError(f"metric must be one of {list(METRIC_MAP.keys())}")
        if weights not in ('uniform', 'distance'):
            raise ValueError("weights must be 'uniform' or 'distance'")

        self.n_neighbors = n_neighbors
        self.metric = metric.lower()
        self.weights = weights

        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None
        self.n_classes_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'CustomKNNClassifier':
        """
        Fit the model using X as training data and y as target labels.
        """
        self.X_train = np.asarray(X, dtype=np.float64)
        self.y_train = np.asarray(y)
        self.classes_ = np.unique(self.y_train)
        self.n_classes_ = len(self.classes_)

        if len(self.X_train) < self.n_neighbors:
            raise ValueError(
                f"n_neighbors={self.n_neighbors} cannot be greater than number of samples={len(self.X_train)}"
            )
        return self

    def kneighbors(self, X: np.ndarray, n_neighbors: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Finds the K-neighbors of a point.
        Returns:
            distances: shape (n_queries, k)
            indices: shape (n_queries, k)
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model must be fitted before calling kneighbors()")

        k = n_neighbors if n_neighbors is not None else self.n_neighbors
        X = np.asarray(X, dtype=np.float64)

        # Compute full pairwise distance matrix between queries X and train X_train
        # dist_matrix shape: (n_queries, n_train)
        dist_matrix = compute_pairwise_distances(X, self.X_train, metric=self.metric)

        # Find indices of top-k smallest distances
        # np.argpartition is O(N) instead of full sort O(N log N)
        k_indices = np.argpartition(dist_matrix, kth=k - 1, axis=1)[:, :k]

        # Sort the top-k to get strictly ascending order of distances
        row_indices = np.arange(len(X))[:, np.newaxis]
        sorted_order = np.argsort(dist_matrix[row_indices, k_indices], axis=1)

        final_indices = k_indices[row_indices, sorted_order]
        final_distances = dist_matrix[row_indices, final_indices]

        return final_distances, final_indices

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return probability estimates for the test data X.
        Shape: (n_queries, n_classes)
        """
        distances, indices = self.kneighbors(X)
        n_queries = len(X)
        probas = np.zeros((n_queries, self.n_classes_))

        class_to_idx = {c: i for i, c in enumerate(self.classes_)}

        for i in range(n_queries):
            neighbor_labels = self.y_train[indices[i]]
            neighbor_dists = distances[i]

            if self.weights == 'uniform':
                for label in neighbor_labels:
                    probas[i, class_to_idx[label]] += 1.0
                probas[i] /= self.n_neighbors
            else:
                # Inverse distance weighting
                weights = 1.0 / (neighbor_dists + 1e-8)
                for w, label in zip(weights, neighbor_labels):
                    probas[i, class_to_idx[label]] += w
                total_w = np.sum(weights)
                if total_w > 0:
                    probas[i] /= total_w
                else:
                    probas[i] = 1.0 / self.n_classes_

        return probas

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the class labels for the provided data.
        """
        probas = self.predict_proba(X)
        max_indices = np.argmax(probas, axis=1)
        return self.classes_[max_indices]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the mean classification accuracy on the given test data and labels.
        """
        y_pred = self.predict(X)
        return float(np.mean(y_pred == y))


class SimilaritySearchEngine:
    """
    Vector similarity search and nearest neighbor retrieval engine.
    Supports Euclidean, Manhattan, and Cosine retrieval.
    """
    def __init__(self, metric: str = 'cosine'):
        self.metric = metric.lower()
        self.database_vectors: Optional[np.ndarray] = None
        self.metadata: List[Dict] = []

    def index(self, vectors: np.ndarray, metadata: Optional[List[Dict]] = None):
        """Index a database of feature vectors."""
        self.database_vectors = np.asarray(vectors, dtype=np.float64)
        if metadata is not None:
            self.metadata = metadata
        else:
            self.metadata = [{"id": i} for i in range(len(vectors))]

    def query(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Retrieve top_k most similar items to query_vector based on selected metric.
        """
        if self.database_vectors is None:
            raise ValueError("No vectors indexed. Call index() first.")

        query_vec = np.asarray(query_vector, dtype=np.float64).reshape(1, -1)
        distances = compute_pairwise_distances(query_vec, self.database_vectors, metric=self.metric)[0]

        top_indices = np.argsort(distances)[:top_k]
        results = []
        for rank, idx in enumerate(top_indices, 1):
            res = {
                "rank": rank,
                "index": int(idx),
                "distance": float(distances[idx]),
                "similarity": float(1.0 / (1.0 + distances[idx])) if self.metric != 'cosine' else float(1.0 - distances[idx]),
                "metadata": self.metadata[idx] if idx < len(self.metadata) else {}
            }
            results.append(res)
        return results
