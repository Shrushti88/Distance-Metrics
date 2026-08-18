"""
Similarity Measurement Using Distance Metrics
Module: src/evaluation.py

Comprehensive evaluation suite for comparing distance metrics (Euclidean,
Manhattan, Cosine) on various benchmark datasets:
- Performance metrics (Accuracy, Precision, Recall, F1-Score)
- Confusion Matrix computation
- K-sweep analysis (evaluating optimal k per metric)
- Execution time / Computational efficiency profiling
- Stratified K-Fold Cross-Validation
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from src.knn_model import CustomKNNClassifier
from src.metrics import METRIC_MAP


def evaluate_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: List[str] = None
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    return {
        'accuracy': float(acc),
        'precision_weighted': float(prec),
        'recall_weighted': float(rec),
        'f1_weighted': float(f1),
        'precision_macro': float(prec_macro),
        'recall_macro': float(rec_macro),
        'f1_macro': float(f1_macro),
        'confusion_matrix': cm,
        'target_names': target_names
    }


def benchmark_single_dataset(
    dataset_dict: Dict,
    k_neighbors: int = 5,
    metrics: List[str] = ['euclidean', 'manhattan', 'cosine'],
    weights: str = 'uniform'
) -> Dict[str, Dict]:
    """
    Evaluate KNN across all distance metrics on a single dataset with time profiling.
    """
    X_train = dataset_dict['X_train']
    X_test = dataset_dict['X_test']
    y_train = dataset_dict['y_train']
    y_test = dataset_dict['y_test']
    target_names = dataset_dict.get('target_names', None)

    results = {}

    for metric in metrics:
        knn = CustomKNNClassifier(n_neighbors=k_neighbors, metric=metric, weights=weights)

        # Fit time
        t0 = time.perf_counter()
        knn.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0

        # Inference time
        t1 = time.perf_counter()
        y_pred = knn.predict(X_test)
        inference_time = time.perf_counter() - t1

        metrics_eval = evaluate_classifier(y_test, y_pred, target_names=target_names)
        metrics_eval['fit_time_sec'] = fit_time
        metrics_eval['inference_time_sec'] = inference_time
        metrics_eval['time_per_query_ms'] = (inference_time / len(X_test)) * 1000.0
        metrics_eval['n_neighbors'] = k_neighbors
        metrics_eval['weights'] = weights
        metrics_eval['y_pred'] = y_pred
        metrics_eval['y_true'] = y_test

        results[metric] = metrics_eval

    return results


def run_cross_validation_benchmark(
    X: np.ndarray,
    y: np.ndarray,
    k_neighbors: int = 5,
    metrics: List[str] = ['euclidean', 'manhattan', 'cosine'],
    n_splits: int = 5,
    random_state: int = 42
) -> Dict[str, Dict]:
    """
    Perform Stratified K-Fold Cross Validation across metrics.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_results = {m: {'accuracies': [], 'f1_scores': [], 'inference_times': []} for m in metrics}

    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        for m in metrics:
            knn = CustomKNNClassifier(n_neighbors=k_neighbors, metric=m)
            knn.fit(X_tr, y_tr)

            t0 = time.perf_counter()
            y_pred = knn.predict(X_te)
            elapsed = time.perf_counter() - t0

            acc = accuracy_score(y_te, y_pred)
            _, _, f1, _ = precision_recall_fscore_support(y_te, y_pred, average='weighted', zero_division=0)

            cv_results[m]['accuracies'].append(acc)
            cv_results[m]['f1_scores'].append(f1)
            cv_results[m]['inference_times'].append(elapsed)

    summary = {}
    for m in metrics:
        summary[m] = {
            'mean_accuracy': float(np.mean(cv_results[m]['accuracies'])),
            'std_accuracy': float(np.std(cv_results[m]['accuracies'])),
            'mean_f1': float(np.mean(cv_results[m]['f1_scores'])),
            'std_f1': float(np.std(cv_results[m]['f1_scores'])),
            'mean_time_sec': float(np.mean(cv_results[m]['inference_times'])),
            'raw_cv': cv_results[m]
        }
    return summary


def evaluate_k_sweep(
    dataset_dict: Dict,
    k_range: range = range(1, 26, 2),
    metrics: List[str] = ['euclidean', 'manhattan', 'cosine']
) -> Dict[str, Dict[int, float]]:
    """
    Evaluate accuracy curves over varying k values for each distance metric.
    """
    X_train = dataset_dict['X_train']
    X_test = dataset_dict['X_test']
    y_train = dataset_dict['y_train']
    y_test = dataset_dict['y_test']

    k_results = {m: {} for m in metrics}

    for k in k_range:
        for m in metrics:
            knn = CustomKNNClassifier(n_neighbors=k, metric=m)
            knn.fit(X_train, y_train)
            acc = knn.score(X_test, y_test)
            k_results[m][k] = float(acc)

    return k_results


def generate_benchmark_summary_table(all_results: Dict[str, Dict[str, Dict]]) -> pd.DataFrame:
    """
    Compile multi-dataset, multi-metric benchmark results into a clean tabular DataFrame.
    """
    rows = []
    for dset_name, metric_dict in all_results.items():
        for metric_name, res in metric_dict.items():
            rows.append({
                'Dataset': dset_name,
                'Distance Metric': METRIC_MAP[metric_name]['name'],
                'Accuracy (%)': round(res['accuracy'] * 100, 2),
                'Weighted F1': round(res['f1_weighted'], 4),
                'Precision': round(res['precision_weighted'], 4),
                'Recall': round(res['recall_weighted'], 4),
                'Latency / Query (ms)': round(res['time_per_query_ms'], 4)
            })
    return pd.DataFrame(rows)
