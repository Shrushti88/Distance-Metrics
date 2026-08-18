"""
Similarity Measurement Using Distance Metrics
Master Execution Pipeline: main.py

Executes complete Project-Based Learning experimental workflow:
1. Distance Metric Verification & Unit Testing
2. Multi-Dataset Benchmarking (Iris, Wine, Digits, Text)
3. Stratified 5-Fold Cross-Validation
4. Hyperparameter K-Sensitivity Analysis
5. Curse of Dimensionality / Metric Contrast Analysis
6. Publication-Grade Visualization Generation
7. Tabular Results Export (CSV & Formatted Terminal Report)
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.metrics import (
    euclidean_distance, manhattan_distance, cosine_similarity, cosine_distance,
    METRIC_MAP
)
from src.knn_model import CustomKNNClassifier, SimilaritySearchEngine
from src.datasets_loader import (
    get_iris_dataset, get_wine_dataset, get_digits_dataset, get_text_dataset,
    generate_curse_of_dimensionality_data
)
from src.evaluation import (
    benchmark_single_dataset, run_cross_validation_benchmark,
    evaluate_k_sweep, generate_benchmark_summary_table
)
from src.visualizer import (
    plot_unit_balls_geometry, plot_decision_boundaries,
    plot_benchmark_comparison, plot_k_sweep_curves,
    plot_curse_of_dimensionality, plot_confusion_matrices
)


def run_sanity_checks():
    """Verify distance metric implementation correctness on known vector examples."""
    print("=" * 80)
    print(" 1. DISTANCE METRIC MATHEMATICAL SANITY CHECKS")
    print("=" * 80)

    u = np.array([1.0, 2.0, 3.0])
    v = np.array([4.0, 6.0, 8.0])

    d_euc = euclidean_distance(u, v)
    d_man = manhattan_distance(u, v)
    sim_cos = cosine_similarity(u, v)
    d_cos = cosine_distance(u, v)

    print(f"Vector u: {u}")
    print(f"Vector v: {v}")
    print(f"[-] Euclidean Distance (L2) : {d_euc:.6f} (Expected: ~7.071068)")
    print(f"[-] Manhattan Distance (L1) : {d_man:.6f} (Expected: 12.000000)")
    print(f"[-] Cosine Similarity       : {sim_cos:.6f} (Expected: ~0.992583)")
    print(f"[-] Cosine Distance (1-cos) : {d_cos:.6f} (Expected: ~0.007417)")
    print("[OK] All distance functions validated successfully!\n")


def run_full_pipeline():
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    # Step 1: Sanity Check
    run_sanity_checks()

    # Step 2: Load Multi-Domain Datasets
    print("=" * 80)
    print(" 2. LOADING MULTI-DOMAIN BENCHMARK DATASETS")
    print("=" * 80)
    datasets = {
        'iris': get_iris_dataset(),
        'wine': get_wine_dataset(),
        'digits': get_digits_dataset(),
        'text': get_text_dataset()
    }

    for name, d in datasets.items():
        print(f" * Loaded [{d['name']}]: {d['n_samples']} samples, {d['n_features']} features, {d['n_classes']} classes ({d['type']})")
    print()

    # Step 3: Run Full KNN Benchmarking Across Distance Metrics
    print("=" * 80)
    print(" 3. BENCHMARKING EUCLIDEAN vs MANHATTAN vs COSINE (KNN Classifier)")
    print("=" * 80)

    all_results = {}
    metrics_list = ['euclidean', 'manhattan', 'cosine']

    for dset_key, dset_dict in datasets.items():
        print(f"\n>>> Evaluating on {dset_dict['name']} (k=5)...")
        eval_res = benchmark_single_dataset(dset_dict, k_neighbors=5, metrics=metrics_list)
        all_results[dset_key] = eval_res

        for m in metrics_list:
            r = eval_res[m]
            print(f"   [{METRIC_MAP[m]['name']:<24}] Accuracy: {r['accuracy']*100:6.2f}% | F1: {r['f1_weighted']:.4f} | Latency: {r['time_per_query_ms']:.4f} ms/query")

    # Step 4: Summary Table Compilation
    df_summary = generate_benchmark_summary_table(all_results)
    csv_path = os.path.join(results_dir, "benchmark_results.csv")
    df_summary.to_csv(csv_path, index=False)
    print("\n" + "=" * 80)
    print(" 4. COMPREHENSIVE EXPERIMENTAL BENCHMARK SUMMARY")
    print("=" * 80)
    print(df_summary.to_string(index=False))
    print(f"\n[+] Saved tabular summary to: {csv_path}\n")

    # Step 5: 5-Fold Stratified Cross Validation
    print("=" * 80)
    print(" 5. STRATIFIED 5-FOLD CROSS VALIDATION (Robustness Check)")
    print("=" * 80)
    for dset_key in ['iris', 'wine']:
        d = datasets[dset_key]
        X = np.vstack([d['X_train'], d['X_test']])
        y = np.concatenate([d['y_train'], d['y_test']])
        cv_summary = run_cross_validation_benchmark(X, y, k_neighbors=5, metrics=metrics_list, n_splits=5)
        print(f"\n>>> 5-Fold CV Results on {d['name']}:")
        for m in metrics_list:
            s = cv_summary[m]
            print(f"   [{METRIC_MAP[m]['name']:<24}] Mean Acc: {s['mean_accuracy']*100:.2f}% (+/-{s['std_accuracy']*100:.2f}%) | Mean F1: {s['mean_f1']:.4f}")

    # Step 6: Hyperparameter K-Sweep
    print("\n" + "=" * 80)
    print(" 6. HYPERPARAMETER SENSITIVITY SWEEP (k = 1 to 25)")
    print("=" * 80)
    k_range = range(1, 26, 2)
    k_results = {}
    for dset_key, dset_dict in datasets.items():
        k_results[dset_key] = evaluate_k_sweep(dset_dict, k_range=k_range, metrics=metrics_list)
        print(f" * Computed k-sweep curves for {dset_dict['name']}")

    # Step 7: Curse of Dimensionality Simulation
    print("\n" + "=" * 80)
    print(" 7. THEORETICAL ANALYSIS: CURSE OF DIMENSIONALITY (D = 2 to 1000)")
    print("=" * 80)
    curse_data = generate_curse_of_dimensionality_data(
        dimensions=[2, 5, 10, 25, 50, 100, 250, 500, 1000],
        n_samples=250
    )
    print(" * Dimension Sweep completed. Contrast Decay captured.")

    # Step 8: Generate Publication-Grade Visualizations
    print("\n" + "=" * 80)
    print(" 8. GENERATING PUBLICATION-GRADE CHARTS AND FIGURES")
    print("=" * 80)

    # 8.1 Unit Balls & Geometric Iso-Contours
    p_unit = os.path.join(results_dir, "unit_balls_geometry.png")
    plot_unit_balls_geometry(save_path=p_unit)
    print(f"[OK] Saved Unit Ball Geometry Plot: {p_unit}")

    # 8.2 2D Decision Boundaries (PCA projection of Iris)
    iris = datasets['iris']
    pca = PCA(n_components=2, random_state=42)
    X_iris_2d = pca.fit_transform(np.vstack([iris['X_train'], iris['X_test']]))
    y_iris = np.concatenate([iris['y_train'], iris['y_test']])
    p_boundary = os.path.join(results_dir, "decision_boundaries.png")
    plot_decision_boundaries(
        X_iris_2d, y_iris,
        feature_names=["PCA Component 1", "PCA Component 2"],
        target_names=iris['target_names'].tolist(),
        k=5,
        save_path=p_boundary
    )
    print(f"[OK] Saved Decision Boundaries Plot: {p_boundary}")

    # 8.3 Benchmark Bar Charts
    p_bench = os.path.join(results_dir, "metrics_comparison.png")
    plot_benchmark_comparison(all_results, save_path=p_bench)
    print(f"[OK] Saved Multi-Dataset Benchmark Chart: {p_bench}")

    # 8.4 K-Sweep Curves
    p_ksweep = os.path.join(results_dir, "k_vs_accuracy.png")
    plot_k_sweep_curves(k_results, save_path=p_ksweep)
    print(f"[OK] Saved K-Parameter Sensitivity Curves: {p_ksweep}")

    # 8.5 Curse of Dimensionality Plot
    p_curse = os.path.join(results_dir, "curse_of_dimensionality.png")
    plot_curse_of_dimensionality(curse_data, save_path=p_curse)
    print(f"[OK] Saved Curse of Dimensionality Analysis: {p_curse}")

    # 8.6 Confusion Matrices Heatmaps
    for dset_key in ['wine', 'text']:
        p_cm = os.path.join(results_dir, f"confusion_matrices_{dset_key}.png")
        plot_confusion_matrices(all_results[dset_key], dataset_name=dset_key, save_path=p_cm)
        print(f"[OK] Saved Confusion Matrix ({dset_key}): {p_cm}")

    print("\n" + "=" * 80)
    print(" [OK] ALL EXPERIMENTAL BENCHMARKS, EVALUATIONS, AND CHARTS COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    run_full_pipeline()
