"""
Similarity Measurement Using Distance Metrics
Module: src/visualizer.py

Generates publication-quality visualizations, plots, and charts:
1. Geometric Unit Balls (L1 diamond, L2 circle, L_inf square, Cosine angle rays)
2. 2D Decision Boundaries for Euclidean, Manhattan, and Cosine
3. Multi-Dataset Performance Comparison Charts
4. K-Parameter Accuracy Curves
5. High-Dimensionality Distance Contrast / Curse of Dimensionality Plot
6. Multi-Metric Confusion Matrix Heatmaps
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from src.knn_model import CustomKNNClassifier
from src.metrics import METRIC_MAP

# Set aesthetic styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'font.family': 'sans-serif'
})


def plot_unit_balls_geometry(save_path: Optional[str] = None):
    """
    Plot unit balls {x : d(0, x) = 1} for L1 (Manhattan), L2 (Euclidean),
    and Linf (Chebyshev) alongside angular Cosine representation.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. L1, L2, L_inf Unit Circles
    ax1 = axes[0]
    theta = np.linspace(0, 2 * np.pi, 500)

    # L2: x^2 + y^2 = 1
    x_l2 = np.cos(theta)
    y_l2 = np.sin(theta)
    ax1.plot(x_l2, y_l2, label=r'Euclidean ($L_2$ norm: $x^2+y^2=1$)', color='#2563eb', lw=2.5)

    # L1: |x| + |y| = 1
    x_l1 = np.array([1, 0, -1, 0, 1])
    y_l1 = np.array([0, 1, 0, -1, 0])
    ax1.plot(x_l1, y_l1, label=r'Manhattan ($L_1$ norm: $|x|+|y|=1$)', color='#dc2626', lw=2.5, linestyle='--')

    # L_inf: max(|x|, |y|) = 1
    x_linf = np.array([1, 1, -1, -1, 1])
    y_linf = np.array([1, -1, -1, 1, 1])
    ax1.plot(x_linf, y_linf, label=r'Chebyshev ($L_\infty$: $\max(|x|,|y|)=1$)', color='#16a34a', lw=2, linestyle=':')

    ax1.axhline(0, color='gray', lw=0.8, alpha=0.6)
    ax1.axvline(0, color='gray', lw=0.8, alpha=0.6)
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_aspect('equal')
    ax1.set_title("Unit Norm Balls: Geometric Iso-Distance Contours", fontweight='bold')
    ax1.set_xlabel("Feature $X_1$")
    ax1.set_ylabel("Feature $X_2$")
    ax1.legend(loc='upper right', framealpha=0.9)

    # 2. Distance Iso-Contours around Center Point (0,0)
    ax2 = axes[1]
    grid_range = np.linspace(-2, 2, 200)
    GX, GY = np.meshgrid(grid_range, grid_range)
    Z_euc = np.sqrt(GX**2 + GY**2)
    Z_man = np.abs(GX) + np.abs(GY)

    cs1 = ax2.contour(GX, GY, Z_euc, levels=[0.5, 1.0, 1.5], colors='#2563eb', linestyles='solid', alpha=0.7)
    cs2 = ax2.contour(GX, GY, Z_man, levels=[0.5, 1.0, 1.5], colors='#dc2626', linestyles='dashed', alpha=0.7)
    ax2.scatter([0], [0], color='black', s=60, zorder=5, label='Origin $(0,0)$')
    ax2.set_aspect('equal')
    ax2.set_title("Iso-Distance Comparison ($L_2$ vs $L_1$)", fontweight='bold')
    ax2.set_xlabel("Feature $X_1$")
    ax2.set_ylabel("Feature $X_2$")
    ax2.plot([], [], color='#2563eb', label=r'Euclidean Contours ($L_2$)')
    ax2.plot([], [], color='#dc2626', linestyle='--', label=r'Manhattan Contours ($L_1$)')
    ax2.legend(loc='upper right', framealpha=0.9)

    # 3. Cosine Angular Similarity Principle
    ax3 = axes[2]
    u = np.array([1.5, 1.0])
    v1 = np.array([3.0, 2.0])  # Collinear (angle = 0)
    v2 = np.array([-1.0, 1.5])  # Orthogonal-ish

    ax3.quiver(0, 0, u[0], u[1], angles='xy', scale_units='xy', scale=1, color='#4f46e5', width=0.015, label='Query Vector $u$')
    ax3.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='#059669', width=0.015, label='Vector $v_1$ (Same Angle, Diff Mag)')
    ax3.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='#d97706', width=0.015, label='Vector $v_2$ (Diff Angle)')

    ax3.axhline(0, color='gray', lw=0.8, alpha=0.6)
    ax3.axvline(0, color='gray', lw=0.8, alpha=0.6)
    ax3.set_xlim(-2.0, 3.5)
    ax3.set_ylim(-1.0, 3.0)
    ax3.set_aspect('equal')
    ax3.set_title("Cosine Distance Principle: Angle vs Magnitude", fontweight='bold')
    ax3.set_xlabel("Feature $X_1$")
    ax3.set_ylabel("Feature $X_2$")
    ax3.legend(loc='upper left', framealpha=0.9)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_decision_boundaries(
    X_2d: np.ndarray,
    y: np.ndarray,
    feature_names: List[str] = ["Feature 1", "Feature 2"],
    target_names: Optional[List[str]] = None,
    k: int = 5,
    save_path: Optional[str] = None
):
    """
    Plot side-by-side 2D decision boundary maps for Euclidean, Manhattan, and Cosine metrics.
    """
    metrics = ['euclidean', 'manhattan', 'cosine']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Create coordinate mesh
    h = 0.05
    x_min, x_max = X_2d[:, 0].min() - 0.6, X_2d[:, 0].max() + 0.6
    y_min, y_max = X_2d[:, 1].min() - 0.6, X_2d[:, 1].max() + 0.6
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    cmap_light = ['#dbeafe', '#fee2e2', '#dcfce7', '#fef3c7']
    palette = ['#1d4ed8', '#b91c1c', '#15803d', '#b45309']

    for i, metric in enumerate(metrics):
        ax = axes[i]
        knn = CustomKNNClassifier(n_neighbors=k, metric=metric)
        knn.fit(X_2d, y)
        Z = knn.predict(grid_points)
        Z = Z.reshape(xx.shape)

        # Plot decision regions
        ax.contourf(xx, yy, Z, alpha=0.35, levels=np.arange(-0.5, len(np.unique(y)) + 0.5, 1), colors=cmap_light[:len(np.unique(y))])
        ax.contour(xx, yy, Z, levels=np.arange(len(np.unique(y))), colors='black', linewidths=0.7, alpha=0.5)

        # Plot training points
        for cl_idx, cl in enumerate(np.unique(y)):
            mask = (y == cl)
            label = target_names[cl_idx] if target_names else f"Class {cl}"
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=palette[cl_idx % len(palette)], label=label, edgecolor='k', s=45, alpha=0.9)

        ax.set_title(f"KNN ({METRIC_MAP[metric]['name']}, $k={k}$)", fontweight='bold')
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
        if i == 0:
            ax.legend(loc='best', framealpha=0.9)

    plt.suptitle("Decision Boundary Topologies Across Distance Metrics", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_benchmark_comparison(all_results: Dict[str, Dict[str, Dict]], save_path: Optional[str] = None):
    """
    Plot grouped bar charts comparing Accuracy and F1-Score across all datasets & metrics.
    """
    dataset_names = list(all_results.keys())
    metrics = ['euclidean', 'manhattan', 'cosine']
    metric_labels = ['Euclidean (L2)', 'Manhattan (L1)', 'Cosine']
    colors = ['#2563eb', '#dc2626', '#059669']

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    x = np.arange(len(dataset_names))
    width = 0.25

    # 1. Accuracy Plot
    ax1 = axes[0]
    for i, (m, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        accs = [all_results[d][m]['accuracy'] * 100 for d in dataset_names]
        bars = ax1.bar(x + (i - 1) * width, accs, width, label=label, color=color, alpha=0.85, edgecolor='black', linewidth=0.6)
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f"{height:.1f}%",
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    ax1.set_title("Classification Accuracy by Metric & Dataset", fontweight='bold')
    ax1.set_ylabel("Test Accuracy (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([d.upper() for d in dataset_names], fontweight='bold')
    ax1.set_ylim(0, 110)
    ax1.legend(loc='lower right', framealpha=0.9)

    # 2. Weighted F1 Score Plot
    ax2 = axes[1]
    for i, (m, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        f1s = [all_results[d][m]['f1_weighted'] for d in dataset_names]
        bars = ax2.bar(x + (i - 1) * width, f1s, width, label=label, color=color, alpha=0.85, edgecolor='black', linewidth=0.6)
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f"{height:.3f}",
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    ax2.set_title("Weighted F1-Score by Metric & Dataset", fontweight='bold')
    ax2.set_ylabel("F1 Score")
    ax2.set_xticks(x)
    ax2.set_xticklabels([d.upper() for d in dataset_names], fontweight='bold')
    ax2.set_ylim(0, 1.15)
    ax2.legend(loc='lower right', framealpha=0.9)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_k_sweep_curves(k_results_dict: Dict[str, Dict[str, Dict[int, float]]], save_path: Optional[str] = None):
    """
    Plot K-parameter sweep line graphs for each dataset.
    """
    n_datasets = len(k_results_dict)
    fig, axes = plt.subplots(1, n_datasets, figsize=(5 * n_datasets, 4.5), sharey=False)
    if n_datasets == 1:
        axes = [axes]

    styles = {
        'euclidean': {'color': '#2563eb', 'marker': 'o', 'label': 'Euclidean (L2)'},
        'manhattan': {'color': '#dc2626', 'marker': 's', 'label': 'Manhattan (L1)'},
        'cosine': {'color': '#059669', 'marker': '^', 'label': 'Cosine'}
    }

    for idx, (dset_name, metrics_k) in enumerate(k_results_dict.items()):
        ax = axes[idx]
        for metric, k_vals in metrics_k.items():
            ks = sorted(list(k_vals.keys()))
            accs = [k_vals[k] * 100 for k in ks]
            ax.plot(ks, accs, **styles[metric], lw=2.0, markersize=5.5)

        ax.set_title(f"{dset_name.upper()} Dataset", fontweight='bold')
        ax.set_xlabel("Number of Neighbors ($k$)")
        ax.set_ylabel("Accuracy (%)" if idx == 0 else "")
        ax.set_xticks(ks)
        ax.legend(loc='best', framealpha=0.9)

    plt.suptitle("Hyperparameter Sensitivity: Accuracy vs. Neighborhood Size ($k$)", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_curse_of_dimensionality(curse_data: Dict, save_path: Optional[str] = None):
    """
    Plot distance metric concentration / contrast decay as dimensions increase from 2 to 1000.
    """
    dims = curse_data['dimensions']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Relative Contrast: (d_max - d_min) / d_min
    ax1 = axes[0]
    ax1.plot(dims, curse_data['euclidean_contrast'], 'o-', color='#2563eb', lw=2.2, label='Euclidean ($L_2$)')
    ax1.plot(dims, curse_data['manhattan_contrast'], 's-', color='#dc2626', lw=2.2, label='Manhattan ($L_1$)')
    ax1.plot(dims, curse_data['cosine_contrast'], '^-', color='#059669', lw=2.2, label='Cosine Distance')
    ax1.set_xscale('log')
    ax1.set_title("Distance Contrast Ratio: $(D_{max} - D_{min}) / D_{min}$", fontweight='bold')
    ax1.set_xlabel("Dimensionality ($D$) [Log Scale]")
    ax1.set_ylabel("Metric Contrast")
    ax1.legend(framealpha=0.9)

    # 2. Relative Standard Deviation: std / mean
    ax2 = axes[1]
    ax2.plot(dims, curse_data['euclidean_mean_ratio'], 'o-', color='#2563eb', lw=2.2, label='Euclidean ($L_2$)')
    ax2.plot(dims, curse_data['manhattan_mean_ratio'], 's-', color='#dc2626', lw=2.2, label='Manhattan ($L_1$)')
    ax2.plot(dims, curse_data['cosine_mean_ratio'], '^-', color='#059669', lw=2.2, label='Cosine Distance')
    ax2.set_xscale('log')
    ax2.set_title(r"Relative Variance: $\sigma_D / \mu_D$ (Concentration Effect)", fontweight='bold')
    ax2.set_xlabel("Dimensionality ($D$) [Log Scale]")
    ax2.set_ylabel(r"$\sigma / \mu$ Ratio")
    ax2.legend(framealpha=0.9)

    plt.suptitle("The Curse of Dimensionality & Metric Degeneracy Analysis", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrices(metric_evals: Dict[str, Dict], dataset_name: str, save_path: Optional[str] = None):
    """
    Plot side-by-side confusion matrix heatmaps for Euclidean, Manhattan, and Cosine metrics.
    """
    metrics = ['euclidean', 'manhattan', 'cosine']
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        cm = metric_evals[metric]['confusion_matrix']
        target_names = metric_evals[metric].get('target_names', None)

        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            cbar=False,
            ax=ax,
            xticklabels=target_names if target_names is not None else "auto",
            yticklabels=target_names if target_names is not None else "auto"
        )
        acc = metric_evals[metric]['accuracy'] * 100
        ax.set_title(f"{METRIC_MAP[metric]['name']}\nAccuracy: {acc:.1f}%", fontweight='bold')
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label" if idx == 0 else "")

    plt.suptitle(f"Confusion Matrix Comparison — {dataset_name.upper()} Dataset", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
