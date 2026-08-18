# Similarity Measurement Using Distance Metrics
> **Pattern Recognition — Project-Based Learning (PBL)**  
> *Implement Euclidean, Manhattan, and Cosine distance measures and compare their performance on datasets.*

---

## 📌 Project Overview
This project delivers an end-to-end, production-grade, and academically rigorous study comparing **Euclidean ($L_2$)**, **Manhattan ($L_1$)**, and **Cosine** distance metrics.

All distance metrics and pattern classification models (K-Nearest Neighbors & Semantic Similarity Search) are **built strictly from scratch in Python / NumPy** without relying on high-level black-box metric wrappers.

---

## 🚀 Quickstart

### 1. Run Complete Benchmark & Generate Visualizations (CLI)
```bash
python main.py
```
This runs sanity checks, loads 4 multi-domain datasets, performs 5-fold cross-validation, executes $k$-parameter sweeps, simulates the Curse of Dimensionality, and saves publication-quality plots to `results/`.

### 2. Launch Interactive Web Dashboard (GUI)
```bash
python web_app/app.py
```
Then open your browser to **[http://127.0.0.1:5000](http://127.0.0.1:5000)** to interactively:
- Calculate step-by-step distance breakdowns on arbitrary vectors.
- Explore live 2D KNN decision boundaries by adjusting metric and $k$.
- Test real-time semantic document search.

### 3. Or 1-Click Runner (Windows)
Double-click `run_demo.bat`.

---

## 📂 Repository Structure
```
Pattern Recognition/
├── src/
│   ├── metrics.py            # Vectorized from-scratch Euclidean, Manhattan, Cosine, Minkowski, Chebyshev
│   ├── knn_model.py          # Custom KNN Classifier & Semantic Search Engine supporting any distance metric
│   ├── datasets_loader.py    # Multi-domain loaders (Iris, Wine, Digits, TF-IDF Text, High-D Synthetic)
│   ├── evaluation.py         # Metrics (Accuracy, Precision, Recall, F1, Confusion Matrix, 5-Fold CV, Latency)
│   └── visualizer.py         # Publication-grade plotting engine
├── web_app/
│   ├── app.py                # Flask REST backend
│   ├── static/               # Dark glassmorphic stylesheet & interactive client logic
│   └── templates/            # Interactive Web UI dashboard
├── notebooks/
│   └── similarity_metrics_walkthrough.ipynb # Step-by-step Jupyter Notebook with math, code, and outputs
├── results/
│   ├── benchmark_results.csv # Tabular benchmark summary
│   ├── unit_balls_geometry.png
│   ├── decision_boundaries.png
│   ├── metrics_comparison.png
│   ├── k_vs_accuracy.png
│   ├── curse_of_dimensionality.png
│   ├── confusion_matrices_wine.png
│   └── confusion_matrices_text.png
├── main.py                   # Master execution pipeline
├── PROJECT_REPORT.md         # Comprehensive academic report for submission
├── run_demo.bat              # 1-click Windows runner
└── README.md
```

---

## 📊 Summary of Experimental Results ($k=5$)

| Dataset | Characteristics | Euclidean ($L_2$) Acc | Manhattan ($L_1$) Acc | Cosine Distance Acc | Optimal Metric |
|---|---|---|---|---|---|
| **Iris** | Low-D Dense (4-D) | **92.11%** | **92.11%** | 78.95% | **Euclidean / Manhattan** |
| **Wine** | Medium-D Dense (13-D) | 93.33% | **97.78%** | **97.78%** | **Manhattan / Cosine** |
| **Optical Digits** | High-D Image Pixels (64-D) | **98.22%** | 98.00% | 98.00% | **Euclidean ($L_2$)** |
| **TF-IDF Text** | High-D Sparse (292-D) | 59.38% | 46.88% | **59.38%** | **Cosine Distance** |

---

## 🎓 Key Academic Takeaways
1. **Geometric Differences:** Euclidean measures spherical distances, Manhattan measures grid/orthogonal distances, and Cosine measures angular orientation.
2. **Curse of Dimensionality:** In higher dimensions ($D > 500$), Euclidean distances concentrate rapidly ($d_{\max} \approx d_{\min}$), whereas Manhattan and Cosine metrics retain higher relative contrast.
3. **Sparse Text Advantage:** Cosine distance is length-invariant, making it vastly superior for document retrieval and Natural Language Processing compared to Manhattan distance.
