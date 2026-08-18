"""
Similarity Measurement Using Distance Metrics
Interactive Web Application: web_app/app.py

Flask backend providing RESTful endpoints for:
- Live vector distance calculations with step-by-step mathematical breakdown
- Interactive 2D KNN decision boundary generation
- Multi-dataset benchmark statistics & confusion matrices
- Semantic document similarity search engine
"""

import os
import sys
import json

# Add project root directory to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.decomposition import PCA

from src.metrics import (
    euclidean_distance, manhattan_distance, cosine_similarity, cosine_distance,
    minkowski_distance, chebyshev_distance, METRIC_MAP
)
from src.knn_model import CustomKNNClassifier, SimilaritySearchEngine
from src.datasets_loader import (
    get_iris_dataset, get_wine_dataset, get_digits_dataset, get_text_dataset
)
from src.evaluation import benchmark_single_dataset

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "static")
)

# Preload datasets for fast interactive responsiveness
CACHED_DATASETS = {
    'iris': get_iris_dataset(),
    'wine': get_wine_dataset(),
    'text': get_text_dataset()
}

# Pre-fit 2D PCA representations for interactive boundary explorer
PCA_CACHE = {}
for name in ['iris', 'wine']:
    d = CACHED_DATASETS[name]
    X_full = np.vstack([d['X_train'], d['X_test']])
    y_full = np.concatenate([d['y_train'], d['y_test']])
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_full)
    PCA_CACHE[name] = {
        'X_2d': X_2d,
        'y': y_full,
        'target_names': [str(c) for c in d.get('target_names', np.unique(y_full))]
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/results/<path:filename>")
def serve_results(filename):
    """Serve generated figures and reports from results directory."""
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    from flask import send_from_directory
    return send_from_directory(results_dir, filename)


@app.route("/api/compute_distance", methods=["POST"])
def api_compute_distance():
    """
    Compute distance metrics between two user-supplied vectors with step-by-step math.
    """
    try:
        data = request.get_json(force=True)
        u = np.array(data.get("u", []), dtype=np.float64)
        v = np.array(data.get("v", []), dtype=np.float64)

        if len(u) == 0 or len(v) == 0:
            return jsonify({"error": "Vectors u and v cannot be empty"}), 400
        if len(u) != len(v):
            return jsonify({"error": f"Vectors must have same dimension. Got {len(u)} and {len(v)}"}), 400

        # Calculations
        d_euc = euclidean_distance(u, v)
        d_man = manhattan_distance(u, v)
        sim_cos = cosine_similarity(u, v)
        d_cos = cosine_distance(u, v)
        d_mink3 = minkowski_distance(u, v, p=3.0)
        d_cheb = chebyshev_distance(u, v)

        # Angle in degrees
        angle_deg = np.degrees(np.arccos(np.clip(sim_cos, -1.0, 1.0)))

        # Math breakdown steps
        diff = u - v
        diff_sq = diff ** 2
        abs_diff = np.abs(diff)

        breakdown = {
            "dim": len(u),
            "diff_vector": diff.tolist(),
            "euclidean": {
                "value": round(d_euc, 6),
                "formula": "sqrt( sum( (u_i - v_i)^2 ) )",
                "sum_squared_diff": round(float(np.sum(diff_sq)), 6),
                "squared_diffs": diff_sq.tolist()
            },
            "manhattan": {
                "value": round(d_man, 6),
                "formula": "sum( |u_i - v_i| )",
                "sum_abs_diff": round(float(np.sum(abs_diff)), 6),
                "abs_diffs": abs_diff.tolist()
            },
            "cosine": {
                "similarity": round(sim_cos, 6),
                "distance": round(d_cos, 6),
                "angle_degrees": round(float(angle_deg), 2),
                "dot_product": round(float(np.dot(u, v)), 6),
                "norm_u": round(float(np.linalg.norm(u)), 6),
                "norm_v": round(float(np.linalg.norm(v)), 6)
            },
            "minkowski_p3": round(d_mink3, 6),
            "chebyshev": round(d_cheb, 6)
        }
        return jsonify({"success": True, "results": breakdown})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/boundary_data", methods=["GET"])
def api_boundary_data():
    """
    Generate 2D grid predictions and dataset points for live interactive canvas plotting.
    """
    try:
        dataset_name = request.args.get("dataset", "iris").lower()
        metric = request.args.get("metric", "euclidean").lower()
        k = int(request.args.get("k", 5))

        if dataset_name not in PCA_CACHE:
            dataset_name = "iris"
        if metric not in METRIC_MAP:
            metric = "euclidean"

        cached = PCA_CACHE[dataset_name]
        X_2d = cached['X_2d']
        y = cached['y']
        target_names = cached['target_names']

        # Fit custom KNN on 2D projection
        knn = CustomKNNClassifier(n_neighbors=k, metric=metric)
        knn.fit(X_2d, y)

        # Generate low-res mesh for smooth canvas rendering
        grid_res = 35
        x_min, x_max = float(X_2d[:, 0].min() - 0.5), float(X_2d[:, 0].max() + 0.5)
        y_min, y_max = float(X_2d[:, 1].min() - 0.5), float(X_2d[:, 1].max() + 0.5)

        gx = np.linspace(x_min, x_max, grid_res)
        gy = np.linspace(y_min, y_max, grid_res)
        xx, yy = np.meshgrid(gx, gy)
        grid_pts = np.c_[xx.ravel(), yy.ravel()]

        preds = knn.predict(grid_pts).tolist()

        points = []
        for i in range(len(X_2d)):
            points.append({
                "x": float(X_2d[i, 0]),
                "y": float(X_2d[i, 1]),
                "label": int(y[i]),
                "className": target_names[int(y[i])] if int(y[i]) < len(target_names) else f"Class {y[i]}"
            })

        return jsonify({
            "success": True,
            "dataset": dataset_name,
            "metric": metric,
            "k": k,
            "bounds": {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max},
            "grid_res": grid_res,
            "grid_preds": preds,
            "points": points,
            "target_names": target_names
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/similarity_search", methods=["POST"])
def api_similarity_search():
    """
    Perform semantic text search across document categories using Cosine vs Euclidean vs Manhattan.
    """
    try:
        data = request.get_json(force=True)
        query_text = data.get("query", "machine learning neural network models")
        metric = data.get("metric", "cosine").lower()
        top_k = int(data.get("top_k", 5))

        text_dset = CACHED_DATASETS['text']
        vectorizer = text_dset['vectorizer']
        target_names = text_dset['target_names']

        # Vectorize query
        q_vec = vectorizer.transform([query_text]).toarray()[0]

        # Use full dataset
        X_all = np.vstack([text_dset['X_train'], text_dset['X_test']])
        y_all = np.concatenate([text_dset['y_train'], text_dset['y_test']])

        metadata = [{"id": i, "category": target_names[y_all[i]]} for i in range(len(X_all))]

        engine = SimilaritySearchEngine(metric=metric)
        engine.index(X_all, metadata=metadata)
        results = engine.query(q_vec, top_k=top_k)

        return jsonify({
            "success": True,
            "query": query_text,
            "metric": metric,
            "results": results
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
