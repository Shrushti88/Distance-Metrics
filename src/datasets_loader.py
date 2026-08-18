"""
Similarity Measurement Using Distance Metrics
Module: src/datasets_loader.py

Dataset loader and preprocessor for evaluating distance metrics across
varying dimensionality, sparsity, and domain characteristics:
- Low-dimensional dense numerical (Iris)
- Medium-dimensional dense numerical (Wine)
- High-dimensional image pixel vectors (Digits / MNIST subset)
- High-dimensional sparse text vectors (TF-IDF Text Corpus)
- Synthetic high-dimensional clusters for Curse of Dimensionality analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.datasets import load_iris, load_wine, load_digits, fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def get_iris_dataset(test_size: float = 0.25, random_state: int = 42, scale: bool = True) -> Dict:
    """
    Load Iris dataset (Low Dimensional: 4 features, 3 classes, 150 samples).
    """
    data = load_iris()
    X, y = data.data, data.target
    feature_names = data.feature_names
    target_names = data.target_names

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return {
        'name': 'Iris Flower Dataset',
        'type': 'Low-Dimensional Numerical',
        'n_features': X.shape[1],
        'n_classes': len(np.unique(y)),
        'n_samples': X.shape[0],
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_names,
        'target_names': target_names
    }


def get_wine_dataset(test_size: float = 0.25, random_state: int = 42, scale: bool = True) -> Dict:
    """
    Load Wine Recognition dataset (Medium Dimensional: 13 features, 3 classes, 178 samples).
    """
    data = load_wine()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return {
        'name': 'Wine Recognition Dataset',
        'type': 'Medium-Dimensional Numerical',
        'n_features': X.shape[1],
        'n_classes': len(np.unique(y)),
        'n_samples': X.shape[0],
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': data.feature_names,
        'target_names': data.target_names
    }


def get_digits_dataset(test_size: float = 0.25, random_state: int = 42, scale: bool = True) -> Dict:
    """
    Load 8x8 Optical Digits dataset (High Dimensional: 64 pixel features, 10 classes, 1797 samples).
    """
    data = load_digits()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if scale:
        # MinMax scaler to preserve 0-1 pixel values
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return {
        'name': 'Optical Digits Dataset',
        'type': 'High-Dimensional Image Pixels (64-D)',
        'n_features': X.shape[1],
        'n_classes': len(np.unique(y)),
        'n_samples': X.shape[0],
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': [f"pixel_{i}" for i in range(X.shape[1])],
        'target_names': [str(i) for i in range(10)]
    }


def get_text_dataset(test_size: float = 0.25, max_features: int = 500, random_state: int = 42) -> Dict:
    """
    Load or generate Text Documents dataset with TF-IDF Vectorization
    (Sparse High-Dimensional Text features).
    """
    # Curated categorized text documents across 4 distinct domains for fast offline execution
    categories = ['tech', 'sports', 'space', 'medicine']
    documents = [
        # Tech
        ("Artificial intelligence and deep learning models are revolutionizing modern computer science software.", 0),
        ("Python, JavaScript, and C++ are widely used programming languages for algorithm implementation.", 0),
        ("Cloud computing and distributed database architectures scale microservices efficiently.", 0),
        ("Neural networks, transformers, and GPU hardware acceleration accelerate machine learning tasks.", 0),
        ("Cybersecurity protocols protect network infrastructure from malicious attacks and data breaches.", 0),
        ("Quantum computing algorithms solve complex cryptographic problems faster than classical computers.", 0),
        ("Software engineering principles emphasize clean code, unit testing, and continuous integration.", 0),
        ("Operating systems manage kernel processes, memory allocation, and CPU scheduling.", 0),
        
        # Sports
        ("The championship football match ended with a stunning last-minute goal in extra time.", 1),
        ("Basketball players focus on shooting percentage, defensive rebounds, and fast break points.", 1),
        ("Tennis Grand Slam tournaments require intense physical endurance, powerful serves, and agility.", 1),
        ("The Olympic marathon runners trained at high altitudes to optimize cardiovascular stamina.", 1),
        ("Baseball pitchers execute curveballs and fastballs to strike out opposing batters.", 1),
        ("Soccer teams utilize tactical formations like 4-3-3 to dominate midfield ball possession.", 1),
        ("Swimming competitions test freestyle stroke technique, underwater flip turns, and speed.", 1),
        ("Cricket test matches test batsman patience, bowler spin variations, and pitch conditions.", 1),

        # Space
        ("NASA space telescope captured high-resolution infrared images of distant galaxy clusters.", 2),
        ("Astronomers discovered an exoplanet orbiting a red dwarf star in the habitable zone.", 2),
        ("Rocket propulsion systems utilize liquid oxygen and cryogenic propellant for orbital insertion.", 2),
        ("The Mars rover collected geological soil samples searching for ancient signs of microbial life.", 2),
        ("Black holes possess gravitational event horizons where escape velocity exceeds light speed.", 2),
        ("Interstellar space missions require deep space radiation shielding and nuclear thermal propulsion.", 2),
        ("Supernova explosions disperse heavy elements like iron, carbon, and gold across nebulae.", 2),
        ("Satellite orbital mechanics follow Kepler's laws of planetary motion and gravitational physics.", 2),

        # Medicine
        ("Clinical trials evaluated the efficacy of novel mRNA vaccines against infectious viral strains.", 3),
        ("Cardiovascular disease prevention involves monitoring cholesterol levels, blood pressure, and diet.", 3),
        ("Antibiotics target bacterial cell wall synthesis while minimizing cytotoxicity to host tissue.", 3),
        ("Neurological MRI scans help diagnose early stage degenerative brain disorders and stroke damage.", 3),
        ("Oncology researchers develop targeted immunotherapies that activate T-cells against tumor cells.", 3),
        ("Pharmacological drug development requires rigorous biochemical phase trials and FDA approval.", 3),
        ("Genomic sequencing identified specific genetic mutations linked to hereditary diseases.", 3),
        ("Surgical procedures utilize robotic laparoscopy for minimally invasive precision operations.", 3),
    ]

    # Duplicate with slight variations to expand sample size for robust train/test split
    expanded_docs = []
    expanded_labels = []
    for doc, label in documents:
        for _ in range(4):
            expanded_docs.append(doc)
            expanded_labels.append(label)

    vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)
    X = vectorizer.fit_transform(expanded_docs).toarray()
    y = np.array(expanded_labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return {
        'name': 'TF-IDF Text Classification Dataset',
        'type': 'High-Dimensional Sparse Text (TF-IDF)',
        'n_features': X.shape[1],
        'n_classes': 4,
        'n_samples': X.shape[0],
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': vectorizer.get_feature_names_out().tolist(),
        'target_names': categories,
        'vectorizer': vectorizer
    }


def generate_curse_of_dimensionality_data(
    dimensions: list = [2, 5, 10, 25, 50, 100, 250, 500, 1000],
    n_samples: int = 200,
    random_state: int = 42
) -> Dict:
    """
    Generate synthetic data across increasing dimensional spaces to measure
    the 'Curse of Dimensionality' and distance metric concentration ratio:
    Contrast = (d_max - d_min) / d_min
    """
    rng = np.random.RandomState(random_state)
    results = {
        'dimensions': dimensions,
        'euclidean_contrast': [],
        'manhattan_contrast': [],
        'cosine_contrast': [],
        'euclidean_mean_ratio': [],
        'manhattan_mean_ratio': [],
        'cosine_mean_ratio': []
    }

    from src.metrics import (
        pairwise_euclidean_distances,
        pairwise_manhattan_distances,
        pairwise_cosine_distances
    )

    for dim in dimensions:
        # Uniformly distributed points in high-dimensional hypercube [0, 1]^D
        X = rng.uniform(0.0, 1.0, size=(n_samples, dim))

        d_euc = pairwise_euclidean_distances(X)
        d_man = pairwise_manhattan_distances(X)
        d_cos = pairwise_cosine_distances(X)

        # Mask upper triangle (excluding zero diagonal)
        triu_indices = np.triu_indices(n_samples, k=1)
        euc_vals = d_euc[triu_indices]
        man_vals = d_man[triu_indices]
        cos_vals = d_cos[triu_indices]

        # Metric contrast: (max - min) / min
        results['euclidean_contrast'].append(float((np.max(euc_vals) - np.min(euc_vals)) / (np.min(euc_vals) + 1e-9)))
        results['manhattan_contrast'].append(float((np.max(man_vals) - np.min(man_vals)) / (np.min(man_vals) + 1e-9)))
        results['cosine_contrast'].append(float((np.max(cos_vals) - np.min(cos_vals)) / (np.min(cos_vals) + 1e-9)))

        # Relative variance / concentration: std / mean
        results['euclidean_mean_ratio'].append(float(np.std(euc_vals) / (np.mean(euc_vals) + 1e-9)))
        results['manhattan_mean_ratio'].append(float(np.std(man_vals) / (np.mean(man_vals) + 1e-9)))
        results['cosine_mean_ratio'].append(float(np.std(cos_vals) / (np.mean(cos_vals) + 1e-9)))

    return results


def load_all_datasets() -> Dict[str, Dict]:
    """Load all standard datasets for comprehensive multi-dataset benchmarking."""
    return {
        'iris': get_iris_dataset(),
        'wine': get_wine_dataset(),
        'digits': get_digits_dataset(),
        'text': get_text_dataset()
    }
