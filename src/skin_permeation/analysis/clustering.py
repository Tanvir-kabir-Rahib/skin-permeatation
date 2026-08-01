from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..artifacts import save_table
from ..data_loading import load_bundle
from ..dependencies import require_module
from ..paths import ProjectPaths
from .applicability import knn_applicability_domain

LOGGER = logging.getLogger(__name__)

sk_cluster = require_module("sklearn.cluster", "Install scikit-learn to run clustering analysis.")
sk_decomposition = require_module("sklearn.decomposition", "Install scikit-learn to run clustering analysis.")
sk_metrics = require_module("sklearn.metrics", "Install scikit-learn to run clustering analysis.")
sk_preprocessing = require_module("sklearn.preprocessing", "Install scikit-learn to run clustering analysis.")
joblib = require_module("joblib", "Install joblib to load saved models.")
plt = require_module("matplotlib.pyplot", "Install matplotlib to generate clustering figures.")
seaborn = require_module("seaborn", "Install seaborn to generate clustering figures.")

KMeans = sk_cluster.KMeans
PCA = sk_decomposition.PCA
silhouette_score = sk_metrics.silhouette_score
calinski_harabasz_score = sk_metrics.calinski_harabasz_score
StandardScaler = sk_preprocessing.StandardScaler


def _predict_drugbank(paths: ProjectPaths, model_path: Path, scaler_path: Path | None) -> pd.DataFrame:
    bundle = load_bundle(paths)
    train = bundle.clean_trial4
    drug = bundle.drugbank_clean.copy()
    feature_columns = [column for column in train.columns if column not in {"logkpl", "Compound", "SMILES"}]
    if "Texpi" not in drug.columns:
        drug["Texpi"] = 310
    model_input = drug[feature_columns]
    model = joblib.load(model_path)
    if scaler_path is not None and str(scaler_path):
        scaler = joblib.load(scaler_path)
        predictions = model.predict(scaler.transform(model_input))
    else:
        predictions = model.predict(model_input)
    output = drug.copy()
    output["predicted_logkpl"] = predictions
    return output


def run_clustering(
    paths: ProjectPaths,
    model_path: Path,
    scaler_path: Path | None,
    cluster_range: range = range(2, 11),
    applicability_threshold_quantile: float = 0.95,
) -> pd.DataFrame:
    output = _predict_drugbank(paths, model_path=model_path, scaler_path=scaler_path)
    descriptor_frame = output.drop(columns=["Name", "SMILES", "predicted_logkpl"])
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(descriptor_frame)

    selection_rows = []
    for k in cluster_range:
        km = KMeans(n_clusters=k, n_init=20, random_state=1)
        labels = km.fit_predict(x_scaled)
        selection_rows.append(
            {
                "k": k,
                "inertia": float(km.inertia_),
                "silhouette": float(silhouette_score(x_scaled, labels)),
                "calinski_harabasz": float(calinski_harabasz_score(x_scaled, labels)),
            }
        )
    selection = pd.DataFrame(selection_rows)
    save_table(selection, paths.reports / "tables" / "drugbank_cluster_selection.csv")

    chosen_k = 4
    km = KMeans(n_clusters=chosen_k, n_init=20, random_state=1)
    output["cluster"] = km.fit_predict(x_scaled)
    pca = PCA(n_components=2, random_state=1)
    projected = pca.fit_transform(x_scaled)
    output["PCA1"] = projected[:, 0]
    output["PCA2"] = projected[:, 1]

    applicability = knn_applicability_domain(
        x_train=load_bundle(paths).clean_trial4.drop(columns=["logkpl", "Compound", "SMILES"]).assign(Texpi=lambda df: df["Texpi"]),
        x_query=output[[column for column in load_bundle(paths).clean_trial4.columns if column not in {"logkpl", "Compound", "SMILES"}]],
        threshold_quantile=applicability_threshold_quantile,
    )
    output = pd.concat([output.reset_index(drop=True), applicability.reset_index(drop=True)], axis=1)
    save_table(output, paths.reports / "tables" / "drugbank_predictions_and_clusters.csv")

    plt.figure(figsize=(9, 7))
    seaborn.scatterplot(data=output, x="PCA1", y="PCA2", hue="cluster", palette="tab10", alpha=0.75)
    plt.title("DrugBank PCA projection with K=4 clusters")
    plt.tight_layout()
    figure_path = paths.figures / "drugbank_pca_clusters.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure_path, dpi=200)
    plt.close()

    cluster_stats = output.groupby("cluster")["predicted_logkpl"].agg(["count", "mean", "std", "min", "max"]).reset_index()
    save_table(cluster_stats, paths.reports / "tables" / "drugbank_cluster_summary.csv")
    return output
