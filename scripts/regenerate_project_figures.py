from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import tempfile
import textwrap
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "skin_permeation_matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "skin_permeation_cache"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu, pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler


LOGGER = logging.getLogger("regenerate_project_figures")

TARGET_CANDIDATES = (
    "logkpl",
    "logkp",
    "log_kp",
    "log kp",
    "experimental_logkp",
    "experimental logkp",
    "skin_permeability",
    "permeability",
    "target",
)
ACTUAL_CANDIDATES = (
    "actual_logkpl",
    "actual_logkp",
    "experimental_logkp",
    "observed_logkp",
    "y_true",
    "actual",
    "observed",
)
PREDICTED_CANDIDATES = (
    "predicted_logkpl",
    "predicted_logkp",
    "prediction_logkp",
    "estimated_logkp",
    "y_pred",
    "predicted",
    "prediction",
)
IDENTIFIER_MARKERS = (
    "compound",
    "name",
    "smiles",
    "drugbank",
    "pubchem",
    "chembl",
    "cas",
    "inchi",
    "inchikey",
    "identifier",
)
GROUP_MARKERS = (
    "atc",
    "therapeutic",
    "drug_class",
    "drug class",
    "category",
    "group",
    "cluster",
    "source",
    "class",
)
LEAKAGE_MARKERS = (
    "predicted",
    "prediction",
    "actual",
    "observed",
    "experimental",
    "residual",
    "absolute_error",
    "error",
    "fold",
    "split",
    "cluster",
    "pca",
    "umap",
    "tsne",
    "applicability",
    "within_domain",
    "mean_knn",
)
PROPERTY_ALIASES = {
    "Molecular weight": ("MW", "MWa", "MolWt", "MolecularWeight", "molecular_weight"),
    "LogP": ("XLogP", "JPLogP", "ALogP", "MLogP", "logKowb", "MolLogP"),
    "Topological polar surface area": ("TopoPSA", "TPSA", "tpsa"),
    "Hydrogen-bond donors": ("nHBDon", "HBD", "NumHDonors"),
    "Hydrogen-bond acceptors": ("nHBAcc", "HBA", "NumHAcceptors"),
    "Lipinski failures": ("LipinskiFailures", "LipinskiViolations"),
    "Aromatic rings": ("nAromRings", "AromaticRings"),
    "Rotatable bonds": ("nRotB", "NumRotatableBonds"),
    "Solubility": ("LogSaqd", "LogSoc", "LogS", "Solubility"),
}
PALETTE = {
    "navy": "#16324F",
    "blue": "#2878B5",
    "teal": "#2A9D8F",
    "gold": "#E9C46A",
    "orange": "#F4A261",
    "red": "#D1495B",
    "gray": "#6B7280",
    "light": "#EFF4F8",
}


@dataclass
class FigureRegistry:
    root: Path
    output_dir: Path
    generated: dict[str, list[str]] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)

    def display_path(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.root.resolve()))
        except ValueError:
            return str(path)

    def generate(self, key: str, callback: Callable[[], Iterable[Path]]) -> None:
        try:
            paths = list(callback())
            if not paths:
                raise ValueError("figure function did not produce output files")
            self.generated[key] = [self.display_path(path) for path in paths]
            LOGGER.info("Generated %s", key)
        except Exception as exc:
            reason = str(exc).strip() or type(exc).__name__
            self.skipped[key] = reason
            LOGGER.warning("Skipped %s: %s", key, reason)

    def write_manifest(self, metadata: dict[str, Any]) -> Path:
        manifest = {
            "metadata": metadata,
            "generated": self.generated,
            "skipped": self.skipped,
        }
        path = self.output_dir / "figure_generation_summary.json"
        path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        return path


@dataclass
class ProjectContext:
    root: Path
    output_dir: Path
    random_state: int
    top_n_features: int
    requested_clusters: int | None
    paper_pdf: Path | None = None
    skin_path: Path | None = None
    descriptor_path: Path | None = None
    external_path: Path | None = None
    prediction_path: Path | None = None
    metrics_path: Path | None = None
    model_dir: Path | None = None
    skin: pd.DataFrame | None = None
    characterization: pd.DataFrame | None = None
    descriptors: pd.DataFrame | None = None
    modeling_descriptors: pd.DataFrame | None = None
    descriptor_report: dict[str, list[str]] = field(default_factory=dict)
    external: pd.DataFrame | None = None
    predictions: pd.DataFrame | None = None
    external_validation: pd.DataFrame | None = None
    external_validation_label: str = "Available external validation"
    metrics: pd.DataFrame | None = None
    target_column: str | None = None
    identifier_columns: list[str] = field(default_factory=list)
    actual_column: str | None = None
    predicted_column: str | None = None
    best_model_name: str | None = None
    best_model_path: Path | None = None
    best_model: Any | None = None
    x_train: pd.DataFrame | None = None
    x_test: pd.DataFrame | None = None
    y_train: pd.Series | None = None
    y_test: pd.Series | None = None
    model_prediction_sets: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def configure_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#374151",
            "axes.linewidth": 0.8,
            "grid.color": "#D7DEE5",
            "grid.linewidth": 0.55,
            "grid.alpha": 0.75,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def resolve_path(root: Path, value: str | Path | None) -> Path | None:
    if value is None or not str(value).strip():
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def first_existing(root: Path, candidates: Iterable[str | Path]) -> Path | None:
    for candidate in candidates:
        path = resolve_path(root, candidate)
        if path is not None and path.exists():
            return path
    return None


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported tabular input format: {path.suffix}")


def find_column(
    frame: pd.DataFrame,
    requested: str | None,
    candidates: Iterable[str],
    contains: Iterable[str] = (),
) -> str | None:
    normalized = {normalize_name(column): str(column) for column in frame.columns}
    if requested:
        match = normalized.get(normalize_name(requested))
        if match:
            return match
        raise ValueError(
            f"Requested column '{requested}' was not found. Available columns: {', '.join(map(str, frame.columns))}"
        )
    for candidate in candidates:
        match = normalized.get(normalize_name(candidate))
        if match:
            return match
    contains_normalized = [normalize_name(marker) for marker in contains]
    matches = [
        str(column)
        for column in frame.columns
        if any(marker in normalize_name(column) for marker in contains_normalized)
    ]
    return matches[0] if len(matches) == 1 else None


def detect_identifier_columns(frame: pd.DataFrame) -> list[str]:
    identifiers = []
    for column in frame.columns:
        normalized = normalize_name(column)
        if any(marker in normalized for marker in IDENTIFIER_MARKERS):
            identifiers.append(str(column))
    return identifiers


def clean_descriptor_matrix(
    frame: pd.DataFrame,
    target_column: str | None,
    identifier_columns: Iterable[str],
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    excluded = set(identifier_columns)
    if target_column:
        excluded.add(target_column)
    candidate_columns = []
    leakage = []
    non_numeric = []
    for column in frame.columns:
        column_text = str(column)
        normalized = normalize_name(column_text)
        if column_text in excluded:
            continue
        if any(normalize_name(marker) in normalized for marker in LEAKAGE_MARKERS):
            leakage.append(column_text)
            continue
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.notna().mean() < 0.80:
            non_numeric.append(column_text)
            continue
        candidate_columns.append(column_text)

    numeric = frame[candidate_columns].apply(pd.to_numeric, errors="coerce")
    all_missing = numeric.columns[numeric.isna().all()].tolist()
    numeric = numeric.drop(columns=all_missing)
    constant = numeric.columns[numeric.nunique(dropna=True) <= 1].tolist()
    numeric = numeric.drop(columns=constant)

    duplicate_columns = []
    seen: dict[int, str] = {}
    keep = []
    for column in numeric.columns:
        hashed = int(pd.util.hash_pandas_object(numeric[column], index=False).sum())
        prior = seen.get(hashed)
        if prior is not None and numeric[column].equals(numeric[prior]):
            duplicate_columns.append(str(column))
        else:
            seen[hashed] = str(column)
            keep.append(str(column))
    numeric = numeric[keep]
    report = {
        "excluded_identifiers_and_target": sorted(excluded),
        "leakage_columns": leakage,
        "non_numeric_columns": non_numeric,
        "all_missing_columns": all_missing,
        "constant_columns": constant,
        "duplicate_columns": duplicate_columns,
    }
    return numeric, report


def modeling_descriptor_matrix(
    frame: pd.DataFrame,
    target_column: str | None,
    identifier_columns: Iterable[str],
) -> pd.DataFrame:
    excluded = set(identifier_columns)
    if target_column:
        excluded.add(target_column)
    candidate_columns = [column for column in frame.columns if column not in excluded]
    numeric = frame[candidate_columns].apply(pd.to_numeric, errors="coerce")
    return numeric.loc[:, numeric.notna().any(axis=0)]


def merge_descriptor_data(
    skin: pd.DataFrame,
    descriptors: pd.DataFrame,
    target_column: str,
    identifiers: list[str],
) -> pd.DataFrame:
    if target_column in descriptors.columns:
        return descriptors.copy()
    shared_identifiers = [column for column in identifiers if column in descriptors.columns]
    for column in shared_identifiers:
        if skin[column].notna().any() and descriptors[column].notna().any():
            left = skin[[column, target_column, *[c for c in identifiers if c != column and c in skin.columns]]].copy()
            merged = left.merge(descriptors, on=column, how="inner")
            if len(merged) >= max(10, int(0.5 * len(skin))):
                return merged
    if len(skin) == len(descriptors):
        output = descriptors.reset_index(drop=True).copy()
        output[target_column] = skin[target_column].reset_index(drop=True)
        for column in identifiers:
            if column in skin.columns and column not in output.columns:
                output[column] = skin[column].reset_index(drop=True)
        return output
    raise ValueError(
        "Descriptor data could not be aligned to the experimental data by identifier or row count."
    )


def augment_characterization_data(root: Path, skin: pd.DataFrame, target_column: str) -> pd.DataFrame:
    output = skin.copy()
    raw_path = first_existing(
        root,
        (
            "data/raw/Skin Permeation.xlsx",
            "data/raw/skin_permeation.xlsx",
            "data/raw/data-original.csv",
        ),
    )
    if raw_path is None:
        return output
    try:
        raw = load_table(raw_path)
        raw_target = find_column(raw, target_column, TARGET_CANDIDATES, contains=("logkp",))
        shared_key = next(
            (
                column
                for column in ("SMILES", "smiles", "Compound", "compound")
                if column in output.columns and column in raw.columns
            ),
            None,
        )
        if shared_key is None:
            return output
        desired = []
        for aliases in PROPERTY_ALIASES.values():
            match = find_column(raw, None, aliases)
            if match and match not in output.columns and match not in desired:
                desired.append(match)
        if not desired:
            return output
        numeric_desired = []
        for column in desired:
            converted = pd.to_numeric(raw[column], errors="coerce")
            if converted.notna().sum() >= 8:
                raw[column] = converted
                numeric_desired.append(column)
        if not numeric_desired:
            return output
        aggregation = {column: "median" for column in numeric_desired}
        auxiliary = raw.groupby(shared_key, dropna=False).agg(aggregation).reset_index()
        output = output.merge(auxiliary, on=shared_key, how="left")
        if raw_target and raw_target != target_column and raw_target in output.columns:
            output = output.drop(columns=[raw_target])
    except Exception as exc:
        LOGGER.info("Could not augment characterization properties from %s: %s", raw_path, exc)
    return output


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        return {}
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return payload if isinstance(payload, dict) else {}


def select_metrics_table(root: Path) -> Path | None:
    priorities = (
        "reports/tables/improved_metrics.csv",
        "reports/tables/strict_grouped_metrics.csv",
        "reports/tables/grouped_metrics.csv",
        "reports/tables/validation_metrics.csv",
        "reports/tables/benchmark_metrics.csv",
        "reports/tables/paper_baseline_metrics.csv",
    )
    selected = first_existing(root, priorities)
    if selected:
        return selected
    candidates = sorted((root / "reports").glob("**/*metrics*.csv")) if (root / "reports").exists() else []
    for path in candidates:
        try:
            frame = pd.read_csv(path, nrows=3)
        except Exception:
            continue
        normalized = {normalize_name(column) for column in frame.columns}
        if "model" in normalized and normalized.intersection({"r2", "rmse", "mae", "mse"}):
            return path
    return None


def choose_best_model(metrics: pd.DataFrame) -> pd.Series:
    if "model" not in metrics.columns:
        raise ValueError("Model metrics table has no 'model' column.")
    candidates = metrics.replace([np.inf, -np.inf], np.nan).dropna(subset=["model"]).copy()
    if "mae" in candidates.columns:
        candidates = candidates.sort_values(
            [column for column in ("mae", "rmse", "r2") if column in candidates.columns],
            ascending=[column != "r2" for column in ("mae", "rmse", "r2") if column in candidates.columns],
        )
    elif "rmse" in candidates.columns:
        candidates = candidates.sort_values("rmse")
    elif "r2" in candidates.columns:
        candidates = candidates.sort_values("r2", ascending=False)
    else:
        raise ValueError("Model metrics table contains none of R2, RMSE, or MAE.")
    return candidates.iloc[0]


def model_path_from_row(root: Path, model_dir: Path | None, row: pd.Series) -> Path | None:
    if "artifact_path" in row and pd.notna(row["artifact_path"]):
        artifact = Path(str(row["artifact_path"]))
        if artifact.exists():
            return artifact
        if model_dir is not None:
            candidate = model_dir / artifact.name
            if candidate.exists():
                return candidate
        candidate = root / "models" / "reproduction" / artifact.parent.name / artifact.name
        if candidate.exists():
            return candidate
    if model_dir is None:
        return None
    stem = re.sub(r"[^A-Za-z0-9]+", "_", str(row["model"])).strip("_")
    candidates = [
        model_dir / f"{stem}.joblib",
        model_dir / f"{stem.replace('_RBF', 'RBF')}.joblib",
        model_dir / f"{stem}.pkl",
        model_dir / f"{stem}.sav",
    ]
    return next((path for path in candidates if path.exists()), None)


def discover_inputs(args: argparse.Namespace) -> ProjectContext:
    root = Path(__file__).resolve().parents[1]
    output_dir = resolve_path(root, args.output_dir) or (root / "figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    context = ProjectContext(
        root=root,
        output_dir=output_dir,
        random_state=args.random_state,
        top_n_features=max(3, args.top_n_features),
        requested_clusters=args.n_clusters,
        paper_pdf=resolve_path(root, args.paper_pdf),
    )

    context.skin_path = resolve_path(root, args.skin_data) or first_existing(
        root,
        (
            "data/final/clean_trial4.csv",
            "data/processed/clean_trial4.csv",
            "data/processed/trial4.csv",
            "data/raw/Skin Permeation.xlsx",
            "train_data.csv",
        ),
    )
    if context.skin_path:
        context.skin = load_table(context.skin_path)
        context.target_column = find_column(
            context.skin,
            args.target_column,
            TARGET_CANDIDATES,
            contains=("logkp",),
        )
        if context.target_column is None:
            raise ValueError(
                f"Could not detect a LogKp target in {context.skin_path}. Pass --target-column explicitly."
            )
        context.identifier_columns = detect_identifier_columns(context.skin)

    context.descriptor_path = resolve_path(root, args.descriptor_data)
    descriptor_source = context.skin
    if context.descriptor_path:
        descriptor_data = load_table(context.descriptor_path)
        if context.skin is None or context.target_column is None:
            descriptor_source = descriptor_data
        else:
            descriptor_source = merge_descriptor_data(
                context.skin,
                descriptor_data,
                context.target_column,
                context.identifier_columns,
            )
    elif context.skin_path:
        context.descriptor_path = context.skin_path

    if descriptor_source is not None:
        if context.target_column is None:
            context.target_column = find_column(
                descriptor_source,
                args.target_column,
                TARGET_CANDIDATES,
                contains=("logkp",),
            )
        context.descriptors, context.descriptor_report = clean_descriptor_matrix(
            descriptor_source,
            context.target_column,
            context.identifier_columns or detect_identifier_columns(descriptor_source),
        )
        context.modeling_descriptors = modeling_descriptor_matrix(
            descriptor_source,
            context.target_column,
            context.identifier_columns or detect_identifier_columns(descriptor_source),
        )
        if context.skin is None:
            context.skin = descriptor_source
            context.identifier_columns = detect_identifier_columns(descriptor_source)
        if context.skin is not None and context.target_column is not None:
            context.characterization = augment_characterization_data(
                root,
                context.skin,
                context.target_column,
            )

    context.external_path = resolve_path(root, args.external_data) or first_existing(
        root,
        (
            "reports/tables/drugbank_predictions_and_clusters.csv",
            "data/processed/drug_bank_clean.csv",
            "data/raw/DrugBank-descriptors.csv",
        ),
    )
    if context.external_path:
        context.external = load_table(context.external_path)

    context.metrics_path = select_metrics_table(root)
    if context.metrics_path:
        context.metrics = pd.read_csv(context.metrics_path)
        best_row = choose_best_model(context.metrics)
        context.best_model_name = str(best_row["model"])

    context.model_dir = resolve_path(root, args.model_dir)
    if context.model_dir is None and context.metrics_path:
        if "improved" in context.metrics_path.name.lower():
            context.model_dir = root / "models" / "reproduction" / "improved"
        elif "benchmark" in context.metrics_path.name.lower():
            context.model_dir = root / "models" / "reproduction" / "benchmark"
        elif "baseline" in context.metrics_path.name.lower():
            context.model_dir = root / "models" / "reproduction" / "paper_baseline"
    if context.metrics is not None:
        best_row = choose_best_model(context.metrics)
        context.best_model_path = model_path_from_row(root, context.model_dir, best_row)

    context.prediction_path = resolve_path(root, args.prediction_data)
    if context.prediction_path is None:
        prediction_candidates = []
        if context.metrics_path and "improved" in context.metrics_path.name.lower():
            prediction_candidates.append("reports/tables/improved_test_errors.csv")
        prediction_candidates.extend(
            (
                "external_test_predictions.csv",
                "reports/tables/test_predictions.csv",
                "reports/tables/predictions.csv",
            )
        )
        context.prediction_path = first_existing(root, prediction_candidates)
    if context.prediction_path:
        context.predictions = load_table(context.prediction_path)
        context.actual_column = find_column(
            context.predictions,
            None,
            ACTUAL_CANDIDATES,
            contains=("actual_logkp", "experimental_logkp", "observed_logkp"),
        )
        context.predicted_column = find_column(
            context.predictions,
            None,
            PREDICTED_CANDIDATES,
            contains=("predicted_logkp", "prediction_logkp"),
        )

    external_validation_path = first_existing(
        root,
        (
            "external_test_predictions.csv",
            "outputs/external_validation/external_test_predictions.csv",
        ),
    )
    if external_validation_path and external_validation_path != context.prediction_path:
        context.external_validation = load_table(external_validation_path)
    elif context.predictions is not None:
        normalized_columns = {normalize_name(column) for column in context.predictions.columns}
        if {"experimentallogkp", "predictedlogkp"}.issubset(normalized_columns):
            context.external_validation = context.predictions.copy()
    validation_model_path = root / "external_validation_model.joblib"
    if context.external_validation is not None and validation_model_path.exists():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                validation_bundle = joblib.load(validation_model_path)
            if isinstance(validation_bundle, dict):
                protocol = validation_bundle.get("protocol")
                split_strategy = validation_bundle.get("split_strategy")
                if protocol:
                    context.external_validation_label = f"Validation artifact ({protocol})"
                elif split_strategy:
                    context.external_validation_label = f"Validation artifact ({split_strategy} split)"
        except Exception as exc:
            LOGGER.info("Could not read external validation model metadata: %s", exc)
    return context


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return [png_path, pdf_path]


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
    )


def available_property_columns(frame: pd.DataFrame) -> list[tuple[str, str]]:
    matches = []
    used = set()
    for label, aliases in PROPERTY_ALIASES.items():
        column = find_column(frame, None, aliases)
        if column and column not in used:
            converted = pd.to_numeric(frame[column], errors="coerce")
            if converted.notna().sum() >= 8 and converted.nunique(dropna=True) > 1:
                matches.append((label, column))
                used.add(column)
    return matches


def plot_workflow(context: ProjectContext) -> list[Path]:
    if context.skin is None or context.target_column is None:
        raise ValueError("experimental dataset and target column are required")
    steps = []
    steps.append(
        (
            "Experimental data",
            f"{len(context.skin):,} observations\nTarget: {context.target_column}",
            PALETTE["navy"],
        )
    )
    if (context.root / "descriptors-generator").exists():
        steps.append(("Descriptor generation", "Java CDK molecular descriptors", PALETTE["blue"]))
    if context.descriptors is not None:
        steps.append(
            (
                "Cleaning and filtering",
                f"{context.descriptors.shape[1]:,} usable descriptors\nIDs, leakage, constants, duplicates removed",
                PALETTE["teal"],
            )
        )
    grouped = bool(context.metrics_path and "improved" in context.metrics_path.name.lower())
    if grouped and "SMILES" in context.skin.columns:
        steps.append(("Data splitting", "Grouped holdout by SMILES", PALETTE["gold"]))
    else:
        steps.append(("Data splitting", "Repository-configured holdout", PALETTE["gold"]))
    if (context.root / "configs" / "improved.yaml").exists():
        steps.append(("Model development", "Cross-validation and hyperparameter search", PALETTE["orange"]))
    if context.metrics is not None:
        steps.append(
            (
                "Model selection",
                f"{len(context.metrics):,} compared models\nSelected: {context.best_model_name}",
                PALETTE["red"],
            )
        )
    if context.predictions is not None:
        steps.append(("Validation", "Holdout errors, residuals, and uncertainty", PALETTE["blue"]))
    if (context.root / "src" / "skin_permeation" / "analysis" / "applicability.py").exists():
        steps.append(("Interpretation", "Feature importance and applicability domain", PALETTE["teal"]))
    if context.external is not None:
        predicted = find_column(context.external, None, PREDICTED_CANDIDATES, contains=("predicted_logkp",))
        detail = f"{len(context.external):,} external compounds"
        if predicted:
            detail += "\nLogKp predictions available"
        steps.append(("External inference", detail, PALETTE["gold"]))
    if context.external is not None and find_column(context.external, None, ("cluster",)) is not None:
        cluster_column = find_column(context.external, None, ("cluster",))
        count = context.external[cluster_column].nunique(dropna=True)
        steps.append(("Chemical-space analysis", f"PCA and K-means\n{count} observed clusters", PALETTE["orange"]))
    if any((context.root / "data").glob("**/*[Aa][Tt][Cc]*")):
        steps.append(("Therapeutic grouping", "ATC-based comparisons", PALETTE["red"]))
    steps.append(("Reporting", "Publication figures and summary manifest", PALETTE["navy"]))

    fig_height = max(9.0, len(steps) * 1.15)
    fig, ax = plt.subplots(figsize=(8.5, fig_height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(steps) + 0.4)
    ax.axis("off")
    y_positions = np.linspace(len(steps) - 0.35, 0.65, len(steps))
    for index, ((title, detail, color), y) in enumerate(zip(steps, y_positions)):
        box = FancyBboxPatch(
            (0.16, y - 0.37),
            0.68,
            0.72,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            facecolor="white",
            edgecolor=color,
            linewidth=1.8,
        )
        ax.add_patch(box)
        ax.add_patch(
            FancyBboxPatch(
                (0.16, y - 0.37),
                0.18,
                0.72,
                boxstyle="round,pad=0.018,rounding_size=0.025",
                facecolor=color,
                edgecolor=color,
                linewidth=1.0,
            )
        )
        ax.text(0.25, y, title, color="white", ha="center", va="center", fontweight="bold", fontsize=9)
        ax.text(0.37, y, detail, color="#1F2937", ha="left", va="center", fontsize=9.2)
        ax.text(0.11, y, str(index + 1), ha="center", va="center", color=color, fontsize=11, fontweight="bold")
        if index < len(steps) - 1:
            next_y = y_positions[index + 1]
            ax.add_patch(
                FancyArrowPatch(
                    (0.5, y - 0.39),
                    (0.5, next_y + 0.39),
                    arrowstyle="-|>",
                    mutation_scale=12,
                    color="#6B7280",
                    linewidth=1.1,
                )
            )
    ax.set_title(
        "Current Skin Permeability Modeling Workflow",
        fontsize=15,
        fontweight="bold",
        color=PALETTE["navy"],
        pad=15,
    )
    ax.text(
        0.5,
        0.01,
        "Constructed from current repository data, configuration, code, and generated artifacts",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=PALETTE["gray"],
    )
    return save_figure(fig, context.output_dir, "fig01_project_workflow")


def plot_dataset_characterization(context: ProjectContext) -> list[Path]:
    if context.characterization is None or context.target_column is None:
        raise ValueError("characterization data and a target column are required")
    frame = context.characterization.copy()
    target = context.target_column
    frame[target] = pd.to_numeric(frame[target], errors="coerce")
    properties = available_property_columns(frame)[:5]
    if not properties:
        raise ValueError("no recognized molecular property columns were found")
    panel_count = 1 + len(properties)
    columns = 3
    rows = math.ceil(panel_count / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(13, 4.0 * rows))
    axes = np.atleast_1d(axes).ravel()

    sns.histplot(frame[target].dropna(), kde=True, color=PALETTE["blue"], ax=axes[0])
    axes[0].axvline(frame[target].median(), color=PALETTE["red"], linestyle="--", linewidth=1.2)
    axes[0].set_title(f"Experimental {target} distribution")
    axes[0].set_xlabel(target)
    axes[0].set_ylabel("Count")
    add_panel_label(axes[0], "A")

    for index, (label, column) in enumerate(properties, start=1):
        plot_frame = frame[[column, target]].apply(pd.to_numeric, errors="coerce").dropna()
        sns.regplot(
            data=plot_frame,
            x=column,
            y=target,
            scatter_kws={"s": 22, "alpha": 0.58, "color": PALETTE["blue"], "edgecolor": "none"},
            line_kws={"color": PALETTE["red"], "linewidth": 1.4},
            ax=axes[index],
        )
        axes[index].set_title(f"{label} vs {target}")
        axes[index].set_xlabel(label)
        axes[index].set_ylabel(target)
        add_panel_label(axes[index], chr(65 + index))

    for ax in axes[panel_count:]:
        ax.axis("off")
    fig.suptitle(
        f"Dataset Characterization ({len(frame):,} project observations)",
        fontsize=14,
        fontweight="bold",
        color=PALETTE["navy"],
        y=1.01,
    )
    fig.tight_layout()
    return save_figure(fig, context.output_dir, "fig02_dataset_characterization")


def target_correlations(context: ProjectContext) -> pd.Series:
    if context.descriptors is None or context.skin is None or context.target_column is None:
        raise ValueError("descriptor matrix and target data are required")
    target = pd.to_numeric(context.skin[context.target_column], errors="coerce")
    if len(target) != len(context.descriptors):
        raise ValueError("descriptor matrix and target vector have different row counts")
    combined = context.descriptors.copy()
    combined["__target__"] = target.to_numpy()
    return combined.corr(numeric_only=True)["__target__"].drop("__target__").dropna()


def plot_descriptor_correlation(context: ProjectContext) -> list[Path]:
    if context.descriptors is None:
        raise ValueError("no cleaned descriptor matrix is available")
    correlations = target_correlations(context).abs().sort_values(ascending=False)
    display_count = min(max(context.top_n_features * 2, 12), 30, len(correlations))
    if display_count < 3:
        raise ValueError("at least three varying numeric descriptors are required")
    selected = correlations.head(display_count).index.tolist()
    corr = context.descriptors[selected].corr()
    size = max(9, min(15, display_count * 0.43))
    fig, ax = plt.subplots(figsize=(size, size * 0.88))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        mask=mask,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.25,
        cbar_kws={"label": "Pearson correlation", "shrink": 0.78},
        ax=ax,
    )
    ax.set_title(
        f"Descriptor Correlation Heatmap\nTop {display_count} descriptors by association with {context.target_column}",
        pad=12,
    )
    ax.tick_params(axis="x", rotation=65, labelsize=7.5)
    ax.tick_params(axis="y", rotation=0, labelsize=7.5)
    fig.tight_layout()
    return save_figure(fig, context.output_dir, "fig03_descriptor_correlation_heatmap")


def choose_relationship_features(context: ProjectContext, limit: int = 6) -> list[str]:
    if context.descriptors is None:
        return []
    selected = []
    for aliases in PROPERTY_ALIASES.values():
        match = find_column(context.descriptors, None, aliases)
        if match and match not in selected:
            selected.append(match)
        if len(selected) >= limit:
            return selected
    for column in target_correlations(context).abs().sort_values(ascending=False).index:
        if column not in selected:
            selected.append(str(column))
        if len(selected) >= limit:
            break
    return selected


def plot_key_relationships(context: ProjectContext) -> list[Path]:
    if context.descriptors is None or context.skin is None or context.target_column is None:
        raise ValueError("descriptor matrix and target data are required")
    selected = choose_relationship_features(context)
    if len(selected) < 2:
        raise ValueError("fewer than two suitable descriptors were found")
    columns = 3
    rows = math.ceil(len(selected) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(13, 3.9 * rows))
    axes = np.atleast_1d(axes).ravel()
    target = pd.to_numeric(context.skin[context.target_column], errors="coerce").reset_index(drop=True)
    for index, (ax, column) in enumerate(zip(axes, selected)):
        values = pd.to_numeric(context.descriptors[column], errors="coerce").reset_index(drop=True)
        plot_frame = pd.DataFrame({"x": values, "y": target}).dropna()
        if len(plot_frame) < 3:
            ax.axis("off")
            continue
        pearson = pearsonr(plot_frame["x"], plot_frame["y"]).statistic
        spearman = spearmanr(plot_frame["x"], plot_frame["y"]).statistic
        sns.regplot(
            data=plot_frame,
            x="x",
            y="y",
            scatter_kws={"s": 24, "alpha": 0.6, "color": PALETTE["teal"], "edgecolor": "none"},
            line_kws={"color": PALETTE["red"], "linewidth": 1.5},
            ax=ax,
        )
        ax.set_title(column)
        ax.set_xlabel(column)
        ax.set_ylabel(context.target_column)
        ax.text(
            0.04,
            0.96,
            f"Pearson r = {pearson:.2f}\nSpearman rho = {spearman:.2f}",
            transform=ax.transAxes,
            va="top",
            fontsize=8.2,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#CBD5E1"},
        )
        add_panel_label(ax, chr(65 + index))
    for ax in axes[len(selected):]:
        ax.axis("off")
    fig.suptitle(
        f"Key Descriptor Relationships with Experimental {context.target_column}",
        fontsize=14,
        fontweight="bold",
        color=PALETTE["navy"],
        y=1.01,
    )
    fig.tight_layout()
    return save_figure(fig, context.output_dir, "fig04_key_descriptor_logkp_relationships")


def plot_model_performance(context: ProjectContext) -> list[Path]:
    if context.metrics is None:
        raise ValueError("no model metrics table was detected")
    metrics = context.metrics.copy()
    metric_columns = [column for column in ("r2", "rmse", "mae", "cv_rmse", "cv_mae", "mse") if column in metrics.columns]
    if not metric_columns:
        raise ValueError("metrics table contains no supported performance columns")
    sort_column = "mae" if "mae" in metrics.columns else "rmse" if "rmse" in metrics.columns else "r2"
    metrics = metrics.sort_values(sort_column, ascending=sort_column != "r2").reset_index(drop=True)
    panel_columns = 2
    panel_rows = math.ceil(len(metric_columns) / panel_columns)
    fig, axes = plt.subplots(panel_rows, panel_columns, figsize=(13, max(5.5, 3.9 * panel_rows)))
    axes = np.atleast_1d(axes).ravel()
    colors = [PALETTE["red"] if model == context.best_model_name else PALETTE["blue"] for model in metrics["model"]]

    for index, (ax, metric) in enumerate(zip(axes, metric_columns)):
        values = pd.to_numeric(metrics[metric], errors="coerce")
        positions = np.arange(len(metrics))
        ax.barh(positions, values, color=colors, alpha=0.88)
        ax.set_yticks(positions, metrics["model"])
        ax.invert_yaxis()
        ax.set_xlabel(metric.upper().replace("_", " "))
        ax.set_title(metric.upper().replace("_", " "))
        finite_positive = values[np.isfinite(values) & (values > 0)]
        if metric != "r2" and len(finite_positive) >= 2 and finite_positive.max() / finite_positive.min() > 10:
            ax.set_xscale("log")
        if metric == "r2" and values.min(skipna=True) < -5:
            ax.set_xscale("symlog", linthresh=1)
            ax.axvline(0, color="#374151", linewidth=0.8)
        for position, value in zip(positions, values):
            if np.isfinite(value):
                ax.text(
                    value,
                    position,
                    f" {value:.3g}",
                    va="center",
                    ha="left",
                    fontsize=7.5,
                    color="#111827",
                )
        add_panel_label(ax, chr(65 + index))

    for ax in axes[len(metric_columns):]:
        ax.axis("off")
    source = context.metrics_path.name if context.metrics_path else "detected metrics"
    fig.suptitle(
        f"Model Performance Comparison\nSource: {source}; selected model highlighted",
        fontsize=14,
        fontweight="bold",
        color=PALETTE["navy"],
        y=1.01,
    )
    fig.tight_layout()
    return save_figure(fig, context.output_dir, "fig05_model_performance_comparison")


def reconstruct_project_split(context: ProjectContext) -> None:
    if context.skin is None or context.target_column is None or context.modeling_descriptors is None:
        raise ValueError("modeling data is unavailable")
    if len(context.skin) != len(context.modeling_descriptors):
        raise ValueError("modeling target and descriptor matrix row counts do not match")
    config = {}
    grouped = False
    if context.metrics_path and "improved" in context.metrics_path.name.lower():
        config = load_yaml(context.root / "configs" / "improved.yaml")
        grouped = "SMILES" in context.skin.columns
    elif context.metrics_path and "benchmark" in context.metrics_path.name.lower():
        config = load_yaml(context.root / "configs" / "benchmark.yaml")
    elif context.metrics_path and "baseline" in context.metrics_path.name.lower():
        config = load_yaml(context.root / "configs" / "paper_baseline.yaml")
    test_size = float(config.get("test_size", 0.20))
    random_state = int(config.get("random_state", context.random_state))
    indices = np.arange(len(context.skin))
    if grouped:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_index, test_index = next(splitter.split(indices, groups=context.skin["SMILES"].astype(str)))
    else:
        train_index, test_index = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
        )
    target = pd.to_numeric(context.skin[context.target_column], errors="coerce")
    context.x_train = context.modeling_descriptors.iloc[train_index]
    context.x_test = context.modeling_descriptors.iloc[test_index]
    context.y_train = target.iloc[train_index]
    context.y_test = target.iloc[test_index]


def load_models_and_predictions(context: ProjectContext) -> None:
    if context.metrics is None:
        return
    reconstruct_project_split(context)
    if context.x_test is None or context.y_test is None:
        return
    ranking_column = "mae" if "mae" in context.metrics.columns else "rmse" if "rmse" in context.metrics.columns else "r2"
    top = context.metrics.sort_values(ranking_column, ascending=ranking_column != "r2").head(3)
    src_path = context.root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    for _, row in top.iterrows():
        model_name = str(row["model"])
        model_path = model_path_from_row(context.root, context.model_dir, row)
        if model_path is None:
            LOGGER.info("No artifact found for top model %s", model_name)
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = joblib.load(model_path)
                predictions = np.asarray(model.predict(context.x_test), dtype=float).reshape(-1)
            context.model_prediction_sets[model_name] = (
                context.y_test.to_numpy(dtype=float),
                predictions,
            )
            if model_name == context.best_model_name:
                context.best_model = model
                context.best_model_path = model_path
        except Exception as exc:
            LOGGER.warning("Could not load/predict with %s from %s: %s", model_name, model_path, exc)

    if (
        context.predictions is not None
        and context.actual_column is not None
        and context.predicted_column is not None
        and context.best_model_name
    ):
        valid = context.predictions[[context.actual_column, context.predicted_column]].apply(
            pd.to_numeric, errors="coerce"
        ).dropna()
        if len(valid) >= 3:
            context.model_prediction_sets[context.best_model_name] = (
                valid[context.actual_column].to_numpy(),
                valid[context.predicted_column].to_numpy(),
            )


def regression_annotation(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    return (
        f"R2 = {r2_score(y_true, y_pred):.3f}\n"
        f"RMSE = {np.sqrt(mean_squared_error(y_true, y_pred)):.3f}\n"
        f"MAE = {mean_absolute_error(y_true, y_pred):.3f}"
    )


def draw_predicted_actual(
    ax: plt.Axes,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    color: str,
) -> None:
    ax.scatter(y_true, y_pred, s=30, alpha=0.72, color=color, edgecolor="white", linewidth=0.35)
    lower = float(np.nanmin([np.nanmin(y_true), np.nanmin(y_pred)]))
    upper = float(np.nanmax([np.nanmax(y_true), np.nanmax(y_pred)]))
    margin = max((upper - lower) * 0.05, 0.1)
    bounds = (lower - margin, upper + margin)
    ax.plot(bounds, bounds, linestyle="--", color="#111827", linewidth=1.1, label="Ideal")
    if len(y_true) >= 3 and np.nanstd(y_true) > 0:
        slope, intercept = np.polyfit(y_true, y_pred, 1)
        line = np.asarray(bounds)
        ax.plot(line, slope * line + intercept, color=PALETTE["red"], linewidth=1.2, label="Fit")
    ax.set_xlim(bounds)
    ax.set_ylim(bounds)
    ax.set_title(title)
    ax.set_xlabel("Actual LogKp")
    ax.set_ylabel("Predicted LogKp")
    ax.text(
        0.04,
        0.96,
        regression_annotation(y_true, y_pred),
        transform=ax.transAxes,
        va="top",
        fontsize=8.2,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#CBD5E1", "alpha": 0.94},
    )


def external_validation_pair(context: ProjectContext) -> tuple[np.ndarray, np.ndarray] | None:
    if context.external_validation is None:
        return None
    actual = find_column(
        context.external_validation,
        None,
        ACTUAL_CANDIDATES,
        contains=("experimental_logkp", "actual_logkp"),
    )
    predicted = find_column(
        context.external_validation,
        None,
        PREDICTED_CANDIDATES,
        contains=("predicted_logkp",),
    )
    if not actual or not predicted:
        return None
    valid = context.external_validation[[actual, predicted]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(valid) < 3:
        return None
    return valid[actual].to_numpy(), valid[predicted].to_numpy()


def plot_predicted_actual(context: ProjectContext) -> list[Path]:
    if not context.model_prediction_sets:
        load_models_and_predictions(context)
    panels = list(context.model_prediction_sets.items())[:3]
    external_pair = external_validation_pair(context)
    if external_pair is not None:
        panels.append((context.external_validation_label, external_pair))
    if not panels:
        raise ValueError("no actual/predicted pairs or usable model artifacts were found")
    columns = 2
    rows = math.ceil(len(panels) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(11, 5.0 * rows))
    axes = np.atleast_1d(axes).ravel()
    colors = [PALETTE["blue"], PALETTE["teal"], PALETTE["orange"], PALETTE["red"]]
    for index, (ax, (name, pair)) in enumerate(zip(axes, panels)):
        y_true, y_pred = pair
        draw_predicted_actual(ax, np.asarray(y_true), np.asarray(y_pred), name, colors[index % len(colors)])
        add_panel_label(ax, chr(65 + index))
    for ax in axes[len(panels):]:
        ax.axis("off")
    fig.suptitle(
        "Predicted vs Actual LogKp",
        fontsize=14,
        fontweight="bold",
        color=PALETTE["navy"],
        y=1.01,
    )
    fig.tight_layout()
    return save_figure(fig, context.output_dir, "fig06_predicted_vs_actual")


def primary_prediction_frame(context: ProjectContext) -> pd.DataFrame:
    if context.predictions is not None and context.actual_column and context.predicted_column:
        frame = context.predictions.copy()
        frame["__actual__"] = pd.to_numeric(frame[context.actual_column], errors="coerce")
        frame["__predicted__"] = pd.to_numeric(frame[context.predicted_column], errors="coerce")
        frame = frame.dropna(subset=["__actual__", "__predicted__"])
        if len(frame) >= 3:
            return frame
    if not context.model_prediction_sets:
        load_models_and_predictions(context)
    if context.best_model_name and context.best_model_name in context.model_prediction_sets:
        y_true, y_pred = context.model_prediction_sets[context.best_model_name]
        return pd.DataFrame({"__actual__": y_true, "__predicted__": y_pred})
    if context.model_prediction_sets:
        y_true, y_pred = next(iter(context.model_prediction_sets.values()))
        return pd.DataFrame({"__actual__": y_true, "__predicted__": y_pred})
    raise ValueError("no primary prediction set is available")


def plot_residual_analysis(context: ProjectContext) -> list[Path]:
    frame = primary_prediction_frame(context)
    frame["__residual__"] = frame["__actual__"] - frame["__predicted__"]
    frame["__absolute_error__"] = frame["__residual__"].abs()
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))

    sns.scatterplot(
        data=frame,
        x="__predicted__",
        y="__residual__",
        color=PALETTE["blue"],
        alpha=0.72,
        s=34,
        ax=axes[0, 0],
    )
    axes[0, 0].axhline(0, color=PALETTE["red"], linestyle="--", linewidth=1.2)
    axes[0, 0].set_title("Residuals vs predicted values")
    axes[0, 0].set_xlabel("Predicted LogKp")
    axes[0, 0].set_ylabel("Residual (actual - predicted)")

    sns.histplot(frame["__residual__"], kde=True, color=PALETTE["teal"], ax=axes[0, 1])
    axes[0, 1].axvline(0, color=PALETTE["red"], linestyle="--", linewidth=1.2)
    axes[0, 1].set_title("Residual distribution")
    axes[0, 1].set_xlabel("Residual")

    sns.scatterplot(
        data=frame,
        x="__actual__",
        y="__absolute_error__",
        color=PALETTE["orange"],
        alpha=0.72,
        s=34,
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("Absolute error across experimental LogKp")
    axes[1, 0].set_xlabel("Experimental LogKp")
    axes[1, 0].set_ylabel("Absolute error")

    property_column = next(
        (
            column
            for aliases in (
                PROPERTY_ALIASES["LogP"],
                PROPERTY_ALIASES["Topological polar surface area"],
                PROPERTY_ALIASES["Molecular weight"],
            )
            for column in [find_column(frame, None, aliases)]
            if column is not None
        ),
        None,
    )
    if property_column:
        property_values = pd.to_numeric(frame[property_column], errors="coerce")
        bands = pd.qcut(property_values, q=4, duplicates="drop")
        box_frame = pd.DataFrame({"Property band": bands.astype(str), "Absolute error": frame["__absolute_error__"]})
        sns.boxplot(data=box_frame, x="Property band", y="Absolute error", color=PALETTE["gold"], ax=axes[1, 1])
        axes[1, 1].set_title(f"Error by {property_column} quartile")
        axes[1, 1].tick_params(axis="x", rotation=25)
    else:
        sorted_errors = np.sort(frame["__absolute_error__"].to_numpy())
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        axes[1, 1].plot(sorted_errors, cumulative, color=PALETTE["red"], linewidth=2)
        axes[1, 1].set_title("Cumulative absolute error")
        axes[1, 1].set_xlabel("Absolute error")
        axes[1, 1].set_ylabel("Cumulative fraction")

    for index, ax in enumerate(axes.ravel()):
        add_panel_label(ax, chr(65 + index))
    fig.suptitle(
        f"Residual and Error Analysis ({context.best_model_name or 'selected model'})",
        fontsize=14,
        fontweight="bold",
        color=PALETTE["navy"],
        y=1.01,
    )
    fig.tight_layout()
    return save_figure(fig, context.output_dir, "fig07_residual_error_analysis")


def unwrap_regression_pipeline(model: Any) -> tuple[Any, Any]:
    fitted = getattr(model, "regressor_", None) or getattr(model, "regressor", None) or model
    if hasattr(fitted, "named_steps"):
        step_names = list(fitted.named_steps)
        estimator = fitted.named_steps[step_names[-1]]
        preprocessor = fitted[:-1] if len(step_names) > 1 else None
        return estimator, preprocessor
    return fitted, None


def transformed_feature_names(preprocessor: Any, input_features: list[str]) -> list[str]:
    if preprocessor is None:
        return input_features
    try:
        return [str(value) for value in preprocessor.get_feature_names_out(input_features)]
    except Exception:
        names = list(input_features)
        if hasattr(preprocessor, "named_steps"):
            for step in preprocessor.named_steps.values():
                if hasattr(step, "get_support"):
                    support = np.asarray(step.get_support(), dtype=bool)
                    if len(support) == len(names):
                        names = [name for name, keep in zip(names, support) if keep]
                elif hasattr(step, "get_feature_names_out"):
                    try:
                        names = [str(value) for value in step.get_feature_names_out(names)]
                    except Exception:
                        pass
        return names


def plot_feature_importance(context: ProjectContext) -> list[Path]:
    if context.best_model is None or context.x_test is None or context.y_test is None:
        load_models_and_predictions(context)
    if context.best_model is None or context.x_test is None or context.y_test is None:
        raise ValueError("the selected fitted model could not be loaded")
    estimator, preprocessor = unwrap_regression_pipeline(context.best_model)
    transformed = (
        preprocessor.transform(context.x_test)
        if preprocessor is not None
        else context.x_test.to_numpy()
    )
    names = transformed_feature_names(preprocessor, list(context.x_test.columns))
    if hasattr(estimator, "feature_importances_"):
        importance = np.asarray(estimator.feature_importances_, dtype=float)
        method = "Native model importance"
    elif hasattr(estimator, "coef_"):
        importance = np.abs(np.asarray(estimator.coef_, dtype=float).reshape(-1))
        method = "Absolute model coefficient"
    else:
        result = permutation_importance(
            context.best_model,
            context.x_test,
            context.y_test,
            scoring="neg_mean_absolute_error",
            n_repeats=15,
            random_state=context.random_state,
            n_jobs=1,
        )
        importance = result.importances_mean
        names = list(context.x_test.columns)
        method = "Permutation importance"
    if len(importance) != len(names):
        names = [f"Feature {index + 1}" for index in range(len(importance))]
    ranking = pd.DataFrame({"feature": names, "importance": importance})
    ranking = ranking.replace([np.inf, -np.inf], np.nan).dropna().sort_values("importance", ascending=False)
    ranking = ranking.head(context.top_n_features).sort_values("importance")
    if ranking.empty:
        raise ValueError("the selected model exposes no usable feature importance values")

    fig, ax = plt.subplots(figsize=(9, max(5.5, 0.35 * len(ranking) + 1.8)))
    ax.barh(ranking["feature"], ranking["importance"], color=PALETTE["teal"])
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    ax.set_title(
        f"{method}: {context.best_model_name}\nTop {len(ranking)} project descriptors"
    )
    for y_position, value in enumerate(ranking["importance"]):
        ax.text(value, y_position, f" {value:.3g}", va="center", fontsize=7.5)
    fig.tight_layout()
    return save_figure(fig, context.output_dir, "fig08_feature_importance_or_shap")


def common_descriptor_matrix(
    context: ProjectContext,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    if context.skin is None or context.external is None or context.target_column is None:
        raise ValueError("training and external datasets are required")
    training_descriptors, _ = clean_descriptor_matrix(
        context.skin,
        context.target_column,
        detect_identifier_columns(context.skin),
    )
    external_descriptors, _ = clean_descriptor_matrix(
        context.external,
        None,
        detect_identifier_columns(context.external),
    )
    common = [column for column in training_descriptors.columns if column in external_descriptors.columns]
    if len(common) < 3:
        raise ValueError("training and external datasets share fewer than three usable descriptors")
    return training_descriptors[common], external_descriptors[common], common


def compute_joint_pca(context: ProjectContext) -> pd.DataFrame:
    training, external, _ = common_descriptor_matrix(context)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    training_imputed = imputer.fit_transform(training)
    external_imputed = imputer.transform(external)
    training_scaled = scaler.fit_transform(training_imputed)
    external_scaled = scaler.transform(external_imputed)
    pca = PCA(n_components=2, random_state=context.random_state)
    training_projected = pca.fit_transform(training_scaled)
    external_projected = pca.transform(external_scaled)
    combined = pd.DataFrame(
        np.vstack([training_projected, external_projected]),
        columns=["PC1", "PC2"],
    )
    combined["Dataset"] = ["Experimental training"] * len(training) + ["External compounds"] * len(external)
    combined.attrs["explained_variance"] = pca.explained_variance_ratio_
    return combined


def plot_dimensionality_reduction(context: ProjectContext) -> list[Path]:
    combined = compute_joint_pca(context)
    variance = combined.attrs["explained_variance"]
    fig, ax = plt.subplots(figsize=(9, 7))
    external_points = combined["Dataset"] == "External compounds"
    ax.scatter(
        combined.loc[external_points, "PC1"],
        combined.loc[external_points, "PC2"],
        s=18,
        alpha=0.28,
        color=PALETTE["orange"],
        label=f"External compounds (n={external_points.sum():,})",
        edgecolor="none",
    )
    ax.scatter(
        combined.loc[~external_points, "PC1"],
        combined.loc[~external_points, "PC2"],
        s=28,
        alpha=0.78,
        color=PALETTE["blue"],
        label=f"Experimental training (n={(~external_points).sum():,})",
        edgecolor="white",
        linewidth=0.3,
    )
    ax.set_xlabel(f"PC1 ({variance[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({variance[1] * 100:.1f}% variance)")
    ax.set_title("PCA Projection of Experimental and External Chemical Space")
    ax.legend(loc="upper left")

    x_lower, x_upper = combined["PC1"].quantile([0.005, 0.995])
    y_lower, y_upper = combined["PC2"].quantile([0.005, 0.995])
    x_margin = max((x_upper - x_lower) * 0.08, 0.5)
    y_margin = max((y_upper - y_lower) * 0.08, 0.5)
    x_bounds = (float(x_lower - x_margin), float(x_upper + x_margin))
    y_bounds = (float(y_lower - y_margin), float(y_upper + y_margin))
    clipped = (
        (combined["PC1"] < x_bounds[0])
        | (combined["PC1"] > x_bounds[1])
        | (combined["PC2"] < y_bounds[0])
        | (combined["PC2"] > y_bounds[1])
    )
    ax.set_xlim(x_bounds)
    ax.set_ylim(y_bounds)
    if clipped.any():
        ax.text(
            0.02,
            0.02,
            f"Main view uses 0.5th-99.5th percentile bounds\n{int(clipped.sum())} points shown in full-range inset",
            transform=ax.transAxes,
            fontsize=7.8,
            va="bottom",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#CBD5E1", "alpha": 0.92},
        )
        inset = inset_axes(ax, width="32%", height="32%", loc="lower right", borderpad=1.1)
        inset.scatter(
            combined.loc[external_points, "PC1"],
            combined.loc[external_points, "PC2"],
            s=5,
            alpha=0.24,
            color=PALETTE["orange"],
            edgecolor="none",
        )
        inset.scatter(
            combined.loc[~external_points, "PC1"],
            combined.loc[~external_points, "PC2"],
            s=7,
            alpha=0.65,
            color=PALETTE["blue"],
            edgecolor="none",
        )
        inset.set_title("Full range", fontsize=7)
        inset.tick_params(labelsize=5.5)
        inset.grid(True, linewidth=0.3)
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.11, top=0.91)
    return save_figure(fig, context.output_dir, "fig09_dimensionality_reduction")


def cluster_external_if_needed(context: ProjectContext) -> tuple[pd.DataFrame, str, str, str]:
    if context.external is None:
        raise ValueError("no external compound dataset was found")
    frame = context.external.copy()
    cluster_column = find_column(frame, None, ("cluster",))
    pca1 = find_column(frame, None, ("PCA1", "PC1"))
    pca2 = find_column(frame, None, ("PCA2", "PC2"))
    if cluster_column and pca1 and pca2:
        return frame, cluster_column, pca1, pca2
    descriptors, _ = clean_descriptor_matrix(frame, None, detect_identifier_columns(frame))
    if descriptors.shape[1] < 3:
        raise ValueError("external data has fewer than three usable descriptors for clustering")
    imputed = SimpleImputer(strategy="median").fit_transform(descriptors)
    scaled = StandardScaler().fit_transform(imputed)
    cluster_count = context.requested_clusters or 4
    if cluster_count < 2 or cluster_count >= len(frame):
        raise ValueError("--n-clusters must be at least 2 and less than the number of external compounds")
    frame["generated_cluster"] = KMeans(
        n_clusters=cluster_count,
        n_init=20,
        random_state=context.random_state,
    ).fit_predict(scaled)
    projected = PCA(n_components=2, random_state=context.random_state).fit_transform(scaled)
    frame["generated_PCA1"] = projected[:, 0]
    frame["generated_PCA2"] = projected[:, 1]
    return frame, "generated_cluster", "generated_PCA1", "generated_PCA2"


def plot_clustering(context: ProjectContext) -> list[Path]:
    frame, cluster_column, pca1, pca2 = cluster_external_if_needed(context)
    selection_path = first_existing(
        context.root,
        ("reports/tables/drugbank_cluster_selection.csv", "reports/tables/cluster_selection.csv"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    if selection_path:
        selection = pd.read_csv(selection_path)
        if {"k", "inertia"}.issubset(selection.columns):
            axes[0].plot(selection["k"], selection["inertia"], marker="o", color=PALETTE["blue"], label="Inertia")
            axes[0].set_xlabel("Number of clusters (k)")
            axes[0].set_ylabel("Inertia", color=PALETTE["blue"])
            axes[0].tick_params(axis="y", labelcolor=PALETTE["blue"])
            if "silhouette" in selection.columns:
                twin = axes[0].twinx()
                twin.plot(
                    selection["k"],
                    selection["silhouette"],
                    marker="s",
                    color=PALETTE["red"],
                    label="Silhouette",
                )
                twin.set_ylabel("Silhouette score", color=PALETTE["red"])
                twin.tick_params(axis="y", labelcolor=PALETTE["red"])
            axes[0].set_title("Cluster-count diagnostics")
        else:
            axes[0].axis("off")
    else:
        counts = frame[cluster_column].value_counts().sort_index()
        axes[0].bar(counts.index.astype(str), counts.values, color=PALETTE["blue"])
        axes[0].set_title("External compound count by cluster")
        axes[0].set_xlabel("Cluster")
        axes[0].set_ylabel("Count")

    sns.scatterplot(
        data=frame,
        x=pca1,
        y=pca2,
        hue=cluster_column,
        palette="tab10",
        alpha=0.68,
        s=28,
        linewidth=0,
        ax=axes[1],
    )
    axes[1].set_title(
        f"External chemical-space clusters (k={frame[cluster_column].nunique(dropna=True)})"
    )
    axes[1].set_xlabel(pca1)
    axes[1].set_ylabel(pca2)
    axes[1].legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    add_panel_label(axes[0], "A")
    add_panel_label(axes[1], "B")
    fig.suptitle(
        "Clustering Analysis of External Compounds",
        fontsize=14,
        fontweight="bold",
        color=PALETTE["navy"],
        y=1.02,
    )
    fig.tight_layout()
    return save_figure(fig, context.output_dir, "fig10_clustering_analysis")


def plot_cluster_properties(context: ProjectContext) -> list[Path]:
    frame, cluster_column, _, _ = cluster_external_if_needed(context)
    properties = available_property_columns(frame)
    predicted = find_column(frame, None, PREDICTED_CANDIDATES, contains=("predicted_logkp",))
    selected = []
    if predicted:
        selected.append(("Predicted LogKp", predicted))
    selected.extend(properties)
    deduplicated = []
    used = set()
    for item in selected:
        if item[1] not in used:
            deduplicated.append(item)
            used.add(item[1])
    selected = deduplicated[:6]
    if len(selected) < 2:
        raise ValueError("fewer than two cluster-comparison properties are available")
    columns = 3
    rows = math.ceil(len(selected) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(13, 3.9 * rows))
    axes = np.atleast_1d(axes).ravel()
    order = sorted(frame[cluster_column].dropna().unique())
    for index, (ax, (label, column)) in enumerate(zip(axes, selected)):
        plot_frame = frame[[cluster_column, column]].copy()
        plot_frame[column] = pd.to_numeric(plot_frame[column], errors="coerce")
        plot_frame = plot_frame.dropna()
        sns.boxplot(
            data=plot_frame,
            x=cluster_column,
            y=column,
            order=order,
            color=PALETTE["light"],
            linecolor=PALETTE["navy"],
            fliersize=1.5,
            ax=ax,
        )
        ax.set_title(label)
        ax.set_xlabel("Cluster")
        ax.set_ylabel(label)
        add_panel_label(ax, chr(65 + index))
    for ax in axes[len(selected):]:
        ax.axis("off")
    fig.suptitle(
        "Molecular Property Comparison Across External Clusters",
        fontsize=14,
        fontweight="bold",
        color=PALETTE["navy"],
        y=1.01,
    )
    fig.tight_layout()
    return save_figure(fig, context.output_dir, "fig11_cluster_property_comparison")


def choose_group_and_value(frame: pd.DataFrame) -> tuple[str, str]:
    predicted = find_column(frame, None, PREDICTED_CANDIDATES, contains=("predicted_logkp",))
    target = predicted or find_column(frame, None, TARGET_CANDIDATES, contains=("logkp",))
    if target is None:
        raise ValueError("no predicted or experimental LogKp column is available")
    normalized_columns = {normalize_name(column): str(column) for column in frame.columns}
    preferred = (
        "ATC_group",
        "ATC",
        "therapeutic_group",
        "drug_class",
        "cluster",
        "category",
        "source",
        "group",
    )
    group = next((normalized_columns.get(normalize_name(candidate)) for candidate in preferred if normalized_columns.get(normalize_name(candidate))), None)
    if group is None:
        group_candidates = [
            str(column)
            for column in frame.columns
            if any(normalize_name(marker) in normalize_name(column) for marker in GROUP_MARKERS)
            and str(column) != target
        ]
        group = group_candidates[0] if group_candidates else None
    if group is None:
        raise ValueError("no ATC, class, category, source, group, or cluster column is available")
    return group, target


def plot_group_comparison(context: ProjectContext) -> list[Path]:
    if context.external is None:
        raise ValueError("no external prediction dataset was found")
    frame = context.external.copy()
    group, value = choose_group_and_value(frame)
    frame[value] = pd.to_numeric(frame[value], errors="coerce")
    frame = frame.dropna(subset=[group, value])
    counts = frame[group].value_counts()
    valid_groups = counts[counts >= 3].index
    frame = frame[frame[group].isin(valid_groups)]
    if frame[group].nunique() < 2:
        raise ValueError(f"fewer than two {group} groups contain at least three LogKp values")
    medians = frame.groupby(group)[value].median().sort_values(ascending=False)
    width = max(9, min(18, 0.55 * len(medians) + 5))
    fig, ax = plt.subplots(figsize=(width, 6))
    sns.boxplot(
        data=frame,
        x=group,
        y=value,
        order=medians.index,
        color=PALETTE["light"],
        linecolor=PALETTE["navy"],
        fliersize=1.5,
        ax=ax,
    )
    sns.stripplot(
        data=frame,
        x=group,
        y=value,
        order=medians.index,
        color=PALETTE["blue"],
        alpha=0.28,
        size=2,
        jitter=0.25,
        ax=ax,
    )
    ax.set_title(f"{value} by {group}")
    ax.set_xlabel(group)
    ax.set_ylabel(value)
    ax.tick_params(axis="x", rotation=60 if len(medians) > 8 else 0)
    fig.tight_layout()
    return save_figure(fig, context.output_dir, "fig12_predicted_logkp_group_comparison")


def adjust_pvalues(pvalues: list[float]) -> np.ndarray:
    try:
        from statsmodels.stats.multitest import multipletests

        return np.asarray(multipletests(pvalues, method="fdr_bh")[1], dtype=float)
    except Exception:
        count = max(len(pvalues), 1)
        return np.minimum(np.asarray(pvalues, dtype=float) * count, 1.0)


def pairwise_group_tests(frame: pd.DataFrame, group: str, value: str) -> pd.DataFrame:
    valid = frame[[group, value]].copy()
    valid[value] = pd.to_numeric(valid[value], errors="coerce")
    valid = valid.dropna()
    counts = valid[group].value_counts()
    groups = sorted(counts[counts >= 3].index, key=str)
    if len(groups) < 2:
        raise ValueError("pairwise testing requires at least two groups with three observations each")
    if len(groups) > 30:
        groups = sorted(groups, key=lambda item: counts[item], reverse=True)[:30]
    rows = []
    for left_index, left in enumerate(groups):
        left_values = valid.loc[valid[group] == left, value].to_numpy()
        for right in groups[left_index + 1 :]:
            right_values = valid.loc[valid[group] == right, value].to_numpy()
            statistic, pvalue = mannwhitneyu(left_values, right_values, alternative="two-sided")
            rows.append(
                {
                    "left": str(left),
                    "right": str(right),
                    "statistic": float(statistic),
                    "p_value": float(pvalue),
                }
            )
    output = pd.DataFrame(rows)
    output["adjusted_p_value"] = adjust_pvalues(output["p_value"].tolist())
    return output


def plot_statistical_heatmap(context: ProjectContext) -> list[Path]:
    if context.external is None:
        raise ValueError("no external prediction dataset was found")
    group, value = choose_group_and_value(context.external)
    tests = pairwise_group_tests(context.external, group, value)
    groups = sorted(set(tests["left"]).union(tests["right"]))
    matrix = pd.DataFrame(np.nan, index=groups, columns=groups)
    np.fill_diagonal(matrix.values, 0.0)
    for row in tests.itertuples():
        transformed = -math.log10(max(float(row.adjusted_p_value), np.finfo(float).tiny))
        matrix.loc[row.left, row.right] = transformed
        matrix.loc[row.right, row.left] = transformed
    size = max(6.5, min(16, len(groups) * 0.55 + 4))
    fig, ax = plt.subplots(figsize=(size, size * 0.88))
    sns.heatmap(
        matrix,
        cmap="mako",
        square=True,
        linewidths=0.35,
        cbar_kws={"label": "-log10(FDR-adjusted p-value)", "shrink": 0.8},
        ax=ax,
    )
    ax.set_title(f"Pairwise Mann-Whitney Comparisons of {value} by {group}")
    ax.tick_params(axis="x", rotation=60)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    return save_figure(fig, context.output_dir, "fig13_group_statistical_heatmap")


def plot_applicability_domain(context: ProjectContext) -> list[Path]:
    if context.external is None:
        raise ValueError("no external compound dataset was found")
    distance = find_column(context.external, None, ("mean_knn_distance", "knn_distance"))
    threshold = find_column(context.external, None, ("applicability_threshold", "ad_threshold"))
    within = find_column(context.external, None, ("within_domain", "in_domain"))
    predicted = find_column(context.external, None, PREDICTED_CANDIDATES, contains=("predicted_logkp",))
    if not distance or not threshold or not within:
        raise ValueError("external data lacks KNN distance, applicability threshold, or within-domain flags")
    frame = context.external.copy()
    frame[distance] = pd.to_numeric(frame[distance], errors="coerce")
    frame[threshold] = pd.to_numeric(frame[threshold], errors="coerce")
    frame = frame.dropna(subset=[distance, threshold, within])
    if frame.empty:
        raise ValueError("no complete applicability-domain rows are available")
    threshold_value = float(frame[threshold].median())
    distance_plot_column = "__log_distance__"
    frame[distance_plot_column] = np.log10(1.0 + frame[distance].clip(lower=0))
    threshold_plot_value = math.log10(1.0 + max(threshold_value, 0))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(
        data=frame,
        x=distance_plot_column,
        hue=within,
        bins=35,
        element="step",
        stat="density",
        common_norm=False,
        palette={True: PALETTE["teal"], False: PALETTE["red"]},
        ax=axes[0],
    )
    axes[0].axvline(threshold_plot_value, color="#111827", linestyle="--", linewidth=1.3, label="AD threshold")
    axes[0].set_title("KNN applicability-domain distance")
    axes[0].set_xlabel("log10(1 + mean KNN distance)")

    if predicted:
        frame[predicted] = pd.to_numeric(frame[predicted], errors="coerce")
        sns.scatterplot(
            data=frame,
            x=distance_plot_column,
            y=predicted,
            hue=within,
            palette={True: PALETTE["teal"], False: PALETTE["red"]},
            alpha=0.6,
            s=24,
            linewidth=0,
            ax=axes[1],
        )
        axes[1].axvline(threshold_plot_value, color="#111827", linestyle="--", linewidth=1.3)
        axes[1].set_title("Predicted LogKp across applicability distance")
        axes[1].set_xlabel("log10(1 + mean KNN distance)")
        axes[1].set_ylabel(predicted)
    else:
        counts = frame[within].value_counts()
        axes[1].bar(
            ["Within domain", "Outside domain"],
            [counts.get(True, 0), counts.get(False, 0)],
            color=[PALETTE["teal"], PALETTE["red"]],
        )
        axes[1].set_title("Applicability-domain classification")
        axes[1].set_ylabel("External compounds")
    for index, ax in enumerate(axes):
        add_panel_label(ax, chr(65 + index))
    within_fraction = frame[within].astype(bool).mean()
    fig.suptitle(
        f"Applicability-Domain Analysis ({within_fraction:.1%} within domain)",
        fontsize=14,
        fontweight="bold",
        color=PALETTE["navy"],
        y=1.02,
    )
    fig.tight_layout()
    return save_figure(fig, context.output_dir, "fig14_applicability_domain")


def metadata_summary(context: ProjectContext) -> dict[str, Any]:
    def display(path: Path | None) -> str | None:
        if path is None:
            return None
        try:
            return str(path.resolve().relative_to(context.root.resolve()))
        except ValueError:
            return str(path)

    return {
        "project_root": ".",
        "paper_pdf": display(context.paper_pdf),
        "paper_pdf_found": bool(context.paper_pdf and context.paper_pdf.exists()),
        "skin_data": display(context.skin_path),
        "descriptor_data": display(context.descriptor_path),
        "external_data": display(context.external_path),
        "prediction_data": display(context.prediction_path),
        "metrics_data": display(context.metrics_path),
        "model_dir": display(context.model_dir),
        "target_column": context.target_column,
        "identifier_columns": context.identifier_columns,
        "descriptor_count": int(context.descriptors.shape[1]) if context.descriptors is not None else 0,
        "modeling_descriptor_count": (
            int(context.modeling_descriptors.shape[1]) if context.modeling_descriptors is not None else 0
        ),
        "best_model": context.best_model_name,
        "best_model_path": display(context.best_model_path),
        "external_validation_label": context.external_validation_label,
        "descriptor_cleaning": context.descriptor_report,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate publication-quality figures from the current skin-permeability "
            "project data, models, validation outputs, and external predictions."
        )
    )
    parser.add_argument("--paper-pdf", default=None, help="Reference paper PDF; recorded only, never copied or extracted.")
    parser.add_argument("--skin-data", default=None, help="Experimental LogKp dataset (CSV/XLSX).")
    parser.add_argument("--descriptor-data", default=None, help="Optional descriptor table separate from the skin dataset.")
    parser.add_argument("--external-data", default=None, help="External/FDA/DrugBank compound or prediction table.")
    parser.add_argument("--prediction-data", default=None, help="Actual/predicted validation table.")
    parser.add_argument("--model-dir", default=None, help="Directory containing fitted project model artifacts.")
    parser.add_argument("--output-dir", default="figures", help="Figure output directory.")
    parser.add_argument("--target-column", default=None, help="Experimental LogKp target column.")
    parser.add_argument("--random-state", type=int, default=2024, help="Reproducible random seed.")
    parser.add_argument("--top-n-features", type=int, default=15, help="Number of features shown in importance plots.")
    parser.add_argument("--n-clusters", type=int, default=None, help="Cluster count used only when cluster labels are absent.")
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    configure_style()
    args = build_parser().parse_args()
    try:
        context = discover_inputs(args)
    except Exception as exc:
        LOGGER.error("Could not initialize figure generation: %s", exc)
        return 2

    if context.paper_pdf and not context.paper_pdf.exists():
        LOGGER.warning(
            "Reference paper PDF was not found at %s. It is optional and is never used as a data source.",
            context.paper_pdf,
        )
    elif context.paper_pdf:
        LOGGER.info("Reference paper recorded as %s; no paper images or values will be extracted.", context.paper_pdf)
    else:
        LOGGER.info("No --paper-pdf supplied. The paper is not required to generate project figures.")

    try:
        load_models_and_predictions(context)
    except Exception as exc:
        LOGGER.warning("Model artifact integration is unavailable: %s", exc)

    registry = FigureRegistry(context.root, context.output_dir)
    registry.generate("fig01_project_workflow", lambda: plot_workflow(context))
    registry.generate("fig02_dataset_characterization", lambda: plot_dataset_characterization(context))
    registry.generate("fig03_descriptor_correlation_heatmap", lambda: plot_descriptor_correlation(context))
    registry.generate("fig04_key_descriptor_logkp_relationships", lambda: plot_key_relationships(context))
    registry.generate("fig05_model_performance_comparison", lambda: plot_model_performance(context))
    registry.generate("fig06_predicted_vs_actual", lambda: plot_predicted_actual(context))
    registry.generate("fig07_residual_error_analysis", lambda: plot_residual_analysis(context))
    registry.generate("fig08_feature_importance_or_shap", lambda: plot_feature_importance(context))
    registry.generate("fig09_dimensionality_reduction", lambda: plot_dimensionality_reduction(context))
    registry.generate("fig10_clustering_analysis", lambda: plot_clustering(context))
    registry.generate("fig11_cluster_property_comparison", lambda: plot_cluster_properties(context))
    registry.generate("fig12_predicted_logkp_group_comparison", lambda: plot_group_comparison(context))
    registry.generate("fig13_group_statistical_heatmap", lambda: plot_statistical_heatmap(context))
    registry.generate("fig14_applicability_domain", lambda: plot_applicability_domain(context))

    manifest_path = registry.write_manifest(metadata_summary(context))
    print("\nFigure generation summary")
    print("=" * 25)
    print(f"Output directory: {registry.display_path(context.output_dir)}")
    print(f"Manifest: {registry.display_path(manifest_path)}")
    print(f"Generated figures: {len(registry.generated)}")
    for key, paths in registry.generated.items():
        print(f"  GENERATED {key}: {', '.join(paths)}")
    print(f"Skipped figures: {len(registry.skipped)}")
    for key, reason in registry.skipped.items():
        print(f"  SKIPPED {key}: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
