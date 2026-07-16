from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .data_loading import load_bundle
from .dependencies import require_module
from .formulas import calculate_formula_logkp
from .paths import ProjectPaths
from .preprocessing import apply_imputation_values, compute_imputation_values

LOGGER = logging.getLogger(__name__)

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "skin-permeation-mpl"))

FEATURE_EXCLUDE_COLUMNS = {"logkpl", "Compound", "SMILES"}
DEFAULT_TEXPI = 310.0
DEFAULT_PLOT_OUTPUT = Path("figures/predictions/predicted_vs_actual_logkp.png")
DEFAULT_PREDICTIONS_OUTPUT = Path("reports/tables/unknown_compound_model_predictions.csv")
DEFAULT_MAVEN_REPOSITORY = Path(".cache/maven-repo")
DEFAULT_MAVEN_EXECUTABLE = os.environ.get("SKIN_PERMEATION_MAVEN_EXECUTABLE", "mvn")

RDKIT_HELPER = """
import json
import sys

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

payload = json.loads(sys.stdin.read())
smiles = payload["smiles"]
mol = Chem.MolFromSmiles(smiles)
if mol is None:
    sys.stderr.write("Invalid SMILES string.\\n")
    raise SystemExit(2)

canonical_smiles = Chem.MolToSmiles(mol)
logp = float(Crippen.MolLogP(mol))
mol_weight = float(Descriptors.MolWt(mol))
summary = {
    "MolWt": mol_weight,
    "MolLogP": logp,
    "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
    "NumHAcceptors": float(Lipinski.NumHAcceptors(mol)),
    "NumHDonors": float(Lipinski.NumHDonors(mol)),
    "NumRotatableBonds": float(Lipinski.NumRotatableBonds(mol)),
    "RingCount": float(rdMolDescriptors.CalcNumRings(mol)),
}
sys.stdout.write(
    json.dumps(
        {
            "input_smiles": smiles,
            "canonical_smiles": canonical_smiles,
            "logp": logp,
            "molecular_weight": mol_weight,
            "descriptor_summary": summary,
        }
    )
)
""".strip()

try:
    from sklearn.exceptions import InconsistentVersionWarning
except Exception:  # pragma: no cover - sklearn versions vary between environments.
    InconsistentVersionWarning = Warning


@dataclass(frozen=True)
class RDKitComputation:
    input_smiles: str
    canonical_smiles: str
    logp: float
    molecular_weight: float
    descriptor_summary: dict[str, float]


@dataclass(frozen=True)
class ModelArtifactSpec:
    name: str
    artifact_path: Path
    scaler_path: Path | None = None
    is_primary: bool = False
    artifact_type: str = "joblib"


@dataclass
class LoadedPredictionModel:
    name: str
    artifact_path: Path
    estimator: Any
    scaler: Any | None
    feature_order: tuple[str, ...] | None
    selected_columns: tuple[str, ...] | None = None
    is_primary: bool = False
    artifact_type: str = "joblib"


@dataclass
class PredictionResult:
    input_smiles: str
    canonical_smiles: str
    texpi: float
    rdkit_logp: float
    molecular_weight: float
    formula_logkp: float
    rdkit_descriptor_summary: dict[str, float]
    feature_frame: pd.DataFrame
    raw_descriptor_frame: pd.DataFrame
    predictions: pd.DataFrame
    plot_path: Path
    skipped_models: list[str]

    @property
    def primary_prediction(self) -> pd.Series:
        return self.predictions.loc[self.predictions["is_primary"]].iloc[0]


def _current_python_has_rdkit() -> bool:
    return importlib.util.find_spec("rdkit") is not None


def _rdkit_python_candidates() -> list[Path]:
    candidates: list[Path] = []
    env_override = os.environ.get("SKIN_PERMEATION_RDKIT_PYTHON")
    if env_override:
        candidates.append(Path(env_override))
    if _current_python_has_rdkit():
        candidates.append(Path(sys.executable))
    candidates.extend(
        [
            Path("/opt/anaconda3/envs/rdkit_env/bin/python"),
            Path("/opt/anaconda3/envs/skin-permeation-repro/bin/python"),
            Path("/opt/anaconda3/envs/skin-perm-benchmark/bin/python"),
        ]
    )
    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = str(candidate)
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(candidate)
    return deduped


def _python_can_import_rdkit(python_executable: Path) -> bool:
    if not python_executable.exists():
        return False
    result = subprocess.run(
        [str(python_executable), "-c", "import rdkit"],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def discover_rdkit_python() -> Path:
    for candidate in _rdkit_python_candidates():
        if _python_can_import_rdkit(candidate):
            return candidate
    raise RuntimeError(
        "RDKit is not available in the current interpreter and no fallback RDKit Python was found. "
        "Set SKIN_PERMEATION_RDKIT_PYTHON to a Python executable that can import rdkit."
    )


def _compute_rdkit_properties_locally(smiles: str) -> RDKitComputation:
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string.")
    logp = float(Crippen.MolLogP(mol))
    molecular_weight = float(Descriptors.MolWt(mol))
    descriptor_summary = {
        "MolWt": molecular_weight,
        "MolLogP": logp,
        "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
        "NumHAcceptors": float(Lipinski.NumHAcceptors(mol)),
        "NumHDonors": float(Lipinski.NumHDonors(mol)),
        "NumRotatableBonds": float(Lipinski.NumRotatableBonds(mol)),
        "RingCount": float(rdMolDescriptors.CalcNumRings(mol)),
    }
    return RDKitComputation(
        input_smiles=smiles,
        canonical_smiles=Chem.MolToSmiles(mol),
        logp=logp,
        molecular_weight=molecular_weight,
        descriptor_summary=descriptor_summary,
    )


def _compute_rdkit_properties_via_subprocess(smiles: str, python_executable: Path) -> RDKitComputation:
    result = subprocess.run(
        [str(python_executable), "-c", RDKIT_HELPER],
        input=json.dumps({"smiles": smiles}),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode == 2:
        raise ValueError("Invalid SMILES string.")
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown RDKit error"
        raise RuntimeError(f"RDKit helper failed when using {python_executable}: {detail}")
    payload = json.loads(result.stdout)
    return RDKitComputation(
        input_smiles=payload["input_smiles"],
        canonical_smiles=payload["canonical_smiles"],
        logp=float(payload["logp"]),
        molecular_weight=float(payload["molecular_weight"]),
        descriptor_summary={key: float(value) for key, value in payload["descriptor_summary"].items()},
    )


def compute_rdkit_properties(smiles: str, python_executable: str | None = None) -> RDKitComputation:
    candidate_smiles = smiles.strip()
    if not candidate_smiles:
        raise ValueError("A non-empty SMILES string is required.")
    if python_executable is None and _current_python_has_rdkit():
        return _compute_rdkit_properties_locally(candidate_smiles)
    rdkit_python = Path(python_executable) if python_executable else discover_rdkit_python()
    return _compute_rdkit_properties_via_subprocess(candidate_smiles, rdkit_python)


def _write_single_smiles_data_original(destination: Path, paths: ProjectPaths, smiles: str, texpi: float) -> None:
    header = pd.read_csv(paths.raw_data / "data-original.csv", nrows=0).columns.tolist()
    row = {column: "" for column in header}
    row["No"] = 1
    row["Compound"] = "User Query"
    row["SMILES"] = smiles
    row["set"] = "query"
    row["Texpi"] = float(texpi)
    row["Reference"] = "Generated for single-SMILES inference"
    frame = pd.DataFrame([row], columns=header)
    frame.to_csv(destination, index=False)


def generate_training_compatible_descriptors(
    paths: ProjectPaths,
    smiles: str,
    texpi: float,
    maven_executable: str = DEFAULT_MAVEN_EXECUTABLE,
) -> pd.DataFrame:
    maven_repo = paths.root / DEFAULT_MAVEN_REPOSITORY
    maven_repo.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="single_smiles_descriptor_runtime_") as tmp_dir:
        runtime_dir = Path(tmp_dir) / "descriptors-generator"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(paths.descriptors_generator / "pom.xml", runtime_dir / "pom.xml")
        shutil.copytree(paths.descriptors_generator / "src", runtime_dir / "src")
        _write_single_smiles_data_original(runtime_dir / "data-original.csv", paths, smiles, texpi)

        command = [
            maven_executable,
            "-q",
            "-DskipTests",
            f"-Dmaven.repo.local={maven_repo}",
            "package",
            "org.codehaus.mojo:exec-maven-plugin:3.5.0:java",
            "-Dexec.mainClass=org.rami.dg.DescriptorsGenerator",
        ]
        try:
            result = subprocess.run(
                command,
                cwd=runtime_dir,
                text=True,
                capture_output=True,
                check=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"Maven executable {maven_executable!r} was not found on PATH.") from exc

        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "unknown Maven/CDK error"
            raise RuntimeError(
                "Descriptor generation failed. The first run may need Maven network access to download CDK dependencies. "
                f"Command: {' '.join(command)}\n{detail}"
            )

        descriptor_path = runtime_dir / "data-descriptors.csv"
        if not descriptor_path.exists():
            raise RuntimeError("Descriptor generation completed without producing data-descriptors.csv.")
        descriptors = pd.read_csv(descriptor_path)
        if descriptors.empty:
            raise RuntimeError("Descriptor generation produced an empty descriptor table.")
        return descriptors.head(1).copy()


def feature_columns_from_training_frame(training_frame: pd.DataFrame) -> list[str]:
    return [column for column in training_frame.columns if column not in FEATURE_EXCLUDE_COLUMNS]


def build_inference_fill_values(bundle) -> dict[str, float]:
    fill_values = compute_imputation_values(bundle.trial4)
    clean_features = bundle.clean_trial4[feature_columns_from_training_frame(bundle.clean_trial4)]
    medians = clean_features.median(numeric_only=True).to_dict()
    for column, value in medians.items():
        fill_values.setdefault(column, float(value))
    if "topoShape" in clean_features.columns:
        fill_values["topoShape"] = float(clean_features["topoShape"].mean())
    fill_values.setdefault("Texpi", float(clean_features["Texpi"].median()) if "Texpi" in clean_features.columns else DEFAULT_TEXPI)
    return fill_values


def align_descriptor_frame_to_feature_columns(
    descriptor_frame: pd.DataFrame,
    feature_columns: list[str],
    fill_values: dict[str, float],
    texpi: float,
) -> pd.DataFrame:
    candidate = descriptor_frame.copy()
    candidate["Texpi"] = float(texpi)
    missing_columns = [column for column in feature_columns if column not in candidate.columns]
    if missing_columns:
        raise ValueError(f"Descriptor frame is missing expected training columns: {missing_columns}")

    feature_frame = candidate.reindex(columns=feature_columns).copy()
    for column in feature_frame.columns:
        feature_frame[column] = pd.to_numeric(feature_frame[column], errors="coerce")
    feature_frame = apply_imputation_values(feature_frame, fill_values)

    remaining_missing = feature_frame.columns[feature_frame.isna().any()].tolist()
    if remaining_missing:
        fallback_values = {column: fill_values[column] for column in remaining_missing if column in fill_values}
        feature_frame = feature_frame.fillna(fallback_values)

    unresolved = feature_frame.columns[feature_frame.isna().any()].tolist()
    if unresolved:
        raise ValueError(f"Could not impute all model features for inference. Missing columns: {unresolved}")
    return feature_frame


def _display_model_name(stem: str) -> str:
    friendly = {
        "SVR_RBF": "SVR (RBF)",
        "Gradient_Boosting": "Gradient Boosting",
        "Decision_Tree": "Decision Tree",
        "MLR_10_features": "MLR (10 features)",
        "StackingRegressor": "Stacking Regressor",
        "best_benchmark_model": "Best Benchmark Model",
    }
    return friendly.get(stem, stem.replace("_", " "))


def _include_keras_models() -> bool:
    return os.environ.get("SKIN_PERMEATION_INCLUDE_KERAS", "").strip().lower() in {"1", "true", "yes"}


def _artifact_path_from_metrics(path_value: str, paths: ProjectPaths) -> Path:
    artifact_path = Path(path_value)
    if not artifact_path.is_absolute():
        artifact_path = paths.root / artifact_path
    return artifact_path


def _benchmark_model_specs(paths: ProjectPaths) -> list[ModelArtifactSpec]:
    benchmark_dir = paths.models / "reproduction" / "benchmark"
    metrics_path = paths.reports / "tables" / "benchmark_metrics.csv"
    specs: list[ModelArtifactSpec] = []
    seen_paths: set[Path] = set()
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        for row in metrics.itertuples(index=False):
            artifact_path = _artifact_path_from_metrics(str(row.artifact_path), paths)
            if artifact_path.suffix != ".joblib" or not artifact_path.exists():
                continue
            specs.append(
                ModelArtifactSpec(
                    name=f"Benchmark {row.model}",
                    artifact_path=artifact_path,
                    is_primary=len(specs) == 0,
                )
            )
            seen_paths.add(artifact_path.resolve())

    if benchmark_dir.exists():
        for artifact_path in sorted(benchmark_dir.glob("*.joblib")):
            if artifact_path.name == "best_benchmark_model.joblib":
                continue
            if artifact_path.resolve() in seen_paths:
                continue
            specs.append(
                ModelArtifactSpec(
                    name=f"Benchmark {_display_model_name(artifact_path.stem)}",
                    artifact_path=artifact_path,
                    is_primary=len(specs) == 0,
                )
            )
            seen_paths.add(artifact_path.resolve())
    return specs


def _paper_baseline_model_specs(paths: ProjectPaths) -> list[ModelArtifactSpec]:
    baseline_dir = paths.models / "reproduction" / "paper_baseline"
    scaler_path = baseline_dir / "paper_baseline_scaler.joblib"
    specs: list[ModelArtifactSpec] = []
    if not baseline_dir.exists():
        return specs
    for artifact_path in sorted(baseline_dir.glob("*.joblib")):
        if artifact_path.name == "paper_baseline_scaler.joblib":
            continue
        specs.append(
            ModelArtifactSpec(
                name=f"Paper Baseline {_display_model_name(artifact_path.stem)}",
                artifact_path=artifact_path,
                scaler_path=scaler_path if scaler_path.exists() else None,
            )
        )
    ann_path = baseline_dir / "ANN" / "ann_model.keras"
    if _include_keras_models() and ann_path.exists():
        specs.append(
            ModelArtifactSpec(
                name="Paper Baseline ANN",
                artifact_path=ann_path,
                scaler_path=scaler_path if scaler_path.exists() else None,
                artifact_type="keras",
            )
        )
    return specs


def _improved_model_specs(paths: ProjectPaths) -> list[ModelArtifactSpec]:
    improved_dir = paths.models / "reproduction" / "improved"
    if not improved_dir.exists():
        return []
    return [
        ModelArtifactSpec(
            name=f"Improved {_display_model_name(artifact_path.stem)}",
            artifact_path=artifact_path,
        )
        for artifact_path in sorted(improved_dir.glob("*.joblib"))
    ]


def _legacy_model_specs(paths: ProjectPaths) -> list[ModelArtifactSpec]:
    scaler_path = paths.models / "scaler.pkl"
    specs = [
        ModelArtifactSpec(
            name="Legacy LGBM",
            artifact_path=paths.models / "LGBMRegressor_model.sav",
            scaler_path=scaler_path,
        ),
        ModelArtifactSpec(
            name="Legacy XGBoost",
            artifact_path=paths.models / "XGBRegressor_model.sav",
            scaler_path=scaler_path,
        ),
        ModelArtifactSpec(
            name="Legacy Gradient Boosting",
            artifact_path=paths.models / "GradientBoostingRegressor_model.sav",
            scaler_path=scaler_path,
        ),
    ]
    if _include_keras_models():
        specs.append(
            ModelArtifactSpec(
                name="Legacy ANN",
                artifact_path=paths.models / "ANN_model.h5",
                scaler_path=scaler_path,
                artifact_type="keras",
            )
        )
    return [spec for spec in specs if spec.artifact_path.exists()]


def available_model_specs(paths: ProjectPaths) -> list[ModelArtifactSpec]:
    specs = [
        *_benchmark_model_specs(paths),
        *_paper_baseline_model_specs(paths),
        *_improved_model_specs(paths),
        *_legacy_model_specs(paths),
    ]
    return [spec for spec in specs if spec.artifact_path.exists()]


def _load_pickle_artifact(path: Path) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=InconsistentVersionWarning)
        warnings.filterwarnings(
            "ignore",
            message=".*If you are loading a serialized model.*",
            category=UserWarning,
        )
        return joblib.load(path)


def _load_keras_artifact(path: Path) -> Any:
    keras_models = require_module("tensorflow.keras.models", "Install tensorflow to load saved ANN models.")
    return keras_models.load_model(path)


def _load_model_artifact(spec: ModelArtifactSpec) -> tuple[Any, tuple[str, ...] | None]:
    if spec.artifact_type == "keras":
        return _load_keras_artifact(spec.artifact_path), None
    artifact = _load_pickle_artifact(spec.artifact_path)
    if isinstance(artifact, dict) and "estimator" in artifact:
        selected_columns = artifact.get("selected_columns")
        selected_tuple = tuple(str(column) for column in selected_columns) if selected_columns is not None else None
        return artifact["estimator"], selected_tuple
    return artifact, None


def _extract_feature_order(estimator: Any, scaler: Any | None) -> tuple[str, ...] | None:
    if scaler is not None and hasattr(scaler, "feature_names_in_"):
        return tuple(str(name) for name in scaler.feature_names_in_)
    if hasattr(estimator, "feature_names_in_"):
        return tuple(str(name) for name in estimator.feature_names_in_)
    fitted_regressor = getattr(estimator, "regressor_", None)
    if fitted_regressor is not None and hasattr(fitted_regressor, "feature_names_in_"):
        return tuple(str(name) for name in fitted_regressor.feature_names_in_)
    return None


def load_available_models(paths: ProjectPaths) -> tuple[list[LoadedPredictionModel], list[str]]:
    loaded_models: list[LoadedPredictionModel] = []
    skipped_models: list[str] = []
    for spec in available_model_specs(paths):
        try:
            estimator, selected_columns = _load_model_artifact(spec)
            scaler = _load_pickle_artifact(spec.scaler_path) if spec.scaler_path else None
            loaded_models.append(
                LoadedPredictionModel(
                    name=spec.name,
                    artifact_path=spec.artifact_path,
                    estimator=estimator,
                    scaler=scaler,
                    feature_order=_extract_feature_order(estimator, scaler),
                    selected_columns=selected_columns,
                    is_primary=spec.is_primary,
                    artifact_type=spec.artifact_type,
                )
            )
        except Exception as exc:  # pragma: no cover - depends on optional local model dependencies.
            skipped_models.append(f"{spec.name}: {exc}")
    if not loaded_models:
        raise RuntimeError("No prediction models could be loaded from the project artifacts.")
    if not any(model.is_primary for model in loaded_models):
        loaded_models[0].is_primary = True
    return loaded_models, skipped_models


def predict_with_model(model: LoadedPredictionModel, feature_frame: pd.DataFrame) -> np.ndarray:
    model_input = feature_frame
    if model.feature_order is not None:
        missing_columns = [column for column in model.feature_order if column not in feature_frame.columns]
        if missing_columns:
            raise ValueError(f"Feature frame is missing model columns: {missing_columns}")
        model_input = feature_frame.loc[:, list(model.feature_order)]
    if model.scaler is not None:
        transformed = model.scaler.transform(model_input)
        if model.feature_order is not None:
            model_input = pd.DataFrame(transformed, columns=list(model.feature_order), index=feature_frame.index)
        else:
            model_input = transformed
    if model.selected_columns is not None:
        if isinstance(model_input, pd.DataFrame):
            missing_columns = [column for column in model.selected_columns if column not in model_input.columns]
            if missing_columns:
                raise ValueError(f"Feature frame is missing selected model columns: {missing_columns}")
            model_input = model_input.loc[:, list(model.selected_columns)]
        elif model.feature_order is not None:
            selected_indices = [list(model.feature_order).index(column) for column in model.selected_columns]
            model_input = np.asarray(model_input)[:, selected_indices]
        else:
            raise ValueError(f"Model {model.name} requires selected columns but no feature order is available.")
    if model.artifact_type == "keras":
        return np.asarray(model.estimator.predict(np.asarray(model_input), verbose=0)).reshape(-1)
    return np.asarray(model.estimator.predict(model_input)).reshape(-1)


def build_reference_scatter(model: LoadedPredictionModel, bundle) -> pd.DataFrame:
    training = bundle.clean_trial4.copy()
    feature_columns = feature_columns_from_training_frame(training)
    x_reference = training[feature_columns]
    y_reference = training["logkpl"]
    predictions = predict_with_model(model, x_reference)
    return pd.DataFrame(
        {
            "actual_logkp": y_reference.to_numpy(),
            "predicted_logkp": predictions,
        }
    )


def save_single_point_prediction_plot(
    reference_frame: pd.DataFrame,
    actual_logkp: float,
    predicted_logkp: float,
    destination: Path,
    title: str,
    show_plot: bool = False,
) -> None:
    plt = require_module("matplotlib.pyplot", "Install matplotlib to generate prediction figures.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    if not reference_frame.empty:
        plt.scatter(
            reference_frame["actual_logkp"],
            reference_frame["predicted_logkp"],
            alpha=0.25,
            color="gray",
            label="Training/reference compounds",
        )
    plt.scatter(
        [actual_logkp],
        [predicted_logkp],
        alpha=0.95,
        color="#1f77b4",
        s=140,
        label="Input compound",
    )
    lower = float(min(reference_frame["actual_logkp"].min() if not reference_frame.empty else actual_logkp, actual_logkp, predicted_logkp))
    upper = float(max(reference_frame["predicted_logkp"].max() if not reference_frame.empty else predicted_logkp, actual_logkp, predicted_logkp))
    plt.plot([lower, upper], [lower, upper], linestyle="--", color="red")
    plt.xlabel("Actual LogKp")
    plt.ylabel("Predicted LogKp")
    plt.title(title)
    if not reference_frame.empty:
        plt.legend()
    plt.tight_layout()
    plt.savefig(destination, dpi=200)
    if show_plot:
        plt.show()
    plt.close()


def prediction_result_to_frame(result: PredictionResult) -> pd.DataFrame:
    rdkit_columns = {f"rdkit_{key}": value for key, value in result.rdkit_descriptor_summary.items()}
    context = {
        "input_smiles": result.input_smiles,
        "canonical_smiles": result.canonical_smiles,
        "texpi": result.texpi,
        "rdkit_logp": result.rdkit_logp,
        "rdkit_molecular_weight": result.molecular_weight,
        "formula_logkp": result.formula_logkp,
        **rdkit_columns,
    }
    output = result.predictions.copy()
    for column, value in context.items():
        output[column] = value
    context_columns = list(context)
    remaining_columns = [column for column in output.columns if column not in context_columns]
    return output.loc[:, [*context_columns, *remaining_columns]]


def save_prediction_result_table(result: PredictionResult, destination: str | Path) -> Path:
    output_path = Path(destination)
    if not output_path.is_absolute():
        output_path = (ProjectPaths.discover().root / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_result_to_frame(result).to_csv(output_path, index=False)
    return output_path


def predict_logkp_from_smiles(
    smiles: str,
    paths: ProjectPaths | None = None,
    texpi: float = DEFAULT_TEXPI,
    plot_output: str | Path = DEFAULT_PLOT_OUTPUT,
    show_plot: bool = False,
    rdkit_python: str | None = None,
    maven_executable: str = DEFAULT_MAVEN_EXECUTABLE,
) -> PredictionResult:
    resolved_paths = paths or ProjectPaths.discover()
    bundle = load_bundle(resolved_paths)
    rdkit_result = compute_rdkit_properties(smiles, python_executable=rdkit_python)
    descriptor_frame = generate_training_compatible_descriptors(
        resolved_paths,
        rdkit_result.canonical_smiles,
        texpi=float(texpi),
        maven_executable=maven_executable,
    )

    feature_columns = feature_columns_from_training_frame(bundle.clean_trial4)
    fill_values = build_inference_fill_values(bundle)
    feature_frame = align_descriptor_frame_to_feature_columns(
        descriptor_frame=descriptor_frame,
        feature_columns=feature_columns,
        fill_values=fill_values,
        texpi=float(texpi),
    )

    formula_logkp = calculate_formula_logkp(rdkit_result.logp, rdkit_result.molecular_weight)
    loaded_models, skipped_models = load_available_models(resolved_paths)

    prediction_rows = []
    reference_frame = pd.DataFrame()
    for model in loaded_models:
        try:
            prediction_value = float(predict_with_model(model, feature_frame)[0])
        except Exception as exc:  # pragma: no cover - depends on local artifact compatibility.
            skipped_models.append(f"{model.name}: prediction failed ({exc})")
            continue
        prediction_rows.append(
            {
                "model": model.name,
                "predicted_logkp": prediction_value,
                "formula_logkp": formula_logkp,
                "delta_vs_formula": prediction_value - formula_logkp,
                "artifact_path": str(model.artifact_path),
                "is_primary": model.is_primary,
            }
        )
        if model.is_primary:
            reference_frame = build_reference_scatter(model, bundle)

    if not prediction_rows:
        raise RuntimeError("All available models failed during prediction.")

    predictions = pd.DataFrame(prediction_rows).sort_values(["is_primary", "model"], ascending=[False, True]).reset_index(drop=True)
    if not predictions["is_primary"].any():
        predictions.loc[predictions.index[0], "is_primary"] = True
    plot_path = Path(plot_output)
    if not plot_path.is_absolute():
        plot_path = (resolved_paths.root / plot_path).resolve()
    primary_prediction = predictions.loc[predictions["is_primary"], "predicted_logkp"].iloc[0]
    save_single_point_prediction_plot(
        reference_frame=reference_frame,
        actual_logkp=formula_logkp,
        predicted_logkp=float(primary_prediction),
        destination=plot_path,
        title="Predicted vs Actual LogKp",
        show_plot=show_plot,
    )

    return PredictionResult(
        input_smiles=smiles,
        canonical_smiles=rdkit_result.canonical_smiles,
        texpi=float(texpi),
        rdkit_logp=rdkit_result.logp,
        molecular_weight=rdkit_result.molecular_weight,
        formula_logkp=formula_logkp,
        rdkit_descriptor_summary=rdkit_result.descriptor_summary,
        feature_frame=feature_frame,
        raw_descriptor_frame=descriptor_frame,
        predictions=predictions,
        plot_path=plot_path,
        skipped_models=skipped_models,
    )


def print_prediction_summary(result: PredictionResult) -> None:
    print(f"Input SMILES: {result.input_smiles}")
    print(f"Canonical SMILES: {result.canonical_smiles}")
    print(f"Texpi (K): {result.texpi:.1f}")
    print(f"RDKit logP: {result.rdkit_logp:.4f}")
    print(f"RDKit molecular weight: {result.molecular_weight:.4f}")
    print("RDKit descriptor summary:")
    for key, value in result.rdkit_descriptor_summary.items():
        print(f"  - {key}: {value:.4f}")
    print(f"Formula-derived LogKp reference: {result.formula_logkp:.4f}")
    print("")
    print("Predictions:")
    for row in result.predictions.itertuples(index=False):
        marker = " [primary]" if row.is_primary else ""
        print(f"  - {row.model}: {row.predicted_logkp:.4f}{marker}")
    if result.skipped_models:
        print("")
        print("Skipped models:")
        for skipped in result.skipped_models:
            print(f"  - {skipped}")
    print("")
    print(f"Plot saved to: {result.plot_path}")


def build_prediction_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict logKp for a new SMILES string using the existing project models.")
    parser.add_argument("--smiles", default="", help="SMILES string to predict. If omitted, the script prompts in the terminal.")
    parser.add_argument("--texpi", type=float, default=None, help=f"Permeation temperature in Kelvin. Defaults to {DEFAULT_TEXPI:g}.")
    parser.add_argument(
        "--plot-output",
        default=str(DEFAULT_PLOT_OUTPUT),
        help="Path to save the predicted-vs-actual logKp plot.",
    )
    parser.add_argument(
        "--predictions-output",
        default=str(DEFAULT_PREDICTIONS_OUTPUT),
        help="CSV path to save RDKit details and per-model logKp predictions.",
    )
    parser.add_argument("--show-plot", action="store_true", help="Display the generated plot in addition to saving it.")
    parser.add_argument("--rdkit-python", default=None, help="Python executable that can import rdkit.")
    parser.add_argument("--maven", default=DEFAULT_MAVEN_EXECUTABLE, help="Maven executable to use for CDK descriptor generation.")
    return parser


def _prompt_for_smiles(default_texpi: float) -> tuple[str, float]:
    smiles = input("Enter a SMILES string: ").strip()
    texpi_raw = input(f"Permeation temperature Texpi in K [{default_texpi:g}]: ").strip()
    texpi = float(texpi_raw) if texpi_raw else default_texpi
    return smiles, texpi


def run_prediction_command(args: argparse.Namespace, paths: ProjectPaths | None = None) -> PredictionResult:
    resolved_paths = paths or ProjectPaths.discover()
    resolved_paths.ensure_runtime_dirs()
    if args.smiles:
        smiles = args.smiles.strip()
        texpi = float(args.texpi) if args.texpi is not None else DEFAULT_TEXPI
    else:
        smiles, texpi = _prompt_for_smiles(DEFAULT_TEXPI)
    if not smiles:
        raise SystemExit("A non-empty SMILES string is required.")

    result = predict_logkp_from_smiles(
        smiles=smiles,
        paths=resolved_paths,
        texpi=texpi,
        plot_output=args.plot_output,
        show_plot=args.show_plot,
        rdkit_python=args.rdkit_python,
        maven_executable=args.maven,
    )
    print_prediction_summary(result)
    if getattr(args, "predictions_output", ""):
        output_path = Path(args.predictions_output)
        if not output_path.is_absolute():
            output_path = (resolved_paths.root / output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_result_to_frame(result).to_csv(output_path, index=False)
        print(f"Prediction table saved to: {output_path}")
    return result


def main(argv: list[str] | None = None) -> PredictionResult:
    parser = build_prediction_parser()
    args = parser.parse_args(argv)
    return run_prediction_command(args)


if __name__ == "__main__":
    main()
