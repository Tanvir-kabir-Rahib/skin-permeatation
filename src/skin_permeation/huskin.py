"""Inference and evaluation pipeline for the HuSkinDB dataset.

The benchmark models in this repository were trained on a 146-column CDK
descriptor panel.  HuSkinDB contains only a small descriptor subset, so this
module follows a provenance-preserving hierarchy:

1. reuse exact, numeric columns already present in HuSkinDB;
2. calculate descriptors with RDKit when an exact, training-compatible
   implementation is available;
3. calculate remaining CDK descriptors with the repository's original Java
   descriptor generator; and
4. use training-reference fill values only for isolated descriptor failures.

No preprocessing object is fitted on HuSkinDB, and the experimental target is
explicitly excluded from model inputs.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import numpy as np
import pandas as pd
from scipy import stats

from .data_loading import load_bundle
from .paths import ProjectPaths
from .prediction import (
    DEFAULT_MAVEN_EXECUTABLE,
    LoadedPredictionModel,
    build_inference_fill_values,
    discover_rdkit_python,
    predict_with_model,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_INPUT_DIR = Path("data/huskinDB")
DEFAULT_OUTPUT_DIR = Path("results/huskinDB")
DEFAULT_MODELS_DIR = Path("models/reproduction/benchmark")
TRAINING_LOGKP_UNIT = "cm/h"
DEFAULT_TARGET_CANDIDATES = (
    "logkp (cm/s)",
    "experimental logkp",
    "experimental_logkp",
    "logkp",
    "logkpl",
)
DEFAULT_SMILES_CANDIDATES = ("smiles", "canonical smiles", "canonical_smiles")
TARGET_NORMALIZED_NAMES = {
    "logkp",
    "logkpl",
    "logkpcms",
    "experimentallogkp",
    "predictedlogkp",
}

MODEL_NAME_OVERRIDES = {
    "RF": "RandomForest",
    "SVR_RBF": "SVR_RBF",
    "Gradient_Boosting": "GradientBoosting",
    "StackingRegressor": "StackingRegressor",
}

# These CDK-named descriptors were independently checked against RDKit across
# every unique SMILES in the benchmark training frame.  Features whose RDKit
# definition differed from CDK (for example Kier2, Kier3, nRotB, and TopoPSA
# for some molecules) are intentionally absent and are generated with CDK.
RDKIT_VALIDATED_CDK_EQUIVALENTS = {
    "nAromRings",
    "nRings3",
    "nRings4",
    "nRings5",
    "nRings6",
    "nRings7",
    "nRings8",
    "nRings9",
    "Zagreb",
    "khs.sLi",
    "khs.ssBe",
    "khs.ssBH",
    "khs.sssB",
    "khs.sCH3",
    "khs.dCH2",
    "khs.tCH",
    "khs.dsCH",
    "khs.aaCH",
    "khs.sssCH",
    "khs.ddC",
    "khs.dssC",
    "khs.aasC",
    "khs.aaaC",
    "khs.sNH3",
    "khs.sNH2",
    "khs.ssNH2",
    "khs.dNH",
    "khs.ssNH",
    "khs.aaNH",
    "khs.tN",
    "khs.dsN",
    "khs.aaN",
    "khs.sssN",
    "khs.ddsN",
    "khs.aasN",
    "khs.ssssN",
    "khs.dO",
    "khs.ssO",
    "khs.aaO",
    "khs.sF",
    "khs.sSiH3",
    "khs.ssSiH2",
    "khs.sssSiH",
    "khs.ssssSi",
    "khs.sPH2",
    "khs.ssPH",
    "khs.sssP",
    "khs.dsssP",
    "khs.sssssP",
    "khs.sSH",
    "khs.dS",
    "khs.ssS",
    "khs.aaS",
    "khs.dssS",
    "khs.ddssS",
    "khs.sCl",
    "khs.sGeH3",
    "khs.ssGeH2",
    "khs.sssGeH",
    "khs.ssssGe",
    "khs.sAsH2",
    "khs.ssAsH",
    "khs.sssAs",
    "khs.sssdAs",
    "khs.sssssAs",
    "khs.sSeH",
    "khs.dSe",
    "khs.ssSe",
    "khs.aaSe",
    "khs.dssSe",
    "khs.ddssSe",
    "khs.sBr",
    "khs.sSnH3",
    "khs.ssSnH2",
    "khs.sssSnH",
    "khs.ssssSn",
    "khs.sPbH3",
    "khs.ssPbH2",
    "khs.sssPbH",
    "khs.ssssPb",
}

RDKIT_BATCH_HELPER = r'''
import json
import math
import sys

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.EState import AtomTypes

payload = json.loads(sys.stdin.read())
rows = payload["rows"]
required = set(payload["required_features"])
validated = set(payload["validated_cdk_equivalents"])
registry = dict(Descriptors.descList)
results = []
raw_cache = {}
descriptor_cache = {}

for row in rows:
    index = row["index"]
    smiles = row.get("smiles")
    item = {"index": index, "canonical_smiles": None, "valid": False,
            "error": None, "descriptors": {}}
    if smiles is None or not str(smiles).strip():
        item["error"] = "Missing SMILES value."
        results.append(item)
        continue
    raw_key = str(smiles).strip()
    if raw_key in raw_cache:
        cached = raw_cache[raw_key]
        item.update({key: value for key, value in cached.items() if key not in {"index", "descriptors"}})
        item["descriptors"] = dict(cached.get("descriptors", {}))
        results.append(item)
        continue
    try:
        mol = Chem.MolFromSmiles(raw_key)
        if mol is None:
            raise ValueError("RDKit could not parse the SMILES string.")
        item["canonical_smiles"] = Chem.MolToSmiles(mol, canonical=True)
        item["valid"] = True
        if item["canonical_smiles"] in descriptor_cache:
            item["descriptors"] = dict(descriptor_cache[item["canonical_smiles"]])
            raw_cache[raw_key] = dict(item)
            results.append(item)
            continue
        descriptors = {}

        # Exact native RDKit descriptor names are supported generically.
        for feature in required.intersection(registry):
            try:
                value = float(registry[feature](mol))
                descriptors[feature] = value if math.isfinite(value) else None
            except Exception:
                descriptors[feature] = None

        if "nAromRings" in required and "nAromRings" in validated:
            descriptors["nAromRings"] = float(rdMolDescriptors.CalcNumAromaticRings(mol))
        rings = mol.GetRingInfo().AtomRings()
        for size in range(3, 10):
            feature = f"nRings{size}"
            if feature in required and feature in validated:
                descriptors[feature] = float(sum(len(ring) == size for ring in rings))
        if "Zagreb" in required and "Zagreb" in validated:
            descriptors["Zagreb"] = float(sum(atom.GetDegree() ** 2 for atom in mol.GetAtoms()))

        khs_features = [feature for feature in required.intersection(validated)
                        if feature.startswith("khs.")]
        if khs_features:
            atom_types = [entry[0] for entry in AtomTypes.TypeAtoms(mol) if entry]
            for feature in khs_features:
                descriptors[feature] = float(atom_types.count(feature[4:]))
        item["descriptors"] = descriptors
        descriptor_cache[item["canonical_smiles"]] = dict(descriptors)
    except Exception as exc:
        item["valid"] = False
        item["error"] = f"{type(exc).__name__}: {exc}"
    raw_cache[raw_key] = dict(item)
    results.append(item)

sys.stdout.write(json.dumps(results, allow_nan=False))
'''.strip()


class HuSkinPipelineError(RuntimeError):
    """Raised when a required inference invariant cannot be satisfied."""


@dataclass
class BenchmarkModel:
    """Loaded benchmark estimator plus recovered inference metadata."""

    display_name: str
    output_name: str
    artifact_path: Path
    estimator: Any
    required_features: tuple[str, ...]
    serialization_format: str
    preprocessing_summary: str
    compatibility_note: str = ""
    prediction_column: str = field(init=False)

    def __post_init__(self) -> None:
        self.prediction_column = f"Predicted_logKp_{self.output_name}"


@dataclass
class DescriptorStats:
    """Per-feature provenance and completeness information."""

    feature: str
    original_column: str | None
    missing_before: int
    missing_after_rdkit: int
    missing_after_processing: int
    reused_count: int = 0
    generated_rdkit_count: int = 0
    generated_cdk_count: int = 0
    imputed_training_count: int = 0
    unsupported_by_rdkit: bool = False
    unsupported_reason: str = ""
    notes: str = ""


@dataclass
class PipelineRunResult:
    """Paths and summary data produced by a HuSkinDB inference run."""

    input_path: Path
    output_dir: Path
    predictions_path: Path
    metrics_path: Path
    descriptor_audit_path: Path
    log_path: Path
    summary_path: Path
    plot_paths: list[Path]
    metrics: pd.DataFrame
    loaded_models: list[BenchmarkModel]
    skipped_models: list[str]
    smiles_column: str
    target_column: str


def normalize_feature_name(name: str) -> str:
    """Normalize harmless case, whitespace, and punctuation differences."""

    return re.sub(r"[^a-z0-9]+", "", str(name).strip().casefold())


def _resolve_path(path: str | Path, root: Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()


def _find_column(columns: Iterable[str], requested: str | None, candidates: Sequence[str]) -> str | None:
    columns_list = [str(column) for column in columns]
    normalized = {normalize_feature_name(column): column for column in columns_list}
    if requested:
        if requested in columns_list:
            return requested
        match = normalized.get(normalize_feature_name(requested))
        if match:
            return match
        return None
    for candidate in candidates:
        match = normalized.get(normalize_feature_name(candidate))
        if match:
            return match
    return None


def find_huskin_dataset(
    paths: ProjectPaths,
    input_path: str | Path | None = None,
    smiles_column: str | None = None,
    target_column: str | None = None,
) -> tuple[Path, str, str]:
    """Locate the HuSkinDB CSV and detect its SMILES and target columns."""

    if input_path is not None:
        candidates = [_resolve_path(input_path, paths.root)]
    else:
        input_dir = paths.root / DEFAULT_INPUT_DIR
        candidates = sorted(input_dir.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No CSV files were found in {paths.root / DEFAULT_INPUT_DIR}.")

    diagnostics: list[str] = []
    for candidate in candidates:
        if not candidate.exists():
            diagnostics.append(f"{candidate}: file does not exist")
            continue
        try:
            columns = pd.read_csv(candidate, nrows=0).columns.tolist()
        except Exception as exc:
            diagnostics.append(f"{candidate}: could not read header ({exc})")
            continue
        detected_smiles = _find_column(columns, smiles_column, DEFAULT_SMILES_CANDIDATES)
        detected_target = _find_column(columns, target_column, DEFAULT_TARGET_CANDIDATES)
        if detected_smiles and detected_target:
            return candidate.resolve(), detected_smiles, detected_target
        diagnostics.append(
            f"{candidate}: SMILES={detected_smiles!r}, experimental logKp={detected_target!r}"
        )
    raise HuSkinPipelineError(
        "No candidate CSV contained both the requested SMILES and experimental logKp columns. "
        + "; ".join(diagnostics)
    )


def load_dataset(path: Path) -> pd.DataFrame:
    """Load HuSkinDB without altering its original columns or row order."""

    frame = pd.read_csv(path)
    if frame.empty:
        raise HuSkinPipelineError(f"The input dataset is empty: {path}")
    return frame


def _friendly_model_name(stem: str, metrics_names: dict[str, str]) -> tuple[str, str]:
    display = metrics_names.get(stem, stem.replace("_", " "))
    output = MODEL_NAME_OVERRIDES.get(stem, re.sub(r"[^A-Za-z0-9]+", "", display))
    return display, output


def discover_model_artifacts(models_dir: Path) -> list[Path]:
    """Discover serialized benchmark estimators in a deterministic order."""

    supported = {".joblib", ".pkl", ".sav", ".keras", ".h5", ".json"}
    return sorted(path for path in models_dir.iterdir() if path.is_file() and path.suffix.lower() in supported)


def _load_sidecar_features(artifact_path: Path) -> tuple[str, ...] | None:
    candidates = [
        artifact_path.with_name(f"{artifact_path.stem}_features.json"),
        artifact_path.with_name("feature_names.json"),
        artifact_path.with_name("features.json"),
    ]
    for path in candidates:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = payload.get("features") if isinstance(payload, dict) else payload
        if isinstance(values, list) and values:
            return tuple(str(value) for value in values)
    return None


def get_required_features(estimator: Any, artifact_path: Path) -> tuple[str, ...]:
    """Recover the exact fitted feature order or raise instead of guessing."""

    candidates = [estimator, getattr(estimator, "regressor_", None), getattr(estimator, "regressor", None)]
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "feature_names_in_"):
            values = tuple(str(value) for value in candidate.feature_names_in_)
            if values:
                return values
    sidecar = _load_sidecar_features(artifact_path)
    if sidecar:
        return sidecar
    raise HuSkinPipelineError(
        f"Cannot determine the required feature names and order for {artifact_path.name}. "
        "Add feature_names_in_ to the fitted artifact or provide an ordered JSON feature list "
        f"at {artifact_path.with_name(artifact_path.stem + '_features.json')}."
    )


def _preprocessing_summary(estimator: Any) -> str:
    if hasattr(estimator, "steps"):
        return "embedded Pipeline: " + " -> ".join(
            f"{name} ({type(step).__name__})" for name, step in estimator.steps
        )
    if type(estimator).__name__ == "TransformedTargetRegressor":
        inner = getattr(estimator, "regressor_", getattr(estimator, "regressor", None))
        return f"embedded target transform ({type(getattr(estimator, 'transformer_', None)).__name__}); {_preprocessing_summary(inner)}"
    if type(estimator).__name__ == "StackingRegressor":
        names = [str(name) for name, _ in getattr(estimator, "estimators", [])]
        return f"embedded stacking estimator ({', '.join(names)}); final={type(getattr(estimator, 'final_estimator_', None)).__name__}"
    return f"estimator only ({type(estimator).__name__}); no separate preprocessor discovered"


def _artifact_compatibility_note(captured_warnings: Sequence[warnings.WarningMessage]) -> str:
    versions: set[tuple[str, str]] = set()
    for warning in captured_warnings:
        match = re.search(
            r"from version ([0-9.]+) when using version ([0-9.]+)",
            str(warning.message),
        )
        if match:
            versions.add((match.group(1), match.group(2)))
    if not versions:
        return ""
    return "; ".join(
        f"scikit-learn artifact version {artifact_version}, runtime {runtime_version}"
        for artifact_version, runtime_version in sorted(versions)
    )


def _metrics_model_name_map(paths: ProjectPaths, models_dir: Path) -> dict[str, str]:
    metrics_path = paths.reports / "tables" / "benchmark_metrics.csv"
    mapping: dict[str, str] = {}
    if not metrics_path.exists():
        return mapping
    metrics = pd.read_csv(metrics_path)
    if not {"model", "artifact_path"}.issubset(metrics.columns):
        return mapping
    for row in metrics.itertuples(index=False):
        stem = Path(str(row.artifact_path)).stem
        mapping[stem] = str(row.model)
    return mapping


def _selected_model(model: BenchmarkModel, selectors: Sequence[str] | None) -> bool:
    if not selectors:
        return True
    aliases = {
        normalize_feature_name(model.display_name),
        normalize_feature_name(model.output_name),
        normalize_feature_name(model.artifact_path.stem),
    }
    return any(normalize_feature_name(selector) in aliases for selector in selectors)


def load_model_and_preprocessor(
    paths: ProjectPaths,
    models_dir: Path,
    model_selectors: Sequence[str] | None = None,
) -> tuple[list[BenchmarkModel], list[str], dict[str, Any]]:
    """Load unique benchmark estimators and recover their preprocessing metadata."""

    if not models_dir.exists():
        raise FileNotFoundError(f"Benchmark model directory does not exist: {models_dir}")
    artifacts = discover_model_artifacts(models_dir)
    names = _metrics_model_name_map(paths, models_dir)
    loaded: list[BenchmarkModel] = []
    skipped: list[str] = []
    json_metadata: dict[str, Any] = {}

    for artifact in artifacts:
        suffix = artifact.suffix.lower()
        if suffix == ".json":
            try:
                json_metadata[artifact.stem] = json.loads(artifact.read_text(encoding="utf-8"))
            except Exception as exc:
                skipped.append(f"{artifact.name}: invalid JSON metadata ({exc})")
            continue
        if artifact.name == "best_benchmark_model.joblib":
            skipped.append(
                "best_benchmark_model.joblib: duplicate convenience alias of the model named in "
                "best_benchmark_model.json; not scored twice"
            )
            continue
        if suffix in {".keras", ".h5"}:
            skipped.append(f"{artifact.name}: Keras artifact loading is not implemented for benchmark discovery")
            continue
        try:
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                estimator = joblib.load(artifact)
            features = get_required_features(estimator, artifact)
            display, output = _friendly_model_name(artifact.stem, names)
            model = BenchmarkModel(
                display_name=display,
                output_name=output,
                artifact_path=artifact,
                estimator=estimator,
                required_features=features,
                serialization_format=suffix.lstrip("."),
                preprocessing_summary=_preprocessing_summary(estimator),
                compatibility_note=_artifact_compatibility_note(captured),
            )
            if _selected_model(model, model_selectors):
                loaded.append(model)
        except Exception as exc:
            skipped.append(f"{artifact.name}: {type(exc).__name__}: {exc}")

    if model_selectors:
        matched = {
            normalize_feature_name(alias)
            for model in loaded
            for alias in (model.display_name, model.output_name, model.artifact_path.stem)
        }
        unmatched = [selector for selector in model_selectors if normalize_feature_name(selector) not in matched]
        if unmatched:
            skipped.append(f"Requested model selectors not matched: {unmatched}")
    if not loaded:
        raise HuSkinPipelineError(
            f"No compatible benchmark estimators could be loaded from {models_dir}. "
            + ("; ".join(skipped) if skipped else "No supported artifacts were found.")
        )
    return loaded, skipped, json_metadata


def ensure_target_excluded(models: Sequence[BenchmarkModel], target_column: str) -> None:
    """Fail closed if any model feature appears to be the experimental target."""

    forbidden = set(TARGET_NORMALIZED_NAMES)
    forbidden.add(normalize_feature_name(target_column))
    for model in models:
        leaked = [feature for feature in model.required_features if normalize_feature_name(feature) in forbidden]
        if leaked:
            raise HuSkinPipelineError(
                f"Target leakage prevented: model {model.display_name} requests target-like features {leaked}."
            )


def calculate_missing_rdkit_descriptors(
    smiles: pd.Series,
    required_features: Sequence[str],
    rdkit_python: str | Path | None = None,
) -> pd.DataFrame:
    """Validate SMILES and calculate supported descriptors in one cached RDKit process."""

    executable = Path(rdkit_python) if rdkit_python else discover_rdkit_python()
    rows = [
        {"index": int(position), "smiles": None if pd.isna(value) else str(value)}
        for position, value in enumerate(smiles.tolist())
    ]
    payload = {
        "rows": rows,
        "required_features": list(dict.fromkeys(str(value) for value in required_features)),
        "validated_cdk_equivalents": sorted(RDKIT_VALIDATED_CDK_EQUIVALENTS),
    }
    result = subprocess.run(
        [str(executable), "-c", RDKIT_BATCH_HELPER],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown RDKit subprocess error"
        raise HuSkinPipelineError(f"RDKit descriptor subprocess failed using {executable}: {detail}")
    try:
        records = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise HuSkinPipelineError(f"RDKit returned invalid JSON: {exc}") from exc
    frame = pd.DataFrame.from_records(records).set_index("index").reindex(range(len(smiles)))
    descriptors = pd.json_normalize(frame["descriptors"].map(lambda value: value or {}))
    descriptors.index = frame.index
    return pd.concat(
        [
            frame.drop(columns=["descriptors"]),
            descriptors,
        ],
        axis=1,
    )


def _write_cdk_input(path: Path, paths: ProjectPaths, canonical_smiles: Sequence[str]) -> None:
    columns = pd.read_csv(paths.raw_data / "data-original.csv", nrows=0).columns.tolist()
    rows: list[dict[str, Any]] = []
    for position, smiles in enumerate(canonical_smiles, start=1):
        row = {column: "" for column in columns}
        row.update(
            {
                "No": position,
                "Compound": f"HuSkinDB {position}",
                "SMILES": smiles,
                "set": "inference",
                "Texpi": 310.0,
                "Reference": "HuSkinDB batch inference",
            }
        )
        rows.append(row)
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def generate_cdk_descriptors_batch(
    paths: ProjectPaths,
    canonical_smiles: Sequence[str],
    maven_executable: str = DEFAULT_MAVEN_EXECUTABLE,
) -> pd.DataFrame:
    """Generate the original training-compatible CDK panel once per unique molecule."""

    unique_smiles = list(dict.fromkeys(str(value) for value in canonical_smiles if str(value)))
    if not unique_smiles:
        return pd.DataFrame(index=pd.Index([], name="canonical_smiles"))
    maven_repo = paths.root / ".cache" / "maven-repo"
    maven_repo.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="huskin_cdk_") as temp_dir:
        runtime = Path(temp_dir) / "descriptors-generator"
        runtime.mkdir(parents=True, exist_ok=True)
        shutil.copy2(paths.descriptors_generator / "pom.xml", runtime / "pom.xml")
        shutil.copytree(paths.descriptors_generator / "src", runtime / "src")
        _write_cdk_input(runtime / "data-original.csv", paths, unique_smiles)
        command = [
            maven_executable,
            "-q",
            "-o",
            "-DskipTests",
            f"-Dmaven.repo.local={maven_repo}",
            "package",
            "org.codehaus.mojo:exec-maven-plugin:3.5.0:java",
            "-Dexec.mainClass=org.rami.dg.DescriptorsGenerator",
        ]
        try:
            result = subprocess.run(
                command,
                cwd=runtime,
                text=True,
                capture_output=True,
                check=False,
            )
        except FileNotFoundError as exc:
            raise HuSkinPipelineError(f"Maven executable {maven_executable!r} was not found.") from exc
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "unknown Maven/CDK error"
            raise HuSkinPipelineError(
                "The repository's CDK descriptor generator failed. "
                f"Command: {' '.join(command)}\n{detail}"
            )
        output = runtime / "data-descriptors.csv"
        if not output.exists():
            raise HuSkinPipelineError("CDK completed without producing data-descriptors.csv.")
        frame = pd.read_csv(output)
    if len(frame) != len(unique_smiles):
        raise HuSkinPipelineError(
            f"CDK generated {len(frame)} rows for {len(unique_smiles)} unique valid molecules."
        )
    frame.insert(0, "canonical_smiles", unique_smiles)
    return frame.set_index("canonical_smiles", drop=True)


def _parse_temperature_kelvin(frame: pd.DataFrame) -> tuple[pd.Series, str | None]:
    candidates = (
        "donor/skin surface temperature (°c)",
        "donor skin surface temperature (°c)",
        "skin surface temperature (°c)",
        "temperature (°c)",
        "texpi",
    )
    source = _find_column(frame.columns, None, candidates)
    values = pd.Series(np.nan, index=frame.index, dtype=float)
    if source is None:
        return values, None
    extracted = frame[source].astype(str).str.extract(r"([-+]?\d+(?:\.\d+)?)", expand=False)
    numeric = pd.to_numeric(extracted, errors="coerce")
    values = numeric.where(numeric >= 200.0, numeric + 273.15)
    return values, source


def _original_feature_column(frame: pd.DataFrame, feature: str, target_column: str, smiles_column: str) -> str | None:
    normalized = normalize_feature_name(feature)
    matches = [
        str(column)
        for column in frame.columns
        if normalize_feature_name(column) == normalized
        and str(column) not in {target_column, smiles_column}
    ]
    return matches[0] if len(matches) == 1 else None


def _union_required_features(models: Sequence[BenchmarkModel]) -> list[str]:
    features: list[str] = []
    seen: set[str] = set()
    for model in models:
        for feature in model.required_features:
            if feature not in seen:
                seen.add(feature)
                features.append(feature)
    return features


def prepare_descriptor_frame(
    dataset: pd.DataFrame,
    models: Sequence[BenchmarkModel],
    smiles_column: str,
    target_column: str,
    paths: ProjectPaths,
    rdkit_python: str | Path | None = None,
    maven_executable: str = DEFAULT_MAVEN_EXECUTABLE,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, DescriptorStats], list[str]]:
    """Build a complete model feature frame with per-feature provenance."""

    required = _union_required_features(models)
    rdkit = calculate_missing_rdkit_descriptors(dataset[smiles_column], required, rdkit_python)
    status = pd.DataFrame(index=dataset.index)
    status["Canonical_SMILES"] = rdkit["canonical_smiles"].to_numpy()
    status["SMILES_Valid"] = rdkit["valid"].fillna(False).astype(bool).to_numpy()
    status["SMILES_Error"] = rdkit["error"].fillna("").astype(str).to_numpy()
    valid_mask = status["SMILES_Valid"]
    duplicate_size = status["Canonical_SMILES"].map(status.loc[valid_mask, "Canonical_SMILES"].value_counts()).fillna(0).astype(int)
    status["Duplicate_Group_Size"] = duplicate_size
    status["Duplicate_Compound"] = duplicate_size.gt(1)

    canonical_unique = status.loc[valid_mask, "Canonical_SMILES"].dropna().drop_duplicates().tolist()
    cdk = generate_cdk_descriptors_batch(paths, canonical_unique, maven_executable)
    bundle = load_bundle(paths)
    fill_values = build_inference_fill_values(bundle)
    feature_series: dict[str, pd.Series] = {}
    audit: dict[str, DescriptorStats] = {}
    unsupported: list[str] = []
    temperature, temperature_source = _parse_temperature_kelvin(dataset)

    for feature in required:
        original_column = _original_feature_column(dataset, feature, target_column, smiles_column)
        if feature == "Texpi" and original_column is None:
            series = temperature.copy()
            original_column = temperature_source
            notes = "Converted Celsius values to Kelvin; values already >=200 were treated as Kelvin."
        elif original_column is not None:
            series = pd.to_numeric(dataset[original_column], errors="coerce")
            notes = "Reused exact normalized column name from HuSkinDB."
        else:
            series = pd.Series(np.nan, index=dataset.index, dtype=float)
            notes = ""
        series = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
        missing_before = int((valid_mask & series.isna()).sum())
        reused_count = int((valid_mask & series.notna()).sum())

        rdkit_generated = 0
        if feature in rdkit.columns:
            candidate = pd.to_numeric(rdkit[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
            fill_mask = valid_mask & series.isna() & candidate.notna()
            series.loc[fill_mask] = candidate.loc[fill_mask]
            rdkit_generated = int(fill_mask.sum())
        missing_after_rdkit = int((valid_mask & series.isna()).sum())

        cdk_generated = 0
        if feature != "Texpi" and feature in cdk.columns and missing_after_rdkit:
            mapped = status["Canonical_SMILES"].map(pd.to_numeric(cdk[feature], errors="coerce"))
            fill_mask = valid_mask & series.isna() & mapped.notna()
            series.loc[fill_mask] = mapped.loc[fill_mask]
            cdk_generated = int(fill_mask.sum())

        imputed = 0
        if feature in fill_values:
            fill_mask = valid_mask & series.isna()
            if fill_mask.any():
                series.loc[fill_mask] = float(fill_values[feature])
                imputed = int(fill_mask.sum())

        remaining = int((valid_mask & series.isna()).sum())
        supported_by_rdkit = feature in rdkit.columns
        unsupported_by_rdkit = not supported_by_rdkit and missing_before > 0
        if feature == "Texpi":
            unsupported_by_rdkit = False
        unsupported_reason = ""
        if unsupported_by_rdkit:
            unsupported_reason = (
                "No exact RDKit implementation consistent with the CDK training definition; "
                "the repository's original CDK generator was used."
            )
            unsupported.append(feature)
        if remaining:
            unsupported_reason = (
                unsupported_reason + f" {remaining} valid rows remain unresolved after all supported processing."
            ).strip()
        feature_series[feature] = series
        audit[feature] = DescriptorStats(
            feature=feature,
            original_column=original_column,
            missing_before=missing_before,
            missing_after_rdkit=missing_after_rdkit,
            missing_after_processing=remaining,
            reused_count=reused_count,
            generated_rdkit_count=rdkit_generated,
            generated_cdk_count=cdk_generated,
            imputed_training_count=imputed,
            unsupported_by_rdkit=unsupported_by_rdkit,
            unsupported_reason=unsupported_reason,
            notes=notes,
        )

    features = pd.DataFrame(feature_series, index=dataset.index)
    unresolved = [feature for feature in required if features.loc[valid_mask, feature].isna().any()]
    if unresolved:
        raise HuSkinPipelineError(
            "Required descriptors remain missing for valid compounds after CSV reuse, RDKit, CDK, "
            f"and training-reference fallback: {unresolved}"
        )
    return features, status, audit, sorted(set(unsupported))


def prepare_model_features(feature_frame: pd.DataFrame, model: BenchmarkModel) -> pd.DataFrame:
    """Return numeric features in the exact order captured by the fitted model."""

    missing = [feature for feature in model.required_features if feature not in feature_frame.columns]
    if missing:
        raise HuSkinPipelineError(f"{model.display_name} is missing required features: {missing}")
    ordered = feature_frame.loc[:, list(model.required_features)].copy()
    for feature in ordered.columns:
        ordered[feature] = pd.to_numeric(ordered[feature], errors="coerce")
    return ordered.replace([np.inf, -np.inf], np.nan)


def _loaded_prediction_model(model: BenchmarkModel) -> LoadedPredictionModel:
    return LoadedPredictionModel(
        name=model.display_name,
        artifact_path=model.artifact_path,
        estimator=model.estimator,
        scaler=None,
        feature_order=model.required_features,
        is_primary=False,
        artifact_type="joblib",
    )


def predict_with_benchmark_model(
    model: BenchmarkModel,
    feature_frame: pd.DataFrame,
    valid_mask: pd.Series,
) -> np.ndarray:
    """Predict only valid rows and retain input row alignment."""

    ordered = prepare_model_features(feature_frame, model)
    valid_input = ordered.loc[valid_mask]
    if valid_input.isna().any().any():
        columns = valid_input.columns[valid_input.isna().any()].tolist()
        raise HuSkinPipelineError(f"{model.display_name} has unresolved values in {columns}")
    predictions = np.full(len(feature_frame), np.nan, dtype=float)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
        warnings.filterwarnings("ignore", message="Could not find the number of physical cores.*")
        values = predict_with_model(_loaded_prediction_model(model), valid_input)
    if len(values) != int(valid_mask.sum()):
        raise HuSkinPipelineError(
            f"{model.display_name} returned {len(values)} predictions for {int(valid_mask.sum())} valid rows."
        )
    predictions[np.flatnonzero(valid_mask.to_numpy())] = values
    return predictions


def calculate_regression_metrics(
    experimental: pd.Series | np.ndarray,
    predicted: pd.Series | np.ndarray,
) -> dict[str, float | int]:
    """Calculate regression metrics safely for missing and small samples."""

    actual = np.asarray(experimental, dtype=float)
    estimate = np.asarray(predicted, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(estimate)
    actual = actual[mask]
    estimate = estimate[mask]
    count = int(len(actual))
    if count == 0:
        return {
            "Evaluated_Compounds": 0,
            "R2": np.nan,
            "RMSE": np.nan,
            "MAE": np.nan,
            "Bias": np.nan,
            "Pearson_r": np.nan,
            "Spearman_rho": np.nan,
        }
    error = estimate - actual
    ss_res = float(np.sum(error**2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if count >= 2 and ss_tot > 0 else np.nan
    pearson = np.nan
    spearman = np.nan
    if count >= 2 and np.std(actual) > 0 and np.std(estimate) > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pearson = float(stats.pearsonr(actual, estimate).statistic)
            spearman = float(stats.spearmanr(actual, estimate).statistic)
    return {
        "Evaluated_Compounds": count,
        "R2": float(r2),
        "RMSE": float(math.sqrt(np.mean(error**2))),
        "MAE": float(np.mean(np.abs(error))),
        "Bias": float(np.mean(error)),
        "Pearson_r": pearson,
        "Spearman_rho": spearman,
    }


def _ensemble_members(
    models: Sequence[BenchmarkModel],
    metadata: dict[str, Any],
) -> list[BenchmarkModel]:
    configured = metadata.get("MeanEnsemble", {}).get("models", [])
    if not configured:
        return list(models)
    configured_norm = {normalize_feature_name(value) for value in configured}
    return [
        model
        for model in models
        if normalize_feature_name(model.display_name) in configured_norm
        or normalize_feature_name(model.output_name) in configured_norm
        or normalize_feature_name(model.artifact_path.stem) in configured_norm
    ]


def resolve_prediction_unit(target_column: str, requested_unit: str = "auto") -> tuple[str, float]:
    """Return output unit and additive log10 offset from native cm/h predictions."""

    unit = requested_unit.casefold()
    if unit == "auto":
        normalized = normalize_feature_name(target_column)
        unit = "cm/s" if "cms" in normalized else "cm/h"
    if unit == "cm/h":
        return "cm/h", 0.0
    if unit == "cm/s":
        return "cm/s", -math.log10(3600.0)
    raise HuSkinPipelineError(f"Unsupported target unit {requested_unit!r}; use auto, cm/s, or cm/h.")


def _error_payload(target: pd.Series, prediction: pd.Series, suffix: str) -> dict[str, pd.Series]:
    error = prediction - target
    return {
        f"Prediction_Error_{suffix}": error,
        f"Absolute_Error_{suffix}": error.abs(),
        f"Squared_Error_{suffix}": error.pow(2),
    }


def generate_predictions(
    dataset: pd.DataFrame,
    feature_frame: pd.DataFrame,
    status: pd.DataFrame,
    models: Sequence[BenchmarkModel],
    target_column: str,
    metadata: dict[str, Any],
    generate_ensemble: bool,
    target_unit: str = "auto",
) -> tuple[pd.DataFrame, pd.DataFrame, list[BenchmarkModel], list[str]]:
    """Run every compatible benchmark model and calculate errors and metrics."""

    calculated = feature_frame.loc[:, [column for column in feature_frame.columns if column not in dataset.columns]]
    output = pd.concat([dataset.copy(), calculated, status.copy()], axis=1).copy()
    target = pd.to_numeric(dataset[target_column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid_mask = status["SMILES_Valid"].astype(bool)
    output_unit, prediction_offset = resolve_prediction_unit(target_column, target_unit)
    successful: list[BenchmarkModel] = []
    failures: list[str] = []
    metric_rows: list[dict[str, Any]] = []
    prediction_payload: dict[str, pd.Series] = {}
    error_payload: dict[str, pd.Series] = {}

    for model in models:
        try:
            native_predictions = predict_with_benchmark_model(model, feature_frame, valid_mask)
        except Exception as exc:
            failures.append(f"{model.display_name}: prediction failed ({type(exc).__name__}: {exc})")
            continue
        predictions = native_predictions + prediction_offset
        prediction_series = pd.Series(predictions, index=dataset.index, dtype=float)
        prediction_payload[model.prediction_column] = prediction_series
        error_payload.update(_error_payload(target, prediction_series, model.output_name))
        metric_rows.append(
            {
                "Model": model.display_name,
                "Prediction_Column": model.prediction_column,
                "Artifact_Path": str(model.artifact_path),
                "Model_Native_Unit": TRAINING_LOGKP_UNIT,
                "Output_Unit": output_unit,
                "Prediction_Unit_Offset": prediction_offset,
                **calculate_regression_metrics(target, predictions),
            }
        )
        successful.append(model)

    if generate_ensemble and len(successful) >= 2:
        members = [
            model
            for model in _ensemble_members(successful, metadata)
            if model.prediction_column in prediction_payload
        ]
        if len(members) >= 2:
            ensemble_column = "Predicted_logKp_Ensemble"
            ensemble_series = pd.DataFrame(
                {model.prediction_column: prediction_payload[model.prediction_column] for model in members}
            ).mean(axis=1, skipna=True)
            ensemble_series.loc[~valid_mask] = np.nan
            prediction_payload[ensemble_column] = ensemble_series
            error_payload.update(_error_payload(target, ensemble_series, "Ensemble"))
            metric_rows.append(
                {
                    "Model": "Ensemble",
                    "Prediction_Column": ensemble_column,
                    "Artifact_Path": "MeanEnsemble.json: " + ", ".join(model.display_name for model in members),
                    "Model_Native_Unit": TRAINING_LOGKP_UNIT,
                    "Output_Unit": output_unit,
                    "Prediction_Unit_Offset": prediction_offset,
                    **calculate_regression_metrics(target, ensemble_series),
                }
            )
        else:
            failures.append("Ensemble: fewer than two configured MeanEnsemble members predicted successfully")

    output = pd.concat(
        [
            output,
            pd.DataFrame(prediction_payload, index=dataset.index),
            pd.DataFrame(error_payload, index=dataset.index),
        ],
        axis=1,
    ).copy()
    prediction_columns = [row["Prediction_Column"] for row in metric_rows]
    processing = pd.DataFrame(
        {
            "Processing_Status": np.where(valid_mask, "predicted", "invalid_smiles"),
            "Processing_Error": status["SMILES_Error"].fillna(""),
        },
        index=dataset.index,
    )
    if prediction_columns:
        no_prediction = valid_mask & output[prediction_columns].isna().all(axis=1)
        processing.loc[no_prediction, "Processing_Status"] = "prediction_failed"
        processing.loc[no_prediction, "Processing_Error"] = "No benchmark model produced a prediction."
    output = pd.concat([output, processing], axis=1)

    metrics = pd.DataFrame(metric_rows)
    if metrics.empty:
        raise HuSkinPipelineError("Every benchmark model failed during prediction. " + "; ".join(failures))
    metrics = metrics.sort_values(["RMSE", "R2"], ascending=[True, False], na_position="last").reset_index(drop=True)
    return output, metrics, successful, failures


def build_descriptor_audit(
    models: Sequence[BenchmarkModel],
    audit: dict[str, DescriptorStats],
) -> pd.DataFrame:
    """Expand feature provenance to one ordered row per model requirement."""

    rows: list[dict[str, Any]] = []
    for model in models:
        for position, feature in enumerate(model.required_features, start=1):
            info = audit[feature]
            rows.append(
                {
                    "Model": model.display_name,
                    "Required_Descriptor": feature,
                    "Feature_Position": position,
                    "Original_CSV_Column": info.original_column or "",
                    "Present_In_Original_CSV": bool(info.original_column),
                    "Reused_Existing_Value_Count": info.reused_count,
                    "Generated_Using_RDKit": info.generated_rdkit_count > 0,
                    "RDKit_Generated_Count": info.generated_rdkit_count,
                    "Generated_Using_CDK": info.generated_cdk_count > 0,
                    "CDK_Generated_Count": info.generated_cdk_count,
                    "Training_Reference_Imputed_Count": info.imputed_training_count,
                    "Missing_Before_Processing": info.missing_before,
                    "Missing_After_RDKit": info.missing_after_rdkit,
                    "Missing_After_Processing": info.missing_after_processing,
                    "Calculation_Succeeded": info.missing_after_processing == 0,
                    "Unsupported_By_RDKit": info.unsupported_by_rdkit,
                    "Unsupported_Reason": info.unsupported_reason,
                    "Affected_Compound_Count": info.missing_before,
                    "Notes": info.notes,
                }
            )
    return pd.DataFrame(rows)


def _configure_matplotlib() -> Any:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "skin-huskin-mpl"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#334155",
            "axes.labelcolor": "#1f2937",
            "axes.titlecolor": "#111827",
            "axes.grid": True,
            "grid.color": "#dbe3ea",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.75,
            "font.size": 10,
            "savefig.bbox": "tight",
        }
    )
    return plt


def _plot_series(metrics: pd.DataFrame, predictions: pd.DataFrame, target_column: str) -> list[tuple[str, str]]:
    return [
        (str(row.Model), str(row.Prediction_Column))
        for row in metrics.itertuples(index=False)
        if str(row.Prediction_Column) in predictions.columns
    ]


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")


def _equal_axis_limits(values: np.ndarray, padding_fraction: float = 0.04) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return (-1.0, 1.0)
    lower = float(finite.min())
    upper = float(finite.max())
    padding = max(0.25, padding_fraction * (upper - lower or 1.0))
    return lower - padding, upper + padding


def create_prediction_plots(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    target_column: str,
    output_dir: Path,
) -> list[Path]:
    """Create publication-quality prediction, residual, and comparison plots."""

    plt = _configure_matplotlib()
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = plot_dir / "plot_manifest.json"
    if manifest_path.exists():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            for name in previous.get("generated_plots", []):
                candidate = plot_dir / Path(str(name)).name
                if candidate.is_file():
                    candidate.unlink()
        except Exception as exc:
            LOGGER.warning("Could not clean plots from the previous generated manifest: %s", exc)
    palette = ["#2563eb", "#d97706", "#7c3aed", "#0891b2", "#be185d", "#4d7c0f", "#475569"]
    markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "h", "*"]
    unit = str(metrics["Output_Unit"].iloc[0]) if "Output_Unit" in metrics.columns else "logKp units"
    actual_all = pd.to_numeric(predictions[target_column], errors="coerce")
    series = _plot_series(metrics, predictions, target_column)
    all_values = [actual_all.to_numpy(dtype=float)]
    all_values.extend(pd.to_numeric(predictions[column], errors="coerce").to_numpy(dtype=float) for _, column in series)
    finite = np.concatenate(all_values)
    finite = finite[np.isfinite(finite)]
    full_axis_limits = _equal_axis_limits(finite)
    if len(finite) >= 20:
        robust_values = finite[
            (finite >= np.quantile(finite, 0.005))
            & (finite <= np.quantile(finite, 0.995))
        ]
        combined_axis_limits = _equal_axis_limits(robust_values)
    else:
        combined_axis_limits = full_axis_limits
    paths: list[Path] = []

    metrics_by_column = metrics.set_index("Prediction_Column")
    for model_name, column in series:
        predicted = pd.to_numeric(predictions[column], errors="coerce")
        mask = actual_all.notna() & predicted.notna()
        metric = metrics_by_column.loc[column]
        safe = _safe_filename(model_name)
        model_axis_limits = _equal_axis_limits(
            np.concatenate(
                [
                    actual_all[mask].to_numpy(dtype=float),
                    predicted[mask].to_numpy(dtype=float),
                ]
            )
        )

        fig, ax = plt.subplots(figsize=(7.2, 6.4))
        ax.scatter(actual_all[mask], predicted[mask], s=28, alpha=0.55, color="#2563eb", edgecolors="white", linewidths=0.35)
        ax.plot(model_axis_limits, model_axis_limits, linestyle="--", color="#334155", linewidth=1.4, label="Identity line")
        ax.set(xlabel=f"Experimental logKp ({unit})", ylabel=f"Predicted logKp ({unit})", title=f"Experimental vs. predicted logKp — {model_name}", xlim=model_axis_limits, ylim=model_axis_limits)
        ax.set_aspect("equal", adjustable="box")
        annotation = (
            f"n = {int(metric['Evaluated_Compounds'])}\n"
            f"R² = {metric['R2']:.3f}\nRMSE = {metric['RMSE']:.3f}\nMAE = {metric['MAE']:.3f}"
        )
        ax.text(0.04, 0.96, annotation, transform=ax.transAxes, va="top", ha="left", bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.95})
        ax.legend(loc="lower right", frameon=False)
        fig.tight_layout()
        destination = plot_dir / f"experimental_vs_predicted_{safe}.png"
        fig.savefig(destination, dpi=300)
        plt.close(fig)
        paths.append(destination)

        residual = actual_all - predicted
        fig, ax = plt.subplots(figsize=(7.2, 5.4))
        ax.scatter(predicted[mask], residual[mask], s=28, alpha=0.55, color="#d97706", edgecolors="white", linewidths=0.35)
        ax.axhline(0.0, linestyle="--", color="#334155", linewidth=1.4)
        ax.set(xlabel=f"Predicted logKp ({unit})", ylabel="Residual (experimental − predicted)", title=f"Residuals — {model_name}")
        fig.tight_layout()
        destination = plot_dir / f"residuals_{safe}.png"
        fig.savefig(destination, dpi=300)
        plt.close(fig)
        paths.append(destination)

    fig, ax = plt.subplots(figsize=(9.0, 7.0))
    for index, (model_name, column) in enumerate(series):
        predicted = pd.to_numeric(predictions[column], errors="coerce")
        mask = actual_all.notna() & predicted.notna()
        ax.scatter(
            actual_all[mask],
            predicted[mask],
            s=18,
            alpha=0.25,
            color=palette[index % len(palette)],
            marker=markers[index % len(markers)],
            label=model_name,
        )
    ax.plot(combined_axis_limits, combined_axis_limits, linestyle="--", color="#111827", linewidth=1.5, label="Identity line")
    ax.set(xlabel=f"Experimental logKp ({unit})", ylabel=f"Predicted logKp ({unit})", title="Experimental vs. predicted logKp — all benchmark models", xlim=combined_axis_limits, ylim=combined_axis_limits)
    ax.set_aspect("equal", adjustable="box")
    outside = sum(
        int(
            (
                (pd.to_numeric(predictions[column], errors="coerce") < combined_axis_limits[0])
                | (pd.to_numeric(predictions[column], errors="coerce") > combined_axis_limits[1])
            ).sum()
        )
        for _, column in series
    )
    if outside:
        ax.text(
            0.02,
            0.98,
            f"Robust comparison window; {outside} model predictions are outside the displayed range.",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            color="#475569",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.95},
        )
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    destination = plot_dir / "combined_model_comparison.png"
    fig.savefig(destination, dpi=300)
    plt.close(fig)
    paths.append(destination)

    ordered = metrics.sort_values("RMSE", ascending=True)
    y = np.arange(len(ordered))
    height = 0.36
    fig, ax = plt.subplots(figsize=(9.0, max(5.5, 0.55 * len(ordered) + 1.5)))
    ax.barh(y - height / 2, ordered["RMSE"], height=height, color="#2563eb", label="RMSE")
    ax.barh(y + height / 2, ordered["MAE"], height=height, color="#d97706", label="MAE")
    ax.set(yticks=y, yticklabels=ordered["Model"], xlabel="Error (logKp units)", title="HuSkinDB model performance comparison")
    ax.set_xlim(left=0)
    ax.legend(frameon=False)
    fig.tight_layout()
    destination = plot_dir / "model_performance_comparison.png"
    fig.savefig(destination, dpi=300)
    plt.close(fig)
    paths.append(destination)

    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    for index, (model_name, column) in enumerate(series):
        error = pd.to_numeric(predictions[column], errors="coerce") - actual_all
        ax.hist(error.dropna(), bins=30, density=True, histtype="step", linewidth=1.4, color=palette[index % len(palette)], label=model_name)
    ax.axvline(0.0, color="#111827", linestyle="--", linewidth=1.3)
    ax.set(xlabel="Prediction error (predicted − experimental)", ylabel="Density", title="Prediction error distributions")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    destination = plot_dir / "prediction_error_distribution.png"
    fig.savefig(destination, dpi=300)
    plt.close(fig)
    paths.append(destination)

    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    ax.hist(actual_all.dropna(), bins=30, density=True, histtype="step", linewidth=2.4, color="#111827", label="Experimental")
    for index, (model_name, column) in enumerate(series):
        ax.hist(pd.to_numeric(predictions[column], errors="coerce").dropna(), bins=30, density=True, histtype="step", linewidth=1.2, color=palette[index % len(palette)], label=model_name)
    ax.set(xlabel=f"logKp ({unit})", ylabel="Density", title="Experimental and predicted logKp distributions")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    destination = plot_dir / "logkp_distributions.png"
    fig.savefig(destination, dpi=300)
    plt.close(fig)
    paths.append(destination)

    fig, ax = plt.subplots(figsize=(12.0, 6.0))
    index_values = np.arange(len(predictions))
    ax.plot(index_values, actual_all, color="#111827", linewidth=1.6, label="Experimental")
    for index, (model_name, column) in enumerate(series):
        ax.plot(index_values, pd.to_numeric(predictions[column], errors="coerce"), color=palette[index % len(palette)], linewidth=0.9, alpha=0.65, label=model_name)
    ax.set(xlabel="Compound row index", ylabel=f"logKp ({unit})", title="Experimental and predicted logKp by compound row")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    destination = plot_dir / "compound_index_comparison.png"
    fig.savefig(destination, dpi=300)
    plt.close(fig)
    paths.append(destination)

    fig, ax = plt.subplots(figsize=(9.0, 6.5))
    for index, (model_name, column) in enumerate(series):
        predicted = pd.to_numeric(predictions[column], errors="coerce")
        mask = actual_all.notna() & predicted.notna()
        mean_value = (actual_all[mask] + predicted[mask]) / 2.0
        difference = predicted[mask] - actual_all[mask]
        ax.scatter(mean_value, difference, s=14, alpha=0.22, color=palette[index % len(palette)], label=model_name)
    ax.axhline(0.0, color="#111827", linestyle="--", linewidth=1.3)
    ax.set(xlabel="Mean of experimental and predicted logKp", ylabel="Difference (predicted − experimental)", title="Bland–Altman-style model comparison")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    destination = plot_dir / "bland_altman_model_comparison.png"
    fig.savefig(destination, dpi=300)
    plt.close(fig)
    paths.append(destination)
    manifest_path.write_text(
        json.dumps({"generated_plots": [path.name for path in paths]}, indent=2),
        encoding="utf-8",
    )
    return paths


def _configure_logging(output_dir: Path, verbose: bool = False) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "huskin_processing.log"
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)
    return log_path


def _summary_text(
    input_path: Path,
    output_dir: Path,
    dataset: pd.DataFrame,
    status: pd.DataFrame,
    smiles_column: str,
    target_column: str,
    models: Sequence[BenchmarkModel],
    skipped: Sequence[str],
    audit: dict[str, DescriptorStats],
    unsupported: Sequence[str],
    metrics: pd.DataFrame,
    plot_paths: Sequence[Path],
) -> str:
    reused = [feature for feature, info in audit.items() if info.reused_count]
    rdkit = [feature for feature, info in audit.items() if info.generated_rdkit_count]
    cdk = [feature for feature, info in audit.items() if info.generated_cdk_count]
    lines = [
        "HuSkinDB benchmark inference summary",
        f"Dataset: {input_path}",
        f"Rows: {len(dataset)}",
        f"SMILES column: {smiles_column}",
        f"Experimental logKp column: {target_column}",
        f"Missing SMILES: {int(dataset[smiles_column].isna().sum() + dataset[smiles_column].astype(str).str.strip().eq('').sum())}",
        f"Valid SMILES: {int(status['SMILES_Valid'].sum())}",
        f"Invalid SMILES: {int((~status['SMILES_Valid']).sum())}",
        f"Duplicate rows by canonical SMILES: {int(status['SMILES_Valid'].sum() - status.loc[status['SMILES_Valid'], 'Canonical_SMILES'].nunique())}",
        f"Duplicate canonical-SMILES groups: {int((status.loc[status['SMILES_Valid'], 'Canonical_SMILES'].value_counts() > 1).sum())}",
        f"Missing experimental logKp: {int(pd.to_numeric(dataset[target_column], errors='coerce').isna().sum())}",
        f"Model native target unit: {TRAINING_LOGKP_UNIT}",
        f"Reported prediction unit: {metrics['Output_Unit'].iloc[0]}",
        f"Applied prediction offset: {metrics['Prediction_Unit_Offset'].iloc[0]:.12f} log10 units",
        "Models loaded successfully:",
    ]
    lines.extend(
        f"- {model.display_name}: {model.artifact_path} [{model.serialization_format}]; "
        f"{len(model.required_features)} features; {model.preprocessing_summary}"
        + (f"; {model.compatibility_note}" if model.compatibility_note else "")
        for model in models
    )
    lines.append("Models/artifacts skipped:")
    lines.extend(f"- {value}" for value in skipped) if skipped else lines.append("- None")
    lines.extend(
        [
            f"Descriptors/features reused or derived from CSV ({len(reused)}): {', '.join(reused) if reused else 'None'}",
            "HuSkinDB LogP and molecular weight were preserved but not used directly: the fitted panel "
            "requires CDK XLogP (not a harmless alias for the supplied LogP), and MW is not among the "
            "146 final benchmark inputs.",
            f"Descriptors generated with RDKit ({len(rdkit)}): {', '.join(rdkit) if rdkit else 'None'}",
            f"Training-compatible descriptors generated with CDK ({len(cdk)}): {', '.join(cdk) if cdk else 'None'}",
            f"Descriptors unsupported by exact RDKit mapping ({len(unsupported)}): {', '.join(unsupported) if unsupported else 'None'}",
            "Metric summary (sorted by RMSE):",
            metrics.to_string(index=False),
            f"Predictions: {output_dir / 'huskin_predictions.csv'}",
            f"Metrics: {output_dir / 'huskin_model_metrics.csv'}",
            f"Descriptor audit: {output_dir / 'huskin_descriptor_audit.csv'}",
            f"Plots generated: {len(plot_paths)} in {output_dir / 'plots'}",
        ]
    )
    return "\n".join(lines) + "\n"


def run_huskin_pipeline(
    input_path: str | Path | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    models_dir: str | Path = DEFAULT_MODELS_DIR,
    smiles_column: str | None = None,
    target_column: str | None = None,
    model_selectors: Sequence[str] | None = None,
    generate_ensemble: bool = True,
    target_unit: str = "auto",
    rdkit_python: str | Path | None = None,
    maven_executable: str = DEFAULT_MAVEN_EXECUTABLE,
    verbose: bool = False,
) -> PipelineRunResult:
    """Execute HuSkinDB inference, evaluation, plotting, and output persistence."""

    paths = ProjectPaths.discover()
    resolved_output = _resolve_path(output_dir, paths.root)
    resolved_models = _resolve_path(models_dir, paths.root)
    log_path = _configure_logging(resolved_output, verbose)
    dataset_path, detected_smiles, detected_target = find_huskin_dataset(
        paths, input_path, smiles_column, target_column
    )
    LOGGER.info("Dataset: %s", dataset_path)
    dataset = load_dataset(dataset_path)
    LOGGER.info("Loaded %d rows and %d original columns.", len(dataset), len(dataset.columns))
    LOGGER.info("Detected SMILES column=%r; experimental logKp column=%r.", detected_smiles, detected_target)
    output_unit, prediction_offset = resolve_prediction_unit(detected_target, target_unit)
    LOGGER.info(
        "Model target unit=%s; output unit=%s; additive prediction offset=%.12f.",
        TRAINING_LOGKP_UNIT,
        output_unit,
        prediction_offset,
    )

    models, skipped, metadata = load_model_and_preprocessor(paths, resolved_models, model_selectors)
    ensure_target_excluded(models, detected_target)
    for model in models:
        LOGGER.info(
            "Loaded %s from %s (%s): %d ordered features; %s",
            model.display_name,
            model.artifact_path,
            model.serialization_format,
            len(model.required_features),
            model.preprocessing_summary,
        )
        if model.compatibility_note:
            LOGGER.warning("Artifact compatibility note for %s: %s", model.display_name, model.compatibility_note)

    feature_frame, status, descriptor_info, unsupported = prepare_descriptor_frame(
        dataset,
        models,
        detected_smiles,
        detected_target,
        paths,
        rdkit_python,
        maven_executable,
    )
    LOGGER.info(
        "SMILES audit: valid=%d invalid=%d duplicate_rows=%d missing_target=%d",
        int(status["SMILES_Valid"].sum()),
        int((~status["SMILES_Valid"]).sum()),
        int(status["SMILES_Valid"].sum() - status.loc[status["SMILES_Valid"], "Canonical_SMILES"].nunique()),
        int(pd.to_numeric(dataset[detected_target], errors="coerce").isna().sum()),
    )
    LOGGER.info(
        "Descriptor sources: reused=%d RDKit=%d CDK=%d training-reference-imputed=%d",
        sum(info.reused_count > 0 for info in descriptor_info.values()),
        sum(info.generated_rdkit_count > 0 for info in descriptor_info.values()),
        sum(info.generated_cdk_count > 0 for info in descriptor_info.values()),
        sum(info.imputed_training_count > 0 for info in descriptor_info.values()),
    )
    if unsupported:
        LOGGER.warning(
            "%d required descriptor definitions have no exact RDKit equivalent; exact project CDK values were used: %s",
            len(unsupported),
            ", ".join(unsupported),
        )

    predictions, metrics, successful, prediction_failures = generate_predictions(
        dataset,
        feature_frame,
        status,
        models,
        detected_target,
        metadata,
        generate_ensemble,
        target_unit,
    )
    skipped.extend(prediction_failures)
    predictions_path = resolved_output / "huskin_predictions.csv"
    metrics_path = resolved_output / "huskin_model_metrics.csv"
    descriptor_audit_path = resolved_output / "huskin_descriptor_audit.csv"
    predictions.to_csv(predictions_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    build_descriptor_audit(models, descriptor_info).to_csv(descriptor_audit_path, index=False)
    plot_paths = create_prediction_plots(predictions, metrics, detected_target, resolved_output)

    summary_path = resolved_output / "huskin_processing_summary.txt"
    summary_path.write_text(
        _summary_text(
            dataset_path,
            resolved_output,
            dataset,
            status,
            detected_smiles,
            detected_target,
            successful,
            skipped,
            descriptor_info,
            unsupported,
            metrics,
            plot_paths,
        ),
        encoding="utf-8",
    )
    LOGGER.info("Saved predictions to %s", predictions_path)
    LOGGER.info("Saved metrics to %s", metrics_path)
    LOGGER.info("Saved descriptor audit to %s", descriptor_audit_path)
    LOGGER.info("Saved %d PNG plots to %s", len(plot_paths), resolved_output / "plots")
    return PipelineRunResult(
        input_path=dataset_path,
        output_dir=resolved_output,
        predictions_path=predictions_path,
        metrics_path=metrics_path,
        descriptor_audit_path=descriptor_audit_path,
        log_path=log_path,
        summary_path=summary_path,
        plot_paths=plot_paths,
        metrics=metrics,
        loaded_models=successful,
        skipped_models=skipped,
        smiles_column=detected_smiles,
        target_column=detected_target,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""

    parser = argparse.ArgumentParser(description="Predict HuSkinDB logKp with saved benchmark models.")
    parser.add_argument("--input", type=Path, help="Input CSV. Defaults to automatic discovery in data/huskinDB/.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR, help="Benchmark model artifact directory.")
    parser.add_argument("--smiles-column", help="Override the detected SMILES column.")
    parser.add_argument("--target-column", help="Override the detected experimental logKp column.")
    parser.add_argument(
        "--target-unit",
        choices=("auto", "cm/s", "cm/h"),
        default="auto",
        help="Prediction output unit. Auto detects '(cm/s)' in the target header; model-native unit is cm/h.",
    )
    parser.add_argument("--model", action="append", dest="models", help="Restrict inference to a model name or artifact stem; repeatable.")
    parser.add_argument(
        "--generate-ensemble",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate the project-defined mean ensemble (default: enabled).",
    )
    parser.add_argument("--rdkit-python", type=Path, help="Python executable able to import RDKit.")
    parser.add_argument("--maven-executable", default=DEFAULT_MAVEN_EXECUTABLE, help="Maven executable for CDK-only descriptors.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    args = build_parser().parse_args(argv)
    try:
        result = run_huskin_pipeline(
            input_path=args.input,
            output_dir=args.output,
            models_dir=args.models_dir,
            smiles_column=args.smiles_column,
            target_column=args.target_column,
            model_selectors=args.models,
            generate_ensemble=args.generate_ensemble,
            target_unit=args.target_unit,
            rdkit_python=args.rdkit_python,
            maven_executable=args.maven_executable,
            verbose=args.verbose,
        )
    except Exception:
        LOGGER.exception("HuSkinDB inference failed.")
        return 1
    LOGGER.info("Completed HuSkinDB inference with %d model rows.", len(result.metrics))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
