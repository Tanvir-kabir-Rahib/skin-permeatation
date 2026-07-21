"""Audit experimental context for HuSkinDB LGBM matches within a tolerance.

The analysis preserves measurement-level conditions and references, while also
summarizing variability across *all* HuSkinDB measurements for every compound
with at least one LGBM prediction inside the requested error tolerance.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)
DEFAULT_SOURCE = Path("data/huskinDB/huskinDB.csv")
DEFAULT_PREDICTIONS = Path("results/huskinDB/huskin_predictions.csv")
DEFAULT_FILTERED = Path(
    "results/huskinDB/within_0.6_LGBM/"
    "huskin_LGBM_predictions_within_0.6.csv"
)
DEFAULT_OUTPUT = Path(
    "results/huskinDB/within_0.6_LGBM/experimental_context"
)
DEFAULT_TOLERANCE = 0.6
EXPERIMENTAL_COLUMN = "logkp (cm/s)"
PREDICTION_COLUMN = "Predicted_logKp_LGBM"

CONDITION_COLUMNS = [
    "skin source type",
    "skin source site",
    "used layer",
    "skin preparation",
    "storage temperature (°C)",
    "storage duration (days)",
    "neat",
    "donor/skin surface temperature (°C)",
    "donor pH",
    "donor type",
    "acceptor temperature (°C)",
    "acceptor pH",
    "acceptor type",
    "cell type",
]
UNAVAILABLE_VALUES = {
    "",
    "unknown",
    "nan",
    "na",
    "n/a",
    "not available",
    "not reported",
    "none",
}


def is_unavailable(series: pd.Series) -> pd.Series:
    """Return a Boolean mask for null or sentinel unavailable values."""

    normalized = series.fillna("").astype(str).str.strip().str.casefold()
    return normalized.isin(UNAVAILABLE_VALUES)


def unique_join(values: Iterable[object], separator: str = " | ") -> str:
    """Join non-empty unique values while preserving their first-seen order."""

    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if not text or text.casefold() in UNAVAILABLE_VALUES or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return separator.join(output)


def doi_url(value: object) -> str:
    """Convert a DOI value to a resolvable URL without changing its identifier."""

    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text or text.casefold() in UNAVAILABLE_VALUES:
        return ""
    if text.startswith(("https://", "http://")):
        return text
    if text.lower().startswith("doi.org/"):
        return f"https://{text}"
    return f"https://doi.org/{text}"


def condition_signature(row: pd.Series) -> str:
    """Create an explicit, auditable experimental-condition signature."""

    return "; ".join(f"{column}={row[column]}" for column in CONDITION_COLUMNS)


def numeric_summary(series: pd.Series, prefix: str) -> dict[str, float]:
    """Return robust descriptive statistics for one numeric measurement series."""

    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {
            f"{prefix}_Min": float("nan"),
            f"{prefix}_Max": float("nan"),
            f"{prefix}_Mean": float("nan"),
            f"{prefix}_Median": float("nan"),
            f"{prefix}_SD": float("nan"),
            f"{prefix}_IQR": float("nan"),
            f"{prefix}_Range": float("nan"),
        }
    minimum = float(values.min())
    maximum = float(values.max())
    return {
        f"{prefix}_Min": minimum,
        f"{prefix}_Max": maximum,
        f"{prefix}_Mean": float(values.mean()),
        f"{prefix}_Median": float(values.median()),
        f"{prefix}_SD": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
        f"{prefix}_IQR": float(values.quantile(0.75) - values.quantile(0.25)),
        f"{prefix}_Range": maximum - minimum,
    }


def variability_band(measurement_count: int, experimental_range: float) -> str:
    """Return an explicit descriptive variability band in log10 units."""

    if measurement_count <= 1 or pd.isna(experimental_range):
        return "Single measurement"
    if experimental_range > 1.0:
        return "High (>1.0 log unit)"
    if experimental_range > 0.5:
        return "Moderate (>0.5 to 1.0 log unit)"
    return "Low (≤0.5 log unit)"


def validate_inputs(
    source: pd.DataFrame,
    predictions: pd.DataFrame,
    filtered: pd.DataFrame,
) -> None:
    """Validate row alignment and required columns before analysis."""

    source_required = {
        "Compound name",
        "Smiles",
        EXPERIMENTAL_COLUMN,
        "reference",
        "DOI",
        "LogP",
        "molecular weight",
        *CONDITION_COLUMNS,
    }
    prediction_required = {
        "Smiles",
        "Canonical_SMILES",
        EXPERIMENTAL_COLUMN,
        PREDICTION_COLUMN,
    }
    filtered_required = {"Source_Data_Row", "Canonical_SMILES"}
    for label, frame, required in (
        ("source", source, source_required),
        ("predictions", predictions, prediction_required),
        ("filtered", filtered, filtered_required),
    ):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"Missing required {label} columns: {missing}")

    if len(source) != len(predictions):
        raise ValueError(
            f"Source/prediction row mismatch: {len(source)} versus {len(predictions)}."
        )
    if not source["Smiles"].astype(str).equals(predictions["Smiles"].astype(str)):
        raise ValueError("Source and prediction SMILES rows are not aligned.")
    source_target = pd.to_numeric(source[EXPERIMENTAL_COLUMN], errors="coerce")
    prediction_target = pd.to_numeric(
        predictions[EXPERIMENTAL_COLUMN], errors="coerce"
    )
    if not np.allclose(source_target, prediction_target, equal_nan=True):
        raise ValueError("Source and prediction experimental targets are not aligned.")


def prepare_measurement_context(
    source: pd.DataFrame,
    predictions: pd.DataFrame,
    filtered: pd.DataFrame,
    tolerance: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return selected measurements, all-compound context, and the full joined data."""

    joined = source.copy()
    joined.insert(0, "Source_Data_Row", np.arange(1, len(joined) + 1, dtype=int))
    joined.insert(3, "Canonical_SMILES", predictions["Canonical_SMILES"])
    joined[PREDICTION_COLUMN] = pd.to_numeric(
        predictions[PREDICTION_COLUMN], errors="coerce"
    )
    experimental = pd.to_numeric(joined[EXPERIMENTAL_COLUMN], errors="coerce")
    joined["Prediction_Error_LGBM"] = joined[PREDICTION_COLUMN] - experimental
    joined["Absolute_Error_LGBM"] = joined["Prediction_Error_LGBM"].abs()
    joined["Within_Tolerance_LGBM"] = joined["Absolute_Error_LGBM"].le(
        tolerance + np.finfo(float).eps
    )
    expected_rows = set(filtered["Source_Data_Row"].astype(int))
    calculated_rows = set(
        joined.loc[joined["Within_Tolerance_LGBM"], "Source_Data_Row"].astype(int)
    )
    if expected_rows != calculated_rows:
        raise ValueError(
            "Calculated tolerance selection does not match the supplied filtered CSV."
        )

    joined["DOI_URL"] = joined["DOI"].map(doi_url)
    joined["Condition_Signature"] = joined.apply(condition_signature, axis=1)
    selected_compounds = set(
        joined.loc[joined["Within_Tolerance_LGBM"], "Canonical_SMILES"]
    )
    context = joined[joined["Canonical_SMILES"].isin(selected_compounds)].copy()
    selected = context[context["Within_Tolerance_LGBM"]].copy()
    return selected, context, joined


def build_compound_summary(context: pd.DataFrame) -> pd.DataFrame:
    """Summarize variability, conditions, and references by canonical compound."""

    rows: list[dict[str, object]] = []
    grouped = context.groupby("Canonical_SMILES", sort=False, dropna=False)
    for canonical_smiles, group in grouped:
        selected = group[group["Within_Tolerance_LGBM"]]
        all_experimental = numeric_summary(
            group[EXPERIMENTAL_COLUMN], "Experimental_All"
        )
        selected_experimental = numeric_summary(
            selected[EXPERIMENTAL_COLUMN], "Experimental_Selected"
        )
        all_predictions = numeric_summary(
            group[PREDICTION_COLUMN], "Prediction_All"
        )
        selected_predictions = numeric_summary(
            selected[PREDICTION_COLUMN], "Prediction_Selected"
        )
        all_measurements = len(group)
        selected_measurements = len(selected)
        experimental_range = all_experimental["Experimental_All_Range"]
        row: dict[str, object] = {
            "Compound_Name": group["Compound name"].iloc[0],
            "Canonical_SMILES": canonical_smiles,
            "SMILES_Values": unique_join(group["Smiles"]),
            "All_Measurements": all_measurements,
            "Selected_Measurements": selected_measurements,
            "Excluded_Measurements": all_measurements - selected_measurements,
            "Selection_Coverage_Percent": 100.0
            * selected_measurements
            / all_measurements,
            "Variability_Band": variability_band(
                all_measurements, experimental_range
            ),
            **all_experimental,
            "Experimental_All_Fold_Range_Kp": (
                10.0**experimental_range
                if pd.notna(experimental_range)
                else float("nan")
            ),
            **selected_experimental,
            **all_predictions,
            "Prediction_All_Unique_Count": int(
                group[PREDICTION_COLUMN].nunique(dropna=True)
            ),
            **selected_predictions,
            "Prediction_Selected_Unique_Count": int(
                selected[PREDICTION_COLUMN].nunique(dropna=True)
            ),
            "Absolute_Error_Selected_Min": float(
                selected["Absolute_Error_LGBM"].min()
            ),
            "Absolute_Error_Selected_Max": float(
                selected["Absolute_Error_LGBM"].max()
            ),
            "Absolute_Error_Selected_Mean": float(
                selected["Absolute_Error_LGBM"].mean()
            ),
            "Condition_Combinations_All": int(
                group["Condition_Signature"].nunique(dropna=True)
            ),
            "Condition_Combinations_Selected": int(
                selected["Condition_Signature"].nunique(dropna=True)
            ),
            "Donor_Surface_Temperatures_All": unique_join(
                group["donor/skin surface temperature (°C)"]
            ),
            "Donor_Surface_Temperatures_Selected": unique_join(
                selected["donor/skin surface temperature (°C)"]
            ),
            "References_All": unique_join(group["reference"], " || "),
            "References_Selected": unique_join(selected["reference"], " || "),
            "Reference_Count_All": int(group["reference"].nunique(dropna=True)),
            "Reference_Count_Selected": int(
                selected["reference"].nunique(dropna=True)
            ),
            "DOIs_All": unique_join(group["DOI"], " || "),
            "DOIs_Selected": unique_join(selected["DOI"], " || "),
            "DOI_Count_All": int(group["DOI"].nunique(dropna=True)),
            "DOI_Count_Selected": int(selected["DOI"].nunique(dropna=True)),
        }
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(
        ["Experimental_All_Range", "All_Measurements", "Compound_Name"],
        ascending=[False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    summary.insert(0, "Variability_Rank", np.arange(1, len(summary) + 1))
    return summary


def top_value_counts(series: pd.Series, limit: int = 5) -> str:
    """Return the most frequent reported values with counts."""

    available = series[~is_unavailable(series)].astype(str).str.strip()
    counts = available.value_counts(dropna=False).head(limit)
    return " | ".join(f"{value} ({count})" for value, count in counts.items())


def build_condition_completeness(
    selected: pd.DataFrame,
    context: pd.DataFrame,
) -> pd.DataFrame:
    """Profile unavailable sentinel values in condition and citation fields."""

    rows: list[dict[str, object]] = []
    for column in [*CONDITION_COLUMNS, "reference", "DOI"]:
        selected_missing = is_unavailable(selected[column])
        context_missing = is_unavailable(context[column])
        rows.append(
            {
                "Field": column,
                "Selected_Rows": len(selected),
                "Selected_Unavailable_Count": int(selected_missing.sum()),
                "Selected_Unavailable_Percent": 100.0
                * float(selected_missing.mean()),
                "Selected_Distinct_Reported_Values": int(
                    selected.loc[~selected_missing, column].nunique(dropna=True)
                ),
                "Selected_Top_Reported_Values": top_value_counts(
                    selected[column]
                ),
                "All_Context_Rows": len(context),
                "All_Context_Unavailable_Count": int(context_missing.sum()),
                "All_Context_Unavailable_Percent": 100.0
                * float(context_missing.mean()),
                "All_Context_Distinct_Reported_Values": int(
                    context.loc[~context_missing, column].nunique(dropna=True)
                ),
                "All_Context_Top_Reported_Values": top_value_counts(
                    context[column]
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "Selected_Unavailable_Percent", ascending=False
    )


def build_reference_index(selected: pd.DataFrame) -> pd.DataFrame:
    """Create a reference/DOI index for the selected measurements."""

    rows: list[dict[str, object]] = []
    grouped = selected.groupby(["reference", "DOI"], sort=True, dropna=False)
    for (reference, doi), group in grouped:
        rows.append(
            {
                "Reference": reference,
                "DOI": doi,
                "DOI_URL": doi_url(doi),
                "Selected_Measurements": len(group),
                "Unique_Compounds": int(
                    group["Canonical_SMILES"].nunique(dropna=True)
                ),
                "Compound_Names": unique_join(group["Compound name"], " || "),
                "Experimental_logKp_Min": float(group[EXPERIMENTAL_COLUMN].min()),
                "Experimental_logKp_Max": float(group[EXPERIMENTAL_COLUMN].max()),
                "Condition_Combinations": int(
                    group["Condition_Signature"].nunique(dropna=True)
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["Selected_Measurements", "Reference"], ascending=[False, True]
    )


def build_readme(
    selected: pd.DataFrame,
    context: pd.DataFrame,
    summary: pd.DataFrame,
    condition_quality: pd.DataFrame,
    source_path: Path,
    predictions_path: Path,
    filtered_path: Path,
    tolerance: float,
) -> pd.DataFrame:
    """Create a compact methodology and findings sheet."""

    repeated = summary[summary["All_Measurements"] > 1]
    partial = summary[summary["Excluded_Measurements"] > 0]
    multiple_predictions = summary[summary["Prediction_All_Unique_Count"] > 1]
    fields_over_half_unknown = int(
        (condition_quality["Selected_Unavailable_Percent"] > 50).sum()
    )
    records = [
        ("Purpose", "Experimental conditions, references, and within-compound variability for LGBM matches."),
        ("Selection rule", f"|Predicted logKp − experimental logKp| ≤ {tolerance:.2f}."),
        ("Selection warning", "Experimental logKp was used to select these rows; this is not unbiased model validation."),
        ("Source dataset", str(source_path)),
        ("Prediction dataset", str(predictions_path)),
        ("Filtered LGBM dataset", str(filtered_path)),
        ("Selected measurement rows", len(selected)),
        ("Selected unique compounds", summary["Canonical_SMILES"].nunique()),
        ("All measurements for selected compounds", len(context)),
        ("Excluded context measurements", len(context) - len(selected)),
        ("Single-measurement compounds", int((summary["All_Measurements"] == 1).sum())),
        ("Repeated-measurement compounds", len(repeated)),
        ("Partially selected compounds", len(partial)),
        ("Compounds with experimental range >0.5 log unit", int((summary["Experimental_All_Range"] > 0.5).sum())),
        ("Compounds with experimental range >1.0 log unit", int((summary["Experimental_All_Range"] > 1.0).sum())),
        ("Compounds with experimental range >2.0 log units", int((summary["Experimental_All_Range"] > 2.0).sum())),
        ("Compounds with multiple LGBM predictions", len(multiple_predictions)),
        ("Maximum within-compound LGBM prediction range", float(summary["Prediction_All_Range"].max())),
        ("LGBM condition feature", "Texpi, mapped from donor/skin surface temperature (°C); the other 145 required features are molecular descriptors."),
        ("Condition fields >50% unavailable in selected rows", fields_over_half_unknown),
        ("Reference coverage", f"{selected['reference'].nunique()} distinct references; no selected reference values marked unknown."),
        ("DOI coverage", f"{selected['DOI'].nunique()} distinct DOI values; no selected DOI values marked unknown."),
        ("Experimental variability", "Calculated across all HuSkinDB measurements for each selected compound, including measurements outside ±0.6."),
        ("Fold range definition", "10^(maximum experimental logKp − minimum experimental logKp)."),
        ("Variability bands", "Single measurement; Low ≤0.5; Moderate >0.5 to 1.0; High >1.0 log10 unit."),
        ("Interpretation limit", "Condition and reference differences are observational and co-vary; they do not establish causal effects on permeability."),
        ("Unknown-value handling", "Nulls and sentinel strings such as unknown, N/A, and not reported are treated as unavailable."),
    ]
    return pd.DataFrame(records, columns=["Item", "Value"])


def ordered_measurement_columns() -> list[str]:
    """Return the user-facing order for measurement-detail sheets."""

    return [
        "Source_Data_Row",
        "Compound name",
        "Smiles",
        "Canonical_SMILES",
        EXPERIMENTAL_COLUMN,
        PREDICTION_COLUMN,
        "Prediction_Error_LGBM",
        "Absolute_Error_LGBM",
        "Within_Tolerance_LGBM",
        *CONDITION_COLUMNS,
        "reference",
        "DOI",
        "DOI_URL",
        "LogP",
        "molecular weight",
        "Condition_Signature",
    ]


def run_analysis(
    source_path: Path,
    predictions_path: Path,
    filtered_path: Path,
    output_dir: Path,
    tolerance: float,
) -> dict[str, Path]:
    """Run the analysis, write auditable CSVs, and return their paths."""

    if tolerance < 0:
        raise ValueError("Tolerance must be non-negative.")
    source = pd.read_csv(source_path)
    predictions = pd.read_csv(predictions_path)
    filtered = pd.read_csv(filtered_path)
    validate_inputs(source, predictions, filtered)
    selected, context, _ = prepare_measurement_context(
        source, predictions, filtered, tolerance
    )
    summary = build_compound_summary(context)
    condition_quality = build_condition_completeness(selected, context)
    references = build_reference_index(selected)
    readme = build_readme(
        selected,
        context,
        summary,
        condition_quality,
        source_path,
        predictions_path,
        filtered_path,
        tolerance,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "README": readme,
        "Selected_Measurements": selected[ordered_measurement_columns()],
        "All_Measurements_Context": context[ordered_measurement_columns()],
        "Compound_Variability": summary,
        "Reference_Index": references,
        "Condition_Data_Quality": condition_quality,
    }
    filenames = {
        "README": "README.csv",
        "Selected_Measurements": "lgbm_within_0.6_selected_measurement_conditions.csv",
        "All_Measurements_Context": "lgbm_within_0.6_all_measurements_context.csv",
        "Compound_Variability": "lgbm_within_0.6_compound_variability.csv",
        "Reference_Index": "lgbm_within_0.6_reference_index.csv",
        "Condition_Data_Quality": "lgbm_within_0.6_condition_data_quality.csv",
    }
    paths: dict[str, Path] = {}
    for name, table in tables.items():
        path = output_dir / filenames[name]
        table.to_csv(path, index=False, float_format="%.8f")
        paths[name] = path

    manifest = {
        "tolerance": tolerance,
        "selected_measurements": len(selected),
        "selected_compounds": len(summary),
        "all_context_measurements": len(context),
        "files": {name: str(path) for name, path in paths.items()},
    }
    manifest_path = output_dir / "analysis_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    paths["Manifest"] = manifest_path
    LOGGER.info(
        "Saved context for %d selected measurements, %d compounds, and %d total context rows to %s.",
        len(selected),
        len(summary),
        len(context),
        output_dir,
    )
    return paths


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(
        description="Audit HuSkinDB experimental context for LGBM tolerance matches."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--filtered", type=Path, default=DEFAULT_FILTERED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    return parser


def main() -> int:
    """Run the command-line interface."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args()
    run_analysis(
        source_path=args.source,
        predictions_path=args.predictions,
        filtered_path=args.filtered,
        output_dir=args.output,
        tolerance=args.tolerance,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
