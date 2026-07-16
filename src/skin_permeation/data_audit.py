from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .artifacts import save_json, save_table
from .data_loading import DatasetBundle, load_bundle
from .paths import ProjectPaths
from .preprocessing import reconstruct_clean_trial4

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetProfile:
    name: str
    rows: int
    columns: int
    duplicate_rows: int
    duplicate_smiles: int | None
    missing_values: int
    target_columns: list[str]
    identifier_columns: list[str]


def profile_dataset(name: str, df: pd.DataFrame) -> DatasetProfile:
    duplicate_smiles = int(df["SMILES"].duplicated().sum()) if "SMILES" in df.columns else None
    target_columns = [column for column in df.columns if column.lower() in {"logkpl", "logjmaxm", "texpi"}]
    identifier_columns = [column for column in df.columns if column.lower() in {"compound", "name", "smiles", "cas no"}]
    return DatasetProfile(
        name=name,
        rows=int(df.shape[0]),
        columns=int(df.shape[1]),
        duplicate_rows=int(df.duplicated().sum()),
        duplicate_smiles=duplicate_smiles,
        missing_values=int(df.isna().sum().sum()),
        target_columns=target_columns,
        identifier_columns=identifier_columns,
    )


def _compare_columns(left_name: str, left: pd.DataFrame, right_name: str, right: pd.DataFrame) -> dict[str, Any]:
    return {
        "left": left_name,
        "right": right_name,
        "shared_columns": sorted(set(left.columns) & set(right.columns)),
        "only_left": sorted(set(left.columns) - set(right.columns)),
        "only_right": sorted(set(right.columns) - set(left.columns)),
    }


def audit_datasets(paths: ProjectPaths) -> dict[str, Any]:
    bundle = load_bundle(paths)
    profiles = [
        profile_dataset("Skin Permeation.xlsx", bundle.skin_workbook),
        profile_dataset("data-original.csv", bundle.data_original),
        profile_dataset("data-descriptors.csv", bundle.data_descriptors),
        profile_dataset("trial4.csv", bundle.trial4),
        profile_dataset("clean_trial4.csv", bundle.clean_trial4),
        profile_dataset("DrugBank-descriptors.csv", bundle.drugbank_descriptors),
        profile_dataset("drug_bank_clean.csv", bundle.drugbank_clean),
    ]

    reconstruction = reconstruct_clean_trial4(bundle.trial4)
    leakage_risks = {
        "global_scaling_before_split": True,
        "repeated_molecule_rows_in_training_dataset": int(bundle.clean_trial4["SMILES"].duplicated().sum()) > 0,
        "naive_random_split_used_in_notebook": True,
        "grouped_validation_recommended": True,
    }

    transformations = [
        {
            "step": "Skin workbook notebook cleanup",
            "evidence": "1. Data Preparation.ipynb drops the last three rows from Skin Permeation.xlsx.",
            "observed_row_change": f"{len(bundle.skin_workbook)} -> {len(bundle.skin_workbook.iloc[:-3, :])}",
        },
        {
            "step": "Descriptor dataset water removal",
            "evidence": "2. Data Preparation Trial 4.ipynb removes rows 445:476 from data-original.csv/data-descriptors.csv.",
            "observed_row_change": f"{len(bundle.data_descriptors)} -> {len(bundle.trial4)}",
        },
        {
            "step": "Duplicate removal and descriptor pruning",
            "evidence": "Deleted notebook 3. Removing Correlated.ipynb drops 28 duplicated rows, then removes 8 manual and highly correlated descriptors.",
            "observed_row_change": f"{len(bundle.trial4)} -> {len(bundle.clean_trial4)}",
            "manual_drop_columns": reconstruction.dropped_manual_features,
            "dropped_correlated_feature_count": len(reconstruction.dropped_correlated_features),
        },
        {
            "step": "DrugBank cleaning",
            "evidence": "1. Data Preparation DrugBank.ipynb drops rows with missing ATSc1 or BCUTw-1l, removes geomShape, then imputes JPLogP/HybRatio/Kier3.",
            "observed_row_change": f"{len(bundle.drugbank_descriptors)} -> {len(bundle.drugbank_clean)}",
        },
    ]

    final_dataset = {
        "dataset": "data/final/clean_trial4.csv",
        "rows": int(bundle.clean_trial4.shape[0]),
        "columns": int(bundle.clean_trial4.shape[1]),
        "modeling_descriptor_count": int(bundle.clean_trial4.drop(columns=["logkpl", "Compound", "SMILES"]).shape[1]),
    }

    report = {
        "profiles": [asdict(profile) for profile in profiles],
        "comparisons": [
            _compare_columns("trial4.csv", bundle.trial4, "clean_trial4.csv", bundle.clean_trial4),
            _compare_columns("data-original.csv", bundle.data_original, "Skin Permeation.xlsx", bundle.skin_workbook),
            _compare_columns("DrugBank-descriptors.csv", bundle.drugbank_descriptors, "drug_bank_clean.csv", bundle.drugbank_clean),
        ],
        "transformations": transformations,
        "leakage_risks": leakage_risks,
        "final_modeling_dataset": final_dataset,
        "reconstructed_feature_drops": {
            "manual": reconstruction.dropped_manual_features,
            "correlated": reconstruction.dropped_correlated_features,
        },
    }
    return report


def run_audit(paths: ProjectPaths) -> dict[str, Any]:
    LOGGER.info("Running data audit.")
    report = audit_datasets(paths)
    profiles_frame = pd.DataFrame(report["profiles"])
    save_table(profiles_frame, paths.reports / "tables" / "dataset_profiles.csv")
    save_json(report, paths.reports / "artifacts" / "data_audit.json")
    return report
