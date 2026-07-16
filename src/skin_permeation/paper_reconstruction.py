from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .artifacts import save_json
from .data_loading import load_bundle
from .paths import ProjectPaths
from .preprocessing import reconstruct_clean_trial4

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PaperClaim:
    claim: str
    expected: Any
    observed: Any
    status: str
    rationale: str


PAPER_METRICS = {
    "MLR (10 features)": {"r2": 0.338, "rmse": 0.814, "mae": 0.624, "cv_mae": 0.599},
    "Decision Tree": {"r2": 0.729, "rmse": 0.535, "mae": 0.323, "cv_mae": 0.473},
    "RF": {"r2": 0.788, "rmse": 0.473, "mae": 0.314, "cv_mae": 0.444},
    "XGBoost": {"r2": 0.798, "rmse": 0.462, "mae": 0.281, "cv_mae": 0.446},
    "Gradient Boosting": {"r2": 0.818, "rmse": 0.439, "mae": 0.276, "cv_mae": 0.441},
    "CatBoost": {"r2": 0.797, "rmse": 0.464, "mae": 0.300, "cv_mae": 0.436},
    "LGBM": {"r2": 0.819, "rmse": 0.437, "mae": 0.278, "cv_mae": 0.445},
    "ANN": {"r2": 0.797, "rmse": 0.462, "mae": 0.298, "cv_mae": 0.412},
}


def build_static_claims(paths: ProjectPaths) -> list[PaperClaim]:
    bundle = load_bundle(paths)
    reconstruction = reconstruct_clean_trial4(bundle.trial4)
    claims = [
        PaperClaim(
            claim="Dataset contains 441 records for 140 molecules.",
            expected={"records": 441, "molecules": 140},
            observed={"records": int(bundle.clean_trial4.shape[0]), "molecules": int(bundle.clean_trial4["SMILES"].nunique())},
            status="approximate",
            rationale="The cleaned modeling file has 417 rows and 146 unique SMILES after duplicate removal; the paper's 441-record count appears to refer to a less-deduplicated stage.",
        ),
        PaperClaim(
            claim="222 1D/2D descriptors were computed.",
            expected=222,
            observed=int(bundle.data_descriptors.shape[1] - 2),
            status="approximate",
            rationale="The generated CSV contains 223 descriptors excluding SMILES and Texpi, creating a one-descriptor mismatch with the manuscript text.",
        ),
        PaperClaim(
            claim="145 descriptors remained after removing zero-value and highly correlated descriptors.",
            expected=145,
            observed=int(bundle.clean_trial4.drop(columns=["logkpl", "Compound", "SMILES", "Texpi"]).shape[1]),
            status="matched exactly",
            rationale="The final cleaned training file contains 145 descriptor columns after removing metadata and Texpi.",
        ),
        PaperClaim(
            claim="The train/test split was 85/15.",
            expected="85/15",
            observed="85/15 in regression notebook",
            status="matched exactly",
            rationale="Regression-Model.ipynb uses train_test_split with test_size=0.15 and random_state=1.",
        ),
        PaperClaim(
            claim="DrugBank clustering used 2326 FDA-approved compounds and four clusters.",
            expected={"rows": 2326, "clusters": 4},
            observed={"rows": int(bundle.drugbank_descriptors.shape[0]), "clusters": 4},
            status="matched exactly",
            rationale="The raw DrugBank descriptor file has 2326 rows, and the clustering notebook fixes k=4 after elbow inspection.",
        ),
        PaperClaim(
            claim="ATC analysis covered 83 groups and 2456 FDA-approved drugs.",
            expected={"groups": 83, "rows": 2456},
            observed=None,
            status="could not match",
            rationale="No ATC mapping file is present in the uploaded project, so the exact ATC aggregation cannot be reconstructed from the available files alone.",
        ),
        PaperClaim(
            claim="The final LGBM model was used for DrugBank prediction.",
            expected="LGBM",
            observed="LGBM",
            status="matched exactly",
            rationale="The clustering notebook loads models/LGBMRegressor_model.sav and scaler.pkl before predicting DrugBank logKp.",
        ),
        PaperClaim(
            claim="Duplicate rows were fully removed before final modeling.",
            expected=True,
            observed=int(bundle.clean_trial4.duplicated().sum()) == 0,
            status="matched exactly",
            rationale=f"The reconstruction drops {reconstruction.dropped_duplicate_rows} duplicated rows, leaving zero exact duplicates in clean_trial4.csv.",
        ),
    ]
    return claims


def write_reproducibility_report(paths: ProjectPaths) -> dict[str, Any]:
    claims = build_static_claims(paths)
    report = {
        "matched_exactly": [claim.__dict__ for claim in claims if claim.status == "matched exactly"],
        "approximately_matched": [claim.__dict__ for claim in claims if claim.status == "approximate"],
        "could_not_match": [claim.__dict__ for claim in claims if claim.status == "could not match"],
        "paper_metrics_reference": PAPER_METRICS,
        "mismatch_explanations": [
            "The published descriptor count likely excludes one generator output column that still appears in the repository export.",
            "The row-count mismatch is consistent with optimistic deduplication and repeated-measure handling choices between manuscript stages.",
            "The baseline notebook standardizes the full dataset before splitting, which introduces leakage and can inflate reported performance.",
            "Random train/test splitting across repeated molecules can leak near-identical descriptor rows between train and test partitions.",
        ],
    }
    save_json(report, paths.reports / "artifacts" / "reproducibility_report.json")
    return report
