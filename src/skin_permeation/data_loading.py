from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .paths import ProjectPaths

LOGGER = logging.getLogger(__name__)


class DataConsistencyError(RuntimeError):
    """Raised when an expected dataset shape or schema does not match the observed data."""


@dataclass(frozen=True)
class DatasetBundle:
    skin_workbook: pd.DataFrame
    data_original: pd.DataFrame
    data_descriptors: pd.DataFrame
    trial4: pd.DataFrame
    clean_trial4: pd.DataFrame
    drugbank_descriptors: pd.DataFrame
    drugbank_clean: pd.DataFrame


def load_bundle(paths: ProjectPaths) -> DatasetBundle:
    LOGGER.info("Loading core datasets from %s", paths.data)
    return DatasetBundle(
        skin_workbook=pd.read_excel(paths.raw_data / "Skin Permeation.xlsx"),
        data_original=pd.read_csv(paths.raw_data / "data-original.csv"),
        data_descriptors=pd.read_csv(paths.raw_data / "data-descriptors.csv"),
        trial4=pd.read_csv(paths.processed_data / "trial4.csv"),
        clean_trial4=pd.read_csv(paths.final_data / "clean_trial4.csv"),
        drugbank_descriptors=pd.read_csv(paths.raw_data / "DrugBank-descriptors.csv"),
        drugbank_clean=pd.read_csv(paths.processed_data / "drug_bank_clean.csv"),
    )


def locate_atc_mapping(paths: ProjectPaths) -> Path | None:
    candidates = [
        *paths.raw_data.glob("*ATC*.csv"),
        *paths.raw_data.glob("*ATC*.xlsx"),
        *paths.processed_data.glob("*ATC*.csv"),
        *paths.processed_data.glob("*ATC*.xlsx"),
        *paths.data.glob("**/*atc*.csv"),
        *paths.data.glob("**/*atc*.xlsx"),
    ]
    return candidates[0] if candidates else None
