from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .artifacts import save_json
from .paths import ProjectPaths

LOGGER = logging.getLogger(__name__)

DESCRIPTOR_CLASS_PATTERN = re.compile(r'"(org\.openscience\.cdk\.qsar\.descriptors\.molecular\.[^"]+)"')


@dataclass(frozen=True)
class DescriptorValidationResult:
    java_descriptor_class_count: int
    generated_column_count: int
    generated_descriptor_count_without_targets: int
    clean_trial4_descriptor_count: int
    paper_reported_descriptor_count_before_filtering: int
    paper_reported_descriptor_count_after_filtering: int


def parse_java_descriptor_classes(java_source: str) -> list[str]:
    return DESCRIPTOR_CLASS_PATTERN.findall(java_source)


def validate_descriptor_inventory(paths: ProjectPaths) -> dict:
    java_source = (paths.descriptors_generator / "src" / "main" / "java" / "org" / "rami" / "dg" / "DescriptorsGenerator.java").read_text(encoding="utf-8")
    descriptor_classes = parse_java_descriptor_classes(java_source)
    generated = pd.read_csv(paths.raw_data / "data-descriptors.csv")
    clean_trial4 = pd.read_csv(paths.final_data / "clean_trial4.csv")
    result = DescriptorValidationResult(
        java_descriptor_class_count=len(descriptor_classes),
        generated_column_count=int(generated.shape[1]),
        generated_descriptor_count_without_targets=int(generated.shape[1] - 2),
        clean_trial4_descriptor_count=int(clean_trial4.drop(columns=["logkpl", "Compound", "SMILES"]).shape[1]),
        paper_reported_descriptor_count_before_filtering=222,
        paper_reported_descriptor_count_after_filtering=145,
    )
    payload = {
        "descriptor_classes": descriptor_classes,
        "summary": result.__dict__,
        "notes": [
            "The Java generator writes SMILES and Texpi before the descriptor panel.",
            "The generated CSV contains 223 descriptor columns after excluding SMILES and Texpi, whereas the paper reports 222 descriptors.",
            "This off-by-one discrepancy likely reflects a descriptor-count reporting mismatch or a descriptor later removed before the manuscript summary.",
            "The final modeling table contains 146 features after excluding logkpl, Compound, and SMILES; one of those features is Texpi, leaving 145 descriptor features and matching the paper.",
        ],
    }
    save_json(payload, paths.reports / "artifacts" / "descriptor_validation.json")
    return payload


def run_java_descriptor_generator(
    paths: ProjectPaths,
    java_executable: str = "java",
    maven_executable: str = "mvn",
) -> None:
    project_dir = paths.descriptors_generator
    LOGGER.info("Running Maven package in %s", project_dir)
    subprocess.run([maven_executable, "-q", "-DskipTests", "package"], cwd=project_dir, check=True)
    subprocess.run([java_executable, "-cp", "target/classes", "org.rami.dg.DescriptorsGenerator"], cwd=project_dir, check=True)
