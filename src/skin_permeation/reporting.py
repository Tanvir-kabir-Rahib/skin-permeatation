from __future__ import annotations

from pathlib import Path

import pandas as pd

from .paths import ProjectPaths


def _render_table(frame: pd.DataFrame) -> str:
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        return "```text\n" + frame.to_string(index=False) + "\n```"


def build_markdown_report(paths: ProjectPaths) -> str:
    baseline_path = paths.reports / "tables" / "paper_baseline_metrics.csv"
    benchmark_path = paths.reports / "tables" / "benchmark_metrics.csv"
    improved_path = paths.reports / "tables" / "improved_metrics.csv"
    audit_path = paths.reports / "artifacts" / "data_audit.json"
    reproducibility_path = paths.reports / "artifacts" / "reproducibility_report.json"

    sections = [
        "# Skin Permeation Reproduction Report",
        "",
        "## Reconstructed workflow",
        "- Descriptor generation: Java CDK 2.8 generator on `data-original.csv` with largest-fragment handling for salts/hydrates.",
        "- Paper baseline preprocessing: water-row removal, descriptor imputation, duplicate removal, manual descriptor removal, correlation pruning, global scaling before split, 85/15 random split.",
        "- Improved preprocessing: grouped split by SMILES, train-only preprocessing, randomized search, bootstrap CIs, and applicability-domain analysis.",
        "",
        "## Generated artifacts",
        f"- Data audit JSON: `{audit_path.relative_to(paths.root)}`",
        f"- Reproducibility report JSON: `{reproducibility_path.relative_to(paths.root)}`",
    ]

    if baseline_path.exists():
        baseline = pd.read_csv(baseline_path)
        sections.extend(["", "## Reproduced baseline metrics", _render_table(baseline)])
    else:
        sections.extend(["", "## Reproduced baseline metrics", "Baseline training has not been executed yet in this environment."])

    if benchmark_path.exists():
        benchmark = pd.read_csv(benchmark_path)
        sections.extend(
            [
                "",
                "## Benchmark-Optimized Metrics",
                "This section uses a naive row split like the paper, but with train-only preprocessing, target transformation, stronger search spaces, and ensemble models. It is designed to challenge the paper's headline performance without adding leakage.",
                _render_table(benchmark),
            ]
        )
    else:
        sections.extend(["", "## Benchmark-Optimized Metrics", "Benchmark training has not been executed yet in this environment."])

    if improved_path.exists():
        improved = pd.read_csv(improved_path)
        sections.extend(["", "## Improved metrics", _render_table(improved)])
    else:
        sections.extend(["", "## Improved metrics", "Improved training has not been executed yet in this environment."])

    sections.extend(
        [
            "",
            "## Key methodological issues",
            "- The notebook baseline scales the full dataset before splitting, which leaks test-set information.",
            "- Multiple rows share the same SMILES but different permeability measurements, so naive random splits can place the same molecule in train and test.",
            "- The repository does not include the ATC mapping file needed to exactly reproduce the paper's ATC analysis.",
            "",
            "## Whether the paper's conclusions hold",
            "- The repository evidence supports the qualitative conclusion that boosted tree models outperform simple linear baselines on this descriptor set.",
            "- The exact headline metrics should be treated cautiously until grouped validation and train-only preprocessing are applied.",
            "",
            "## Where the improved pipeline is stronger",
            "- It separates molecule groups across train and test when repeated molecules are present.",
            "- It tunes models without leaking preprocessing statistics from the holdout set.",
            "- It adds uncertainty intervals, applicability-domain checks, and structured error analysis.",
            "",
            "## Performance interpretation",
            "- If the benchmark-optimized section outperforms the paper while the grouped section does not, that usually means the paper's split design was easier rather than the chemistry becoming more predictable.",
            "- The grouped metrics remain the more defensible estimate of real-world generalization to unseen molecules.",
        ]
    )
    return "\n".join(sections) + "\n"
