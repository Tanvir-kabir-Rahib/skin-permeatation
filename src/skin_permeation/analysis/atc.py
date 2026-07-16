from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ..artifacts import save_table
from ..data_loading import locate_atc_mapping
from ..dependencies import require_module
from ..paths import ProjectPaths

LOGGER = logging.getLogger(__name__)

scipy_stats = require_module("scipy.stats", "Install scipy to run ATC analysis.")
statsmodels_multitest = require_module(
    "statsmodels.stats.multitest",
    "Install statsmodels to run ATC multiple-testing correction.",
)
plt = require_module("matplotlib.pyplot", "Install matplotlib to generate ATC figures.")
seaborn = require_module("seaborn", "Install seaborn to generate ATC figures.")

kruskal = scipy_stats.kruskal
mannwhitneyu = scipy_stats.mannwhitneyu
multipletests = statsmodels_multitest.multipletests


def _load_mapping(mapping_path: Path) -> pd.DataFrame:
    if mapping_path.suffix.lower() == ".csv":
        return pd.read_csv(mapping_path)
    return pd.read_excel(mapping_path)


def run_atc_analysis(paths: ProjectPaths, predicted_drugbank: pd.DataFrame) -> dict[str, pd.DataFrame]:
    mapping_path = locate_atc_mapping(paths)
    if mapping_path is None:
        raise FileNotFoundError(
            "No ATC mapping file was found in data/. Add a CSV/XLSX containing DrugBank identifiers and ATC codes."
        )
    mapping = _load_mapping(mapping_path)
    key_candidates = [column for column in mapping.columns if column.lower() in {"name", "drug_name", "compound", "drugbank_name"}]
    atc_candidates = [column for column in mapping.columns if "atc" in column.lower()]
    if not key_candidates or not atc_candidates:
        raise ValueError("ATC mapping file must contain a drug-name column and at least one ATC-code column.")
    key_column = key_candidates[0]
    atc_column = atc_candidates[0]
    merged = predicted_drugbank.merge(mapping[[key_column, atc_column]], left_on="Name", right_on=key_column, how="inner")
    merged["ATC_group"] = merged[atc_column].astype(str).str[:3]
    filtered = merged.groupby("ATC_group").filter(lambda frame: len(frame) >= 3).copy()
    group_values = [group["predicted_logkpl"].to_numpy() for _, group in filtered.groupby("ATC_group")]
    overall = kruskal(*group_values)

    pairwise_rows = []
    groups = sorted(filtered["ATC_group"].unique())
    for i, left_group in enumerate(groups):
        left_values = filtered.loc[filtered["ATC_group"] == left_group, "predicted_logkpl"]
        for right_group in groups[i + 1 :]:
            right_values = filtered.loc[filtered["ATC_group"] == right_group, "predicted_logkpl"]
            statistic, p_value = mannwhitneyu(left_values, right_values, alternative="two-sided")
            pairwise_rows.append(
                {
                    "left_group": left_group,
                    "right_group": right_group,
                    "u_statistic": statistic,
                    "p_value": p_value,
                }
            )
    pairwise = pd.DataFrame(pairwise_rows)
    if not pairwise.empty:
        pairwise["p_value_fdr"] = multipletests(pairwise["p_value"], method="fdr_bh")[1]
        pairwise["significant_fdr"] = pairwise["p_value_fdr"] < 0.05

    distributions = filtered.groupby("ATC_group")["predicted_logkpl"].agg(["count", "mean", "std", "median"]).reset_index()
    save_table(distributions, paths.reports / "tables" / "atc_group_distributions.csv")
    save_table(pairwise, paths.reports / "tables" / "atc_pairwise_tests.csv")

    plt.figure(figsize=(max(10, len(groups) * 0.3), 6))
    order = distributions.sort_values("median", ascending=False)["ATC_group"]
    seaborn.boxplot(data=filtered, x="ATC_group", y="predicted_logkpl", order=order)
    plt.xticks(rotation=90)
    plt.title("Predicted LogKp by ATC group")
    plt.tight_layout()
    plt.savefig(paths.figures / "atc_group_boxplot.png", dpi=200)
    plt.close()

    heatmap_source = pairwise.pivot(index="left_group", columns="right_group", values="p_value_fdr")
    if not heatmap_source.empty:
        plt.figure(figsize=(10, 8))
        seaborn.heatmap(heatmap_source, cmap="viridis_r")
        plt.title("ATC pairwise FDR-adjusted p-values")
        plt.tight_layout()
        plt.savefig(paths.figures / "atc_pairwise_heatmap.png", dpi=200)
        plt.close()

    return {
        "filtered_predictions": filtered,
        "distributions": distributions,
        "pairwise": pairwise,
        "overall_test": pd.DataFrame([{"statistic": overall.statistic, "p_value": overall.pvalue}]),
    }
