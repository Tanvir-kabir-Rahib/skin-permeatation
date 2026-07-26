# Skin Permeation Reproduction

This project rebuilds the repository for the paper "Predictive modeling of skin permeability for molecules: Investigating FDA-approved drug permeability with various AI algorithms" into a reproducible Python package with a faithful paper baseline, a benchmark-optimized pipeline aimed at stronger headline metrics on a naive split, and a stricter grouped evaluation pipeline.

## Project structure

```text
configs/
data/
descriptors-generator/
figures/
models/
reports/
scripts/
src/skin_permeation/
tests/
```

## Workflow guide

See [FDA_APPROVED_DRUG_CLASSIFICATION_WORKFLOW.md](FDA_APPROVED_DRUG_CLASSIFICATION_WORKFLOW.md) for the end-to-end workflow covering model development, DrugBank prediction, clustering, and optional ATC-based FDA drug classification.

## What the code reconstructs

- Data audit across every uploaded CSV/XLSX dataset.
- Java CDK descriptor-generation validation and optional Java wrapper execution.
- Paper-style baseline preprocessing:
  - remove water rows `445:476`,
  - impute `topoShape` by mean,
  - impute `HybRatio`, `JPLogP`, `Kier3` by median,
  - remove 28 exact duplicate rows,
  - remove eight manually flagged descriptors,
  - remove highly correlated descriptors at `|r| >= 0.95`,
  - global scaling before the split to mirror the original notebook,
  - 85/15 random split with `random_state=1`.
- Baseline model training for MLR, Decision Tree, RF, XGBoost, Gradient Boosting, CatBoost, LightGBM, ANN, SVR, and Lasso.
- Benchmark-optimized model training with train-only feature filtering, target transformation, larger search spaces, and stronger ensemble models.
- Improved grouped validation, hyperparameter tuning, stacking, bootstrap intervals, error analysis, DrugBank prediction, clustering, and optional ATC analysis if an ATC mapping file is supplied.

## Environment setup

### Conda

```bash
conda env create -f environment.yml
conda activate skin-permeation-repro
```

### Pip

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements_reproduction.txt
```

## Run commands

### Full pipeline

```bash
python scripts/run_full_pipeline.py full-run --baseline-config configs/paper_baseline.yaml --benchmark-config configs/benchmark.yaml --improved-config configs/improved.yaml
```

### Individual stages

```bash
python scripts/run_full_pipeline.py audit-data
python scripts/run_full_pipeline.py validate-descriptors
python scripts/run_full_pipeline.py reproducibility-report
python scripts/run_full_pipeline.py run-baseline --config configs/paper_baseline.yaml
python scripts/run_full_pipeline.py run-benchmark --config configs/benchmark.yaml
python scripts/run_full_pipeline.py run-improved --config configs/improved.yaml
python scripts/run_full_pipeline.py cluster-drugbank --model-path models/reproduction/benchmark/best_benchmark_model.joblib
python scripts/run_full_pipeline.py run-atc --predictions reports/tables/drugbank_predictions_and_clusters.csv
```

### Java descriptor regeneration

Requires Java 18+ and Maven on `PATH`.

```bash
python scripts/run_java_descriptors.py --java java --maven mvn
```

## External QSAR/QSPR validation

Use `skin_permeation.external_validation.run_external_validation` to validate an external prediction CSV with experimental and predicted logKp values. The validator writes `external_validation_metrics.csv`, `external_validation_metrics.xlsx`, a journal-ready table, and `experimental_vs_predicted_logKp.png` to `outputs/external_validation/`.

```python
from skin_permeation.external_validation import run_external_validation

summary_table = run_external_validation(
    csv_path="external_test_predictions.csv",
    actual_col="Experimental_logKp",
    pred_col="Predicted_logKp",
    train_csv_path="train_data.csv",
    train_target_col="Experimental_logKp",
)
```

Metric meanings in simple terms:

- `R2_ext` measures how much external-set variance is explained by the predictions.
- `RMSE_ext` is the typical prediction error in logKp units, with larger errors penalized more strongly.
- `MAE_ext` is the average absolute prediction error in logKp units.
- `Q2_F1`, `Q2_F2`, and `Q2_F3` are external predictivity checks that compare prediction error against reference variation in the training or external response values.
- `CCC_ext` measures both correlation and agreement, so it penalizes predictions that are correlated but systematically shifted.
- `r_m^2` checks whether high correlation remains credible after accounting for origin-forced agreement.
- Golbraikh-Tropsha parameters (`R0^2`, `R0'^2`, `k`, `k'`, and the R2/R0 ratios) test whether predicted and experimental values agree without large slope bias.

`R2_ext` alone is not enough because a model can have a decent squared correlation while still showing biased slopes, shifted predictions, poor concordance, or unacceptable absolute error. The combined metrics give a more defensible view of whether a skin-permeability QSAR/QSPR model predicts unseen compounds, not just whether predicted and experimental values move in roughly the same direction.

For journal reporting, include the generated table with columns `Metric`, `Value`, `Recommended threshold`, and `Interpretation`, and cite the metric definitions used for external QSAR/QSPR validation in the Methods or Model Validation section. A concise reporting sentence is: "The external validation results indicate whether the developed skin-permeability model satisfies the recommended QSAR/QSPR predictivity criteria for unseen compounds."

## Expected outputs

- `reports/artifacts/data_audit.json`
- `reports/artifacts/descriptor_validation.json`
- `reports/artifacts/reproducibility_report.json`
- `reports/tables/dataset_profiles.csv`
- `reports/tables/paper_baseline_metrics.csv`
- `reports/tables/paper_vs_baseline_comparison.csv`
- `reports/tables/benchmark_metrics.csv`
- `reports/tables/benchmark_vs_paper_comparison.csv`
- `reports/tables/improved_metrics.csv`
- `reports/tables/drugbank_predictions_and_clusters.csv`
- `reports/tables/drugbank_cluster_selection.csv`
- `reports/tables/drugbank_cluster_summary.csv`
- `reports/tables/atc_group_distributions.csv` and `reports/tables/atc_pairwise_tests.csv` when an ATC mapping file is available
- `figures/` plots for predicted-vs-actual, clustering, and ATC summaries
- `models/reproduction/` fitted models and scalers
- `REPORT.md`

## Regenerating publication figures

The Abdallah et al. paper is used only as a reference for broad figure categories and
visual presentation. The generated figures are built from this repository's current
datasets, preprocessing, model results, validation artifacts, DrugBank predictions,
clustering, and applicability-domain outputs. They do not copy the paper's images,
values, model rankings, cluster assignments, captions, or conclusions.

Run the complete project-aware figure pipeline with:

```bash
python scripts/regenerate_project_figures.py
```

By default, the script detects:

- `data/final/clean_trial4.csv` as the experimental LogKp and descriptor dataset
- `reports/tables/improved_metrics.csv` as the preferred grouped-validation model comparison
- `reports/tables/improved_test_errors.csv` as the preferred holdout prediction artifact
- `models/reproduction/improved/` as the fitted-model directory
- `reports/tables/drugbank_predictions_and_clusters.csv` as the external prediction dataset

All inputs can be overridden:

```bash
python scripts/regenerate_project_figures.py \
  --paper-pdf path/to/reference-paper.pdf \
  --skin-data data/final/clean_trial4.csv \
  --descriptor-data data/final/clean_trial4.csv \
  --external-data reports/tables/drugbank_predictions_and_clusters.csv \
  --metrics-data reports/tables/improved_metrics.csv \
  --prediction-data reports/tables/improved_test_errors.csv \
  --model-dir models/reproduction/improved \
  --output-dir figures \
  --target-column logkpl \
  --random-state 2024 \
  --top-n-features 15
```

To regenerate figures for the benchmark-optimized models instead of the grouped
improved models, point the script at the benchmark metrics and model artifacts:

```bash
python scripts/regenerate_project_figures.py \
  --metrics-data reports/tables/benchmark_metrics.csv \
  --model-dir models/reproduction/benchmark \
  --output-dir figures/benchmark_models
```

The paper PDF is optional and is recorded for documentation only. It is never used as
a scientific data source and no images are extracted from it.

Each supported figure is saved to `figures/` as both a 300 dpi PNG and a PDF. Optional
figures are skipped independently when their required data are unavailable. The command
prints the exact reason for every skip and writes the same generated/skipped inventory
to `figures/figure_generation_summary.json`.

## Assumptions and limitations

- The uploaded repository does not contain the ATC mapping used in the paper, so ATC reproduction is implemented but requires an added mapping file.
- The paper reports 222 descriptors, but the repository CSV export contains 223 descriptor columns after removing `SMILES` and `Texpi`; the code records this mismatch explicitly.
- The faithful baseline intentionally preserves the leakage-prone scaling and naive split logic from the notebook for comparison purposes.
- The benchmark-optimized pipeline is intended to maximize predictive performance on a paper-style naive split while still keeping preprocessing inside the training workflow.
- The improved pipeline is the scientifically preferred result and may show lower headline scores if grouped evaluation reduces optimistic leakage.
