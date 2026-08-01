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
python scripts/run_full_pipeline.py run-benchmark --config configs/benchmark_aggressive.yaml
python scripts/run_full_pipeline.py run-improved --config configs/improved.yaml
python scripts/run_full_pipeline.py cluster-drugbank --model-path models/reproduction/benchmark/best_benchmark_model.joblib --applicability-threshold-quantile 0.97
python scripts/run_full_pipeline.py run-atc --predictions reports/tables/drugbank_predictions_and_clusters.csv
```

### Java descriptor regeneration

Requires Java 18+ and Maven on `PATH`.

```bash
python scripts/run_java_descriptors.py --java java --maven mvn
```

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

## Assumptions and limitations

- The uploaded repository does not contain the ATC mapping used in the paper, so ATC reproduction is implemented but requires an added mapping file.
- The paper reports 222 descriptors, but the repository CSV export contains 223 descriptor columns after removing `SMILES` and `Texpi`; the code records this mismatch explicitly.
- The faithful baseline intentionally preserves the leakage-prone scaling and naive split logic from the notebook for comparison purposes.
- The benchmark-optimized pipeline is intended to maximize predictive performance on a paper-style naive split while still keeping preprocessing inside the training workflow.
- The improved pipeline is the scientifically preferred result and may show lower headline scores if grouped evaluation reduces optimistic leakage.
