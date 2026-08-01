# Methodology Audit

## 1. Repository Files Inspected

The methodology was derived from direct inspection of the following repository material.

### Project documentation and configuration

- `README.md`
- `README_reproduction.md`
- `REPORT.md`
- `FDA_APPROVED_DRUG_CLASSIFICATION_WORKFLOW.md`
- `environment.yml`
- `requirements.txt`
- `requirements_reproduction.txt`
- `configs/paper_baseline.yaml`
- `configs/benchmark.yaml`
- `configs/benchmark_aggressive.yaml`
- `configs/improved.yaml`

### Data and descriptor-generation files

- `data/raw/Skin Permeation.xlsx`
- `data/raw/data-original.csv`
- `data/raw/data-descriptors.csv`
- `data/raw/DrugBank-descriptors.csv`
- `data/processed/trial4.csv`
- `data/processed/drug_bank_clean.csv`
- `data/final/clean_trial4.csv`
- `descriptors-generator/READ ME.md`
- `descriptors-generator/pom.xml`
- `descriptors-generator/src/main/java/org/rami/dg/DescriptorsGenerator.java`
- `descriptors-generator/src/main/java/org/rami/dg/SMILESCleaner.java`

### Workflow and analysis code

- all Python modules under `src/skin_permeation/`
- all model modules under `src/skin_permeation/modeling/`
- all downstream modules under `src/skin_permeation/analysis/`
- `scripts/run_full_pipeline.py`
- `scripts/run_java_descriptors.py`
- `scripts/generate_external_validation_inputs.py`
- `scripts/train_external_validation_model.py`
- `scripts/render_rdkit_notebook.py`
- `scripts/regenerate_project_figures.py`
- `predict_logkp.py`

### Tests and generated evidence

- all files under `tests/`
- `reports/artifacts/data_audit.json`
- `reports/artifacts/descriptor_validation.json`
- `reports/artifacts/reproducibility_report.json`
- model metric, error, cluster, and prediction tables under `reports/tables/`
- extended-validation tables under `outputs/external_validation/`
- `external_validation_model.joblib`
- `models/reproduction/benchmark/best_benchmark_model.json`
- `figures/figure_generation_summary.json`
- `reports/logs/pipeline.log`

No `.ipynb` file is present. `scripts/render_rdkit_notebook.py` can generate an RDKit demonstration notebook, but the generated notebook itself is not checked in.

The Abdallah et al. reference PDF was not found in the repository or the supplied attachment directory. The methodology therefore uses no text, values, or figures from that paper.

## 2. Datasets Found

| Dataset | Rows | Columns | Confirmed role |
|---|---:|---:|---|
| `Skin Permeation.xlsx` | 449 | 241 | Experimental workbook |
| `data-original.csv` | 476 | 20 | Java descriptor-generator input |
| `data-descriptors.csv` | 476 | 225 | Raw CDK descriptor output |
| `trial4.csv` | 445 | 226 | Pre-filter modeling table |
| `clean_trial4.csv` | 417 | 149 | Final supervised model table |
| `DrugBank-descriptors.csv` | 2,326 | 225 | Raw external compound descriptors |
| `drug_bank_clean.csv` | 2,293 | 224 | Clean external prediction table |
| `train_data.csv` | 354 | 1 | Training targets for current extended validation run |
| `external_test_predictions.csv` | 63 | 4 | Current random-row held-out predictions |
| `drugbank_predictions_and_clusters.csv` | 2,293 | 232 | DrugBank predictions, clusters, PCA, and domain flags |

The final experimental table contains 417 observations, 180 distinct compound names, 146 unique SMILES, no exact duplicate rows, no missing cells, and 146 numerical model inputs including `Texpi`.

## 3. Scripts and Modules Defining the Workflow

- Data loading and audit: `data_loading.py`, `data_audit.py`
- Preprocessing and filtering: `preprocessing.py`, `modeling/transforms.py`
- Descriptor inventory and Java execution: `descriptors.py`, `run_java_descriptors.py`
- Split logic: `splitters.py`, `validation_model.py`
- Baseline modeling: `modeling/baseline.py`, `modeling/ann.py`
- Benchmark modeling: `modeling/benchmark.py`
- Grouped improved modeling: `modeling/improved.py`
- Extended validation metrics: `external_validation.py`
- Validation-ready training: `validation_model.py`
- Validation file conversion: `validation_data.py`
- DrugBank prediction, PCA, and clustering: `analysis/clustering.py`
- Applicability domain: `analysis/applicability.py`
- Conditional ATC analysis: `analysis/atc.py`
- RDKit/CDK unknown-molecule prediction: `prediction.py`, `predict_logkp.py`
- Publication figures: `regenerate_project_figures.py`
- Main orchestration: `cli.py`, `run_full_pipeline.py`

## 4. Models Detected

The repository contains implementations or artifacts for:

- multiple linear regression;
- Lasso;
- ElasticNet;
- decision tree;
- random forest;
- Extra Trees;
- gradient boosting;
- histogram gradient boosting;
- XGBoost;
- CatBoost;
- LightGBM;
- RBF support-vector regression;
- artificial neural network;
- linear and RidgeCV stacking;
- arithmetic mean ensembles;
- calibrated single models; and
- calibrated out-of-fold Ridge ensembles.

## 5. Validation Methods Detected

- 85:15 random holdout with seed 1 for the baseline;
- random 85:15 benchmark holdout, with current artifacts matching seed 22;
- 85:15 grouped-SMILES holdout with seed 2024 for the improved track;
- five-fold GroupKFold hyperparameter search in the improved track;
- repeated K-fold benchmark search;
- training-only out-of-fold candidate selection and calibration;
- optional strict 80:20 grouped validation with seed 42;
- current executed 85:15 random validation with seed 1;
- R2, RMSE, MAE, cross-validation MAE/RMSE;
- 1,000-resample MAE bootstrap intervals;
- Q2_F1, Q2_F2, Q2_F3, CCC, r_m-squared, and Golbraikh-Tropsha diagnostics;
- residual and molecular-property error analysis; and
- five-nearest-neighbor applicability-domain analysis.

No Y-randomization, nested cross-validation, scaffold split, learning curve, leverage/Williams plot, or standardized-residual domain was found.

## 6. Fully Confirmed Methodology Elements

- File names, dimensions, target, identifiers, and final feature count.
- The final model input contains 145 molecular descriptors plus `Texpi`.
- Exact-duplicate removal from 445 to 417 rows.
- Java 18, Maven, and CDK 2.8 descriptor generation.
- Largest-fragment handling for disconnected SMILES in Java.
- The named imputation strategies for `topoShape`, `HybRatio`, `JPLogP`, and `Kier3`.
- DrugBank reduction from 2,326 to 2,293 rows by removing missing `ATSc1`/`BCUTw-1l` rows.
- DrugBank removal of `geomShape` and imputation of the three remaining missing descriptor types.
- Baseline, benchmark, improved, and validation-ready model families.
- Improved grouped split size and absence of SMILES overlap.
- Current external-validation artifact metadata: `paper-reproduction`, random 85:15 split, seed 1.
- Current validation model: calibrated RBF-SVR.
- Current extended-validation metrics and acceptance results.
- RDKit validation/canonicalization and CDK-based 146-feature inference.
- Fixed DrugBank `Texpi=310`.
- K-means evaluation for `k=2..10`, final fixed `k=4`, and PCA projection.
- Five-nearest-neighbor applicability-domain rule with a configurable training-distance percentile threshold.
- ATC code path, minimum group size of three, Kruskal-Wallis, Mann-Whitney, and FDR logic.
- Figure names, formats, and current manifest status.

## 7. Items Requiring Manual Confirmation

1. **Exact descriptor-pruning history.** The current reconstruction function produces 150 columns, while `clean_trial4.csv` has 149 and retains a different subset of correlated descriptors.

   `[Confirm the exact notebook/code and ordered drop list used to create clean_trial4.csv.]`

2. **Single final/deployment model.** Current artifacts identify different models for different purposes:
   - grouped result by MAE: LGBM;
   - CV-selected benchmark artifact: CatBoost;
   - primary unknown-SMILES prediction: Gradient Boosting;
   - extended validation model: uncalibrated RBF-SVR selected by the training-only calibration guard.

   `[Confirm which model should be called the final model in the thesis.]`

3. **DrugBank model revision.** The DrugBank prediction table predates the latest benchmark model files, and its predictions do not match the current serialized models exactly.

   `[Confirm or regenerate the DrugBank predictions with the intended final model.]`

4. **Meaning of external validation.** The current `external_test_predictions.csv` is a held-out subset of `clean_trial4.csv`, not an independently sourced experimental dataset. It uses a random row split; `validation_protocol.json` records 33 overlapping SMILES and 54 overlapping validation rows.

   `[Confirm whether a genuinely independent experimental LogKp dataset exists outside this repository.]`

5. **Latest benchmark configuration.** Saved CatBoost parameters match `benchmark_aggressive.yaml`, but the metric table does not store its originating config path.

   `[Confirm that benchmark_aggressive.yaml is the intended latest benchmark protocol.]`

6. **ATC source.** No ATC mapping file is present, so ATC results were not generated.

   `[Provide the ATC mapping source/file if ATC analysis should be reported as completed.]`

7. **Reference PDF.** The paper PDF described in the request was not present in the accessible repository or attachment directory.

   `[Provide its path only if a separate stylistic review is still required.]`

8. **RDKit environment.** RDKit is not pinned in `requirements.txt` or `environment.yml`; prediction depends on a compatible active interpreter or `SKIN_PERMEATION_RDKIT_PYTHON`.

   `[Confirm the RDKit version and environment to report.]`

9. **Inference temperature.** Unknown-molecule and DrugBank prediction default to `Texpi=310 K`.

   `[Confirm that 310 K is the intended deployment temperature for all external molecules.]`

10. **Saved-model software versions.** The regenerated validation artifact records Python 3.13.5 and scikit-learn 1.6.1 in both the model bundle and `validation_protocol.json`.

    `[Use the recorded versions when archiving or transferring the serialized model.]`

11. **External-validation fold count.** The regenerated `external_validation_model.joblib` and `validation_protocol.json` record `cv_folds=5`.
