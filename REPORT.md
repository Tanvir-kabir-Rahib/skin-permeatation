# Skin Permeation Reproduction Report

## Reconstructed workflow
- Descriptor generation: Java CDK 2.8 generator on `data-original.csv` with largest-fragment handling for salts/hydrates.
- Paper baseline preprocessing: water-row removal, descriptor imputation, duplicate removal, manual descriptor removal, correlation pruning, global scaling before split, 85/15 random split.
- Improved preprocessing: grouped split by SMILES, train-only preprocessing, randomized search, bootstrap CIs, and applicability-domain analysis.

## Generated artifacts
- Data audit JSON: `reports/artifacts/data_audit.json`
- Reproducibility report JSON: `reports/artifacts/reproducibility_report.json`

## Reproduced baseline metrics
| model             | artifact_path                                                                                                               |       r2 |     rmse |      mae |     cv_mae |   mae_ci_low |   mae_ci_high |
|:------------------|:----------------------------------------------------------------------------------------------------------------------------|---------:|---------:|---------:|-----------:|-------------:|--------------:|
| Gradient Boosting | /Users/tanvir/Desktop/skin/skin-permeation-reproduction-project/models/reproduction/paper_baseline/Gradient_Boosting.joblib | 0.812448 | 0.445116 | 0.27805  |   0.4398   |     0.205001 |      0.375007 |
| LGBM              | /Users/tanvir/Desktop/skin/skin-permeation-reproduction-project/models/reproduction/paper_baseline/LGBM.joblib              | 0.816714 | 0.440025 | 0.28415  |   0.44577  |     0.212607 |      0.376623 |
| SVR (RBF)         | /Users/tanvir/Desktop/skin/skin-permeation-reproduction-project/models/reproduction/paper_baseline/SVR_RBF.joblib           | 0.798792 | 0.461036 | 0.288847 |   0.434982 |     0.212478 |      0.386405 |
| CatBoost          | /Users/tanvir/Desktop/skin/skin-permeation-reproduction-project/models/reproduction/paper_baseline/CatBoost.joblib          | 0.794954 | 0.465413 | 0.296828 |   0.430819 |     0.218546 |      0.393171 |
| XGBoost           | /Users/tanvir/Desktop/skin/skin-permeation-reproduction-project/models/reproduction/paper_baseline/XGBoost.joblib           | 0.780234 | 0.481828 | 0.304727 |   0.45     |     0.221184 |      0.407846 |
| Decision Tree     | /Users/tanvir/Desktop/skin/skin-permeation-reproduction-project/models/reproduction/paper_baseline/Decision_Tree.joblib     | 0.768069 | 0.494984 | 0.313529 |   0.47154  |     0.225192 |      0.417536 |
| Lasso             | /Users/tanvir/Desktop/skin/skin-permeation-reproduction-project/models/reproduction/paper_baseline/Lasso.joblib             | 0.791245 | 0.469602 | 0.314364 |   0.536562 |     0.23821  |      0.414591 |
| ANN               | /Users/tanvir/Desktop/skin/skin-permeation-reproduction-project/models/reproduction/paper_baseline/ANN/ann_model.keras      | 0.759328 | 0.504225 | 0.31731  | nan        |   nan        |    nan        |
| RF                | /Users/tanvir/Desktop/skin/skin-permeation-reproduction-project/models/reproduction/paper_baseline/RF.joblib                | 0.787385 | 0.473925 | 0.317768 |   0.441131 |     0.237018 |      0.409383 |
| MLR (10 features) | /Users/tanvir/Desktop/skin/skin-permeation-reproduction-project/models/reproduction/paper_baseline/MLR_10_features.joblib   | 0.646034 | 0.611495 | 0.430272 |   0.521624 |   nan        |    nan        |

## Benchmark-Optimized Metrics
This section uses a naive row split like the paper, but with train-only preprocessing, target transformation, stronger search spaces, and ensemble models. It is designed to challenge the paper's headline performance without adding leakage.
| model                | artifact_path                                                                                                             |       r2 |     rmse |      mae |    cv_rmse |   mae_ci_low |   mae_ci_high |     cv_mae |   tuning_rmse |
|:---------------------|:--------------------------------------------------------------------------------------------------------------------------|---------:|---------:|---------:|-----------:|-------------:|--------------:|-----------:|--------------:|
| SVR (RBF)            | /Users/tanvir/Desktop/skin/skin-permeation-reproduction-project/models/reproduction/benchmark/SVR_RBF.joblib              | 0.775213 | 0.535553 | 0.334524 |   0.6413   |     0.23939  |      0.445856 |   0.433505 |      0.6413   |
| XGBoost              | /Users/tanvir/Desktop/skin/skin-permeation-reproduction-project/models/reproduction/benchmark/XGBoost.joblib              | 0.758877 | 0.554673 | 0.358803 |   0.636949 |     0.26069  |      0.466172 |   0.448735 |      0.636949 |
| ExtraTrees           | /Users/tanvir/Desktop/skin/skin-permeation-reproduction-project/models/reproduction/benchmark/ExtraTrees.joblib           | 0.758793 | 0.55477  | 0.36553  |   0.625746 |     0.269427 |      0.467851 |   0.442107 |      0.625746 |
| Mean Ensemble        | /Users/tanvir/Desktop/skin/skin-permeation-reproduction-project/models/reproduction/benchmark/MeanEnsemble.json           | 0.757824 | 0.555883 | 0.363258 | nan        |   nan        |    nan        | nan        |      0.633097 |
| CatBoost             | /Users/tanvir/Desktop/skin/skin-permeation-reproduction-project/models/reproduction/benchmark/CatBoost.joblib             | 0.757611 | 0.556127 | 0.384529 |   0.62613  |     0.288923 |      0.483088 |   0.437803 |      0.62613  |
| LGBM                 | /Users/tanvir/Desktop/skin/skin-permeation-reproduction-project/models/reproduction/benchmark/LGBM.joblib                 | 0.752909 | 0.561496 | 0.36915  |   0.638041 |     0.269225 |      0.47401  |   0.448183 |      0.638041 |
| RF                   | /Users/tanvir/Desktop/skin/skin-permeation-reproduction-project/models/reproduction/benchmark/RF.joblib                   | 0.751344 | 0.56327  | 0.363974 |   0.629842 |     0.262714 |      0.469511 |   0.438673 |      0.629842 |
| Gradient Boosting    | /Users/tanvir/Desktop/skin/skin-permeation-reproduction-project/models/reproduction/benchmark/Gradient_Boosting.joblib    | 0.751038 | 0.563618 | 0.374159 |   0.647527 |     0.278387 |      0.481688 |   0.4443   |      0.647527 |
| HistGradientBoosting | /Users/tanvir/Desktop/skin/skin-permeation-reproduction-project/models/reproduction/benchmark/HistGradientBoosting.joblib | 0.739343 | 0.576704 | 0.376722 |   0.638618 |     0.279548 |      0.487575 |   0.446684 |      0.638618 |
| Stacking Regressor   | /Users/tanvir/Desktop/skin/skin-permeation-reproduction-project/models/reproduction/benchmark/StackingRegressor.joblib    | 0.689405 | 0.629528 | 0.411225 |   0.647629 |     0.306368 |      0.533172 |   0.469736 |      0.647629 |
| ElasticNet           | /Users/tanvir/Desktop/skin/skin-permeation-reproduction-project/models/reproduction/benchmark/ElasticNet.joblib           | 0.651621 | 0.66672  | 0.459063 |   0.678514 |     0.35004  |      0.586891 |   0.490377 |      0.678514 |

## Improved metrics
| model              | artifact_path                                                                                                         |          r2 |      rmse |      mae |    cv_mae |   mae_ci_low |   mae_ci_high |
|:-------------------|:----------------------------------------------------------------------------------------------------------------------|------------:|----------:|---------:|----------:|-------------:|--------------:|
| LGBM               | /Users/tanvir/Desktop/skin/skin-permeation-reproduction-project/models/reproduction/improved/LGBM.joblib              |    0.44472  |  0.690512 | 0.47565  |  0.721285 |     0.348898 |      0.613595 |
| CatBoost           | /Users/tanvir/Desktop/skin/skin-permeation-reproduction-project/models/reproduction/improved/CatBoost.joblib          |    0.518062 |  0.643296 | 0.478918 |  0.673125 |     0.369127 |      0.597653 |
| RF                 | /Users/tanvir/Desktop/skin/skin-permeation-reproduction-project/models/reproduction/improved/RF.joblib                |    0.396804 |  0.719688 | 0.481349 |  0.644471 |     0.341125 |      0.623297 |
| Gradient Boosting  | /Users/tanvir/Desktop/skin/skin-permeation-reproduction-project/models/reproduction/improved/Gradient_Boosting.joblib |    0.433688 |  0.697338 | 0.497189 |  0.700683 |     0.373664 |      0.627407 |
| XGBoost            | /Users/tanvir/Desktop/skin/skin-permeation-reproduction-project/models/reproduction/improved/XGBoost.joblib           |    0.383292 |  0.727704 | 0.51285  |  0.689616 |     0.383524 |      0.649928 |
| Stacking Regressor | /Users/tanvir/Desktop/skin/skin-permeation-reproduction-project/models/reproduction/improved/StackingRegressor.joblib |    0.302764 |  0.773758 | 0.557196 |  0.672429 |     0.422321 |      0.704599 |
| Decision Tree      | /Users/tanvir/Desktop/skin/skin-permeation-reproduction-project/models/reproduction/improved/Decision_Tree.joblib     |    0.138038 |  0.860318 | 0.587187 |  0.740326 |     0.424956 |      0.759265 |
| Lasso              | /Users/tanvir/Desktop/skin/skin-permeation-reproduction-project/models/reproduction/improved/Lasso.joblib             |    0.177434 |  0.840428 | 0.594522 |  0.761721 |     0.453227 |      0.752018 |
| SVR (RBF)          | /Users/tanvir/Desktop/skin/skin-permeation-reproduction-project/models/reproduction/improved/SVR_RBF.joblib           |    0.310009 |  0.769727 | 0.651898 |  0.713917 |     0.550659 |      0.756409 |
| MLR                | /Users/tanvir/Desktop/skin/skin-permeation-reproduction-project/models/reproduction/improved/MLR.joblib               | -364.365    | 17.7125   | 8.17242  | 12.3229   |     4.13988  |     12.4685   |

## Key methodological issues
- The notebook baseline scales the full dataset before splitting, which leaks test-set information.
- Multiple rows share the same SMILES but different permeability measurements, so naive random splits can place the same molecule in train and test.
- The repository does not include the ATC mapping file needed to exactly reproduce the paper's ATC analysis.

## Whether the paper's conclusions hold
- The repository evidence supports the qualitative conclusion that boosted tree models outperform simple linear baselines on this descriptor set.
- The exact headline metrics should be treated cautiously until grouped validation and train-only preprocessing are applied.

## Where the improved pipeline is stronger
- It separates molecule groups across train and test when repeated molecules are present.
- It tunes models without leaking preprocessing statistics from the holdout set.
- It adds uncertainty intervals, applicability-domain checks, and structured error analysis.

## Performance interpretation
- If the benchmark-optimized section outperforms the paper while the grouped section does not, that usually means the paper's split design was easier rather than the chemistry becoming more predictable.
- The grouped metrics remain the more defensible estimate of real-world generalization to unseen molecules.
