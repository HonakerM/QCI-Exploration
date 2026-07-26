# QCI-Exploration

Tools for exploring QCi's systems including the [Dirac-3 quantum-ready optimizer](https://quantumcomputinginc.com/products/commercial-products/dirac-3).
This repository will contain various scripts for various classical and quantum
classifiers. 

Currently the repository implements scripts to compare a classical **XGBoost**
classifier against **QBoost** from QCi, plus file-based ensemble testing for
`ensemble_fraud.py`.


## Project structure

```
QCI-Exploration/
├── README.md
├── requirements.txt          #
├── xgboost_fraud.py          # Train/evaluate XGBoost for fraud
├── ensemble_fraud.py         # Train/evaluate pluggable ensemble classifiers from YAML test files
├── compare_results.py        # Compare saved results across runs
└── common/
    ├── qci.py                 # QCi API client factory
    ├── logging.py              # Shared logging setup
    └── binary_classification   # Shared binary classification libraries
```

## Installation

1. Use Python 3.11+.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. To run `ensemble_fraud.py` with `cvqboost`, you'll also need access to a
  QCi Dirac-3 account. Create a `.env` file in the project root with your
  credentials or set environment variables.

   ```env
   QCI_TOKEN=your-api-token
   QCI_API_URL=https://api.qci-prod.com
   ```

  `ensemble_fraud.py` loads this automatically at startup.

4. You can test your QCI connection by running the `qci_status.py` script:

  ```bash
  $ python3 .\qci_status.py
  2026-07-15 16:31:44 INFO - csample: No Access
  2026-07-15 16:31:44 INFO - dirac: Trail with 594 remaining 
  ```

## Preparing your data

Both scripts expect one or more CSV files containing:

- A **class/label column** identifying fraud vs. non-fraud (or true/false)
  rows. The default column name is `Class`; override it with
  `--class-override` if your dataset uses a different name (e.g.
  `FraudFound`).
- Any number of feature columns with either raw values or categories. If
  you use categories the scripts will automatically discretize via target 
  (mean) encoding. These are auto-detected and used to engineer eight additional
  `Comp_*` aggregate features (sum, min, max, avg, std, etc.).
- Optionally, an **`id` column** and any **additional feature columns**
  (e.g. `Amount`, `Time`) you'd like included as-is.

You can pass a `--train-file` only, a `--test-file` only, or both — if both
are given they're combined before the train/test split is performed
internally.

## Running the code

### XGBoost

```bash
python3 xgboost_fraud.py --train-file "./data/mlg-ulb/train.csv" --test-file "./data/mlg-ulb/test.csv"
```

### File-based ensemble testing

> [!WARNING]
> Each `ensemble_fraud.py` run that actually submits to Dirac-3 consumes
> paid QPU allocation (~1 QPU second, ~$0.22/run at time of writing). Use
> `--dry-run` to validate your data pipeline first without submitting a job.

```bash
python3 .\ensemble_fraud.py test-file .\tests\mlg-ulb\ensemble\classical_qboost\optim_lbfgs.yaml
```

That YAML fixture looks like this:

```yaml
algorithm: classical_qboost
classifier:
  optimization_method: "L-BFGS-B"
  include_smu_params: true
  lambda_coef: 0.0
  num_samples: 1
  weak_cls_schedule: 1
  weak_cls_strategy: sequential
  weak_cls_type: lg
data:
  additional_feature_names:
  - Amount
  - Time
  class_name: Class
  enforce_equal_samples: true
  engineered_feature_names:
  - Comp_Sum
  - Comp_Min
  - Comp_Max
  - Comp_Avg
  - Comp_Std
  - Comp_Pos
  - Comp_Neg
  - Comp_Var
  index_column: id
  model_file: null
  model_name_override: "classical tnc 1k oversample"
  non_fraud_sample_size: 1000
  over_sample_percentage: 1.0
  random_state: 42
  should_over_sample: true
  test_file: data\mlg-ulb\test.csv
  test_size: 0.3
  train_file: data\mlg-ulb\train.csv
  v_feature_names: []
```

To run a whole folder of tests, point `test-folder` at the directory:

```bash
python3 .\ensemble_fraud.py test-folder .\tests\mlg-ulb\ensemble\classical_qboost
```

### Comparison

```bash
python3 .\compare_results.py .\results\mlg-ulb\ensemble\classical_qboost .\results\mlg-ulb\ensemble\cvqboost
```

### Using a differently-shaped dataset

Datasets that don't use the default `Class` label column, or that don't
have extra columns like `Amount`/`Time` to include, can be pointed at with
`--class-override` and `--no-additional-features`:

```bash
python3 xgboost_fraud.py --train-file "./data/car_fraud/carclaims.csv" --class-override "FraudFound" --no-additional-features
python3 .\ensemble_fraud.py test-file .\tests\mlg-ulb\ensemble\classical_qboost\weakcls_knn.yaml
```

## Examples

### Binary Classification with a Tabular Credit Card Fraud Dataset

#### Data

From Kaggle: [https://www.kaggle.com/competitions/playground-series-s3e4/data](https://www.kaggle.com/competitions/playground-series-s3e4/data)

#### Results

![comparison](./results/mlg-ulb/ensemble/comparison_metrics.png)
![comparison](./results/mlg-ulb/ensemble/comparison_roc.png)
![comparison](./results/mlg-ulb/ensemble/comparison_timing.png)

### Vehicle Insurance Fraud Detection

#### Data

From Kaggle: [https://www.kaggle.com/datasets/khusheekapoor/vehicle-insurance-fraud-detection](https://www.kaggle.com/datasets/khusheekapoor/vehicle-insurance-fraud-detection)