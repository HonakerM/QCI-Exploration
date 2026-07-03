# Credit Card Fraud Detection — XGBoost vs CVQBoost

Comparative analysis of classical (XGBoost) and quantum-enhanced (CVQBoost on QCi Dirac-3)
machine learning for credit card fraud detection.
Refactored from a Jupyter notebook into a clean, typed Python project.

## Project Structure

```
fraud_detection/
├── config.py            # All config as dataclasses (DataConfig, XGBoostConfig, CVQBoostConfig)
├── results.py           # Result dataclasses (DataSplit, ModelResults, ClassificationMetrics)
├── data_loader.py       # CSV loading + V_* feature engineering
├── data_prep.py         # Balancing, label encoding {-1,+1}, train/test split
├── evaluation.py        # Shared metric computation (used by both model trainers)
├── visualization.py     # ROC curves + AUC/LogLoss bar charts
├── models/
│   ├── __init__.py
│   ├── xgboost_model.py   # XGBoost trainer (labels: {0, 1})
│   └── cvqboost_model.py  # CVQBoost trainer (labels: {-1, +1}, hits Dirac-3)
└── main.py              # Pipeline entry point
```

## Data

Download `train.csv` and `test.csv` from the
[Kaggle Playground Series S3E4](https://www.kaggle.com/competitions/playground-series-s3e4/data)
competition and place them in the same directory as `main.py`.

## Setup

Requires Python 3.11 (quantum libraries are pinned to 3.11).

```bash
python -3.11 -m venv .venv --clear
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate
pip install --upgrade pip
pip install -r requirements.txt
```

Create a `.env` file for QCi credentials (only needed for quantum runs):

```
QCI_TOKEN=your_token_here
QCI_API_URL=https://api.qci-prod.com
```

## Usage

```bash
# Classical XGBoost only — no QPU cost, safe to run anytime
python main.py

# Include CVQBoost on Dirac-3 — ⚠️ consumes ~1 QPU second (~$0.22/run)
python main.py --enable-quantum

# Save plots to PNG files
python main.py --save-plots

# Both flags
python main.py --enable-quantum --save-plots
```

## Configuration

All tuneable settings are dataclasses in `config.py`:

| Class           | Controls                                               |
|-----------------|--------------------------------------------------------|
| `DataConfig`    | File paths, test split fraction, sampling size         |
| `XGBoostConfig` | All XGBoost hyperparameters                            |
| `CVQBoostConfig`| QBoost hyperparameters + credential env-var names      |

## Expected Results

| Model    | AUC    | Training time      |
|----------|--------|--------------------|
| XGBoost  | ~0.887 | ~1–2 seconds       |
| CVQBoost | ~0.882 | ~2–320 seconds     |

## Label Convention

| Stage              | Labels  | Note                                      |
|--------------------|---------|-------------------------------------------|
| Raw Kaggle data    | {0, 1}  | Class column                              |
| After `prep_data`  | {-1, +1}| Both splits use this internally           |
| XGBoost input      | {0, 1}  | `xgboost_labels()` re-scales before fit   |
| CVQBoost input     | {-1, +1}| Passed directly from `prep_data` output   |

## Credits

- Original Kaggle notebook: [PSS 3, Episode 4, Quick Start, GBDT](https://www.kaggle.com/code/cv13j0/pss-3-episode-4-quick-start-gbdt/notebook)
  by Carlos V. Montenegro (cv13j0)
- Quantum extension + feature engineering: Harold Kimmey, IT DevOps Engineer Lead, Progressive Insurance
