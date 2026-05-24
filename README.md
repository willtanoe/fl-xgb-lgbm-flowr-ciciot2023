# Federated Learning + XGBoost/LightGBM for Intrusion Detection

**Network Intrusion Detection on CICIoT2023 using Federated Learning with Tree-Based Models**

---

## Hardware

- **CPU**: Ryzen 7 5700X (8 cores / 16 threads)
- **RAM**: 16 GB
- **GPU**: NVIDIA RTX 3060 Ti 8 GB

---

## Dataset: CICIoT2023

[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-blue)](https://www.kaggle.com/datasets/himadri07/ciciot2023)

The **CICIoT2023** dataset is a comprehensive and modern dataset designed for research in Internet of Things (IoT) security, particularly for intrusion detection and anomaly detection systems. Released by the **Canadian Institute for Cybersecurity (CIC)**, this dataset reflects real-world IoT network traffic and attack scenarios, providing a valuable resource for machine learning and cybersecurity research.

**Key Features:**
- Contains a mix of normal and malicious IoT network traffic
- Includes **34 distinct attack types**, covering modern and advanced cyber threat scenarios
- Provides labeled data suitable for supervised machine learning models
- Offers extracted network flow features (e.g., packet size, duration, flags, statistical summaries) for traffic classification and anomaly detection

**Data splits** (pre-partitioned by source):

| Split | Rows | Features | Description |
|-------|------|----------|-------------|
| `train.csv` | 5,491,972 | 46 | Training data |
| `validation.csv` | 1,176,852 | 46 | Validation data |
| `test.csv` | 1,176,852 | 46 | Test data (held-out) |

**Label distribution**: 34 classes (33 attack types + Benign), heavily imbalanced.

**Features** (46 columns):
- Flow stats: `flow_duration`, `Duration`, `Rate`, `Srate`, `Drate`
- Packet flags: `fin_flag_number`, `syn_flag_number`, `rst_flag_number`, etc.
- Protocol counts: `HTTP`, `HTTPS`, `DNS`, `TCP`, `UDP`, etc.
- Statistical: `Min`, `Max`, `AVG`, `Std`, `IAT`, `Number`
- Derived: `Magnitue`, `Radius`, `Covariance`, `Variance`, `Weight`

> **Credit**: Dataset sourced from [Kaggle — CICIoT2023 by Himadri07](https://www.kaggle.com/datasets/himadri07/ciciot2023).  
> Original paper: Sharafaldin et al. "CICIoT2023: A Real-Time Dataset for IoT Intrusion Detection."

---

## Project Structure

```
fl-xgb-thesis/
├── src/                    # Configuration only
│   ├── __init__.py
│   └── config.py           # XGB_PARAMS, LGB_PARAMS, paths, hardware settings
├── flwr_app/               # Flower Application (Round-Robin FL)
│   ├── pyproject.toml      # Flower chart / FAB build config
│   ├── config.py           # n_jobs=1, TREES_PER_ROUND=20, FL_ROUNDS=10
│   ├── client_app.py       # ClientApp: incremental XGBoost/LightGBM
│   └── server_app.py       # ServerApp: Round-Robin strategy, metric logging
├── notebooks/              # Main pipeline (run in order)
│   ├── 01_eda.ipynb                # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb      # Cleaning, scaling, encode labels
│   ├── 03_data_analysis.ipynb      # Deeper data exploration
│   ├── 04_fl_setup.ipynb           # Client partitioning (Dirichlet non-IID)
│   ├── 05_baseline_comparison.ipynb# Centralized baselines (XGBoost, LightGBM)
│   ├── 06_fl_training.ipynb        # FL training: flwr run (Round-Robin)
│   ├── 07_evaluation.ipynb         # Full test-set evaluation + FL vs centralized
│   ├── 08_hyperparameter_tuning.ipynb # RandomizedSearchCV
│   └── 09_statistical_analysis.ipynb  # Significance tests
├── results/                # Output: figures, logs
│   ├── figures/            # Plots (13 PNGs)
│   └── logs/               # FL history JSONs, evaluation results
├── README.md
└── requirements.txt
```

---

## Pipeline

### Phase 1: Preprocessing

```
01_eda.ipynb → 02_preprocessing.ipynb → 03_data_analysis.ipynb
```

Creates `train.parquet`, `val.parquet`, `test.parquet`, scaler, label encoder.

### Phase 2: Client Setup

```
04_fl_setup.ipynb
```

Partitions training data into 5 non-IID client splits via Dirichlet(α=1.0).

### Phase 3: Baseline Comparison

```
05_baseline_comparison.ipynb
```

Trains centralized XGBoost and LightGBM on 200k subsample, evaluates on test set.
Saves trained models to `results/models/` for downstream notebooks.

### Phase 4: Federated Learning

```
06_fl_training.ipynb
```

**Round-Robin Incremental FL** — 1 client per round adds 20 trees to the global model.
- Via `flwr run` command in subprocess (or directly from terminal)
- Model parameters (bytes) exchanged between server and clients
- Evaluation metrics saved to `results/logs/fl_history_{xgb,lgb}.json`

Alternative terminal command:
```bash
flwr run /home/willtanoe/Documents/fl-xgb-thesis/flwr_app --stream \
  --federation-config "client-resources-num-cpus=1 num-supernodes=5"
```

### Phase 5: Evaluation

```
07_evaluation.ipynb
```

Full metrics on test set: F1 Macro, F1 Weighted, Accuracy, per-class F1, confusion matrix.
Loads centralized models from NB05 and FL models from NB06 for FL-vs-centralized comparison.

### Phase 6: Hyperparameter Tuning

```
08_hyperparameter_tuning.ipynb
```

RandomizedSearchCV with 3-fold cross-validation on XGBoost and LightGBM.
Uses 15% subsample of training data for faster tuning.

### Phase 7: Statistical Analysis

```
09_statistical_analysis.ipynb
```

Significance tests (paired t-test, Wilcoxon, McNemar) comparing XGBoost vs LightGBM.
Uses bootstrapped test-set predictions for robust comparison.

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **FL Framework** | Flower 1.15+ | Production-grade Simulation Runtime |
| **FL Strategy** | Round-Robin (custom) | 1 client per round, incremental tree addition; avoids parameter averaging on tree bytes |
| **Incremental Training** | `xgb.train(xgb_model=prev)` | Each round adds `TREES_PER_ROUND=20` trees to the global model |
| **XGBoost** | `tree_method="hist"`, `device="cpu"` | CPU-only to keep GPU free for display |
| **LightGBM** | `device="cpu"`, `force_col_wise=True` | CPU-only with efficient column histograms |
| **Non-IID** | Dirichlet(α=1.0) | Realistic IoT data heterogeneity |
| **Class Imbalance** | No weighting (F1 macro reports honestly) | 34 classes, heavy imbalance |
| **Primary Metric** | F1 Macro | Honest for imbalanced multi-class |
| **Data Subsample** | `TRAIN_SUBSAMPLE_RATIO=0.25` | 25% of train rows for faster iteration |
| **Memory Safety** | `val_sample.parquet` (10k rows), FL `n_jobs=1` | Prevents OOM on 16 GB system |

---

## Configuration

Key settings in `src/config.py`:

```python
TRAIN_SUBSAMPLE_RATIO = 0.25   # 25% of data for faster iteration (set 1.0 for final)
FL_ROUNDS = 10                 # Total FL rounds (10 × 20 trees = 200 total)
NUM_CLIENTS = 5                # Number of FL clients
DIRICHLET_ALPHA = 1.0          # Non-IID partitioning (lower = more skewed)
```

Key settings in `flwr_app/config.py`:

```python
CPU_THREADS = 1                # FL client n_jobs for XGBoost & LightGBM (centralized uses 8)
FL_ROUNDS = 10                 # Must match src/config.py
TREES_PER_ROUND = 20           # Trees added per client per round (total: 10×20=200)
```

---

## Quick Start

```bash
# Phase 1–2: Preprocessing + Client Setup
01_eda.ipynb → 02_preprocessing.ipynb → 03_data_analysis.ipynb → 04_fl_setup.ipynb

# Phase 3: Baseline Training
05_baseline_comparison.ipynb

# Phase 4: Federated Learning
06_fl_training.ipynb

# Phase 5–7: Evaluation, Tuning & Statistical Analysis
07_evaluation.ipynb → 08_hyperparameter_tuning.ipynb → 09_statistical_analysis.ipynb

# Or train FL from terminal:
cd flwr_app
flwr run . --stream \
  --federation-config "client-resources-num-cpus=1 num-supernodes=5"

# Or from terminal (without notebook):
flwr run /home/willtanoe/Documents/fl-xgb-thesis/flwr_app --stream \
  --federation-config "client-resources-num-cpus=1 num-supernodes=5"

# For LightGBM:
flwr run /home/willtanoe/Documents/fl-xgb-thesis/flwr_app --stream \
  --federation-config "client-resources-num-cpus=1 num-supernodes=5" \
  --run-config 'model_type="lgb"'
```

---

## Requirements

```
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
scikit-learn>=1.3.0
xgboost>=2.1.0
lightgbm>=4.3.0
flwr[simulation]>=1.14.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.3.0
tqdm>=4.65.0
pyarrow>=14.0
```

---

## Training Notes

- **Resource Safety**: `n_jobs=CPU_THREADS` (8) for centralized models; `n_jobs=1` in `flwr_app/config.py` for FL clients to prevent CPU contention.
- **FL Efficiency**: Round-Robin runs 1 client per round, eliminating OOM risk from 5 concurrent clients.
- **Full data**: Set `TRAIN_SUBSAMPLE_RATIO=1.0` in `src/config.py`, then re-run notebooks 02, 04, 05, and 06.

---

## Expected Results

| Model | Accuracy | F1 Macro |
|-------|----------|----------|
| **Centralized XGBoost** | ~99% | ~72% |
| **Centralized LightGBM** | ~99% | ~70% |
| **FL XGBoost** (Round-Robin, 10×20 trees) | ~95-99% | ~60-72% |
| **FL LightGBM** (Round-Robin, 10×20 trees) | ~95-99% | ~60-72% |

F1 Macro is lower than Accuracy due to class imbalance (34 classes). FL models typically underperform centralized by 5-10% F1 due to non-IID data partitioning.

---

## References

1. **CICIoT2023**: Sharafaldin et al. "CICIoT2023: A Real-Time Dataset for IoT Intrusion Detection." IEEE ICASSP 2023.
2. **XGBoost**: Chen & Guestrin (2016). "XGBoost: A Scalable Tree Boosting System." KDD.
3. **LightGBM**: Ke et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." NeurIPS.
4. **FedAvg**: McMahan et al. (2017). "Communication-Efficient Learning of Deep Networks from Decentralized Data." AISTATS.
5. **Flower**: Beutel et al. (2020). "Flower: A Friendly Federated Learning Framework."
6. **FedProx**: Li et al. (2020). "Federated Optimization in Heterogeneous Networks." MLSys.
