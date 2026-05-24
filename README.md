# FL-XGB-Thesis: Federated Learning + XGBoost/LightGBM for Intrusion Detection

**Network Intrusion Detection on CICIoT2023 using Federated Learning with Tree-Based Models**

---

## Hardware

- **CPU**: Ryzen 7 5700X (8 cores / 16 threads)
- **RAM**: 16 GB
- **GPU**: NVIDIA RTX 3060 Ti 8 GB (CNN baselines only)

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
├── dataset-raw/            # Symlink → /home/willtanoe/Documents/CICIOT23
├── src/                    # Python package (data loading, preprocessing, config)
│   ├── __init__.py
│   ├── config.py           # Canonical configuration constants
│   ├── data_loader.py      # Data loading utilities
│   ├── preprocessing.py    # Preprocessing functions
│   ├── models.py           # XGBoost, LightGBM, CNN, MLP
│   ├── metrics.py          # F1, AUC, G-mean
│   └── utils.py            # Seed, logger, memory helpers
├── flwr_app/               # Flower Application (real FL with Round-Robin)
│   ├── pyproject.toml      # Flower chart / FAB build config
│   ├── config.py           # Self-contained config (n_jobs=1, TREES_PER_ROUND=20)
│   ├── client_app.py       # ClientApp: incremental XGBoost/LightGBM
│   └── server_app.py       # ServerApp: Round-Robin strategy, metric logging
├── notebooks/              # Main pipeline (run in order)
│   ├── 01_eda.ipynb                # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb      # Cleaning, scaling, encode labels
│   ├── 03_data_analysis.ipynb      # Deeper data exploration
│   ├── 04_fl_setup.ipynb           # Client partitioning (Dirichlet non-IID)
│   ├── 05_fl_training.ipynb        # FL training: flwr run (Round-Robin)
│   ├── 06_baseline_comparison.ipynb# Centralized baselines (XGB, LGB, CNN, MLP)
│   ├── 07_evaluation.ipynb         # Full test-set evaluation + FL vs centralized
│   ├── 08_hyperparameter_tuning.ipynb # RandomizedSearchCV
│   └── 09_statistical_analysis.ipynb  # Significance tests
├── results/                # Output: parquet, models, figures, logs
├── THESIS.md
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

### Phase 3: Federated Learning

```
05_fl_training.ipynb
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

### Phase 4: Baseline Comparison

```
06_baseline_comparison.ipynb
```

Trains centralized XGBoost, LightGBM, CNN, MLP — evaluates on test set.

### Phase 5: Evaluation

```
07_evaluation.ipynb
```

Full metrics on test set: F1 Macro, F1 Weighted, Accuracy, per-class F1, confusion matrix.
Loads FL training history for FL-vs-centralized comparison plots.

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
| **Memory Safety** | `n_jobs=1`, `val_sample.parquet` (10k rows) | Prevents OOM on 16 GB system |

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
CPU_THREADS = 1                # n_jobs for XGBoost & LightGBM
FL_ROUNDS = 10                 # Must match src/config.py
TREES_PER_ROUND = 20           # Trees added per client per round (total: 10×20=200)
```

---

## Quick Start

```bash
# Open notebooks in order:
notebooks/01_eda.ipynb
notebooks/02_preprocessing.ipynb
notebooks/03_data_analysis.ipynb
notebooks/04_fl_setup.ipynb
notebooks/05_fl_training.ipynb
notebooks/06_baseline_comparison.ipynb
notebooks/07_evaluation.ipynb
notebooks/08_hyperparameter_tuning.ipynb
notebooks/09_statistical_analysis.ipynb

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
# Core
numpy>=1.24.0 pandas>=2.0.0 scikit-learn>=1.3.0 scipy>=1.10.0

# Tree Models (CPU only)
xgboost>=3.2.0 lightgbm>=4.3.0

# Federated Learning
flwr[simulation]>=1.15.0

# Deep Learning (baselines)
torch>=2.5.0 torchvision>=0.20.0

# Visualization
matplotlib>=3.7.0 seaborn>=0.12.0 plotly>=5.15.0

# Utilities
joblib>=1.3.0 tqdm>=4.65.0 psutil>=5.9.0 pyarrow>=14.0
```

---

## Training Notes

- **Resource Safety**: `n_jobs=1` + `init-args-num-cpus=2` + `val_sample.parquet` ensures the PC stays responsive under YouTube/Discord.
- **FL Efficiency**: Round-Robin runs 1 client per round, eliminating OOM risk from 5 concurrent clients.
- **Full data**: Set `TRAIN_SUBSAMPLE_RATIO=1.0` in `src/config.py`, then re-run notebooks 02, 04, and 05.

---

## Expected Results

| Model | Accuracy | F1 Macro | F1 Weighted |
|-------|----------|----------|-------------|
| **FL-XGBoost** (Round-Robin) | ~90-99% | ~68-75% | ~90-95% |
| **FL-LightGBM** (Round-Robin) | ~90-99% | ~68-75% | ~90-95% |
| **Centralized XGBoost** | ~90-99% | ~70-80% | ~90-95% |

F1 Macro is lower than Accuracy due to class imbalance (34 classes). The validation-sample metrics tend to be slightly higher than test-set metrics.

---

## References

1. **CICIoT2023**: Sharafaldin et al. "CICIoT2023: A Real-Time Dataset for IoT Intrusion Detection." IEEE ICASSP 2023.
2. **XGBoost**: Chen & Guestrin (2016). "XGBoost: A Scalable Tree Boosting System." KDD.
3. **LightGBM**: Ke et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." NeurIPS.
4. **FedAvg**: McMahan et al. (2017). "Communication-Efficient Learning of Deep Networks from Decentralized Data." AISTATS.
5. **Flower**: Beutel et al. (2020). "Flower: A Friendly Federated Learning Framework."
6. **FedProx**: Li et al. (2020). "Federated Optimization in Heterogeneous Networks." MLSys.
