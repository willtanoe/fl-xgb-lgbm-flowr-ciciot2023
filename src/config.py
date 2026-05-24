"""FL-XGB: Configuration constants."""

from pathlib import Path

BASE_DIR    = Path(__file__).parent.parent
DATA_DIR    = BASE_DIR / "dataset-raw"
RESULTS_DIR = BASE_DIR / "results"
LOGS_DIR    = BASE_DIR / "logs"

TRAIN_CSV  = DATA_DIR / "train" / "train.csv"
VAL_CSV    = DATA_DIR / "validation" / "validation.csv"
TEST_CSV   = DATA_DIR / "test" / "test.csv"

CPU_THREADS    = 8
GPU_DEVICE     = "cuda"
XGB_DEVICE     = "cpu"
LGB_DEVICE     = "cpu"

NUM_CLIENTS    = 5
FL_ROUNDS      = 10
LOCAL_EPOCHS   = 5
FRACTION_FIT   = 1.0

DIRICHLET_ALPHA = 1.0

XGB_PARAMS = {
    "objective":          "multi:softprob",
    "num_class":          34,
    "tree_method":        "hist",
    "device":             "cpu",
    "max_depth":          8,
    "learning_rate":      0.1,
    "n_estimators":       200,
    "subsample":          0.8,
    "colsample_bytree":   0.8,
    "min_child_weight":    1,
    "reg_alpha":          0.1,
    "reg_lambda":         1.0,
    "random_state":       42,
    "verbosity":          0,
    "n_jobs":             CPU_THREADS,
}

LGB_PARAMS = {
    "objective":          "multiclass",
    "num_class":          34,
    "boosting_type":      "gbdt",
    "device":             "cpu",
    "max_depth":          8,
    "num_leaves":         127,
    "learning_rate":      0.1,
    "n_estimators":       200,
    "subsample":          0.8,
    "colsample_bytree":   0.8,
    "min_child_samples":  20,
    "reg_alpha":          0.1,
    "reg_lambda":         1.0,
    "random_state":       42,
    "verbose":            -1,
    "n_jobs":             CPU_THREADS,
    "force_col_wise":     True,
}

CNN_PARAMS = {
    "input_dim":          46,
    "num_classes":        34,
    "hidden_dims":        [256, 128, 64],
    "dropout":            0.3,
    "batch_size":         4096,
    "epochs":             30,
    "lr":                 1e-3,
    "weight_decay":       1e-4,
    "device":             GPU_DEVICE,
}

LOW_MEM_MODE          = True
TRAIN_SUBSAMPLE_RATIO = 0.25

SEED = 42
