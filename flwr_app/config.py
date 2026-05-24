"""Flower App configuration (FL-specific settings)."""

# Hardware
CPU_THREADS = 1

# Federated Learning
NUM_CLIENTS = 5
FL_ROUNDS = 10
TREES_PER_ROUND = 20      # trees added per client per round (total: 10×20=200)
FRACTION_FIT = 1.0
LOCAL_EPOCHS = 5

# XGBoost params
XGB_PARAMS = {
    "objective":          "multi:softprob",
    "num_class":          34,
    "tree_method":        "hist",
    "device":             "cpu",
    "max_depth":          8,
    "learning_rate":      0.1,
    "subsample":          0.8,
    "colsample_bytree":   0.8,
    "min_child_weight":   1,
    "reg_alpha":          0.1,
    "reg_lambda":         1.0,
    "random_state":       42,
    "verbosity":          0,
    "n_jobs":             CPU_THREADS,
}

# LightGBM params
LGB_PARAMS = {
    "objective":          "multiclass",
    "num_class":          34,
    "boosting_type":      "gbdt",
    "device":             "cpu",
    "max_depth":          8,
    "num_leaves":         127,
    "learning_rate":      0.1,
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

SEED = 42
