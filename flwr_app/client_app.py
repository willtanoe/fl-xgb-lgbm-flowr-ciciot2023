import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb

from flwr.common import Context
from flwr.client import NumPyClient, Client
from flwr.clientapp import ClientApp

from config import XGB_PARAMS, LGB_PARAMS, SEED, NUM_CLIENTS, TREES_PER_ROUND

BASE_DIR = Path("/home/willtanoe/Documents/fl-xgb-thesis")
PREPROCESSED_DIR = BASE_DIR / "results" / "preprocessed"
CLIENTS_DIR = BASE_DIR / "results" / "clients"

with open(os.path.join(PREPROCESSED_DIR, "metadata.json")) as f:
    metadata = json.load(f)
feature_cols = metadata["feature_cols"]

_val_sample = pd.read_parquet(os.path.join(PREPROCESSED_DIR, "val_sample.parquet"))
X_val = _val_sample[feature_cols].values
y_val = _val_sample["label"].values


def _deserialize_xgb(params):
    """Deserialize Flower parameters → XGBoost Booster."""
    if not params or not params[0].size:
        return None
    model_bytes = params[0].tobytes()
    model = xgb.Booster(model_file=None)
    model.load_model(bytearray(model_bytes))
    return model

def _serialize_xgb(model):
    """XGBoost Booster → Flower parameters."""
    model_bytes = model.save_raw()
    return [np.frombuffer(model_bytes, dtype=np.uint8)]

def _deserialize_lgb(params):
    if not params or not params[0].size:
        return None
    model_str = params[0].tobytes().decode("utf-8")
    return lgb.Booster(model_str=model_str)

def _serialize_lgb(model):
    model_str = model.model_to_string()
    model_bytes = model_str.encode("utf-8")
    return [np.frombuffer(model_bytes, dtype=np.uint8)]


class XGBNumPyClient(NumPyClient):
    def __init__(self, cid):
        self.cid = cid
        client_dir = os.path.join(CLIENTS_DIR, f"client_{cid}")
        df = pd.read_parquet(os.path.join(client_dir, "data.parquet"))
        self.X_train = df[feature_cols].values
        self.y_train = df["label"].values
        self.X_stop = X_val
        self.y_stop = y_val
        self.params = XGB_PARAMS.copy()

    def fit(self, parameters, config):
        rnd = config.get("round", "?")
        n_samples = len(self.X_train)
        print(f"  └─ [Client {self.cid}] XGB fit (round {rnd}, {n_samples} samples, +{TREES_PER_ROUND} trees)")

        prev_model = _deserialize_xgb(parameters)
        dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        model = xgb.train(
            self.params, dtrain,
            num_boost_round=TREES_PER_ROUND,
            xgb_model=prev_model,
        )
        return _serialize_xgb(model), n_samples, {}

    def evaluate(self, parameters, config):
        model = _deserialize_xgb(parameters)
        if model is None:
            return 0.0, len(self.X_stop), {}
        dtest = xgb.DMatrix(self.X_stop)
        y_pred_proba = model.predict(dtest)
        y_pred = np.argmax(y_pred_proba, axis=1)
        from sklearn.metrics import f1_score, accuracy_score
        f1 = f1_score(self.y_stop, y_pred, average="macro")
        acc = accuracy_score(self.y_stop, y_pred)
        rnd = config.get("round", "?")
        print(f"  └─ [Client {self.cid}] XGB eval (round {rnd}) → F1={f1:.4f}, Acc={acc:.4f}")
        return float(1.0 - acc), len(self.X_stop), {"f1_macro": float(f1), "accuracy": float(acc)}


class LGBNumPyClient(NumPyClient):
    def __init__(self, cid):
        self.cid = cid
        client_dir = os.path.join(CLIENTS_DIR, f"client_{cid}")
        df = pd.read_parquet(os.path.join(client_dir, "data.parquet"))
        self.X_train = df[feature_cols].values
        self.y_train = df["label"].values
        self.X_stop = X_val
        self.y_stop = y_val
        self.params = LGB_PARAMS.copy()

    def fit(self, parameters, config):
        rnd = config.get("round", "?")
        n_samples = len(self.X_train)
        print(f"  └─ [Client {self.cid}] LGB fit (round {rnd}, {n_samples} samples, +{TREES_PER_ROUND} trees)")

        prev_model = _deserialize_lgb(parameters)
        train_data = lgb.Dataset(self.X_train, label=self.y_train)
        model = lgb.train(
            self.params, train_data,
            num_boost_round=TREES_PER_ROUND,
            init_model=prev_model,
        )
        return _serialize_lgb(model), n_samples, {}

    def evaluate(self, parameters, config):
        model = _deserialize_lgb(parameters)
        if model is None:
            return 0.0, len(self.X_stop), {}
        y_pred_proba = model.predict(self.X_stop)
        y_pred = np.argmax(y_pred_proba, axis=1)
        from sklearn.metrics import f1_score, accuracy_score
        f1 = f1_score(self.y_stop, y_pred, average="macro")
        acc = accuracy_score(self.y_stop, y_pred)
        rnd = config.get("round", "?")
        print(f"  └─ [Client {self.cid}] LGB eval (round {rnd}) → F1={f1:.4f}, Acc={acc:.4f}")
        return float(1.0 - acc), len(self.X_stop), {"f1_macro": float(f1), "accuracy": float(acc)}


def client_fn(context: Context) -> Client:
    cid = context.node_id % NUM_CLIENTS
    model_type = context.run_config.get("model_type", "xgb")
    if model_type == "lgb":
        return LGBNumPyClient(cid).to_client()
    return XGBNumPyClient(cid).to_client()


app = ClientApp(client_fn=client_fn)
