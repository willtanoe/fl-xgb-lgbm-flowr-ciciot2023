import os
import json
import time
from pathlib import Path
import numpy as np

from flwr.common import Context
from flwr.server import ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.serverapp import ServerApp

from config import FL_ROUNDS, FRACTION_FIT, NUM_CLIENTS, TREES_PER_ROUND

BASE_DIR = Path("/home/willtanoe/Documents/fl-xgb-thesis")
LOGS_DIR = BASE_DIR / "results" / "logs"
MODELS_DIR = BASE_DIR / "results" / "models"
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


class RoundRobinStrategy(FedAvg):
    """Round-Robin FL: 1 client fit per round, all clients evaluate.
    
    Overrides aggregate_fit to skip weighted averaging of raw model bytes.
    Saves final model after last round.
    """

    def __init__(self, model_type, num_rounds, **kwargs):
        super().__init__(**kwargs)
        self.model_type = model_type
        self.num_rounds = num_rounds

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        _, fit_res = results[0]
        total_examples = fit_res.num_examples
        print(f"  └─ Global model updated (+{TREES_PER_ROUND} trees, {total_examples} samples)")

        if server_round == self.num_rounds:
            try:
                if self.model_type == "xgb":
                    import xgboost as xgb
                    model_bytes = np.array(fit_res.parameters.tensors[0]).tobytes()
                    booster = xgb.Booster(model_file=None)
                    booster.load_model(bytearray(model_bytes))
                    booster.save_model(os.path.join(MODELS_DIR, "fl_xgboost.ubj"))
                    print(f"  └─ Saved FL XGBoost: results/models/fl_xgboost.ubj")
                elif self.model_type == "lgb":
                    import lightgbm as lgb
                    model_str = np.array(fit_res.parameters.tensors[0]).tobytes().decode("utf-8")
                    model = lgb.Booster(model_str=model_str)
                    model.save_model(os.path.join(MODELS_DIR, "fl_lightgbm.txt"))
                    print(f"  └─ Saved FL LightGBM: results/models/fl_lightgbm.txt")
            except Exception as e:
                print(f"  └─ Failed to save final model: {e}")

        return fit_res.parameters, {}


def server_fn(context: Context):
    model_type = context.run_config.get("model_type", "xgb")
    num_rounds = context.run_config.get("num_rounds", FL_ROUNDS)
    metrics_path = os.path.join(LOGS_DIR, f"fl_history_{model_type}.json")

    print(f"\n{'=' * 60}")
    print(f"ROUND-ROBIN FL — {model_type.upper()} ({num_rounds} rounds, "
          f"+{TREES_PER_ROUND} trees per round = {num_rounds * TREES_PER_ROUND} total)")
    print(f"{'=' * 60}")

    metrics_data = {
        "model_type": model_type,
        "trees_per_round": TREES_PER_ROUND,
        "num_rounds": num_rounds,
        "rounds": [],
        "f1_macro": [],
        "accuracy": [],
        "per_client": [],
        "time_elapsed": [],
    }
    start_time = time.time()
    current_round = 0

    def metrics_aggregation_fn(results):
        nonlocal current_round
        current_round += 1

        per_client_log = []
        f1_scores = []
        accs = []
        for num_examples, metrics in results:
            f1 = metrics.get("f1_macro", 0)
            acc = metrics.get("accuracy", 0)
            f1_scores.append(f1)
            accs.append(acc)
            per_client_log.append({"f1_macro": f1, "accuracy": acc, "num_examples": num_examples})

        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        avg_acc = sum(accs) / len(accs) if accs else 0.0

        metrics_data["rounds"].append(current_round)
        metrics_data["f1_macro"].append(avg_f1)
        metrics_data["accuracy"].append(avg_acc)
        metrics_data["per_client"].append(per_client_log)
        metrics_data["time_elapsed"].append(time.time() - start_time)

        with open(metrics_path, "w") as f:
            json.dump(metrics_data, f, indent=2)

        total_trees = current_round * TREES_PER_ROUND
        print(f"  Round {current_round} ({total_trees} trees): "
              f"F1={avg_f1:.4f}, Acc={avg_acc:.4f}")
        for i, (fe, m) in enumerate(results):
            cid_tag = f"ClientEval {i}"
            print(f"    └─ {cid_tag}: F1={m.get('f1_macro',0):.4f}, "
                  f"Acc={m.get('accuracy',0):.4f} ({fe} samples)")

        return {"f1_macro": avg_f1, "accuracy": avg_acc}

    strategy = RoundRobinStrategy(
        model_type=model_type, num_rounds=num_rounds,
        fraction_fit=0.2,        # ~1 client per round (0.2 × 5 = 1)
        fraction_evaluate=1.0,   # all clients evaluate
        min_fit_clients=1,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        on_fit_config_fn=lambda rnd: {"round": rnd},
        on_evaluate_config_fn=lambda rnd: {"round": rnd},
        evaluate_metrics_aggregation_fn=metrics_aggregation_fn,
    )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


server_app = ServerApp(server_fn=server_fn)
