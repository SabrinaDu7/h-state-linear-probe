import logging
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
import tyro
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))

from config import Config, binarize
from data import load_eval_trajectories # type: ignore
from metrics import compute_all_metrics
from plot import plot_fold_metrics

optuna.logging.set_verbosity(optuna.logging.WARNING)
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _make_objective(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
    def objective(trial: optuna.Trial) -> float:
        C = trial.suggest_float("C", 1e-4, 1e2, log=True)
        clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs", class_weight="balanced")
        t0 = time.time()
        clf.fit(X_train, y_train)
        trial.set_user_attr("fit_time", time.time() - t0)
        scores = clf.predict_proba(X_val)[:, 1].astype(np.float32)
        return compute_all_metrics(
            torch.from_numpy(y_val.astype(np.int64)),
            torch.from_numpy(binarize(scores).astype(np.int64)),
            torch.from_numpy(scores),
        )["auc_roc"]

    return objective


def load_and_flatten(data_paths: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one or more trajectories.pt files and concatenate into a single array.

    X: [sum(B_i*T_i), H] float32
    y: [sum(B_i*T_i)]    int64
    groups: [sum(B_i*T_i)] int64 — globally unique trajectory IDs (no cross-dataset leakage)
    """
    Xs, ys, group_chunks = [], [], []
    group_offset = 0
    for path in data_paths:
        t0 = time.time()
        data = load_eval_trajectories(path)
        log.info(f"Loaded {path.parent.name} in {time.time()-t0:.2f}s")

        hidden: np.ndarray = data["hidden_states"].numpy()  # [B, T, H]
        labels: np.ndarray = data["labels"].numpy()         # [B, T]
        B, T, H = hidden.shape
        log.info(f"  B={B}, T={T}, H={H} — {B*T} samples")

        Xs.append(hidden.reshape(B * T, H).astype(np.float32))
        ys.append(labels.reshape(B * T).astype(np.int64))
        group_chunks.append(np.repeat(np.arange(B) + group_offset, T))
        group_offset += B

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    groups = np.concatenate(group_chunks, axis=0)
    log.info(f"Combined: {X.shape[0]} samples total")
    return X, y, groups


def eval_metrics(clf: LogisticRegression, scaler: StandardScaler,
                 X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Apply scaler+clf to X and return all metrics dict."""
    X_scaled = scaler.transform(X)
    scores = clf.predict_proba(X_scaled)[:, 1].astype(np.float32)
    preds = binarize(scores).astype(np.int64)
    return compute_all_metrics(
        torch.from_numpy(y),
        torch.from_numpy(preds),
        torch.from_numpy(scores),
    )


def train(cfg: Config) -> list[dict]:
    set_seed(cfg.seed)
    cfg.probes_dir.mkdir(parents=True, exist_ok=True)
    cfg.metrics_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading data from {cfg.data_paths} ...")
    t0 = time.time()
    X, y, groups = load_and_flatten(cfg.data_paths)
    t_load = time.time() - t0

    cv = StratifiedGroupKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
    fold_metrics: list[dict] = []
    all_trials: list[pd.DataFrame] = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
        log.info(f"\n--- Fold {fold} | train={len(train_idx)}, val={len(val_idx)} ---")
        X_tr_raw, X_val_raw = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        t_std = time.time()
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_raw)
        X_val = scaler.transform(X_val_raw)
        t_std = time.time() - t_std

        t_optuna = time.time()
        study = optuna.create_study(direction="maximize")
        study.optimize(_make_objective(X_tr, y_tr, X_val, y_val), n_trials=cfg.n_trials, show_progress_bar=True)
        t_optuna = time.time() - t_optuna

        avg_fit_ms = np.mean([t.user_attrs.get("fit_time", 0) for t in study.trials]) * 1000
        best_C = study.best_params["C"]
        log.info(f"  Optuna: {cfg.n_trials} trials in {t_optuna:.1f}s, avg_fit={avg_fit_ms:.0f}ms, best_C={best_C:.2e}")

        trials_df = pd.DataFrame([{
            "fold": fold, "trial": t.number, "C": t.params["C"],
            "val_auc_roc": t.value, "fit_ms": t.user_attrs.get("fit_time", 0) * 1000,
        } for t in study.trials])
        all_trials.append(trials_df)

        clf = LogisticRegression(C=best_C, max_iter=2000, solver="lbfgs", class_weight='balanced')
        clf.fit(X_tr, y_tr)

        train_m = eval_metrics(clf, scaler, X_tr_raw, y_tr)
        val_m = eval_metrics(clf, scaler, X_val_raw, y_val)
        log.info(f"  Train: {train_m}")
        log.info(f"  Val:   {val_m}")

        fold_metrics.append({
            "fold": fold,
            "best_C": best_C,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "t_load_s": t_load,
            "t_std_s": t_std,
            "t_optuna_s": t_optuna,
            "avg_fit_ms": avg_fit_ms,
            "train": train_m,
            "val": val_m,
        })

        torch.save(
            {
                "coef": clf.coef_,
                "intercept": clf.intercept_,
                "scaler_mean": scaler.mean_,
                "scaler_scale": scaler.scale_,
                "best_C": best_C,
                "fold": fold,
                "val_auc_roc": study.best_value,
            },
            cfg.probes_dir / f"{cfg.dataset_name}_fold{fold}.pt",
        )

    # Save metrics parquet (upsert by dataset_name + fold)
    metrics_rows = [{
        "dataset_name": cfg.dataset_name, "fold": m["fold"],
        "best_C": m["best_C"], "n_train": m["n_train"], "n_val": m["n_val"],
        "t_load_s": m["t_load_s"], "t_std_s": m["t_std_s"],
        "t_optuna_s": m["t_optuna_s"], "avg_fit_ms": m["avg_fit_ms"],
        **{f"train_{k}": v for k, v in m["train"].items()},
        **{f"val_{k}": v for k, v in m["val"].items()},
    } for m in fold_metrics]
    metrics_df = pd.DataFrame(metrics_rows)
    if cfg.metrics_path.exists():
        existing = pd.read_parquet(cfg.metrics_path)
        existing = existing[existing["dataset_name"] != cfg.dataset_name]
        metrics_df = pd.concat([existing, metrics_df], ignore_index=True)
    metrics_df.to_parquet(cfg.metrics_path, index=False)
    log.info(f"Saved metrics → {cfg.metrics_path}")

    # Save trials parquet (upsert by dataset_name + fold)
    trials_df = pd.concat(all_trials, ignore_index=True)
    trials_df.insert(0, "dataset_name", cfg.dataset_name)
    if cfg.trials_path.exists():
        existing = pd.read_parquet(cfg.trials_path)
        existing = existing[existing["dataset_name"] != cfg.dataset_name]
        trials_df = pd.concat([existing, trials_df], ignore_index=True)
    trials_df.to_parquet(cfg.trials_path, index=False)
    log.info(f"Saved trials  → {cfg.trials_path}")

    return fold_metrics


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    fold_metrics = train(cfg)
    plot_fold_metrics(fold_metrics, cfg)
