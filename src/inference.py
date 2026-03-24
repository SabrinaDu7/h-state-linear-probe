import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import tyro
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))

from probe import eval_metrics, load_and_flatten

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")


@dataclass
class InferenceConfig:
    # Trained probe to load (identifies which .pt files to use)
    probe_dataset_step: int = 1008
    probe_dataset_goal: int = 147

    # Dataset to evaluate on
    eval_dataset_step: int = 1208
    eval_dataset_goal: int = 147

    # Shared paths (should match what was used during training)
    data_dir: Path = Path("data/")
    probes_dir: Path = Path("outputs/probes/")
    metrics_dir: Path = Path("outputs/metrics/")

    @property
    def probe_dataset_name(self) -> str:
        return f"probe_step{self.probe_dataset_step}_goal{self.probe_dataset_goal}"

    @property
    def eval_dataset_name(self) -> str:
        return f"probe_step{self.probe_dataset_step}_goal{self.probe_dataset_goal}_eval_step{self.eval_dataset_step}_goal{self.eval_dataset_goal}"

    @property
    def eval_data_path(self) -> Path:
        return self.data_dir / f"data_cur_lroom_step{self.eval_dataset_step}_goal{self.eval_dataset_goal}" / "trajectories.pt"

    @property
    def metrics_path(self) -> Path:
        return self.metrics_dir / f"{self.eval_dataset_name}_metrics.parquet"


def _load_probe(checkpoint_path: Path) -> tuple[LogisticRegression, StandardScaler]:
    """Reconstruct clf and scaler from a saved probe .pt file."""
    ckpt = torch.load(checkpoint_path, weights_only=False)
    clf = LogisticRegression()
    clf.coef_ = ckpt["coef"]
    clf.intercept_ = ckpt["intercept"]
    clf.classes_ = np.array([0, 1])

    scaler = StandardScaler()
    scaler.mean_ = ckpt["scaler_mean"]
    scaler.scale_ = ckpt["scaler_scale"]
    return clf, scaler


def evaluate(cfg: InferenceConfig) -> list[dict]:
    cfg.metrics_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Eval data:  {cfg.eval_data_path}")
    log.info(f"Probe source: {cfg.probe_dataset_name}")

    X, y, _ = load_and_flatten(cfg.eval_data_path)

    probe_files = sorted(cfg.probes_dir.glob(f"{cfg.probe_dataset_name}_fold*.pt"))
    if not probe_files:
        raise FileNotFoundError(f"No probe files found for {cfg.probe_dataset_name} in {cfg.probes_dir}")
    log.info(f"Found {len(probe_files)} probe(s): {[p.name for p in probe_files]}")

    fold_metrics: list[dict] = []
    for probe_path in probe_files:
        fold = int(probe_path.stem.split("_fold")[-1])
        clf, scaler = _load_probe(probe_path)
        m = eval_metrics(clf, scaler, X, y)
        log.info(f"  Fold {fold}: {m}")
        fold_metrics.append({"fold": fold, "n_eval": len(y), **m})

    metrics_df = pd.DataFrame([{
        "probe_dataset_name": cfg.probe_dataset_name,
        "eval_dataset_name": cfg.eval_dataset_name,
        **row,
    } for row in fold_metrics])
    if cfg.metrics_path.exists():
        existing = pd.read_parquet(cfg.metrics_path)
        existing = existing[existing["eval_dataset_name"] != cfg.eval_dataset_name]
        metrics_df = pd.concat([existing, metrics_df], ignore_index=True)
    metrics_df.to_parquet(cfg.metrics_path, index=False)
    log.info(f"Saved metrics → {cfg.metrics_path}")

    return fold_metrics


if __name__ == "__main__":
    cfg = tyro.cli(InferenceConfig)
    evaluate(cfg)
