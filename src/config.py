from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MASK_THRESHOLD = 0.5


def binarize(scores: np.ndarray) -> np.ndarray:
    return (scores >= MASK_THRESHOLD).astype(int)


@dataclass
class Config:
    dataset_step: int = 1208
    dataset_goals: list[int] = field(default_factory=lambda: [147])
    data_dir: Path = Path("data/")
    probes_dir: Path = Path("outputs/probes/")
    plots_dir: Path = Path("outputs/plots/")
    metrics_dir: Path = Path("outputs/metrics/")

    # Training
    n_splits: int = 5
    n_trials: int = 15
    seed: int = 42
    logo: bool = False  # leave-one-goal-out CV instead of StratifiedGroupKFold

    @property
    def dataset_name(self) -> str:
        goals_str = "_".join(str(g) for g in self.dataset_goals)
        suffix = "_logo" if self.logo else ""
        return f"probe_step{self.dataset_step}_goal{goals_str}{suffix}"

    @property
    def data_paths(self) -> list[Path]:
        return [
            self.data_dir / f"data_cur_lroom_step{self.dataset_step}_goal{g}" / "trajectories.pt"
            for g in self.dataset_goals
        ]

    @property
    def metrics_path(self) -> Path:
        return self.metrics_dir / f"{self.dataset_name}_metrics.parquet"

    @property
    def trials_path(self) -> Path:
        return self.metrics_dir / f"{self.dataset_name}_trials.parquet"
