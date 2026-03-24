import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from config import Config


def plot_fold_metrics(fold_metrics: list[dict], cfg: Config) -> None:
    """Plot train/val accuracy, F1, and AUC-ROC per fold with mean ± std overlaid."""
    cfg.plots_dir.mkdir(parents=True, exist_ok=True)

    metrics = ["accuracy", "f1", "auc_roc"]
    titles = ["Accuracy", "F1", "AUC-ROC"]
    n_folds = len(fold_metrics)
    folds = [m["fold"] for m in fold_metrics]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    fig.suptitle(cfg.dataset_name, fontsize=12)

    for ax, metric, title in zip(axes, metrics, titles):
        train_vals = np.array([m["train"][metric] for m in fold_metrics])
        val_vals = np.array([m["val"][metric] for m in fold_metrics])

        x = np.arange(n_folds)
        width = 0.35
        ax.bar(x - width / 2, train_vals, width, label="Train", alpha=0.8)
        ax.bar(x + width / 2, val_vals, width, label="Val", alpha=0.8)

        # Mean ± std lines
        for vals, color, split in [(train_vals, "C0", "train"), (val_vals, "C1", "val")]:
            mean, std = vals.mean(), vals.std()
            ax.axhline(mean, color=color, linestyle="--", linewidth=1.2,
                       label=f"{split} mean={mean:.3f}±{std:.3f}")
            ax.axhspan(mean - std, mean + std, color=color, alpha=0.1)

        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Fold {f}" for f in folds], rotation=15)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7)

    plt.tight_layout()
    out_path = cfg.plots_dir / f"{cfg.dataset_name}.png"
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Plot saved to {out_path}")


def plot_val_auc_vs_C(combined_trials_df: pd.DataFrame, out_path: Path) -> None:
    """Plot val AUC-ROC vs C (log x-axis) for each dataset.

    Shows scatter of all Optuna trial points and a binned mean trend line.
    """
    datasets = sorted(combined_trials_df["dataset_name"].unique())
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(datasets)))

    fig, ax = plt.subplots(figsize=(8, 5))

    for color, ds_name in zip(colors, datasets):
        df = combined_trials_df[combined_trials_df["dataset_name"] == ds_name].copy()
        ax.scatter(df["C"], df["val_auc_roc"], alpha=0.25, s=18, color=color)

        # Bin C on log scale, compute mean ± std per bin
        log_edges = np.logspace(np.log10(df["C"].min()), np.log10(df["C"].max()), 12)
        df["bin"] = pd.cut(df["C"], bins=log_edges, include_lowest=True)
        grouped = df.groupby("bin", observed=True)["val_auc_roc"].agg(["mean", "std"])
        # geometric-mean bin centres
        bin_centres = np.array([
            np.sqrt(iv.left * iv.right) for iv in grouped.index
        ])
        valid = grouped["mean"].notna().values
        means = grouped["mean"].values[valid]
        stds = grouped["std"].fillna(0).values[valid]
        xs = bin_centres[valid]

        ax.plot(xs, means, "o-", color=color, label=ds_name, linewidth=1.8, markersize=5)
        ax.fill_between(xs, means - stds, means + stds, color=color, alpha=0.15)

    ax.set_xscale("log")
    ax.set_xlabel("C (regularisation strength)")
    ax.set_ylabel("Val AUC-ROC")
    ax.set_title("Val AUC-ROC vs C — Optuna trials")
    ax.grid(True, alpha=0.3)
    ax.legend(title="dataset")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Plot saved to {out_path}")
