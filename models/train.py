#!/usr/bin/env python3
"""Train PyTorch MLP regressors on pooled AF3 pair representations.

The HDF5 file is expected to contain row-aligned datasets named by PAIR_KEYS,
``system_name``, ``label``, and one dataset per regression metric.  Fold
assignments come from ``build_splits.py``: a TSV with ``system_name``,
``uniprot_a``, ``uniprot_b`` and ``fold_0`` ... ``fold_N`` columns whose values
are ``train`` or ``test``.

EVALUATION PROTOCOL
-------------------
The folds are target-protein holdout with NON-DISJOINT test sets, not k-fold
cross-validation.  A pair is test in fold k if either protein's sequence cluster
is a fold-k holdout target, so a pair whose two proteins are targets in
different folds is tested in BOTH folds.  Two consequences:

  * The headline number is the mean across folds with the observed spread.  That
    spread is NOT a standard error: the folds share most of their training rows,
    so the per-fold estimates are correlated and std/sqrt(k) would understate
    the true uncertainty.
  * The pooled scatter deduplicates by HDF5 row before scoring, averaging the
    predictions for any pair tested in more than one fold.  Plain concatenation
    would double-count roughly 60% of pairs, and not a random 60%.

Test pairs are additionally split by difficulty regime, by how many of a pair's
two proteins the model saw during training:

  * C1 -- both seen.  Impossible under target-protein holdout; tracked so a
    broken split shows up loudly instead of silently inflating scores.
  * C2 -- one seen, one held out.  The easier, majority case.
  * C3 -- neither seen.  The harder case, and the one that matches screening
    genuinely novel protein pairs.

Predictions are clipped to each metric's definitional [0, 1] range before
scoring.  Because the bound is known a priori rather than read off the data,
clipping can only move a prediction closer to the truth, so the clipped scores
are a better estimate of accuracy rather than a flattering one.  The fraction
of raw predictions falling outside the range, and the largest overshoot, are
reported so the clipping never hides a calibration problem.

Example:
    python train.py --hdf5 embeddings.h5 --folds fold_assignments.tsv \
        --metrics all --pooling both --n-folds 5 --save-models

    # local figures only, no experiment tracking
    python train.py --hdf5 embeddings.h5 --folds fold_assignments.tsv --no-wandb
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import h5py
import matplotlib
matplotlib.use("Agg")  # headless backend; must be set before pyplot is imported
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, r2_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:  # optional dependency, so --no-wandb works on a bare environment
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PAIR_KEYS = ["pair_row_mean", "pair_row_max", "pair_col_mean", "pair_col_max"]
INTERFACE_METRICS = ["ipSAE", "ipSAE_d0dom", "ipTM_af", "pDockQ2", "LIS", "ipTM_d0chn"]
# INTERFACE_METRICS = ["ipSAE_d0dom", "ipTM_d0chn", "ipTM_af"]

# Difficulty regimes for a test pair, by how many of its two proteins the model
# saw during training. See module docstring.
REGIMES = ("C1", "C2", "C3")
REGIME_HELP = {
    "C1": "both proteins seen in train (should be 0 under target holdout)",
    "C2": "one protein seen in train, one held out",
    "C3": "neither protein seen in train",
}
REGIME_COLORS = {"C1": "#d62728", "C2": "#1f77b4", "C3": "#ff7f0e"}
# Below this many rows, R2 is dominated by the variance denominator rather than
# by prediction quality, so only the count is reported.
MIN_REGIME_ROWS = 30


@dataclass(frozen=True)
class TrainConfig:
    hidden_dims: tuple[int, ...]
    weight_decay: float
    learning_rate: float = 1e-3
    batch_size: int = 256
    dropout: float = 0.0

    @property
    def label(self) -> str:
        dims = "x".join(map(str, self.hidden_dims))
        # return (f"{metric}")
        return (f"h{dims}_wd{self.weight_decay:g}_lr{self.learning_rate:g}"
                f"_bs{self.batch_size}_do{self.dropout:g}")


# Definitional ranges, known a priori — not derived from the data. Clip so metric stays betwen 0 and 1.
METRIC_BOUNDS = {m: (0.0, 1.0) for m in INTERFACE_METRICS}


def pair_keys_for_pooling(pooling: str) -> list[str]:
    if pooling == "both":
        return list(PAIR_KEYS)
    return [key for key in PAIR_KEYS if key.endswith(f"_{pooling}")]


def resolve_metrics(metrics: Sequence[str]) -> list[str]:
    return list(INTERFACE_METRICS) if len(metrics) == 1 and metrics[0].lower() == "all" else list(metrics)


def decode_names(values: np.ndarray) -> np.ndarray:
    return np.asarray([v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values])


def inspect_hdf5(path: str | Path, pair_keys: Sequence[str], metrics: Sequence[str]):
    """Validate row alignment and return names, labels, targets, and input size."""
    required = [*pair_keys, "system_name", "label", *metrics]
    with h5py.File(path, "r") as handle:
        missing = [key for key in required if key not in handle]
        if missing:
            raise KeyError(f"Missing HDF5 datasets {missing}; available: {list(handle.keys())}")
        n = len(handle["system_name"])
        bad = {key: len(handle[key]) for key in required if len(handle[key]) != n}
        if bad:
            raise ValueError(f"HDF5 datasets are not row aligned (expected {n} rows): {bad}")
        dims = []
        for key in pair_keys:
            if handle[key].ndim != 2:
                raise ValueError(f"{key} must have shape [samples, features], got {handle[key].shape}")
            dims.append(handle[key].shape[1])
        names = decode_names(handle["system_name"][:])
        labels = np.asarray(handle["label"][:], dtype=np.float32)
        targets = {metric: np.asarray(handle[metric][:], dtype=np.float32) for metric in metrics}
    if len(set(names)) != len(names):
        raise ValueError("system_name must be unique so fold rows can be aligned unambiguously")
    return names, labels, targets, int(sum(dims))


def align_folds(system_names: np.ndarray, folds_path: str | Path) -> pd.DataFrame:
    folds = pd.read_csv(folds_path, sep="\t")
    if "system_name" not in folds:
        raise ValueError("Fold TSV must contain a system_name column")
    if folds["system_name"].duplicated().any():
        raise ValueError("Fold TSV contains duplicate system_name values")
    aligned = folds.set_index("system_name").reindex(system_names)

    # Restrict the NaN checks to fold columns. Checking every column means the
    # always-populated uniprot_a/uniprot_b keep isna().all() from ever being
    # true, which silently disables the missing-assignment warning.
    fold_columns = [column for column in aligned.columns if column.startswith("fold_")]
    if not fold_columns:
        raise ValueError("Fold TSV has no fold_* columns")
    missing = aligned.index[aligned[fold_columns].isna().all(axis=1)].tolist()
    if missing:
        log.warning("%d HDF5 systems have no fold assignment (e.g. %s)", len(missing), missing[:3])
    # Partial rows silently drop from a single fold's train/test sets, so surface them too.
    partial = int((aligned[fold_columns].isna().any(axis=1)
                   & ~aligned[fold_columns].isna().all(axis=1)).sum())
    if partial:
        log.warning("%d systems have some but not all fold columns populated", partial)

    if {"uniprot_a", "uniprot_b"} - set(aligned.columns):
        log.warning("Fold TSV lacks uniprot_a/uniprot_b; C2/C3 regime reporting disabled")
    return aligned


def feature_stats(path: str | Path, keys: Sequence[str], indices: np.ndarray):
    """Compute per-feature mean/std using only the specified HDF5 rows."""
    indices = np.asarray(indices, dtype=np.int64)
    if not len(indices):
        raise ValueError("Cannot compute normalization statistics from zero rows")
    # h5py requires increasing fancy indices. Fold masks naturally supply these,
    # and unique also protects callers constructing indices another way.
    indices = np.unique(indices)
    means, stds = [], []
    with h5py.File(path, "r") as handle:
        for key in keys:
            total = np.zeros(handle[key].shape[1], dtype=np.float64)
            total_sq = np.zeros_like(total)
            for start in range(0, len(indices), 4096):
                block = np.asarray(handle[key][indices[start:start + 4096]], dtype=np.float64)
                if not np.isfinite(block).all():
                    raise ValueError(f"Non-finite values found in latent feature {key}")
                total += block.sum(axis=0)
                total_sq += np.square(block).sum(axis=0)
            key_mean = total / len(indices)
            variance = np.maximum(total_sq / len(indices) - np.square(key_mean), 0.0)
            means.append(key_mean)
            stds.append(np.sqrt(variance))
    mean = np.concatenate(means).astype(np.float32)
    std = np.concatenate(stds).astype(np.float32)
    std[std < 1e-7] = 1.0
    return mean, std


class LatentPairDataset(Dataset):
    """Loads all rows for this fold/split in one bulk HDF5 read, then serves from RAM.

    Avoids re-hitting disk on every __getitem__ call across every epoch — fine for
    pooled pair representations, which are small enough to fit in memory even for
    thousands of systems.
    """

    def __init__(self, path, pair_keys, target_key, indices, mean, std, labels):
        indices = np.unique(np.asarray(indices, dtype=np.int64))  # h5py fancy-indexing needs increasing order
        with h5py.File(path, "r") as handle:
            blocks = [np.asarray(handle[key][indices], dtype=np.float32) for key in pair_keys]
            targets = np.asarray(handle[target_key][indices], dtype=np.float32)

        X = np.concatenate(blocks, axis=1)
        mean_t = torch.as_tensor(mean, dtype=torch.float32)
        std_t = torch.as_tensor(std, dtype=torch.float32)
        self.X = (torch.from_numpy(X) - mean_t) / std_t
        self.y = torch.from_numpy(targets)
        self.labels = torch.as_tensor(np.asarray(labels)[indices], dtype=torch.float32)
        # Post-unique row order. A loader with shuffle=False walks the dataset in
        # this order, so predictions line up with these HDF5 rows -- which is
        # what lets us look up uniprot_a/uniprot_b for regimes and deduplicate
        # pairs that appear in more than one fold's test set.
        self.row_indices = indices

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        return self.X[item], self.y[item], self.labels[item]

    def close(self):
        pass  # nothing to close; kept so existing call sites stay valid


class PairMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=(256, 128), dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden_dims:
            layers.extend([nn.Linear(previous, width), nn.ReLU()])
            if dropout:
                layers.append(nn.Dropout(dropout))
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(dataset, batch_size, shuffle, workers, seed, device):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
        pin_memory=device.type == "cuda", persistent_workers=workers > 0,
        generator=generator,
    )


# --------------------------------------------------------------------------- #
# Test-pair difficulty regimes
# --------------------------------------------------------------------------- #

def classify_test_regimes(aligned: pd.DataFrame, train_indices: np.ndarray,
                          test_row_indices: np.ndarray):
    """Label each test row C1/C2/C3 by how many of its proteins are in training.

    The training protein set is taken from the whole train partition, matching
    the guarantee build_splits.py enforces. Rows held back for early-stopping
    validation are included: they come from the train partition, and a protein
    seen only there still influences model selection.

    Returns None when the fold table has no uniprot columns.
    """
    if {"uniprot_a", "uniprot_b"} - set(aligned.columns):
        return None
    protein_a = aligned["uniprot_a"].to_numpy()
    protein_b = aligned["uniprot_b"].to_numpy()

    train_proteins = set(protein_a[train_indices]) | set(protein_b[train_indices])
    train_proteins.discard(np.nan)

    seen_a = np.fromiter((p in train_proteins for p in protein_a[test_row_indices]),
                         dtype=bool, count=len(test_row_indices))
    seen_b = np.fromiter((p in train_proteins for p in protein_b[test_row_indices]),
                         dtype=bool, count=len(test_row_indices))
    n_seen = seen_a.astype(np.int8) + seen_b.astype(np.int8)
    return np.array(["C3", "C2", "C1"], dtype=object)[n_seen]


def regime_breakdown(regimes, y, pred, labels) -> dict:
    """Per-regime metrics. Regimes too small for a stable R2 report only n."""
    if regimes is None:
        return {}
    out = {}
    for name in REGIMES:
        mask = regimes == name
        count = int(mask.sum())
        if not count:
            continue
        entry = {"n": count}
        if count >= MIN_REGIME_ROWS:
            entry.update(regression_metrics(y[mask], pred[mask], labels[mask]))
        out[name] = entry
    return out


def present_regimes(regimes) -> list:
    """Regimes actually present, largest first, so rarer ones draw on top."""
    if regimes is None:
        return []
    names = [name for name in REGIMES if (regimes == name).any()]
    names.sort(key=lambda name: -int((regimes == name).sum()))
    return names

def bounds_for(metric: str) -> tuple[float, float]:
    """All AF3 interface metrics are [0, 1] by definition.

    Returning the definitional bound by default, rather than looking it up in a
    dict keyed on INTERFACE_METRICS, means trimming that list (or passing
    --metrics with a name not in it) can't silently turn clipping into a no-op
    and report out_of_range as 0%.
    """
    return METRIC_BOUNDS.get(metric, (0.0, 1.0))

# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def _save(fig, out_path: Path | None):
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    return fig


def plot_topn_recall(recalls, out_path: Path | None):
    """Plot cutoff recall and its exact denominators for the supplied metrics."""
    fig, ax = plt.subplots(figsize=(7, 5))
    cutoffs = set()
    for metric, results in recalls.items():
        percentages = [100 * fraction for fraction in results]
        cutoffs.update(percentages)
        values = [entry["recall"] for entry in results.values()]
        ax.plot(percentages, values, marker="o", label=metric)
        for pct, entry in zip(percentages, results.values()):
            recall = entry["recall"]
            # Undefined cutoffs have no marker, but still show their denominator.
            ax.annotate(
                f"n_top={entry['n_top']}" + (" (undefined)" if not np.isfinite(recall) else ""),
                (pct, recall if np.isfinite(recall) else 0),
                xytext=(0, -14 if recall > 0.9 else 8),
                textcoords="offset points", ha="center", fontsize=8,
            )
    ax.set_xticks(sorted(cutoffs))
    ax.set_xlabel("N%")
    ax.set_ylabel("recall")
    ax.set_ylim(0, 1)
    ax.set_title("top-N recall (interim, imbalanced set)")
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    return _save(fig, out_path)


def plot_loss_curves(history, title: str, out_path: Path | None, test_mse: float | None = None):
    """Per-epoch train/validation MSE, with the early-stopping epoch marked.

    Test loss is not tracked per epoch on purpose (that would invite selecting on
    test), so the final test MSE is drawn as a single horizontal reference line.
    """
    train_loss = np.asarray(history["train_loss"], dtype=float)
    val_loss = np.asarray(history["val_loss"], dtype=float)
    epochs = np.arange(1, len(train_loss) + 1)

    fig, ax = plt.subplots(figsize=(5.4, 3.9), constrained_layout=True)
    ax.plot(epochs, train_loss, label="train", linewidth=1.4, color="#1f77b4")
    ax.plot(epochs, val_loss, label="validation", linewidth=1.4, color="#ff7f0e")
    if len(val_loss):
        best = int(np.argmin(val_loss))
        ax.axvline(best + 1, color="0.45", linestyle=":", linewidth=1.1,
                   label=f"best epoch = {best + 1}")
    if test_mse is not None and np.isfinite(test_mse):
        ax.axhline(test_mse, color="crimson", linestyle="--", linewidth=1.1,
                   label=f"test MSE = {test_mse:.4g}")

    ax.set_xlabel("epoch")
    ax.set_ylabel("mean squared error")
    positive = np.concatenate([train_loss, val_loss])
    if positive.size and np.all(positive > 0):
        ax.set_yscale("log")  # log scale makes late-epoch convergence legible
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7.5, frameon=False)
    ax.grid(alpha=0.25, linewidth=0.5)
    return _save(fig, out_path)


def plot_pred_vs_true(y, pred, metric: str, title: str, out_path: Path | None,
                      r2: float | None = None, rho: float | None = None,
                      bounds: tuple | None = None, regimes=None, breakdown=None):
    """Two panels: predicted vs true with a y=x line, and residuals vs true.

    `pred` should be the RAW (unclipped) predictions. Clipped points pile into a
    hard stripe along the bound that reads as an artifact, whereas the raw
    overshoot is visible and diagnostic against the drawn bound lines.

    When regimes are supplied, C2 and C3 are drawn in different colors with
    per-regime R2 in the legend, so the easy and hard halves of the test set are
    separable at a glance.
    """
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    residuals = pred - y

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), constrained_layout=True)

    # Test sets run to ~30k pairs, which turns a plain scatter into a solid
    # block, so markers are small and very transparent and rasterized.
    names = present_regimes(regimes)
    if not names:
        groups = [(None, np.ones(len(y), dtype=bool), "#1f77b4")]
    else:
        groups = [(name, regimes == name, REGIME_COLORS[name]) for name in names]

    ax = axes[0]
    for name, mask, color in groups:
        if name is None:
            label = None
        else:
            label = f"{name} (n={int(mask.sum())}"
            entry = (breakdown or {}).get(name, {})
            if "r2" in entry and np.isfinite(entry["r2"]):
                label += f", $R^2$={entry['r2']:.3f}"
            label += ")"
        ax.scatter(y[mask], pred[mask], s=6, alpha=0.15, edgecolor="none",
                   color=color, label=label, rasterized=True)

    lo = float(min(y.min(), pred.min()))
    hi = float(max(y.max(), pred.max()))
    span = hi - lo if hi > lo else 1.0
    lo, hi = lo - 0.03 * span, hi + 0.03 * span
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y = x")
    if bounds is not None:
        for edge in bounds:
            if lo <= edge <= hi:
                ax.axhline(edge, color="0.5", linestyle=":", linewidth=1)
        ax.plot([], [], color="0.5", linestyle=":", linewidth=1, label="clip bounds")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"true {metric}")
    ax.set_ylabel(f"predicted {metric} (raw)")
    annotation = [f"n = {len(y)}"]
    if r2 is not None and np.isfinite(r2):
        annotation.append(f"$R^2$ = {r2:.3f}")
    if rho is not None and np.isfinite(rho):
        annotation.append(rf"$\rho$ = {rho:.3f}")
    ax.text(0.04, 0.96, "\n".join(annotation), transform=ax.transAxes,
            va="top", ha="left", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.75, linewidth=0))
    legend = ax.legend(fontsize=7, frameon=False, loc="lower right", markerscale=2.5)
    for handle in getattr(legend, "legend_handles", getattr(legend, "legendHandles", [])):
        handle.set_alpha(1.0)
    ax.grid(alpha=0.25, linewidth=0.5)

    ax = axes[1]
    for name, mask, color in groups:
        ax.scatter(y[mask], residuals[mask], s=6, alpha=0.15, edgecolor="none",
                   color=color, rasterized=True)
    ax.axhline(0.0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel(f"true {metric}")
    ax.set_ylabel("residual (predicted - true)")
    ax.grid(alpha=0.25, linewidth=0.5)

    fig.suptitle(title, fontsize=9)
    return _save(fig, out_path)


def plot_calibration_by_regime(y, pred, regimes, metric: str, title: str,
                               out_path: Path | None, n_bins: int = 10,
                               bounds: tuple | None = None):
    """Mean prediction vs mean truth within bins of the true value, per regime.

    A single R2 says how wrong the model is on average; this says WHERE it is
    wrong. For screening, the bins near the top of the range matter most -- a
    model that compresses high-scoring pairs toward the mean will rank poorly
    even at a respectable R2. Bin counts are drawn below so sparse bins are not
    over-read.
    """
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    lo, hi = bounds if bounds is not None else (float(y.min()), float(y.max()))
    edges = np.linspace(lo, hi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    names = present_regimes(regimes) or [None]
    fig, axes = plt.subplots(2, 1, figsize=(5.8, 5.4), height_ratios=[3, 1],
                             sharex=True, constrained_layout=True)

    ax = axes[0]
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="perfect calibration")
    for name in names:
        mask = np.ones(len(y), dtype=bool) if name is None else (regimes == name)
        color = "#1f77b4" if name is None else REGIME_COLORS[name]
        # np.digitize puts values above the last edge in an extra bin; clip so
        # y == hi lands in the final real bin instead of being dropped.
        which = np.clip(np.digitize(y[mask], edges[1:-1]), 0, n_bins - 1)
        means, spreads, counts = [], [], []
        for index in range(n_bins):
            cell = which == index
            counts.append(int(cell.sum()))
            if cell.sum() >= 5:
                means.append(float(pred[mask][cell].mean()))
                spreads.append(float(pred[mask][cell].std()))
            else:
                means.append(np.nan)
                spreads.append(np.nan)
        means = np.asarray(means)
        spreads = np.asarray(spreads)
        finite = np.isfinite(means)
        ax.errorbar(centers[finite], means[finite], yerr=spreads[finite],
                    fmt="o-", markersize=4, linewidth=1.3, capsize=2.5,
                    color=color, label=name or "all test pairs")
        axes[1].plot(centers, counts, "o-", markersize=3, linewidth=1,
                     color=color, label=name or "all")

    ax.set_ylabel(f"mean predicted {metric}")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7.5, frameon=False)
    ax.grid(alpha=0.25, linewidth=0.5)

    axes[1].set_xlabel(f"true {metric} (binned)")
    axes[1].set_ylabel("pairs per bin")
    axes[1].set_yscale("symlog")
    axes[1].grid(alpha=0.25, linewidth=0.5)
    return _save(fig, out_path)


def recycle_tag(args) -> str:
    """Identify embedding and score provenance in output names."""
    return f"__emb-r{args.input_recycle}_score-r{args.target_recycle}"


def figure_dir_for(args, metric: str, config: TrainConfig) -> Path | None:
    if not args.figures:
        return None
    path = Path(args.output_dir) / "figures" / f"{args.pooling}_{metric}" / config.label
    path.mkdir(parents=True, exist_ok=True)
    return path


def log_figure(wandb_run, key: str, fig):
    if wandb_run is not None and wandb is not None:
        wandb_run.log({key: wandb.Image(fig)})


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    predictions, targets, labels = [], [], []
    for x, y, label in loader:
        predictions.append(model(x.to(device, non_blocking=True)).cpu().numpy())
        targets.append(y.numpy())
        labels.append(label.numpy())
    return tuple(np.concatenate(values) for values in (predictions, targets, labels))


def fit_model(model, train_loader, val_loader, config, epochs, patience, device,
              wandb_run=None, log_prefix=""):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loss_fn = nn.MSELoss()
    best_loss, best_state, stale = math.inf, None, 0
    history = {"train_loss": [], "val_loss": []}
    for epoch in range(epochs):
        model.train()
        total, count = 0.0, 0
        for x, y, _ in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            # Deliberately NOT clipped: clamping inside the loss zeroes the
            # gradient for every out-of-range prediction, so the model gets no
            # pressure to come back into range and can drift further out.
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
            count += len(y)
        train_loss = total / max(count, 1)
        val_pred, val_y, _ = predict(model, val_loader, device)
        val_loss = float(np.mean((val_pred - val_y) ** 2))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if wandb_run is not None:
            wandb_run.log({
                f"{log_prefix}train_loss": train_loss,
                f"{log_prefix}val_loss": val_loss,
                f"{log_prefix}epoch": epoch,
            })

        if val_loss < best_loss - 1e-8:
            best_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def split_train_validation(indices, fraction, seed):
    indices = np.asarray(indices, dtype=np.int64).copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    n_val = max(1, int(round(len(indices) * fraction)))
    if n_val >= len(indices):
        raise ValueError("A fold needs at least two finite-target training rows")
    return indices[n_val:], indices[:n_val]


def compute_topn_recall(y_true, y_pred, fractions=(0.01, 0.05, 0.10, 0.20)):
    """Return {fraction: {recall, n_top}} for aligned, scalar predictions.

    Ties are broken by input order (stable descending sort), keeping exactly
    round(fraction * n) pairs. Recall is NaN when that denominator is zero,
    including empty inputs; tiny sets are not silently promoted to top-1.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.ndim != 1 or y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must be aligned one-dimensional arrays")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("y_true and y_pred must contain only finite values")
    true_order = np.argsort(-y_true, kind="stable")
    pred_order = np.argsort(-y_pred, kind="stable")
    results = {}
    for fraction in fractions:
        if not 0 < fraction <= 1:
            raise ValueError("fractions must be in (0, 1]")
        n_top = round(float(fraction) * len(y_true))
        overlap = np.intersect1d(true_order[:n_top], pred_order[:n_top]).size
        results[fraction] = {
            "recall": float(overlap / n_top) if n_top else float("nan"),
            "n_top": n_top,
        }
    return results


def regression_metrics(y, pred, labels):
    r2 = r2_score(y, pred) if len(y) > 1 else np.nan
    rho = spearmanr(pred, y)[0] if len(y) > 1 else np.nan  # index [0], not .statistic — works across scipy versions
    auprc = average_precision_score(labels, pred) if len(np.unique(labels)) > 1 else np.nan
    return {"r2": r2, "spearman": rho, "auprc": auprc, "base_rate": float(labels.mean())}


def run_one_fold(path, metric, labels, train_indices, test_indices, input_dim,
                 pair_keys, config, args, device, fold_seed, fold=None,
                 wandb_run=None, log_prefix="", figure_dir=None, aligned=None):
    train_indices = train_indices[np.isfinite(args.targets[metric][train_indices])]
    test_indices = test_indices[np.isfinite(args.targets[metric][test_indices])]
    if len(train_indices) < 2 or not len(test_indices):
        return None
    fit_indices, val_indices = split_train_validation(train_indices, args.val_fraction, fold_seed)
    stats_key = (metric, fold_seed)
    if stats_key not in args.stats_cache:
        args.stats_cache[stats_key] = feature_stats(path, pair_keys, fit_indices)
    mean, std = args.stats_cache[stats_key]
    datasets = {
        name: LatentPairDataset(path, pair_keys, metric, idx, mean, std, labels)
        for name, idx in (("train", fit_indices), ("val", val_indices), ("test", test_indices))
    }
    loaders = {
        name: make_loader(ds, config.batch_size, name == "train", args.workers, fold_seed, device)
        for name, ds in datasets.items()
    }
    seed_everything(fold_seed)
    model = PairMLP(input_dim, config.hidden_dims, config.dropout).to(device)
    history = fit_model(model, loaders["train"], loaders["val"], config, args.epochs, args.patience, device,
                        wandb_run=wandb_run, log_prefix=log_prefix)

    lo, hi = bounds_for(metric)

    # Train predictions are clipped too, so r2_train and the train/test gap
    # compare like with like.
    train_pred_raw, train_y, _ = predict(model, loaders["train"], device)
    train_pred = np.clip(train_pred_raw, lo, hi)

    test_pred_raw, test_y, test_labels = predict(model, loaders["test"], device)
    out_of_range = float(np.mean((test_pred_raw < lo) | (test_pred_raw > hi)))
    overshoot = float(np.max(np.abs(test_pred_raw - np.clip(test_pred_raw, lo, hi))))
    test_pred = np.clip(test_pred_raw, lo, hi)

    # The test loader has shuffle=False, so predictions come back in dataset
    # order, which is row_indices -- letting us map each prediction to its pair.
    test_rows = datasets["test"].row_indices
    regimes = None
    breakdown = {}
    if aligned is not None:
        regimes = classify_test_regimes(aligned, train_indices, test_rows)
        breakdown = regime_breakdown(regimes, test_y, test_pred, test_labels)
        leaked = breakdown.get("C1", {}).get("n", 0)
        if leaked:
            log.warning("fold %s: %d test pairs have BOTH proteins in train (C1). "
                        "Target-protein holdout should make this impossible — check the split.",
                        fold, leaked)

    result = regression_metrics(test_y, test_pred, test_labels)
    test_mse = float(np.mean((test_pred - test_y) ** 2))
    result.update({
        "n_train": len(fit_indices), "n_val": len(val_indices), "n_test": len(test_indices),
        "r2_train": r2_score(train_y, train_pred) if len(train_y) > 1 else np.nan,
        "epochs": len(history["train_loss"]), "history": history,
        "test_mse": test_mse, "test_y": test_y, "test_pred": test_pred,
        "test_pred_raw": test_pred_raw, "test_rows": test_rows,
        "regimes": regimes, "by_regime": breakdown,
        "out_of_range": out_of_range, "overshoot": overshoot,
    })

    if figure_dir is not None or wandb_run is not None:
        stem = ("fold" if fold is None else f"fold{fold}") + recycle_tag(args)
        tag = "" if fold is None else f" | fold {fold}"
        subtitle = f"{metric} | {config.label}{tag}"
        fig_loss = plot_loss_curves(
            history, f"training loss — {subtitle}",
            None if figure_dir is None else figure_dir / f"{stem}_loss.png",
            test_mse=test_mse,
        )
        log_figure(wandb_run, f"figures/{log_prefix}loss_curves", fig_loss)
        plt.close(fig_loss)

        fig_scatter = plot_pred_vs_true(
            test_y, test_pred_raw, metric, f"held-out predictions — {subtitle}",
            None if figure_dir is None else figure_dir / f"{stem}_pred_vs_true.png",
            r2=result["r2"], rho=result["spearman"], bounds=(lo, hi),
            regimes=regimes, breakdown=breakdown,
        )
        log_figure(wandb_run, f"figures/{log_prefix}pred_vs_true", fig_scatter)
        plt.close(fig_scatter)

        if regimes is not None:
            fig_calib = plot_calibration_by_regime(
                test_y, test_pred, regimes, metric,
                f"calibration by regime — {subtitle}",
                None if figure_dir is None else figure_dir / f"{stem}_calibration_by_regime.png",
                bounds=METRIC_BOUNDS.get(metric),
            )
            log_figure(wandb_run, f"figures/{log_prefix}calibration_by_regime", fig_calib)
            plt.close(fig_calib)

    for dataset in datasets.values():
        dataset.close()
    return result


def start_wandb_run(args, metric: str, config: TrainConfig, n_folds: int):
    if not args.wandb:
        return None
    if wandb is None:
        raise RuntimeError("wandb is not installed; install it or pass --no-wandb")
    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        group=metric,  # groups every config for this metric together in the UI
        name=f"{metric}_{args.pooling}_{config.label}{recycle_tag(args)}"
             + (f"_fold{args.fold}" if args.fold is not None else ""),
        config={"metric": metric, "pooling": args.pooling, "n_folds": args.n_folds,
                "epochs": args.epochs, "patience": args.patience,
                "val_fraction": args.val_fraction, "seed": args.seed,
                "fold": args.fold, **asdict(config)},
    )
    # Each fold restarts at epoch 0, so give every fold its own x-axis. Without
    # this, wandb's implicit global step makes fold curves run end-to-end instead
    # of overlaying.
    for fold in range(n_folds):
        run.define_metric(f"fold_{fold}/epoch")
        run.define_metric(f"fold_{fold}/*", step_metric=f"fold_{fold}/epoch")
    return run


def deduplicate_across_folds(results) -> pd.DataFrame:
    """One row per pair, averaging predictions for pairs tested in several folds.

    Under non-disjoint target holdout a pair whose two proteins are targets in
    different folds is tested in both. Plain concatenation would give it two
    points in the pooled scatter and double its weight in the pooled R2 -- and
    the twice-tested pairs are a specific subpopulation, not a random sample.
    Averaging the duplicate predictions restores one point per pair.
    """
    frame = pd.DataFrame({
        "row": np.concatenate([r["test_rows"] for r in results]),
        "y": np.concatenate([r["test_y"] for r in results]),
        "pred": np.concatenate([r["test_pred"] for r in results]),
        "pred_raw": np.concatenate([r["test_pred_raw"] for r in results]),
        "regime": np.concatenate([
            r["regimes"] if r["regimes"] is not None
            else np.full(len(r["test_rows"]), None, dtype=object)
            for r in results
        ]),
    })
    grouped = frame.groupby("row", as_index=False).agg(
        y=("y", "first"),                # identical across folds for a given row
        pred=("pred", "mean"),
        pred_raw=("pred_raw", "mean"),
        # A pair can be C2 in one fold and C3 in another; take the harder label
        # so the pooled regime split never overstates what the model was shown.
        regime=("regime", lambda values: "C3" if "C3" in set(values) else
                (sorted(set(values) - {None})[0] if set(values) - {None} else None)),
        n_appearances=("pred", "size"),
    )
    return grouped


def evaluate_config(path, metric, labels, aligned, input_dim, pair_keys, config, args, device):
    run = start_wandb_run(args, metric, config, args.n_folds)
    figure_dir = figure_dir_for(args, metric, config)
    suffix = "" if args.fold is None else f"_fold{args.fold}"
    results = []
    selected_folds = range(args.n_folds) if args.fold is None else [args.fold]
    for fold in selected_folds:
        column = f"fold_{fold}"
        if column not in aligned:
            log.warning("%s is absent from the fold table; skipping", column)
            continue
        values = aligned[column].to_numpy()
        train_indices = np.flatnonzero(values == "train")
        test_indices = np.flatnonzero(values == "test")
        result = run_one_fold(
            path, metric, labels, train_indices, test_indices, input_dim,
            pair_keys, config, args, device, args.seed + fold, fold=fold,
            wandb_run=run, log_prefix=f"fold_{fold}/", figure_dir=figure_dir,
            aligned=aligned,
        )
        if result is not None:
            result["fold"] = fold
            results.append(result)
            if getattr(args, "save_predictions", False):
                pooled = deduplicate_across_folds(results)
                save_prediction_tables(pooled, results, metric, labels, aligned, config, args)
            regime_text = "  ".join(
                f"{name}: n={entry['n']}, R2={entry.get('r2', float('nan')):.3f}"
                for name, entry in result["by_regime"].items()
            )
            log.info("    fold %d: test R2 %.3f (train %.3f); %.2f%% of raw predictions "
                     "outside [%.3g, %.3g], max overshoot %.4f",
                     fold, result["r2"], result["r2_train"],
                     100.0 * result["out_of_range"],
                     *bounds_for(metric),
                     result["overshoot"])
            if regime_text:
                log.info("             %s", regime_text)

    if results:
        pooled = deduplicate_across_folds(results)
        duplicated = int((pooled["n_appearances"] > 1).sum())
        pooled_y = pooled["y"].to_numpy()
        pooled_pred = pooled["pred"].to_numpy()
        pooled_raw = pooled["pred_raw"].to_numpy()
        pooled_regimes = (pooled["regime"].to_numpy()
                          if pooled["regime"].notna().any() else None)
        pooled_r2 = r2_score(pooled_y, pooled_pred) if len(pooled_y) > 1 else np.nan
        pooled_rho = spearmanr(pooled_pred, pooled_y)[0] if len(pooled_y) > 1 else np.nan
        log.info("  pooled over folds: %d distinct pairs (%d tested in >1 fold, "
                 "predictions averaged); R2 %.3f", len(pooled), duplicated, pooled_r2)

        # Rank raw predictions: clipping would collapse distinct scores into ties.
        # Pooling already averages repeat test predictions to count each pair once.
        topn_recall = compute_topn_recall(pooled_y, pooled_raw)
        for fraction, entry in topn_recall.items():
            log.info("  %s top%g recall (interim): %.4f; n_top=%d",
                     metric, 100 * fraction, entry["recall"], entry["n_top"])
        if run is not None:
            run.log({
                **{f"recall/{metric}/top{100 * fraction:g}": entry["recall"]
                   for fraction, entry in topn_recall.items()},
                "recall/table": wandb.Table(
                    columns=["metric", "fraction", "n_top", "recall"],
                    data=[[metric, fraction, entry["n_top"], entry["recall"]]
                          for fraction, entry in topn_recall.items()],
                ),
            })
        if figure_dir is not None or run is not None:
            fig_recall = plot_topn_recall(
                {metric: topn_recall},
                None if figure_dir is None else figure_dir / f"pooled{suffix}{recycle_tag(args)}_topn_recall.png",
            )
            log_figure(run, "figures/topn_recall", fig_recall)
            plt.close(fig_recall)

        pooled_breakdown = {}
        if pooled_regimes is not None:
            pooled_labels = np.asarray(labels)[pooled["row"].to_numpy()]
            pooled_breakdown = regime_breakdown(pooled_regimes, pooled_y, pooled_pred, pooled_labels)

        if figure_dir is not None or run is not None:
            fig_pooled = plot_pred_vs_true(
                pooled_y, pooled_raw, metric,
                f"pooled predictions, deduplicated — {metric} | {config.label}",
                None if figure_dir is None else figure_dir / f"pooled{suffix}{recycle_tag(args)}_pred_vs_true.png",
                r2=pooled_r2, rho=pooled_rho, bounds=METRIC_BOUNDS.get(metric),
                regimes=pooled_regimes, breakdown=pooled_breakdown,
            )
            log_figure(run, "figures/pooled_pred_vs_true", fig_pooled)
            plt.close(fig_pooled)

            if pooled_regimes is not None:
                fig_calib = plot_calibration_by_regime(
                    pooled_y, pooled_pred, pooled_regimes, metric,
                    f"pooled calibration by regime — {metric} | {config.label}",
                    None if figure_dir is None else figure_dir / f"pooled{suffix}{recycle_tag(args)}_calibration_by_regime.png",
                    bounds=METRIC_BOUNDS.get(metric),
                )
                log_figure(run, "figures/pooled_calibration_by_regime", fig_calib)
                plt.close(fig_calib)

        if run is not None:
            summary = {
                "test_r2_mean": float(np.nanmean([r["r2"] for r in results])),
                "test_r2_std": float(np.nanstd([r["r2"] for r in results])),
                "test_r2_min": float(np.nanmin([r["r2"] for r in results])),
                "test_r2_max": float(np.nanmax([r["r2"] for r in results])),
                "test_spearman_mean": float(np.nanmean([r["spearman"] for r in results])),
                "test_auprc_mean": float(np.nanmean([r["auprc"] for r in results])),
                "train_r2_mean": float(np.nanmean([r["r2_train"] for r in results])),
                "test_mse_mean": float(np.nanmean([r["test_mse"] for r in results])),
                "pooled_dedup_r2": float(pooled_r2),
                "pooled_dedup_spearman": float(pooled_rho),
                "pooled_n_pairs": int(len(pooled)),
                "pooled_n_duplicated": duplicated,
                "out_of_range_mean": float(np.nanmean([r["out_of_range"] for r in results])),
                "max_overshoot": float(np.nanmax([r["overshoot"] for r in results])),
                "epochs_mean": float(np.mean([r["epochs"] for r in results])),
                "n_folds_used": len(results),
            }
            for name in REGIMES:
                entries = [r["by_regime"][name] for r in results if name in r.get("by_regime", {})]
                if not entries:
                    continue
                summary[f"{name}_n_mean"] = float(np.mean([e["n"] for e in entries]))
                for key in ("r2", "spearman", "auprc"):
                    values = [e[key] for e in entries if key in e]
                    if values:
                        summary[f"{name}_{key}_mean"] = float(np.nanmean(values))
            if "C2_r2_mean" in summary and "C3_r2_mean" in summary:
                # The headline generalization number: how much worse the model
                # is when it has seen neither protein before.
                summary["C2_minus_C3_r2"] = summary["C2_r2_mean"] - summary["C3_r2_mean"]
            run.summary.update(summary)
            run.log({
                "per_fold": wandb.Table(
                    columns=["fold", "r2", "r2_train", "spearman", "auprc",
                             "test_mse", "epochs", "n_train", "n_val", "n_test",
                             "out_of_range", "overshoot",
                             "C2_n", "C2_r2", "C2_spearman", "C3_n", "C3_r2", "C3_spearman"],
                    data=[[r["fold"], r["r2"], r["r2_train"], r["spearman"], r["auprc"],
                           r["test_mse"], r["epochs"], r["n_train"], r["n_val"], r["n_test"],
                           r["out_of_range"], r["overshoot"],
                           r["by_regime"].get("C2", {}).get("n"),
                           r["by_regime"].get("C2", {}).get("r2"),
                           r["by_regime"].get("C2", {}).get("spearman"),
                           r["by_regime"].get("C3", {}).get("n"),
                           r["by_regime"].get("C3", {}).get("r2"),
                           r["by_regime"].get("C3", {}).get("spearman")]
                          for r in results],
                )
            })

    if run is not None:
        run.finish()
    return results


def save_prediction_tables(pooled, results, metric, labels, aligned, config, args):
    """Update single-metric TSVs after each completed fold.

    Pooled ``fold`` contains sorted, comma-separated zero-based fold IDs when a
    pair was held out more than once. Per-fold files retain individual predictions
    for stability checks; pooled predictions use the existing averaging policy.
    """
    directory = Path(args.output_dir) / f"predictions_{Path(args.hdf5).stem}" / f"{args.pooling}_{metric}" / config.label
    directory.mkdir(parents=True, exist_ok=True)

    def write(frame, filename):
        frame = frame.copy()
        rows = frame["row"].to_numpy(dtype=int)
        frame["system_name"] = aligned.index.to_numpy()[rows]
        frame["label"] = np.asarray(labels)[rows]
        for column in ("uniprot_a", "uniprot_b"):
            if column in aligned:
                frame[column] = aligned[column].to_numpy()[rows]
        frame["metric"] = metric
        frame["pooling"] = args.pooling
        frame["config"] = config.label
        frame["prediction_kind"] = "cv_held_out"
        destination = directory / filename
        temporary = destination.with_suffix(".tsv.tmp")
        frame.to_csv(temporary, sep="\t", index=False)
        temporary.replace(destination)

    for result in results:
        write(pd.DataFrame({
            "row": result["test_rows"], "y": result["test_y"],
            "pred": result["test_pred"], "pred_raw": result["test_pred_raw"],
            "regime": result["regimes"], "fold": result["fold"],
        }), f"fold_{result['fold']}{recycle_tag(args)}.tsv")
    suffix = "" if args.fold is None else f"_fold{args.fold}"
    fold_membership = {}
    for result in results:
        for row in result["test_rows"]:
            fold_membership.setdefault(row, set()).add(result["fold"])
    pooled = pooled.copy()
    pooled["fold"] = pooled["row"].map({
        row: ",".join(str(fold) for fold in sorted(folds))
        for row, folds in fold_membership.items()
    })
    write(pooled, f"pooled{suffix}{recycle_tag(args)}.tsv")
    log.info("Saved held-out prediction tables to %s", directory)


def refit_and_save(path, metric, labels, input_dim, pair_keys, config, args, device, output_path):
    indices = np.flatnonzero(np.isfinite(args.targets[metric]))
    fit_indices, val_indices = split_train_validation(indices, args.val_fraction, args.seed)
    mean, std = feature_stats(path, pair_keys, fit_indices)
    train_ds = LatentPairDataset(path, pair_keys, metric, fit_indices, mean, std, labels)
    val_ds = LatentPairDataset(path, pair_keys, metric, val_indices, mean, std, labels)
    train_loader = make_loader(train_ds, config.batch_size, True, args.workers, args.seed, device)
    val_loader = make_loader(val_ds, config.batch_size, False, args.workers, args.seed, device)
    seed_everything(args.seed)
    model = PairMLP(input_dim, config.hidden_dims, config.dropout).to(device)
    history = fit_model(model, train_loader, val_loader, config, args.epochs, args.patience, device)
    torch.save({
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "input_dim": input_dim, "hidden_dims": config.hidden_dims, "dropout": config.dropout,
        "feature_mean": mean, "feature_std": std, "pair_keys": list(pair_keys),
        "metric": metric, "config": asdict(config), "epochs": len(history["train_loss"]),
        "input_recycle": args.input_recycle, "target_recycle": args.target_recycle,
        "hdf5": str(Path(path).resolve()),
        # Consumers should clip to these bounds to match how the model was scored.
        "clip_bounds": METRIC_BOUNDS.get(metric),
    }, output_path)

    figure_dir = figure_dir_for(args, metric, config)
    if figure_dir is not None:
        fig = plot_loss_curves(
            history, f"final refit — {metric} | {config.label}",
            figure_dir / f"final_refit{recycle_tag(args)}_loss.png",
        )
        plt.close(fig)

    train_ds.close()
    val_ds.close()


def load_sweep(path: str | None) -> list[TrainConfig]:
    if path is None:
        return SWEEP
    with open(path, encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list) or not rows:
        raise ValueError("--sweep-json must contain a non-empty JSON list")
    return [TrainConfig(hidden_dims=tuple(row["hidden_dims"]), **{k: v for k, v in row.items() if k != "hidden_dims"}) for row in rows]


def main(args):
    if args.n_folds < 1:
        raise ValueError("--n-folds must be positive")
    if args.fold is not None and not 0 <= args.fold < args.n_folds:
        raise ValueError("--fold must be between 0 and --n-folds minus 1")
    metrics = resolve_metrics(args.metrics)
    pair_keys = pair_keys_for_pooling(args.pooling)
    names, labels, targets, input_dim = inspect_hdf5(args.hdf5, pair_keys, metrics)
    args.targets = targets  # avoid repeatedly loading small target vectors
    args.stats_cache = {}  # normalization is identical across configs for a metric/fold
    aligned = align_folds(names, args.folds)
    if args.fold is not None and f"fold_{args.fold}" not in aligned:
        raise ValueError(f"fold_{args.fold} is absent from the fold table")
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep = load_sweep(args.sweep_json)
    # Parallel --fold jobs would otherwise all race on the same summary paths and
    # only the last writer's results would survive.
    suffix = "" if args.fold is None else f"_fold{args.fold}"
    log.info("Loaded %d systems; input dim %d; device %s", len(names), input_dim, device)

    cross_metric = []
    for metric in metrics:
        rows, config_results = [], {}
        log.info("Metric %s", metric)
        for config in sweep:
            folds = evaluate_config(
                args.hdf5, metric, labels, aligned, input_dim, pair_keys, config, args, device
            )
            if not folds:
                continue
            config_results[config.label] = folds
            test_r2 = float(np.nanmean([row["r2"] for row in folds]))
            train_r2 = float(np.nanmean([row["r2_train"] for row in folds]))
            row = {
                "pooling": args.pooling, "config": config.label,
                "fold": args.fold, "n_folds_used": len(folds),
                "r2_test": test_r2, "r2_train": train_r2, "gap": train_r2 - test_r2,
                "r2_test_std": float(np.nanstd([r["r2"] for r in folds])),
                "spearman": np.nanmean([row_["spearman"] for row_ in folds]),
                "auprc": np.nanmean([row_["auprc"] for row_ in folds]),
                "out_of_range": np.nanmean([row_["out_of_range"] for row_ in folds]),
                "max_overshoot": np.nanmax([row_["overshoot"] for row_ in folds]),
                "epochs": np.mean([row_["epochs"] for row_ in folds]),
            }
            for name in REGIMES:
                entries = [r["by_regime"][name] for r in folds if name in r.get("by_regime", {})]
                row[f"{name}_n"] = float(np.mean([e["n"] for e in entries])) if entries else 0.0
                for key in ("r2", "spearman", "auprc"):
                    values = [e[key] for e in entries if key in e]
                    row[f"{name}_{key}"] = float(np.nanmean(values)) if values else np.nan
            row["C2_minus_C3_r2"] = row.get("C2_r2", np.nan) - row.get("C3_r2", np.nan)
            rows.append(row)
            log.info("  %-35s test R2 %.3f; train R2 %.3f; %.2f%% clipped; "
                     "C2 R2 %.3f, C3 R2 %.3f (gap %.3f)",
                     config.label, test_r2, train_r2, 100.0 * row["out_of_range"],
                     row.get("C2_r2", float("nan")), row.get("C3_r2", float("nan")),
                     row["C2_minus_C3_r2"])
        if not rows:
            log.warning("No usable folds for %s", metric)
            continue
        summary = pd.DataFrame(rows).sort_values("r2_test", ascending=False)
        summary.to_csv(output_dir / f"mlp_pair_{args.pooling}_{metric}{suffix}{recycle_tag(args)}_sweep_summary.tsv",
                       sep="\t", index=False)
        best_row = summary.iloc[0]
        best_config = next(config for config in sweep if config.label == best_row["config"])
        cross_metric.append({"metric": metric, **best_row.to_dict()})
        log.info("Best %s config: %s (test R2 %.3f)", metric, best_config.label, best_row["r2_test"])
        if args.save_models and args.fold is None:
            model_path = output_dir / f"latentppi_{args.pooling}_{metric}{recycle_tag(args)}_model.pt"
            refit_and_save(
                args.hdf5, metric, labels, input_dim, pair_keys, best_config,
                args, device, model_path,
            )
            log.info("Saved %s", model_path)

    cross = pd.DataFrame(cross_metric)
    if not cross.empty:
        cross.sort_values("r2_test", ascending=False).to_csv(
            output_dir / f"mlp_pair_{args.pooling}_all_metrics{suffix}{recycle_tag(args)}_summary.tsv",
            sep="\t", index=False
        )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train separate PyTorch MLPs to predict metrics from pooled PAIR latents",
        epilog=("Sweep JSON is a list of objects, for example: "
                "[{\"hidden_dims\":[256,128],\"weight_decay\":0.0001,"
                "\"learning_rate\":0.001,\"batch_size\":256,\"dropout\":0.0}]")
    )
    parser.add_argument("--hdf5", required=True)
    parser.add_argument("--folds", required=True)
    parser.add_argument("--metrics", nargs="+", default=["all"])
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--fold", type=int,
                        help="evaluate only this zero-based fold; skips final refit")
    parser.add_argument("--input-recycle", help="recycle the embedding data came from", required=True)
    parser.add_argument("--target-recycle", help="recycle the target metric data came from", required=True)
    parser.add_argument("--pooling", choices=["mean", "max", "both"], default="max")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=0, help="DataLoader worker processes")
    parser.add_argument("--device", help="e.g. cuda, cuda:1, or cpu (default: auto)", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sweep-json", help="optional JSON file replacing the built-in SWEEP")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--save-models", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-predictions", action=argparse.BooleanOptionalAction, default=True,
                        help="save per-fold and pooled held-out TSVs under <output-dir>/predictions")
    parser.add_argument("--figures", action=argparse.BooleanOptionalAction, default=True,
                        help="write diagnostic PNGs under <output-dir>/figures")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False,
                        help="mirror scalars and figures to Weights & Biases")
    parser.add_argument("--wandb-project", default="af3-2k-r1-r10")
    parser.add_argument("--wandb-entity")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
