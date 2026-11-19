#!/usr/bin/env python3
"""Predict from train.py checkpoints, or recompute recall from saved TSVs.

See models/INFERENCE.md for examples and the distinction between checkpoint
inference and held-out predictions. The recall command needs only NumPy/pandas.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_checkpoint(path):
    """Load tensor/NumPy bundles with the project's older PyTorch as well as 2.6+."""
    import codecs
    import collections
    import pickle
    import types
    import torch

    known = {("collections", "OrderedDict"): collections.OrderedDict,
             ("_codecs", "encode"): codecs.encode,
             ("numpy", "ndarray"): np.ndarray, ("numpy", "dtype"): np.dtype}
    for module in ("numpy.core.multiarray", "numpy._core.multiarray"):
        known[module, "_reconstruct"] = np.core.multiarray._reconstruct
        known[module, "scalar"] = np.core.multiarray.scalar
    for name in ("_rebuild_tensor", "_rebuild_tensor_v2", "_rebuild_tensor_v3", "_rebuild_parameter"):
        function = getattr(torch._utils, name, None)
        if function is not None:
            known["torch._utils", name] = function

    class RestrictedUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if (module, name) not in known:
                raise pickle.UnpicklingError(f"Unsupported checkpoint global: {module}.{name}")
            return known[module, name]

    restricted = types.ModuleType("prediction_checkpoint_pickle")
    restricted.Unpickler = RestrictedUnpickler
    restricted.load = lambda handle, **kwargs: RestrictedUnpickler(handle, **kwargs).load()
    return torch.load(path, map_location="cpu", weights_only=False, pickle_module=restricted)


def predict_checkpoint(checkpoint, hdf5, batch_size=1024, device="cpu"):
    import h5py
    import torch
    if __package__:
        from .train import PairMLP, decode_names
    else:
        from train import PairMLP, decode_names

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    bundle = load_checkpoint(checkpoint)
    model = PairMLP(bundle["input_dim"], bundle["hidden_dims"], bundle.get("dropout", 0.0))
    model.load_state_dict(bundle["state_dict"])
    model.to(device).eval()
    mean = np.asarray(bundle["feature_mean"], dtype=np.float32)
    std = np.asarray(bundle["feature_std"], dtype=np.float32)
    if (mean.shape != (bundle["input_dim"],) or std.shape != mean.shape
            or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any()):
        raise ValueError("Invalid checkpoint normalization statistics")
    metric = bundle["metric"]
    keys = bundle["pair_keys"]
    with h5py.File(hdf5, "r") as handle:
        names = decode_names(handle["system_name"][:])
        n = len(names)
        if len(set(names)) != n:
            raise ValueError("system_name must be unique")
        for key in keys:
            if handle[key].ndim != 2 or len(handle[key]) != n:
                raise ValueError(f"{key} must have shape [number of pairs, features]")
        if sum(handle[key].shape[1] for key in keys) != bundle["input_dim"]:
            raise ValueError("HDF5 feature width does not match checkpoint input_dim")
        frame = pd.DataFrame({"row": np.arange(n), "system_name": names})
        for column, key in (("y", metric), ("label", "label"),
                            ("uniprot_a", "uniprot_a"), ("uniprot_b", "uniprot_b"),
                            ("neg_source", "neg_source")):
            if key in handle:
                values = handle[key][:]
                if values.shape != (n,):
                    raise ValueError(f"{key} must be a row-aligned vector")
                frame[column] = decode_names(values) if values.dtype.kind in "OSU" else values
        for column in ("y", "label"):
            if column not in frame:
                frame[column] = np.nan
        raw = np.empty(n, dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, n, batch_size):
                stop = min(start + batch_size, n)
                features = np.concatenate([handle[key][start:stop] for key in keys], axis=1)
                features = (features.astype(np.float32) - mean) / std
                if not np.isfinite(features).all():
                    raise ValueError(f"Non-finite features in rows {start}:{stop}")
                raw[start:stop] = model(torch.from_numpy(features).to(device)).cpu().numpy()
    if not np.isfinite(raw).all():
        raise ValueError("Model produced non-finite predictions")
    bounds = bundle.get("clip_bounds")
    frame["pred_raw"] = raw
    frame["pred"] = raw if bounds is None else np.clip(raw, *bounds)
    frame["metric"] = metric
    frame["prediction_kind"] = "checkpoint_inference"
    frame["checkpoint"] = str(Path(checkpoint).resolve())
    return frame


def recall_table(frame, fractions=(0.01, 0.05, 0.1, 0.2), top_k=(), thresholds=(),
                 positive="label", true_threshold=None, score="pred_raw"):
    """Top-N overlap, or recall/precision of positives selected by rank or score.

    Fractions use round(fraction * n), matching train.py. Ties keep TSV order.
    Missing truth/labels are excluded and reported in n_excluded. Zero positive
    denominators give NaN; selecting no pairs gives zero recall if positives exist.
    """
    if any(not 0 < f <= 1 for f in fractions):
        raise ValueError("fractions must be in (0, 1]")
    if any(k < 0 or int(k) != k for k in top_k):
        raise ValueError("top-k must contain nonnegative integers")
    if any(not np.isfinite(t) for t in thresholds):
        raise ValueError("thresholds must be finite")
    if positive not in ("label", "true-threshold", "top-n"):
        raise ValueError("Unknown positive definition")
    if positive == "top-n" and thresholds:
        raise ValueError("top-n overlap supports fractions/top-k, not score thresholds")
    if positive == "true-threshold" and (true_threshold is None or not np.isfinite(true_threshold)):
        raise ValueError("true-threshold requires a finite --true-threshold")
    for column in ("metric", "config", "pooling", "prediction_kind", "checkpoint"):
        if column in frame and frame[column].nunique(dropna=False) > 1:
            raise ValueError(f"Filter to one {column} before computing recall")
    identity = "system_name" if "system_name" in frame else "row"
    if identity in frame and frame[identity].duplicated().any():
        raise ValueError("Duplicate pairs: use the deduplicated pooled table")
    truth_column = "label" if positive == "label" else "y"
    values = frame[[score, truth_column]].apply(pd.to_numeric, errors="raise")
    valid = np.isfinite(values).all(axis=1)
    values = values.loc[valid]
    n = len(values)
    scores = values[score].to_numpy()
    truth = values[truth_column].to_numpy()
    if positive == "label" and not np.isin(truth, [0, 1]).all():
        raise ValueError("Binary labels must be 0 or 1 (missing labels may be NaN)")
    order = np.argsort(-scores, kind="stable")
    true_order = np.argsort(-truth, kind="stable")
    cutoffs = [("fraction", f, round(f * n)) for f in fractions]
    cutoffs += [("top_k", k, min(k, n)) for k in top_k]
    cutoffs += [("threshold", t, None) for t in thresholds]
    records = []
    for selection, cutoff, count in cutoffs:
        selected = np.zeros(n, dtype=bool)
        if selection == "threshold":
            selected = scores >= cutoff
        else:
            selected[order[:count]] = True
        if positive == "top-n":
            positives = np.zeros(n, dtype=bool)
            positives[true_order[:count]] = True
        else:
            positives = truth == 1 if positive == "label" else truth >= true_threshold
        hits = int((selected & positives).sum())
        n_selected, n_positive = int(selected.sum()), int(positives.sum())
        records.append({
            "positive": positive, "true_threshold": true_threshold, "score": score,
            "selection": selection, "cutoff": cutoff, "n_pairs": n,
            "n_excluded": len(frame) - n, "n_selected": n_selected,
            "n_positive": n_positive, "hits": hits,
            "recall": hits / n_positive if n_positive else np.nan,
            "precision": hits / n_selected if n_selected else np.nan,
        })
    return pd.DataFrame(records)


def write_table(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    predict = commands.add_parser("predict", help="Run a saved train.py model on pooled HDF5 features")
    predict.add_argument("--checkpoint", required=True, help="train.py .pt checkpoint")
    predict.add_argument("--hdf5", required=True)
    predict.add_argument("--output", required=True)
    predict.add_argument("--batch-size", type=int, default=1024)
    predict.add_argument("--device", default="cpu")
    recall = commands.add_parser("recall", help="Evaluate a saved prediction TSV without running a model")
    recall.add_argument("--predictions", required=True)
    recall.add_argument("--output", help="Optional recall TSV; also prints the table")
    recall.add_argument("--positive", choices=["label", "true-threshold", "top-n"], default="label")
    recall.add_argument("--true-threshold", type=float)
    recall.add_argument("--fractions", type=float, nargs="*", default=[0.01, 0.05, 0.1, 0.2],
                        help="Fractional budgets; pass with no values to disable")
    recall.add_argument("--top-k", type=int, nargs="*", default=[])
    recall.add_argument("--thresholds", type=float, nargs="*", default=[])
    recall.add_argument("--score", choices=["pred_raw", "pred"], default="pred_raw")
    recall.add_argument("--regime", choices=["C1", "C2", "C3"])
    args = parser.parse_args(argv)
    if args.command == "predict":
        frame = predict_checkpoint(args.checkpoint, args.hdf5, args.batch_size, args.device)
        write_table(frame, args.output)
        print(f"Saved {len(frame)} pairs to {args.output}. These are checkpoint predictions; "
              "they are not cross-validation held-out predictions.")
    else:
        frame = pd.read_csv(args.predictions, sep="\t")
        if args.regime:
            frame = frame.loc[frame["regime"] == args.regime]
        result = recall_table(frame, args.fractions, args.top_k, args.thresholds,
                              args.positive, args.true_threshold, args.score)
        if args.output:
            write_table(result, args.output)
        print(result.to_string(index=False))


if __name__ == "__main__":
    main()
