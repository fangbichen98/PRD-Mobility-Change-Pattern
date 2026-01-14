import os
import json
import csv
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from utils import setup_logger, ensure_dir, save_json, save_numpyz


logger = setup_logger(__name__)


def _labels_path(data_dir: str, override_path: str | None = None) -> str:
    # Support both label.csv and labels.csv
    if override_path:
        p = override_path if os.path.isabs(override_path) else os.path.join(data_dir, override_path)
        if not os.path.exists(p):
            raise FileNotFoundError(f"labels file not found: {p}")
        return p
    for name in ["labels.csv", "label.csv"]:
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError("labels.csv or label.csv not found in data dir")


def load_labels(data_dir: str, labels_path: str | None = None) -> pd.DataFrame:
    p = _labels_path(data_dir, labels_path)
    df = pd.read_csv(p)
    # Standardize columns
    need = ["grid_id", "lon", "lat", "label"]
    cols = {c: c.strip() for c in df.columns}
    df = df.rename(columns=cols)
    for c in need:
        if c not in df.columns:
            raise KeyError(f"Expected column '{c}' in labels CSV: {p}")
    # Ensure label in [1..9], convert to int
    df = df.copy()
    df["label"] = df["label"].astype(int)
    return df[need]


def build_node_index_from_grids(grid_ids: List[int]) -> Tuple[Dict[int, int], np.ndarray]:
    grid_ids_sorted = sorted(set(int(g) for g in grid_ids))
    grid_to_idx = {gid: i for i, gid in enumerate(grid_ids_sorted)}
    idx_to_grid = np.array(grid_ids_sorted, dtype=np.int64)
    return grid_to_idx, idx_to_grid


def find_first_week(csv_path: str, date_col: str = "date_dt") -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Scan the CSV to find the minimum date actually present and return [min, min+21days).
    Note: previously used 7 days; expanded to 21 days per new requirement.
    """
    min_date = None
    for chunk in pd.read_csv(csv_path, usecols=[date_col], parse_dates=[date_col], chunksize=2_000_000):
        cmin = chunk[date_col].min()
        if pd.isna(cmin):
            continue
        min_date = cmin if min_date is None else min(min_date, cmin)
        # Early stop heuristic: if we already saw at least 1e7 rows? Skip; keep simple
    if min_date is None:
        raise RuntimeError(f"No dates found in {csv_path}")
    start = pd.Timestamp(min_date.date())
    end = start + pd.Timedelta(days=21)
    return start, end


def _alloc_ts(num_nodes: int, T: int = 168, C: int = 2) -> np.ndarray:
    return np.zeros((num_nodes, T, C), dtype=np.float32)


def aggregate_time_series_for_week(
    csv_path: str,
    grid_to_idx: Dict[int, int],
    start: pd.Timestamp,
    end: pd.Timestamp,
    date_col: str = "date_dt",
    time_col: str = "time_",
    o_col: str = "o_grid",
    d_col: str = "d_grid",
    flow_col: str = "num_total",
    chunksize: int = 2_000_000,
) -> np.ndarray:
    """Build per-grid inbound/outbound time series with 168 steps (7 days x 24h).

    Returns: np.ndarray [N, 168, 2] where channels are [inbound, outbound].
    """
    num_nodes = len(grid_to_idx)
    X = _alloc_ts(num_nodes)

    date_parser = [date_col]
    usecols = [date_col, time_col, o_col, d_col, flow_col]

    for chunk in pd.read_csv(csv_path, usecols=usecols, parse_dates=date_parser, chunksize=chunksize):
        # Filter to week
        mask = (chunk[date_col] >= start) & (chunk[date_col] < end)
        if not mask.any():
            continue
        c = chunk.loc[mask, [date_col, time_col, o_col, d_col, flow_col]].copy()
        # Drop rows with missing grid ids before casting
        c = c.dropna(subset=[o_col, d_col])
        # Ensure dtypes
        c[o_col] = c[o_col].astype(int)
        c[d_col] = c[d_col].astype(int)
        c[time_col] = c[time_col].astype(int)
        c[flow_col] = c[flow_col].astype(float)

        # Compute day offset [0..6] and time idx [0..167]
        c["day_idx"] = (c[date_col].dt.normalize() - start).dt.days
        c = c[(c["day_idx"] >= 0) & (c["day_idx"] < 7) & (c[time_col] >= 0) & (c[time_col] < 24)]
        c["tidx"] = c["day_idx"] * 24 + c[time_col]

        # Outbound aggregation
        out = (
            c.groupby([o_col, "tidx"], as_index=False)[flow_col]
            .sum()
            .rename(columns={o_col: "grid", flow_col: "flow"})
        )
        # Inbound aggregation
        inc = (
            c.groupby([d_col, "tidx"], as_index=False)[flow_col]
            .sum()
            .rename(columns={d_col: "grid", flow_col: "flow"})
        )

        # Fill X
        for df, chan in ((inc, 0), (out, 1)):
            # Only keep grids present in mapping
            df = df[df["grid"].isin(grid_to_idx.keys())]
            if df.empty:
                continue
            gi = df["grid"].map(grid_to_idx).values
            ti = df["tidx"].values
            fv = df["flow"].values.astype(np.float32)
            # Accumulate (in case of multiple chunks)
            for k in range(len(df)):
                X[gi[k], ti[k], chan] += fv[k]

    # Standardize per-grid per-channel (z-score across time)
    # Avoid division by 0
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True) + 1e-6
    X = (X - mean) / std
    return X


def build_dynamic_graphs_for_week(
    csv_path: str,
    grid_to_idx: Dict[int, int],
    start: pd.Timestamp,
    end: pd.Timestamp,
    date_col: str = "date_dt",
    time_col: str = "time_",
    o_col: str = "o_grid",
    d_col: str = "d_grid",
    flow_col: str = "num_total",
    chunksize: int = 2_000_000,
    min_weight: float = 0.0,
    topk_per_source: int | None = 20,
) -> List[Data]:
    """Build a 24-snapshot dynamic graph sequence; each snapshot is the hour-of-day
    average across the 7 days. Only include nodes from labels mapping.
    Edge attribute stores average weight for that hour.
    """
    # Accumulator: dict[(hour, o_idx, d_idx)] = sum_weight
    acc: Dict[Tuple[int, int, int], float] = {}

    date_parser = [date_col]
    usecols = [date_col, time_col, o_col, d_col, flow_col]

    for chunk in pd.read_csv(csv_path, usecols=usecols, parse_dates=date_parser, chunksize=chunksize):
        mask = (chunk[date_col] >= start) & (chunk[date_col] < end)
        if not mask.any():
            continue
        c = chunk.loc[mask, [date_col, time_col, o_col, d_col, flow_col]].copy()
        # Drop rows with missing grid ids before casting
        c = c.dropna(subset=[o_col, d_col])
        c[o_col] = c[o_col].astype(int)
        c[d_col] = c[d_col].astype(int)
        c[time_col] = c[time_col].astype(int)
        c[flow_col] = c[flow_col].astype(float)

        # Keep only edges where both nodes are labeled (to bound graph size)
        c = c[c[o_col].isin(grid_to_idx.keys()) & c[d_col].isin(grid_to_idx.keys())]
        if c.empty:
            continue

        # Hour-of-day average across 7 days => sum now, divide later
        c = c[(c[time_col] >= 0) & (c[time_col] < 24)]
        # Map to indices and aggregate per (hour, o, d)
        c["o_idx"] = c[o_col].map(grid_to_idx)
        c["d_idx"] = c[d_col].map(grid_to_idx)
        agg = c.groupby([time_col, "o_idx", "d_idx"], as_index=False)[flow_col].sum()
        for row in agg.itertuples(index=False):
            h = int(getattr(row, time_col))
            o = int(getattr(row, "o_idx"))
            d = int(getattr(row, "d_idx"))
            w = float(getattr(row, flow_col))
            acc[(h, o, d)] = acc.get((h, o, d), 0.0) + w

    # Build 24 Data snapshots
    num_nodes = len(grid_to_idx)
    snapshots: List[Data] = []
    for h in range(24):
        # Group by source, then keep top-k per source by weight
        by_src: Dict[int, List[Tuple[int, float]]] = {}
        for (hh, o, d), s in acc.items():
            if hh != h:
                continue
            w = s / 7.0
            if w <= min_weight:
                continue
            lst = by_src.get(o)
            if lst is None:
                lst = []
                by_src[o] = lst
            lst.append((d, w))

        edges: List[Tuple[int, int]] = []
        weights: List[float] = []
        for o, lst in by_src.items():
            if topk_per_source is not None and len(lst) > topk_per_source:
                # keep top-k by weight
                lst.sort(key=lambda x: x[1], reverse=True)
                lst = lst[:topk_per_source]
            for d, w in lst:
                edges.append((o, d))
                weights.append(w)

        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(weights, dtype=torch.float).unsqueeze(-1)  # shape [E,1]
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float)

        # Dummy node features (learnable emb will be used in model); keep zeros placeholder
        x = torch.zeros((num_nodes, 1), dtype=torch.float)
        snapshots.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))

    return snapshots


def collect_active_grids_week(
    csv_path: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    date_col: str = "date_dt",
    time_col: str = "time_",
    o_col: str = "o_grid",
    d_col: str = "d_grid",
    chunksize: int = 2_000_000,
) -> set:
    """Collect set of grid ids that appear as origin or destination within [start, end)."""
    active = set()
    usecols = [date_col, time_col, o_col, d_col]
    for chunk in pd.read_csv(csv_path, usecols=usecols, parse_dates=[date_col], chunksize=chunksize):
        mask = (chunk[date_col] >= start) & (chunk[date_col] < end)
        if not mask.any():
            continue
        c = chunk.loc[mask, [o_col, d_col]].copy()
        # Drop rows with missing grid ids before casting
        c = c.dropna(subset=[o_col, d_col])
        c[o_col] = c[o_col].astype(int)
        c[d_col] = c[d_col].astype(int)
        active.update(c[o_col].unique().tolist())
        active.update(c[d_col].unique().tolist())
    return active


def resolve_columns(csv_path: str) -> Tuple[str, str, str, str, str]:
    """Resolve column names from CSV header allowing common aliases.
    Returns: (date_col, time_col, o_col, d_col, flow_col)
    """
    with open(csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
    header = [h.strip() for h in header]

    def pick(cands):
        for c in cands:
            if c in header:
                return c
        return None

    date_col = pick(["date_dt", "date", "date_str", "day"])
    time_col = pick(["time_", "time", "hour", "hr"])
    o_col = pick(["o_grid", "o_grid_500", "o", "o_id", "origin", "o_grid_id"])
    d_col = pick(["d_grid", "d_grid_500", "d", "d_id", "destination", "d_grid_id"])
    flow_col = pick(["num_total", "flow", "count", "volume", "trips"])

    missing = [n for n, v in [("date_col", date_col), ("time_col", time_col), ("o_col", o_col), ("d_col", d_col), ("flow_col", flow_col)] if v is None]
    if missing:
        raise ValueError(f"Could not resolve columns in {csv_path}. Missing: {missing}. Header={header}")
    return date_col, time_col, o_col, d_col, flow_col


def run_preprocess(data_dir: str, processed_dir: str, *, year_2021_path: str | None = None, year_2024_path: str | None = None, labels_path: str | None = None) -> None:
    ensure_dir(processed_dir)
    labels_df = load_labels(data_dir, labels_path)

    # Discover first-week windows for both years
    # Remap years: now use 2021 and 2024 as the pair
    csv21 = year_2021_path if year_2021_path else os.path.join(data_dir, "2021.csv")
    csv24 = year_2024_path if year_2024_path else os.path.join(data_dir, "2024.csv")
    if not os.path.exists(csv21) or not os.path.exists(csv24):
        raise FileNotFoundError("Missing 2021.csv or 2024.csv (or provided paths) in data_dir")
    logger.info("Scanning first-week windows for both years (2021 and 2024) ...")
    start21, end21 = find_first_week(csv21)
    start24, end24 = find_first_week(csv24)
    logger.info(f"2021 week: {start21.date()} to {end21.date()} | 2024 week: {start24.date()} to {end24.date()}")

    # Resolve column names (allow alias)
    cols21 = resolve_columns(csv21)
    cols24 = resolve_columns(csv24)
    # Active node sets per year (within week)
    logger.info("Collecting active grids within the week for each year ...")
    # Pass only (date, time, o, d)
    active21 = collect_active_grids_week(csv21, start21, end21, cols21[0], cols21[1], cols21[2], cols21[3])
    active24 = collect_active_grids_week(csv24, start24, end24, cols24[0], cols24[1], cols24[2], cols24[3])

    # Graph nodes = intersection across years (do NOT intersect with labels)
    graph_keep = sorted(active21 & active24)
    logger.info(f"Active 2021: {len(active21)}, Active 2024: {len(active24)}, Graph nodes (intersection): {len(graph_keep)}")
    if len(graph_keep) == 0:
        raise RuntimeError("Empty intersection of active nodes across years.")

    graph_grid_to_idx, graph_idx_to_grid = build_node_index_from_grids(graph_keep)
    save_json({"grid_to_idx": graph_grid_to_idx, "num_nodes": len(graph_grid_to_idx)}, os.path.join(processed_dir, "graph_node_mapping.json"))
    np.save(os.path.join(processed_dir, "graph_idx_to_grid.npy"), graph_idx_to_grid)

    # Filter labels to those present in the graph nodes (for training subset)
    labels_f = labels_df[labels_df["grid_id"].isin(set(graph_keep))].copy()
    labels_f = labels_f.sort_values("grid_id")
    labels_f.to_csv(os.path.join(processed_dir, "labels_filtered.csv"), index=False)

    # Build label mapping for time series (train subset only)
    label_grid_to_idx, label_idx_to_grid = build_node_index_from_grids(labels_f["grid_id"].tolist())
    save_json({"grid_to_idx": label_grid_to_idx, "num_nodes": len(label_grid_to_idx)}, os.path.join(processed_dir, "label_node_mapping.json"))
    np.save(os.path.join(processed_dir, "label_idx_to_grid.npy"), label_idx_to_grid)

    # Record selection meta
    save_json(
        {
            "start21": str(start21.date()), "end21": str(end21.date()),
            "start24": str(start24.date()), "end24": str(end24.date()),
            "active21": len(active21), "active24": len(active24),
            "graph_nodes": len(graph_keep), "labeled_kept": len(labels_f),
        },
        os.path.join(processed_dir, "node_selection.json")
    )

    # Build time series for labeled subset (compact arrays for training)
    for year, (csv_path, start, end, cols) in ((2021, (csv21, start21, end21, cols21)), (2024, (csv24, start24, end24, cols24))):
        logger.info(f"Building time series for {year} (labeled subset) ...")
        X = aggregate_time_series_for_week(csv_path, label_grid_to_idx, start, end, *cols)
        save_numpyz(os.path.join(processed_dir, f"time_series_{year}.npz"), X=X)
        logger.info(f"Saved time series for {year} -> {X.shape}")

    # Build dynamic graphs for graph node set (full intersection)
    for year, (csv_path, start, end, cols) in ((2021, (csv21, start21, end21, cols21)), (2024, (csv24, start24, end24, cols24))):
        logger.info(f"Building dynamic graphs for {year} (graph intersection set) ...")
        snapshots = build_dynamic_graphs_for_week(csv_path, graph_grid_to_idx, start, end, *cols, topk_per_source=10)
        torch.save(snapshots, os.path.join(processed_dir, f"graphs_{year}.pt"))
        logger.info(f"Saved {len(snapshots)} graph snapshots for {year}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess OD flows into time series and dynamic graphs (2021 vs 2024)")
    parser.add_argument("--data_dir", type=str, default=os.path.join(os.path.dirname(__file__), "..", "data"))
    parser.add_argument("--out_dir", type=str, default=os.path.join(os.path.dirname(__file__), "..", "processed"))
    args = parser.parse_args()

    run_preprocess(
        os.path.abspath(args.data_dir),
        os.path.abspath(args.out_dir),
        year_2021_path=args.year_2021_path,
        year_2024_path=args.year_2024_path,
        labels_path=args.labels_path,
    )


# -------------------------
# Additional utilities used after initial preprocessing
# -------------------------
def build_graph_timeseries_only(data_dir: str, processed_dir: str) -> None:
    """Build time series for ALL graph-intersection nodes (not just labeled subset).

    Requires that run_preprocess has been executed at least once to materialize:
    - processed/graph_node_mapping.json (graph_grid_to_idx)
    - processed/node_selection.json (start/end week per year)
    """
    ensure_dir(processed_dir)

    # Load mapping for graph nodes
    import json as _json
    with open(os.path.join(processed_dir, "graph_node_mapping.json"), "r") as f:
        graph_grid_to_idx = {int(k): int(v) for k, v in _json.load(f)["grid_to_idx"].items()}

    # Resolve week windows
    sel_path = os.path.join(processed_dir, "node_selection.json")
    if os.path.exists(sel_path):
        with open(sel_path, "r") as f:
            sel = _json.load(f)
        start18 = pd.Timestamp(sel["start18"]) ; end18 = pd.Timestamp(sel["end18"])
        start21 = pd.Timestamp(sel["start21"]) ; end21 = pd.Timestamp(sel["end21"])
        logger.info(f"Using saved week windows: 2018 [{start18.date()}..{end18.date()}), 2021 [{start21.date()}..{end21.date()})")
    else:
        # Fallback: rescan CSVs
        csv18 = os.path.join(data_dir, "2018.csv")
        csv21 = os.path.join(data_dir, "2021.csv")
        logger.info("node_selection.json not found; rescanning CSVs to find first-week windows ...")
        start18, end18 = find_first_week(csv18)
        start21, end21 = find_first_week(csv21)

    # Resolve column names for both CSVs
    csv18 = os.path.join(data_dir, "2018.csv")
    csv21 = os.path.join(data_dir, "2021.csv")
    cols18 = resolve_columns(csv18)
    cols21 = resolve_columns(csv21)

    # Build full-graph time series
    for year, (csv_path, start, end, cols) in ((2018, (csv18, start18, end18, cols18)), (2021, (csv21, start21, end21, cols21))):
        logger.info(f"Building time series for {year} (graph intersection set, all {len(graph_grid_to_idx)} nodes) ...")
        X = aggregate_time_series_for_week(csv_path, graph_grid_to_idx, start, end, *cols)
        save_numpyz(os.path.join(processed_dir, f"time_series_graph_{year}.npz"), X=X)
        logger.info(f"Saved full-graph time series for {year} -> {X.shape}")


def save_graph_nodes_lonlat(data_dir: str, processed_dir: str) -> None:
    """Save lon/lat for all graph-intersection nodes into processed_dir/graph_nodes_lonlat.csv.

    Uses grid metadata CSV at data_dir/grid_metadata/PRD_grid_metadata.csv.
    """
    meta_path = os.path.join(data_dir, "grid_metadata", "PRD_grid_metadata.csv")
    if not os.path.exists(meta_path):
        logger.warning(f"Grid metadata not found at {meta_path}; skip saving lon/lat for graph nodes.")
        return

    import json as _json
    with open(os.path.join(processed_dir, "graph_node_mapping.json"), "r") as f:
        graph_grid_to_idx = {int(k): int(v) for k, v in _json.load(f)["grid_to_idx"].items()}
    keep = set(graph_grid_to_idx.keys())

    meta = pd.read_csv(meta_path)
    cols = {c: c.strip() for c in meta.columns}
    meta = meta.rename(columns=cols)
    need = ["grid_id", "lon", "lat"]
    for c in need:
        if c not in meta.columns:
            raise KeyError(f"Expected column '{c}' in grid metadata: {meta_path}")
    meta = meta[need]
    meta = meta[meta["grid_id"].isin(keep)].copy()
    out_p = os.path.join(processed_dir, "graph_nodes_lonlat.csv")
    meta.to_csv(out_p, index=False)
    logger.info(f"Saved lon/lat for graph nodes -> {out_p} ({len(meta)} rows)")
