"""
Predict all grids using cached data and best model
Generate full-region visualizations with English labels
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon
import torch
import pickle
import json
import os
import sys
import math
import glob
from tqdm import tqdm
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.append('src')

from models.enhanced_dual_branch_model import EnhancedDualBranchModel
import config
from viz_config import AREA_NAME_EN, CITY_NAME_EN, CLASS_COLORS, CLASS_NAMES, DEFAULT_DPI, EXPORT_DPI, EXPERIMENT_DIR, FONT_FAMILY

# Set matplotlib parameters
plt.rcParams['font.family'] = FONT_FAMILY
plt.rcParams['figure.dpi'] = DEFAULT_DPI


def add_north_arrow(ax, x=0.94, y=0.85, size=0.08):
    """Add a journal-style black/white compass north arrow."""
    # Outer black needle (north-pointing)
    outer = np.array([
        [x, y + size],
        [x - size * 0.20, y - size * 0.55],
        [x, y - size * 0.30],
        [x + size * 0.20, y - size * 0.55],
    ])

    # Inner white face to mimic standard compass style
    inner = np.array([
        [x, y + size * 0.82],
        [x - size * 0.06, y - size * 0.40],
        [x, y - size * 0.22],
    ])

    ax.add_patch(Polygon(outer, closed=True, transform=ax.transAxes,
                         facecolor='black', edgecolor='black', linewidth=0.8, zorder=10))
    ax.add_patch(Polygon(inner, closed=True, transform=ax.transAxes,
                         facecolor='white', edgecolor='none', zorder=11))
    ax.text(x, y + size * 1.18, 'N', transform=ax.transAxes,
            ha='center', va='bottom', fontsize=16, fontweight='bold', color='black', zorder=12)


def add_scalebar(ax, lon_min, lon_max, lat_min, lat_max, length_km=20, position='left'):
    """Add a simple geographic scale bar using local latitude"""
    lat_ref = (lat_min + lat_max) / 2.0
    km_per_deg_lon = 111.32 * math.cos(math.radians(lat_ref))
    if km_per_deg_lon <= 0:
        return

    length_deg = length_km / km_per_deg_lon
    if position == 'right':
        x0 = lon_max - (lon_max - lon_min) * 0.06 - length_deg
    elif position == 'center':
        x0 = lon_min + (lon_max - lon_min - length_deg) * 0.5
    else:
        x0 = lon_min + (lon_max - lon_min) * 0.06
    y0 = lat_min + (lat_max - lat_min) * 0.035
    x1 = x0 + length_deg

    ax.plot([x0, x1], [y0, y0], color='black', linewidth=3)
    ax.plot([x0, x0], [y0 - 0.001, y0 + 0.001], color='black', linewidth=2)
    ax.plot([x1, x1], [y0 - 0.001, y0 + 0.001], color='black', linewidth=2)
    ax.text((x0 + x1) / 2, y0 + (lat_max - lat_min) * 0.012, f'{length_km} km',
            ha='center', va='bottom', fontsize=12, fontweight='bold',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1.2))


def apply_geographic_axes_style(ax, lon_min, lon_max, lat_min, lat_max, scalebar_km=20, scalebar_position='left'):
    """Apply map style with approximate true geographic scale"""
    lat_ref = (lat_min + lat_max) / 2.0
    ax.set_aspect(1.0 / max(math.cos(math.radians(lat_ref)), 1e-6))
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    add_scalebar(ax, lon_min, lon_max, lat_min, lat_max, length_km=scalebar_km, position=scalebar_position)
    add_north_arrow(ax)

def _print_cached_data_summary(data: dict):
    """Print summary of loaded cached data."""
    print(f"✓ Loaded cached data")
    print(f"  - Grids with features: {len(data['change_features'])}")
    print(f"  - Total grids in mapping: {len(data['grid_id_to_idx'])}")
    print(f"  - Graph 2021 edges: {data['graphs_2021'][0][0].shape[1]}")
    print(f"  - Graph 2024 edges: {data['graphs_2024'][0][0].shape[1]}")
    if 'train_flow_grid_ids' in data:
        print(f"  - Training-flow grids: {len(data['train_flow_grid_ids'])}")
    if data.get('train_flow_label_hash'):
        print(f"  - Training-flow label hash: {data['train_flow_label_hash']}")
    if data.get('train_flow_total_samples') is not None:
        print(f"  - Training-flow total samples: {data['train_flow_total_samples']}")
    if data.get('train_flow_cache_file'):
        print(f"  - Training-flow cache source: {data['train_flow_cache_file']}")
    if data.get('edge_feature_mode'):
        print(f"  - Edge feature mode: {data['edge_feature_mode']}")


def load_split_cached_data(feature_path: str, graph_path: str) -> dict:
    """Load and merge the two-level split cache (features + graph variant)."""
    print(f"Loading split cache:")
    print(f"  features : {feature_path}")
    print(f"  graphs   : {graph_path}")
    with open(feature_path, 'rb') as f:
        feat = pickle.load(f)
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)

    # Respect explicit None (full-grid cache with no train mask) vs absent key
    _sentinel = object()
    _tfgi = feat.get('train_flow_grid_ids', _sentinel)
    if _tfgi is _sentinel:
        # key absent — fall back to flows_2021 keys (old training caches)
        train_flow_grid_ids = sorted(feat.get('flows_2021', {}).keys())
    else:
        # key present (may be None for full-grid caches without a train mask)
        train_flow_grid_ids = _tfgi
    label_hash = feat.get('label_file_hash') or feat.get('train_flow_label_hash')

    data = {
        # from feature cache
        'change_features':          feat['change_features'],
        'flows_2021':               feat.get('flows_2021', {}),
        'flows_2024':               feat.get('flows_2024', {}),
        'train_flow_grid_ids':      train_flow_grid_ids,
        'train_flow_total_samples': len(train_flow_grid_ids) if train_flow_grid_ids is not None else None,
        'train_flow_label_hash':    label_hash,
        'label_file_hash':          label_hash,
        # from graph variant cache
        'graphs_2021':              graph['graphs_2021'],
        'graphs_2024':              graph['graphs_2024'],
        'grid_id_to_idx':           graph['grid_id_to_idx'],
        'edge_index':               graph.get('edge_index'),
        'edge_weights':             graph.get('edge_weights'),
        'edge_feature_mode':        graph.get('edge_feature_mode', 'flow_only'),
        # provenance
        'train_flow_cache_file':    f"split:{os.path.basename(feature_path)}+{os.path.basename(graph_path)}",
        'is_full_grid_cache':       feat.get('is_full_grid_cache', False),
    }
    _print_cached_data_summary(data)
    return data


def load_cached_data(cache_path_or_pair):
    """Load cached data — accepts either a monolithic path (str) or a split-cache pair (tuple)."""
    if isinstance(cache_path_or_pair, tuple):
        feature_path, graph_path = cache_path_or_pair
        return load_split_cached_data(feature_path, graph_path)

    cache_path = cache_path_or_pair
    print(f"Loading cached data from {cache_path}...")
    with open(cache_path, 'rb') as f:
        data = pickle.load(f)
    _print_cached_data_summary(data)
    return data


def resolve_train_flow_grids(cached_data: dict):
    """Resolve which nodes should receive temporal/raw-flow inputs at inference time."""
    explicit_train_grids = cached_data.get('train_flow_grid_ids')
    if explicit_train_grids is not None:
        return set(explicit_train_grids), 'train_flow_grid_ids'

    flows_2021 = cached_data.get('flows_2021', {})
    flows_2024 = cached_data.get('flows_2024', {})
    fallback_grids = set(flows_2021.keys()) | set(flows_2024.keys())
    print(
        "WARNING: cache is missing explicit train_flow_grid_ids; "
        "falling back to flows_* keys, which may not match training semantics."
    )
    return fallback_grids, 'flows_keys_fallback'


def use_strict_train_flow_mask() -> bool:
    """Whether to force inference inputs to match the training-flow mask exactly."""
    value = os.environ.get('STRICT_TRAIN_FLOW_MASK', '').strip().lower()
    return value in {'1', 'true', 'yes', 'y'}


def use_train_flow_mask_for_spatial_raw(manifest: dict = None) -> bool:
    """Whether raw spatial node features should be masked to the training-flow support."""
    value = os.environ.get('SPATIAL_RAW_MASK_MODE', '').strip().lower()
    if value in {'all', 'full', 'none', '0', 'false', 'no'}:
        return False
    if value in {'train', 'mask', 'strict', '1', 'true', 'yes'}:
        return True

    node_mode = infer_node_mode_from_manifest(manifest) if manifest else None
    return node_mode in {'raw_temporal_mean', 'raw_temporal_graph_stats'}


def load_experiment_manifest(experiment_dir: Path):
    """Load training metadata from test_results for consistency checks."""
    manifest_path = experiment_dir / 'metrics' / 'test_results.json'
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest file: {manifest_path}")

    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    return manifest


def infer_spatial_model_from_manifest(manifest: dict) -> str:
    spatial_type = (
        manifest.get('model_architecture', {})
        .get('spatial_branch', {})
        .get('type', '')
        .upper()
    )
    if 'EVOLVEGCN' in spatial_type:
        return 'EVOLVEGCN'
    if 'GINE' in spatial_type:
        return 'GINE'
    if 'WGCN' in spatial_type:
        return 'WGCN'
    if 'SAGE' in spatial_type:
        return 'SAGE'
    if 'GAT' in spatial_type:
        return 'GAT'
    return 'GCN'


def infer_node_mode_from_manifest(manifest: dict):
    return (
        manifest.get('model_architecture', {})
        .get('spatial_branch', {})
        .get('node_feature_mode')
    )


def infer_edge_feature_mode_from_manifest(manifest: dict) -> str:
    return (
        manifest.get('model_architecture', {})
        .get('spatial_branch', {})
        .get('edge_feature_mode')
        or manifest.get('data_config', {}).get('graph_edge_feature_mode')
        or 'flow_only'
    )


def cache_matches_train_flow_manifest(cached_data: dict, manifest: dict) -> bool:
    expected = manifest.get('data_info', {})
    expected_total = expected.get('total_samples')
    expected_hash = expected.get('label_file_hash')

    train_flow_grid_ids = cached_data.get('train_flow_grid_ids')
    if train_flow_grid_ids is None:
        return False

    if expected_total is not None and len(train_flow_grid_ids) != int(expected_total):
        return False

    cache_hash = cached_data.get('train_flow_label_hash') or cached_data.get('label_file_hash')
    if expected_hash and cache_hash and cache_hash != expected_hash:
        return False

    cache_total = cached_data.get('train_flow_total_samples')
    if cache_total is not None and expected_total is not None and int(cache_total) != int(expected_total):
        return False

    return True


def validate_inference_consistency(cached_data: dict, manifest: dict, cache_path: str):
    """
    Validate whether inference cache is consistent with training setup.

    We fail fast on critical mismatches to prevent misleading visualization outputs.
    """
    expected = manifest.get('data_info', {})
    expected_e2021 = expected.get('graph_2021_edges')
    expected_e2024 = expected.get('graph_2024_edges')

    actual_e2021 = int(cached_data['graphs_2021'][0][0].shape[1])
    actual_e2024 = int(cached_data['graphs_2024'][0][0].shape[1])

    mismatches = []
    if expected_e2021 is not None and int(expected_e2021) != actual_e2021:
        mismatches.append(f"graph_2021_edges expected {expected_e2021} but got {actual_e2021}")
    if expected_e2024 is not None and int(expected_e2024) != actual_e2024:
        mismatches.append(f"graph_2024_edges expected {expected_e2024} but got {actual_e2024}")

    trained_spatial_model = infer_spatial_model_from_manifest(manifest)
    runtime_spatial_model = getattr(config, 'SPATIAL_MODEL', 'GCN')
    if trained_spatial_model != runtime_spatial_model:
        mismatches.append(
            f"SPATIAL_MODEL expected {trained_spatial_model} but current config is {runtime_spatial_model}"
        )

    trained_node_mode = infer_node_mode_from_manifest(manifest)
    runtime_node_mode = getattr(config, 'SPATIAL_NODE_FEATURE_MODE', 'ones')
    if trained_node_mode and trained_node_mode != runtime_node_mode:
        mismatches.append(
            f"SPATIAL_NODE_FEATURE_MODE expected {trained_node_mode} but current config is {runtime_node_mode}"
        )

    trained_edge_mode = infer_edge_feature_mode_from_manifest(manifest)
    runtime_edge_mode = getattr(config, 'GINE_EDGE_FEATURE_MODE', 'flow_only')
    if trained_spatial_model == 'GINE' and trained_edge_mode != runtime_edge_mode:
        mismatches.append(
            f"GINE_EDGE_FEATURE_MODE expected {trained_edge_mode} but current config is {runtime_edge_mode}"
        )

    # Cross-check edge_feature_mode stored in cache against what the manifest expects.
    # This catches the exact bug where a wrong cache (e.g. flow_only) is used for a
    # model trained with flow_distance_direction, silently corrupting spatial branch output.
    cache_edge_mode = cached_data.get('edge_feature_mode')
    if cache_edge_mode and cache_edge_mode != trained_edge_mode:
        mismatches.append(
            f"cache edge_feature_mode={cache_edge_mode!r} but manifest expects {trained_edge_mode!r}"
        )

    is_full_grid_cache = cached_data.get('is_full_grid_cache', False)
    if trained_node_mode in {'raw_temporal_mean', 'raw_temporal_graph_stats'} and not is_full_grid_cache:
        if 'train_flow_grid_ids' not in cached_data:
            mismatches.append(
                "cache missing explicit train_flow_grid_ids required to reproduce training-time raw-flow masking"
            )
        elif not cache_matches_train_flow_manifest(cached_data, manifest):
            expected_total = expected.get('total_samples')
            expected_hash = expected.get('label_file_hash')
            actual_total = len(cached_data.get('train_flow_grid_ids', []))
            actual_hash = cached_data.get('train_flow_label_hash') or cached_data.get('label_file_hash')
            mismatches.append(
                "training-flow mask provenance mismatch "
                f"(expected label_hash={expected_hash}, total_samples={expected_total}; "
                f"got label_hash={actual_hash}, train_flow_grids={actual_total})"
            )
    if is_full_grid_cache:
        print("  (full-grid inference cache: train-flow provenance check skipped)")

    if mismatches:
        detail = '\n  - '.join(mismatches)
        raise RuntimeError(
            "Inference/visualization consistency check failed.\n"
            f"Cache file: {cache_path}\n"
            f"Experiment: {EXPERIMENT_DIR}\n"
            f"Mismatches:\n  - {detail}\n"
            "Please use a cache generated with the same graph/model settings as training."
        )


def resolve_cache_paths_from_manifest(default_cache_path: str, manifest: dict):
    """
    Pick a cache whose graph edge counts (and edge_feature_mode) match the manifest.

    Returns either:
      - str                  — monolithic cache path (old format)
      - (str, str)           — (feature_cache_path, graph_variant_cache_path) split format

    Priority:
      1. VIS_CACHE_PATH env var → return as monolithic (unchanged)
      2. Old-style monolithic caches in data/cache/dual_year_data_*.pkl
      3. NEW: split caches in data/cache/graphs/ + data/cache/features/
    """
    if os.environ.get('VIS_CACHE_PATH'):
        return default_cache_path

    expected = manifest.get('data_info', {})
    expected_e2021 = expected.get('graph_2021_edges')
    expected_e2024 = expected.get('graph_2024_edges')
    trained_node_mode = infer_node_mode_from_manifest(manifest)
    trained_edge_mode = infer_edge_feature_mode_from_manifest(manifest)
    require_train_flow_match = trained_node_mode in {'raw_temporal_mean', 'raw_temporal_graph_stats'}

    if expected_e2021 is None or expected_e2024 is None:
        return default_cache_path

    expected_e2021 = int(expected_e2021)
    expected_e2024 = int(expected_e2024)

    # --- 1. Try old-style monolithic caches (backward compat) ---
    candidate_paths = []
    if os.path.exists(default_cache_path):
        candidate_paths.append(default_cache_path)
    candidate_paths.extend(sorted(glob.glob('data/cache/dual_year_data_*.pkl')))

    best_mono_path = None
    best_mono_score = None

    for path in candidate_paths:
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            actual_e2021 = int(data['graphs_2021'][0][0].shape[1])
            actual_e2024 = int(data['graphs_2024'][0][0].shape[1])
            if actual_e2021 != expected_e2021 or actual_e2024 != expected_e2024:
                continue

            train_flow_match = cache_matches_train_flow_manifest(data, manifest)
            if require_train_flow_match and not train_flow_match:
                continue

            feature_count = len(data.get('change_features', {}))
            has_train_mask = 'train_flow_grid_ids' in data
            score = (
                1 if train_flow_match else 0,
                1 if has_train_mask else 0,
                feature_count,
            )
            if best_mono_score is None or score > best_mono_score:
                best_mono_score = score
                best_mono_path = path
        except Exception:
            continue

    # --- 2. Try new split caches ---
    best_split = None      # (feat_path, graph_path)
    best_split_score = None

    graph_variant_paths = sorted(glob.glob('data/cache/graphs/dual_year_graph_variant_*.pkl'))
    feature_paths = sorted(glob.glob('data/cache/features/dual_year_features_*.pkl'))

    for graph_path in graph_variant_paths:
        try:
            with open(graph_path, 'rb') as f:
                graph_data = pickle.load(f)
            actual_e2021 = int(graph_data['graphs_2021'][0][0].shape[1])
            actual_e2024 = int(graph_data['graphs_2024'][0][0].shape[1])
            if actual_e2021 != expected_e2021 or actual_e2024 != expected_e2024:
                continue
            cache_edge_mode = graph_data.get('edge_feature_mode', 'flow_only')
            if cache_edge_mode != trained_edge_mode:
                continue
        except Exception:
            continue

        # Find best matching feature cache
        # Priority: full-grid cache > training cache with provenance match > others
        for feat_path in feature_paths:
            try:
                with open(feat_path, 'rb') as f:
                    feat_data = pickle.load(f)
                feature_count = len(feat_data.get('change_features', {}))
                if feature_count == 0:
                    continue

                is_full_grid = feat_data.get('is_full_grid_cache', False)

                if not is_full_grid:
                    # Build a minimal proxy dict for cache_matches_train_flow_manifest
                    proxy = {
                        'train_flow_grid_ids': feat_data.get('train_flow_grid_ids',
                                                              sorted(feat_data.get('flows_2021', {}).keys())),
                        'train_flow_label_hash': feat_data.get('label_file_hash') or feat_data.get('train_flow_label_hash'),
                        'label_file_hash':       feat_data.get('label_file_hash') or feat_data.get('train_flow_label_hash'),
                        'train_flow_total_samples': len(feat_data.get('train_flow_grid_ids',
                                                                       sorted(feat_data.get('flows_2021', {}).keys()))),
                    }
                    train_flow_match = cache_matches_train_flow_manifest(proxy, manifest)
                    if require_train_flow_match and not train_flow_match:
                        continue
                else:
                    train_flow_match = False  # full-grid cache has no label provenance

                has_train_mask = 'train_flow_grid_ids' in feat_data
                # full-grid cache scores highest on feature_count dimension;
                # use is_full_grid as the top-level tiebreaker
                score = (
                    1 if is_full_grid else 0,       # prefer full-grid for visualization
                    1 if train_flow_match else 0,
                    1 if has_train_mask else 0,
                    feature_count,
                )
                if best_split_score is None or score > best_split_score:
                    best_split_score = score
                    best_split = (feat_path, graph_path)
            except Exception:
                continue

    # --- Choose best result ---
    # Prefer split cache when it has a better or equal score (it carries edge_feature_mode)
    if best_split is not None:
        feat_path, graph_path = best_split
        print(
            f"Auto-selected split cache: "
            f"{os.path.basename(feat_path)} + {os.path.basename(graph_path)} "
            f"(score={best_split_score}, edge_mode={trained_edge_mode}, "
            f"expected_edges=({expected_e2021},{expected_e2024}))"
        )
        return best_split

    if best_mono_path is not None:
        print(
            f"Auto-selected VIS cache by edge match: {best_mono_path} "
            f"(score={best_mono_score}, "
            f"expected_edges=({expected_e2021},{expected_e2024}))"
        )
        return best_mono_path

    return default_cache_path

def infer_gine_use_spatial_coords_from_checkpoint(model_path: str, manifest: dict) -> bool:
    """
    Infer GINE_USE_SPATIAL_COORDS from the first GINE layer weight shape in the checkpoint.

    The first GINE layer input dim = LAPLACIAN_PE_DIM [+ RAW_TEMPORAL_GRAPH_STATS_DIM] [+ 2 if spatial_coords].
    We compare the actual checkpoint dim against what we'd expect without spatial coords.
    """
    spatial_type = infer_spatial_model_from_manifest(manifest)
    if spatial_type != 'GINE':
        return False

    laplacian_pe_dim = (
        manifest.get('model_architecture', {})
        .get('spatial_branch', {})
        .get('laplacian_pe_dim')
        or getattr(config, 'LAPLACIAN_PE_DIM', 16)
    )
    node_mode = infer_node_mode_from_manifest(manifest)
    RAW_TEMPORAL_GRAPH_STATS_DIM = 10
    base_dim = int(laplacian_pe_dim)
    if node_mode == 'raw_temporal_graph_stats':
        base_dim += RAW_TEMPORAL_GRAPH_STATS_DIM

    try:
        ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        sd = ckpt.get('model_state_dict', ckpt)
        w = sd.get('spatial_branch.gine_layers.0.nn.0.weight')
        if w is None:
            return getattr(config, 'GINE_USE_SPATIAL_COORDS', False)
        actual_dim = w.shape[1]
        use_coords = (actual_dim == base_dim + 2)
        print(f"Inferred GINE_USE_SPATIAL_COORDS={use_coords} "
              f"(checkpoint first-layer input={actual_dim}, base_dim={base_dim})")
        return use_coords
    except Exception as e:
        print(f"[WARN] Could not infer GINE_USE_SPATIAL_COORDS from checkpoint: {e}")
        return getattr(config, 'GINE_USE_SPATIAL_COORDS', False)


def infer_temporal_model_from_manifest(manifest: dict) -> str:
    return (
        manifest.get('model_architecture', {})
        .get('temporal_branch', {})
        .get('temporal_model', 'LSTM')
        .upper()
    )


def load_model(model_path, device, manifest: dict = None):
    """Load the best trained model"""
    print(f"\nLoading model from {model_path}...")

    temporal_model = infer_temporal_model_from_manifest(manifest) if manifest else 'LSTM'
    print(f"Using TEMPORAL_MODEL from manifest: {temporal_model}")

    model = EnhancedDualBranchModel(
        temporal_input_size=config.TEMPORAL_INPUT_SIZE,
        hidden_size=config.FUSION_HIDDEN_SIZE,
        num_classes=config.NUM_CLASSES,
        num_time_steps=config.TIME_STEPS,
        dropout=0.4,
        spatial_model=config.SPATIAL_MODEL,
        temporal_model=temporal_model,
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    sd = checkpoint.get('model_state_dict', checkpoint)

    # Restore buffer keys that strict=False would silently drop.
    # _node_coords is saved in the checkpoint but not pre-registered on the
    # freshly-constructed model, so load_state_dict marks it as "unexpected"
    # and discards it — causing a 16-vs-18 dim mismatch at inference time.
    for buf_key in ('_node_coords', '_wamd_node_features_2021', '_wamd_node_features_2024'):
        if buf_key in sd:
            model.register_buffer(buf_key, sd[buf_key].to(device))
            print(f"  Restored buffer: {buf_key} {tuple(sd[buf_key].shape)}")

    missing, unexpected = model.load_state_dict(sd, strict=False)

    if missing:
        print(f"  WARNING: {len(missing)} missing keys in checkpoint")
    # Filter out the buffers we already restored so the count is accurate
    unexpected = [k for k in unexpected if k not in ('_node_coords', '_wamd_node_features_2021', '_wamd_node_features_2024')]
    if unexpected:
        print(f"  WARNING: {len(unexpected)} unexpected keys in checkpoint")

    model.eval()
    print(f"✓ Model loaded successfully")
    return model

def predict_grids_with_features(model, cached_data, device, batch_size=64):
    """Predict only grids that have features"""
    print(f"\nPredicting grids with features...")

    # Get grids with features
    grid_ids = list(cached_data['change_features'].keys())
    print(f"  Total grids to predict: {len(grid_ids)}")

    # Prepare data
    all_temporal_2021 = []
    all_temporal_2024 = []

    for grid_id in grid_ids:
        features = cached_data['change_features'][grid_id]  # (168, 4)
        temporal_2021 = features[:, :2]  # (168, 2)
        temporal_2024 = features[:, 2:]  # (168, 2)
        all_temporal_2021.append(temporal_2021)
        all_temporal_2024.append(temporal_2024)

    all_temporal_2021 = torch.FloatTensor(np.array(all_temporal_2021)).to(device)  # (N, 168, 2)
    all_temporal_2024 = torch.FloatTensor(np.array(all_temporal_2024)).to(device)  # (N, 168, 2)

    # Get node indices
    node_indices = torch.LongTensor([cached_data['grid_id_to_idx'][gid] for gid in grid_ids]).to(device)

    # Get graphs
    edge_index_2021, edge_attr_2021 = cached_data['graphs_2021'][0]
    edge_index_2024, edge_attr_2024 = cached_data['graphs_2024'][0]

    edge_index_2021 = torch.from_numpy(edge_index_2021).long().to(device)
    edge_attr_2021 = torch.from_numpy(edge_attr_2021).float().to(device)
    edge_index_2024 = torch.from_numpy(edge_index_2024).long().to(device)
    edge_attr_2024 = torch.from_numpy(edge_attr_2024).float().to(device)

    graphs_2021 = [(edge_index_2021, edge_attr_2021)]
    graphs_2024 = [(edge_index_2024, edge_attr_2024)]

    num_nodes = len(cached_data['grid_id_to_idx'])

    # Create full size temporal tensors for model inputs.
    # Default behavior for full-grid visualization is to use all available full-grid
    # temporal/raw-flow features. The training-flow mask is still tracked for provenance
    # and can be enforced explicitly via STRICT_TRAIN_FLOW_MASK=1 when needed.
    full_temporal_2021 = torch.zeros((num_nodes, 168, 2), dtype=torch.float32).to(device)
    full_temporal_2024 = torch.zeros((num_nodes, 168, 2), dtype=torch.float32).to(device)

    flows_2021 = cached_data.get('flows_2021', {})
    flows_2024 = cached_data.get('flows_2024', {})
    train_flow_grids, train_flow_source = resolve_train_flow_grids(cached_data)
    strict_train_flow_mask = use_strict_train_flow_mask()
    spatial_raw_train_mask = use_train_flow_mask_for_spatial_raw(load_experiment_manifest(Path(EXPERIMENT_DIR)))

    filled_temporal = 0

    for i, grid_id in enumerate(grid_ids):
        if not strict_train_flow_mask or grid_id in train_flow_grids:
            node_idx = cached_data['grid_id_to_idx'][grid_id]
            full_temporal_2021[node_idx] = all_temporal_2021[i]
            full_temporal_2024[node_idx] = all_temporal_2024[i]
            filled_temporal += 1

    print(
        f"  Grids filled into full_temporal ({'strict training-flow mask' if strict_train_flow_mask else 'all cached full-grid features'}): "
        f"{filled_temporal} / {len(grid_ids)}"
    )
    if strict_train_flow_mask:
        print(f"  - strict mask source: {train_flow_source}")
    else:
        print(f"  - training-flow provenance retained for validation only: {len(train_flow_grids)} grids from {train_flow_source}")

    # Raw flows for spatial raw_temporal_mean mode.
    # Keep this aligned with training semantics by default: during training the
    # raw spatial node features only existed on the sampled training-support grids.
    full_raw_temporal_2021 = torch.zeros((num_nodes, 168, 2), dtype=torch.float32).to(device)
    full_raw_temporal_2024 = torch.zeros((num_nodes, 168, 2), dtype=torch.float32).to(device)
    filled_raw_2021 = 0
    for grid_id, flow in flows_2021.items():
        if grid_id in cached_data['grid_id_to_idx'] and (not spatial_raw_train_mask or grid_id in train_flow_grids):
            idx = cached_data['grid_id_to_idx'][grid_id]
            full_raw_temporal_2021[idx] = torch.tensor(np.array(flow), dtype=torch.float32).to(device)
            filled_raw_2021 += 1
    filled_raw_2024 = 0
    for grid_id, flow in flows_2024.items():
        if grid_id in cached_data['grid_id_to_idx'] and (not spatial_raw_train_mask or grid_id in train_flow_grids):
            idx = cached_data['grid_id_to_idx'][grid_id]
            full_raw_temporal_2024[idx] = torch.tensor(np.array(flow), dtype=torch.float32).to(device)
            filled_raw_2024 += 1

    print(
        f"  Grids filled into full_raw_temporal ({'training-flow support' if spatial_raw_train_mask else 'all cached raw features'}): "
        f"{filled_raw_2021} / {len(flows_2021)} (2021), "
        f"{filled_raw_2024} / {len(flows_2024)} (2024)"
    )
    if spatial_raw_train_mask:
        print(f"  - spatial raw mask source: {train_flow_source}")

    # Predict in batches
    all_predictions = []

    model.eval()
    with torch.no_grad():
        n_samples = len(grid_ids)
        for i in tqdm(range(0, n_samples, batch_size), desc="Predicting"):
            batch_end = min(i + batch_size, n_samples)
            batch_node_indices = node_indices[i:batch_end]

            try:
                logits = model(
                    x_2021=full_temporal_2021,
                    x_2024=full_temporal_2024,
                    graphs_2021=graphs_2021,
                    graphs_2024=graphs_2024,
                    num_nodes=num_nodes,
                    node_indices=batch_node_indices,
                    raw_x_2021=full_raw_temporal_2021,
                    raw_x_2024=full_raw_temporal_2024,
                )

                predictions = torch.argmax(logits, dim=1).cpu().numpy() + 1
                all_predictions.extend(predictions)
            except Exception as e:
                print(f"\nError in batch {i}-{batch_end}: {e}")
                # Use fallback: assign most common class
                all_predictions.extend([5] * (batch_end - i))

    print(f"✓ Predicted {len(all_predictions)} grids")
    return np.array(all_predictions), grid_ids

# ---------------------------------------------------------------------------
# Node filtering thresholds — adjust these to tune which grids are shown
# ---------------------------------------------------------------------------
# Condition 1 (hard, always on): both years have zero total flow → exclude.
#   These nodes carry no information whatsoever.
#
# Condition 2 (soft, default OFF): either year has zero total flow → exclude.
#   WARNING: turning this ON will remove semantically valid classes:
#     - Class 6 (Growth Diffusion): 2021 flow=0, 2024 flow>0  (new activity)
#     - Class 8 (Decline Aggregation): 2021 flow>0, 2024 flow=0 (vanished activity)
#     - Class 1 (Stable Static): many labeled nodes have zero flow in both years
#   Set to True only if you want a conservative "both years active" subset.
FILTER_EITHER_ZERO = False            # condition-2 switch

# Condition 3 (soft): minimum total flow required for EACH year that has flow.
#   Applied only to years with non-zero flow (so Class 6/8 one-sided flow is
#   evaluated against the year that actually has flow).  Set to 0 to disable.
FILTER_MIN_FLOW_PER_YEAR = 0          # type-3 threshold (total flow > this)

# Condition 4 (soft): minimum non-zero hours required for EACH year that has
#   flow.  Set to 0 to disable.
FILTER_MIN_ACTIVE_HOURS_PER_YEAR = 1  # type-4 threshold

# Condition 5 (optional, currently disabled): exclude grids that are
#   unidirectional (only inflow OR only outflow in both years).
#   Set to True to enable.
FILTER_UNIDIRECTIONAL = False

# Condition 6 (optional, currently disabled): exclude grids whose
#   year-over-year total-flow ratio is more extreme than this factor.
#   e.g. 50 means exclude if flow_2024/flow_2021 > 50 or < 1/50.
#   Only meaningful when FILTER_EITHER_ZERO=False (both years have flow).
#   Set to None to disable.
FILTER_MAX_FLOW_RATIO = None          # e.g. 50.0 to enable


def filter_grids_by_flow(predictions, grid_ids, cached_data):
    """
    Filter out grids whose flow data makes their predicted label unreliable.

    Hard exclusion (always applied):
      1. Both years have zero total flow — no signal at all.

    Soft exclusions (controlled by module-level flags/thresholds above):
      2. Either year has zero total flow (FILTER_EITHER_ZERO).
         OFF by default: Class 6 (Growth Diffusion, 2021=0→2024>0) and
         Class 8 (Decline Aggregation, 2021>0→2024=0) are semantically valid
         one-sided-flow classes and must NOT be excluded here.
      3. Either active year's total flow <= FILTER_MIN_FLOW_PER_YEAR.
      4. Either active year has fewer than FILTER_MIN_ACTIVE_HOURS_PER_YEAR
         non-zero hours.

    Optional exclusions (disabled by default):
      5. Unidirectional nodes (FILTER_UNIDIRECTIONAL).
      6. Extreme year-over-year flow ratio (FILTER_MAX_FLOW_RATIO).

    Parameters
    ----------
    predictions : np.ndarray  shape (N,)
    grid_ids    : list[int]   length N
    cached_data : dict        must contain 'change_features'

    Returns
    -------
    filtered_predictions : np.ndarray
    filtered_grid_ids    : list[int]
    filter_stats         : dict  — counts for each exclusion reason
    """
    change_features = cached_data['change_features']

    keep_mask = np.ones(len(grid_ids), dtype=bool)
    stats = {
        'total_input': len(grid_ids),
        'excluded_both_zero': 0,
        'excluded_either_zero': 0,
        'excluded_low_flow': 0,
        'excluded_low_active_hours': 0,
        'excluded_unidirectional': 0,
        'excluded_extreme_ratio': 0,
    }

    for i, grid_id in enumerate(grid_ids):
        if not keep_mask[i]:
            continue

        feat = change_features[grid_id]          # (168, 4)
        # feat columns: [inflow_2021_log, outflow_2021_log, inflow_2024_log, outflow_2024_log]
        # Recover approximate raw flow from log features: exp(x) - 1
        flow_2021 = np.expm1(feat[:, :2])        # (168, 2)  inflow+outflow 2021
        flow_2024 = np.expm1(feat[:, 2:])        # (168, 2)  inflow+outflow 2024

        total_2021 = flow_2021.sum()
        total_2024 = flow_2024.sum()

        # --- Condition 1 (hard): both years zero → no signal at all ---
        if total_2021 == 0 and total_2024 == 0:
            keep_mask[i] = False
            stats['excluded_both_zero'] += 1
            continue

        # --- Condition 2 (soft, default OFF): either year zero ---
        # Disabled by default because Class 6 (Growth Diffusion) has 2021=0
        # and Class 8 (Decline Aggregation) has 2024=0 by definition.
        if FILTER_EITHER_ZERO and (total_2021 == 0 or total_2024 == 0):
            keep_mask[i] = False
            stats['excluded_either_zero'] += 1
            continue

        # For conditions 3 & 4, evaluate only the year(s) that have flow.
        # If a year is zero, skip its check (it's a valid one-sided pattern).
        active_totals = [t for t in (total_2021, total_2024) if t > 0]
        active_flows  = [f for f, t in zip((flow_2021, flow_2024),
                                            (total_2021, total_2024)) if t > 0]

        # --- Condition 3 (soft): minimum total flow for active years ---
        if FILTER_MIN_FLOW_PER_YEAR > 0:
            if any(t <= FILTER_MIN_FLOW_PER_YEAR for t in active_totals):
                keep_mask[i] = False
                stats['excluded_low_flow'] += 1
                continue

        # --- Condition 4 (soft): minimum active hours for active years ---
        if FILTER_MIN_ACTIVE_HOURS_PER_YEAR > 0:
            for f in active_flows:
                if (f.sum(axis=1) > 0).sum() < FILTER_MIN_ACTIVE_HOURS_PER_YEAR:
                    keep_mask[i] = False
                    stats['excluded_low_active_hours'] += 1
                    break
            if not keep_mask[i]:
                continue

        # --- Condition 5 (optional): unidirectional nodes ---
        if FILTER_UNIDIRECTIONAL:
            inflow_2021  = flow_2021[:, 0].sum()
            outflow_2021 = flow_2021[:, 1].sum()
            inflow_2024  = flow_2024[:, 0].sum()
            outflow_2024 = flow_2024[:, 1].sum()
            uni_2021 = (inflow_2021 == 0) or (outflow_2021 == 0)
            uni_2024 = (inflow_2024 == 0) or (outflow_2024 == 0)
            if uni_2021 and uni_2024:
                keep_mask[i] = False
                stats['excluded_unidirectional'] += 1
                continue

        # --- Condition 6 (optional): extreme year-over-year ratio ---
        # Only meaningful when both years have flow.
        if FILTER_MAX_FLOW_RATIO is not None and total_2021 > 0 and total_2024 > 0:
            ratio = total_2024 / total_2021
            if ratio > FILTER_MAX_FLOW_RATIO or ratio < 1.0 / FILTER_MAX_FLOW_RATIO:
                keep_mask[i] = False
                stats['excluded_extreme_ratio'] += 1
                continue

    filtered_predictions = predictions[keep_mask]
    filtered_grid_ids = [gid for gid, k in zip(grid_ids, keep_mask) if k]
    stats['kept'] = int(keep_mask.sum())
    stats['excluded_total'] = stats['total_input'] - stats['kept']

    print(f"\n[Node Filter] Results:")
    print(f"  Input grids              : {stats['total_input']}")
    print(f"  Excluded (both zero)     : {stats['excluded_both_zero']}")
    print(f"  Excluded (one zero, cond2={FILTER_EITHER_ZERO}): {stats['excluded_either_zero']}")
    print(f"  Excluded (low flow ≤{FILTER_MIN_FLOW_PER_YEAR})    : {stats['excluded_low_flow']}")
    print(f"  Excluded (active hrs <{FILTER_MIN_ACTIVE_HOURS_PER_YEAR})  : {stats['excluded_low_active_hours']}")
    if FILTER_UNIDIRECTIONAL:
        print(f"  Excluded (unidirectional): {stats['excluded_unidirectional']}")
    if FILTER_MAX_FLOW_RATIO is not None:
        print(f"  Excluded (ratio >{FILTER_MAX_FLOW_RATIO}x)  : {stats['excluded_extreme_ratio']}")
    print(f"  Kept grids               : {stats['kept']} ({stats['kept']/stats['total_input']*100:.1f}%)")

    return filtered_predictions, filtered_grid_ids, stats


def create_full_prediction_dataframe(predictions, grid_ids, grid_metadata_path):
    """Create dataframe with predictions and coordinates"""
    print(f"\nCreating prediction dataframe...")

    # Load grid metadata
    grid_meta = pd.read_csv(grid_metadata_path)

    # Create predictions dataframe
    pred_df = pd.DataFrame({
        'grid_id': grid_ids,
        'predicted_label': predictions,
    })

    # Merge with metadata
    result_df = pred_df.merge(
        grid_meta[['grid_id', 'lon', 'lat', 'city_name', 'area_name']],
        on='grid_id',
        how='left'
    )

    result_df['city_name_en'] = result_df['city_name'].map(CITY_NAME_EN).fillna(result_df['city_name'])
    result_df['area_name_en'] = result_df['area_name'].map(AREA_NAME_EN).fillna(result_df['area_name'])

    print(f"✓ Created dataframe with {len(result_df)} grids")
    print(f"  - Cities: {result_df['city_name'].nunique()}")
    print(f"  - Areas: {result_df['area_name'].nunique()}")

    return result_df

def plot_full_region_map(pred_df, output_dir):
    """Plot full region map with all predictions"""
    print(f"\nPlotting full region map...")

    fig, ax = plt.subplots(figsize=(20, 16))
    lon_min = pred_df['lon'].min() - 0.05
    lon_max = pred_df['lon'].max() + 0.05
    lat_min = pred_df['lat'].min() - 0.05
    lat_max = pred_df['lat'].max() + 0.05

    for class_id in range(1, 10):
        mask = pred_df['predicted_label'] == class_id
        if mask.sum() > 0:
            class_data = pred_df[mask]
            ax.scatter(
                class_data['lon'],
                class_data['lat'],
                c=CLASS_COLORS[class_id],
                s=15,
                alpha=0.7,
                label=CLASS_NAMES[class_id],
                edgecolors='none'
            )

    ax.set_xlabel('Longitude', fontsize=20, fontweight='bold')
    ax.set_ylabel('Latitude', fontsize=20, fontweight='bold')
    ax.set_title('Mobility Pattern Changes in Shenzhen-Dongguan-Huizhou Region\n(2021-2024)',
                 fontsize=24, fontweight='bold', pad=20)

    # Single merged legend (journal-friendly, avoids excessive right-side blocks)
    merged_handles = [
        mpatches.Patch(color=CLASS_COLORS[class_id], label=CLASS_NAMES[class_id])
        for class_id in range(1, 10)
    ]
    ax.legend(handles=merged_handles, title='Legend',
              loc='upper left', bbox_to_anchor=(1.02, 1.0), ncol=1,
              fontsize=15, title_fontsize=17, frameon=True, framealpha=0.95, shadow=True)

    ax.grid(True, alpha=0.3, linestyle='--')
    apply_geographic_axes_style(ax, lon_min, lon_max, lat_min, lat_max, scalebar_km=20)

    plt.tight_layout()

    output_path = f"{output_dir}/full_region_prediction_map.png"
    output_pdf = f"{output_dir}/full_region_prediction_map.pdf"
    plt.savefig(output_path, dpi=EXPORT_DPI, bbox_inches='tight')
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"✓ Saved to {output_path}")
    print(f"✓ Saved to {output_pdf}")
    plt.close()

def plot_pattern_group_maps(pred_df, output_dir):
    """Plot separate maps for each pattern group"""
    print(f"\nPlotting pattern group maps...")

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    lon_min = pred_df['lon'].min() - 0.05
    lon_max = pred_df['lon'].max() + 0.05
    lat_min = pred_df['lat'].min() - 0.05
    lat_max = pred_df['lat'].max() + 0.05

    groups = [
        ('Stable Patterns', [1, 2, 3]),
        ('Growth Patterns', [4, 5, 6]),
        ('Decline Patterns', [7, 8, 9]),
    ]

    for idx, (title, classes) in enumerate(groups):
        ax = axes[idx]

        for class_id in classes:
            mask = pred_df['predicted_label'] == class_id
            if mask.sum() > 0:
                class_data = pred_df[mask]
                ax.scatter(
                    class_data['lon'],
                    class_data['lat'],
                    c=CLASS_COLORS[class_id],
                    s=20,
                    alpha=0.7,
                    label=CLASS_NAMES[class_id],
                    edgecolors='white',
                    linewidths=0.3
                )

        ax.set_xlabel('Longitude', fontsize=14, fontweight='bold')
        ax.set_ylabel('Latitude', fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=17, fontweight='bold')
        ax.legend(loc='upper left', fontsize=11, frameon=True)
        ax.grid(True, alpha=0.3, linestyle='--')

        apply_geographic_axes_style(ax, lon_min, lon_max, lat_min, lat_max, scalebar_km=20, scalebar_position='center')

    plt.suptitle('Mobility Pattern Changes by Group (2021-2024)',
                 fontsize=22, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = f"{output_dir}/pattern_group_maps.png"
    output_pdf = f"{output_dir}/pattern_group_maps.pdf"
    plt.savefig(output_path, dpi=EXPORT_DPI, bbox_inches='tight')
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"✓ Saved to {output_path}")
    print(f"✓ Saved to {output_pdf}")
    plt.close()

def plot_city_comparison(pred_df, output_dir):
    """Plot city-wise comparison as three separate city maps"""
    print(f"\nPlotting city comparison...")

    city_slug = {
        'Shenzhen': 'shenzhen',
        'Dongguan': 'dongguan',
        'Huizhou': 'huizhou',
    }

    for city in sorted(pred_df['city_name_en'].dropna().unique()):
        city_data = pred_df[pred_df['city_name_en'] == city]
        if city_data.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 10))
        lon_min = city_data['lon'].min() - 0.03
        lon_max = city_data['lon'].max() + 0.03
        lat_min = city_data['lat'].min() - 0.03
        lat_max = city_data['lat'].max() + 0.03

        for class_id in range(1, 10):
            mask = city_data['predicted_label'] == class_id
            if mask.sum() > 0:
                class_data = city_data[mask]
                ax.scatter(
                    class_data['lon'],
                    class_data['lat'],
                    c=CLASS_COLORS[class_id],
                    s=26,
                    alpha=0.75,
                    label=CLASS_NAMES[class_id],
                    edgecolors='white',
                    linewidths=0.35
                )

        ax.set_xlabel('Longitude', fontsize=16, fontweight='bold')
        ax.set_ylabel('Latitude', fontsize=16, fontweight='bold')
        ax.set_title(f'{city}: Mobility Pattern Changes (2021-2024)\n({len(city_data)} grids)',
                     fontsize=20, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        apply_geographic_axes_style(
            ax,
            lon_min,
            lon_max,
            lat_min,
            lat_max,
            scalebar_km=10,
            scalebar_position='center'
        )

        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=10, frameon=True)
        plt.tight_layout()

        slug = city_slug.get(city, city.lower().replace(' ', '_'))
        output_path = f"{output_dir}/city_map_{slug}.png"
        output_pdf = f"{output_dir}/city_map_{slug}.pdf"
        plt.savefig(output_path, dpi=EXPORT_DPI, bbox_inches='tight')
        plt.savefig(output_pdf, bbox_inches='tight')
        print(f"✓ Saved to {output_path}")
        print(f"✓ Saved to {output_pdf}")
        plt.close()

def plot_class_distribution(pred_df, output_dir):
    """Plot class distribution statistics"""
    print(f"\nPlotting class distribution...")

    # Overall distribution data
    class_counts = pred_df['predicted_label'].value_counts().sort_index()
    colors = [CLASS_COLORS[i] for i in class_counts.index]

    all_handles = [
        mpatches.Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i])
        for i in range(1, 10)
    ]

    # City-wise distribution
    city_class_counts = pred_df.groupby(['city_name_en', 'predicted_label']).size().unstack(fill_value=0)
    city_class_pct = city_class_counts.div(city_class_counts.sum(axis=1), axis=0) * 100

    # Area-wise (District) distribution
    area_class_counts = pred_df.groupby(['area_name_en', 'predicted_label']).size().unstack(fill_value=0)
    area_class_pct = area_class_counts.div(area_class_counts.sum(axis=1), axis=0) * 100

    for class_id in range(1, 10):
        if class_id not in city_class_pct.columns:
            city_class_pct[class_id] = 0
        if class_id not in area_class_pct.columns:
            area_class_pct[class_id] = 0

    city_class_pct = city_class_pct[[i for i in range(1, 10)]]
    area_class_pct = area_class_pct[[i for i in range(1, 10)]]

    # ---------------- Figure A: Overall only ----------------
    fig_a, ax_a = plt.subplots(figsize=(15, 8))
    ax_a.bar(range(len(class_counts)), class_counts.values,
             color=colors, edgecolor='white', linewidth=0.9)
    ax_a.set_xlabel('Pattern Class', fontsize=16, fontweight='bold')
    ax_a.set_ylabel('Number of Grids', fontsize=16, fontweight='bold')
    ax_a.set_title('Overall Pattern Distribution', fontsize=19, fontweight='bold')
    ax_a.set_xticks(range(len(class_counts)))
    ax_a.set_xticklabels([CLASS_NAMES[i] for i in class_counts.index], rotation=20, ha='right', fontsize=12)
    ax_a.tick_params(axis='y', labelsize=12)
    ax_a.grid(True, alpha=0.30, axis='y')
    fig_a.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.23)

    output_a_png = f"{output_dir}/class_distribution_overall.png"
    output_a_pdf = f"{output_dir}/class_distribution_overall.pdf"
    fig_a.savefig(output_a_png, dpi=EXPORT_DPI, bbox_inches='tight')
    fig_a.savefig(output_a_pdf, bbox_inches='tight')
    print(f"✓ Saved to {output_a_png}")
    print(f"✓ Saved to {output_a_pdf}")
    plt.close(fig_a)

    # ---------------- Figure B: City + District ----------------
    fig_b, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(18, 7.0))

    # Bottom-left: city-wise distribution
    x_city = np.arange(len(city_class_pct))
    bottom_city = np.zeros(len(city_class_pct))
    for class_id in range(1, 10):
        values = city_class_pct[class_id].values
        ax_left.bar(x_city, values, bottom=bottom_city, color=CLASS_COLORS[class_id],
                    edgecolor='white', linewidth=0.4)
        bottom_city += values

    ax_left.set_xlabel('City', fontsize=15, fontweight='bold')
    ax_left.set_ylabel('Percentage (%)', fontsize=15, fontweight='bold')
    ax_left.set_title('City-wise Pattern Distribution', fontsize=17, fontweight='bold')
    ax_left.set_xticks(x_city)
    ax_left.set_xticklabels(city_class_pct.index, fontsize=12)
    ax_left.tick_params(axis='y', labelsize=12)
    ax_left.grid(True, alpha=0.28, axis='y')
    ax_left.set_ylim(0, 100)

    # Bottom-right: district-wise distribution
    x_area = np.arange(len(area_class_pct))
    bottom_area = np.zeros(len(area_class_pct))
    for class_id in range(1, 10):
        values = area_class_pct[class_id].values
        ax_right.bar(x_area, values, bottom=bottom_area, color=CLASS_COLORS[class_id],
                     edgecolor='white', linewidth=0.35)
        bottom_area += values

    ax_right.set_xlabel('District / Area', fontsize=15, fontweight='bold')
    ax_right.set_ylabel('Percentage (%)', fontsize=15, fontweight='bold')
    ax_right.set_title('District-wise Pattern Distribution', fontsize=17, fontweight='bold')
    ax_right.set_xticks(x_area)
    ax_right.set_xticklabels(area_class_pct.index, rotation=35, ha='right', fontsize=11)
    ax_right.tick_params(axis='y', labelsize=12)
    ax_right.grid(True, alpha=0.28, axis='y')
    ax_right.set_ylim(0, 100)

    # Shared legend on the right side
    fig_b.legend(
        handles=all_handles,
        loc='center left',
        bbox_to_anchor=(0.83, 0.5),
        ncol=1,
        fontsize=10,
        frameon=True,
        title='Legend',
        title_fontsize=11
    )
    fig_b.subplots_adjust(left=0.07, right=0.80, top=0.89, bottom=0.19, wspace=0.26)
    output_b_png = f"{output_dir}/class_distribution_city_district.png"
    output_b_pdf = f"{output_dir}/class_distribution_city_district.pdf"
    fig_b.savefig(output_b_png, dpi=EXPORT_DPI, bbox_inches='tight')
    fig_b.savefig(output_b_pdf, bbox_inches='tight')
    print(f"✓ Saved to {output_b_png}")
    print(f"✓ Saved to {output_b_pdf}")
    plt.close(fig_b)

    # Keep backward-compatible combined file name pointing to bottom figure layout
    compat_png = f"{output_dir}/class_distribution.png"
    compat_pdf = f"{output_dir}/class_distribution.pdf"
    import shutil
    shutil.copyfile(output_b_png, compat_png)
    shutil.copyfile(output_b_pdf, compat_pdf)
    print(f"✓ Updated compatibility file: {compat_png}")
    print(f"✓ Updated compatibility file: {compat_pdf}")

    # Save statistics
    stats_path = f"{output_dir}/prediction_statistics.csv"
    city_class_pct.to_csv(stats_path)
    print(f"✓ Saved statistics to {stats_path}")

def save_predictions(pred_df, output_dir):
    """Save predictions to CSV"""
    print(f"\nSaving predictions...")

    output_path = f"{output_dir}/all_grids_predictions.csv"
    pred_df.to_csv(output_path, index=False)
    print(f"✓ Saved predictions to {output_path}")

    summary = {
        'total_grids': int(len(pred_df)),
        'cities': int(pred_df['city_name'].nunique()),
        'areas': int(pred_df['area_name'].nunique()),
        'class_distribution': {int(k): int(v) for k, v in pred_df['predicted_label'].value_counts().sort_index().to_dict().items()},
    }

    summary_path = f"{output_dir}/prediction_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary to {summary_path}")


def main():
    """Main function"""
    # For full-grid inference, always use all cached raw features (not just training-flow grids).
    # This prevents the spatial branch from receiving all-zero node features for non-training grids,
    # which would cause prediction collapse to a single class.
    if not os.environ.get('SPATIAL_RAW_MASK_MODE'):
        os.environ['SPATIAL_RAW_MASK_MODE'] = 'all'

    # Configuration
    cache_path = os.environ.get(
        'VIS_CACHE_PATH',
        "/root/workspace/Graph_Deep_Learning/20251001-PRD_18-21-24-mobility_change_pattern/analysis/PRD-Mobility-Change-Pattern/data/cache/dual_year_data_all_grids.pkl"
    )
    model_path = str(EXPERIMENT_DIR / 'models' / 'best_model.pth')
    grid_metadata_path = "data/grid_metadata/sgh_grid_metadata.csv"

    # Create output directory
    output_dir = str(EXPERIMENT_DIR / 'model_predictions')
    os.makedirs(output_dir, exist_ok=True)

    print("="*80)
    print("Full Region Prediction with Best Model (Using Cached Data)")
    print("="*80)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load manifest from experiment outputs and align runtime config.
    manifest = load_experiment_manifest(EXPERIMENT_DIR)
    cache_result = resolve_cache_paths_from_manifest(cache_path, manifest)
    config.SPATIAL_MODEL = infer_spatial_model_from_manifest(manifest)
    manifest_node_mode = (
        manifest.get('model_architecture', {})
        .get('spatial_branch', {})
        .get('node_feature_mode')
    )
    manifest_edge_mode = infer_edge_feature_mode_from_manifest(manifest)
    if manifest_node_mode:
        config.SPATIAL_NODE_FEATURE_MODE = manifest_node_mode
    config.GINE_EDGE_FEATURE_MODE = manifest_edge_mode
    # Infer GINE_USE_SPATIAL_COORDS from checkpoint weight shape (not recorded in manifest)
    config.GINE_USE_SPATIAL_COORDS = infer_gine_use_spatial_coords_from_checkpoint(model_path, manifest)
    print(f"Using SPATIAL_MODEL from manifest: {config.SPATIAL_MODEL}")
    print(f"Using SPATIAL_NODE_FEATURE_MODE from manifest: {getattr(config, 'SPATIAL_NODE_FEATURE_MODE', 'ones')}")
    print(f"Using GINE_EDGE_FEATURE_MODE from manifest: {getattr(config, 'GINE_EDGE_FEATURE_MODE', 'flow_only')}")
    print(f"Using GINE_USE_SPATIAL_COORDS (inferred): {getattr(config, 'GINE_USE_SPATIAL_COORDS', False)}")

    # Load cached data
    cached_data = load_cached_data(cache_result)

    # Fail fast if cache/model/training settings do not match.
    cache_path_str = (
        f"{cache_result[0]}+{cache_result[1]}" if isinstance(cache_result, tuple) else cache_result
    )
    validate_inference_consistency(cached_data, manifest, cache_path_str)
    print("✓ Inference consistency check passed")

    # Load model
    model = load_model(model_path, device, manifest=manifest)

    # Predict
    predictions, grid_ids = predict_grids_with_features(model, cached_data, device, batch_size=64)

    # Filter out grids with unreliable flow data
    predictions, grid_ids, filter_stats = filter_grids_by_flow(predictions, grid_ids, cached_data)

    # Create dataframe
    pred_df = create_full_prediction_dataframe(predictions, grid_ids, grid_metadata_path)

    # Generate visualizations
    print("\n" + "="*80)
    print("Generating Visualizations (All English)")
    print("="*80)

    plot_full_region_map(pred_df, output_dir)
    plot_pattern_group_maps(pred_df, output_dir)
    plot_city_comparison(pred_df, output_dir)
    plot_class_distribution(pred_df, output_dir)

    # Save predictions
    save_predictions(pred_df, output_dir)

    print("\n" + "="*80)
    print("✓ All predictions and visualizations completed!")
    print(f"  Output directory: {output_dir}")
    print("="*80)

    # Print summary
    print("\nPrediction Summary:")
    print(f"  - Total grids: {len(pred_df)}")
    print(f"  - Cities: {pred_df['city_name'].nunique()}")
    print(f"  - Areas: {pred_df['area_name'].nunique()}")
    print("\nClass Distribution:")
    for class_id in range(1, 10):
        count = (pred_df['predicted_label'] == class_id).sum()
        pct = count / len(pred_df) * 100
        print(f"  {CLASS_NAMES[class_id]}: {count} ({pct:.2f}%)")

if __name__ == "__main__":
    main()
