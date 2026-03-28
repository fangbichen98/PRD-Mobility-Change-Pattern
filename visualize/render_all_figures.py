from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import plot_classification_performance
import plot_confusion_matrix
import predict_with_cache
from viz_config import AREA_NAME_EN, CITY_NAME_EN, EXPERIMENT_DIR, METRICS_DIR, MODEL_PREDICTIONS_DIR


def ensure_english_columns(pred_df: pd.DataFrame) -> pd.DataFrame:
    pred_df = pred_df.copy()
    if 'city_name_en' not in pred_df.columns:
        pred_df['city_name_en'] = pred_df['city_name'].map(CITY_NAME_EN).fillna(pred_df['city_name'])
    if 'area_name_en' not in pred_df.columns:
        pred_df['area_name_en'] = pred_df['area_name'].map(AREA_NAME_EN).fillna(pred_df['area_name'])
    return pred_df


def _remove_both_year_zero_flow_grids(pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop grids that have zero total flow in BOTH 2021 and 2024.

    Priority:
    1) If prediction CSV already contains total flow columns, use them directly.
    2) Otherwise, recover totals from cache change_features by grid_id.
    """
    pred_df = pred_df.copy()

    # Fast path: use flow totals if present in prediction CSV.
    if {'total_flow_2021', 'total_flow_2024'}.issubset(pred_df.columns):
        keep_mask = (pred_df['total_flow_2021'] > 0) | (pred_df['total_flow_2024'] > 0)
        removed = int((~keep_mask).sum())
        filtered = pred_df.loc[keep_mask].copy()
        print(f"Removed both-year-zero-flow grids from prediction CSV: {removed}")
        return filtered

    # Fallback: recover from cached change_features keyed by grid_id.
    if 'grid_id' not in pred_df.columns:
        print("[WARN] Cannot apply both-year-zero-flow filtering: missing grid_id column.")
        return pred_df

    try:
        manifest = predict_with_cache.load_experiment_manifest(EXPERIMENT_DIR)
        default_cache_path = os.environ.get(
            'VIS_CACHE_PATH',
            str(Path('data/cache/dual_year_data_all_grids.pkl')),
        )
        cache_result = predict_with_cache.resolve_cache_paths_from_manifest(default_cache_path, manifest)
        cached_data = predict_with_cache.load_cached_data(cache_result)
        change_features = cached_data.get('change_features', {})
    except (FileNotFoundError, KeyError, RuntimeError) as exc:
        print(f"[WARN] Failed to load cache for zero-flow filtering: {exc}")
        return pred_df

    keep_mask = []
    missing_count = 0
    for grid_id in pred_df['grid_id'].tolist():
        feature = change_features.get(grid_id)
        if feature is None:
            keep_mask.append(True)
            missing_count += 1
            continue

        flow_2021 = feature[:, :2]
        flow_2024 = feature[:, 2:]
        # Original features are log(1+flow); recover raw total with expm1.
        total_2021 = float((np.expm1(flow_2021)).sum())
        total_2024 = float((np.expm1(flow_2024)).sum())
        keep_mask.append((total_2021 > 0.0) or (total_2024 > 0.0))

    keep_mask = pd.Series(keep_mask, index=pred_df.index)
    removed = int((~keep_mask).sum())
    filtered = pred_df.loc[keep_mask].copy()
    print(f"Removed both-year-zero-flow grids from cache features: {removed}")
    if missing_count:
        print(f"[WARN] {missing_count} grids missing in cache change_features; kept by default.")
    return filtered


def render_from_existing_predictions(predictions_path: Path, output_dir: Path) -> None:
    pred_df = pd.read_csv(predictions_path)
    pred_df = ensure_english_columns(pred_df)
    pred_df = _remove_both_year_zero_flow_grids(pred_df)

    predict_with_cache.plot_full_region_map(pred_df, str(output_dir))
    predict_with_cache.plot_pattern_group_maps(pred_df, str(output_dir))
    predict_with_cache.plot_city_comparison(pred_df, str(output_dir))
    predict_with_cache.plot_class_distribution(pred_df, str(output_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Render all publication figures for the configured experiment.'
    )
    parser.add_argument(
        '--force-inference',
        action='store_true',
        help='Ignore existing all_grids_predictions.csv and rerun full-grid inference before plotting.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions_path = MODEL_PREDICTIONS_DIR / 'all_grids_predictions.csv'

    print(f'Experiment directory: {EXPERIMENT_DIR}')

    # --- Spatial prediction maps (require all-grids cache) ---
    try:
        if args.force_inference or not predictions_path.exists():
            reason = 'forced rerun' if args.force_inference else 'missing prediction CSV'
            print(f'Running full prediction pipeline due to {reason}...')
            predict_with_cache.main()
        else:
            print(f'Reusing existing predictions: {predictions_path}')
            render_from_existing_predictions(predictions_path, MODEL_PREDICTIONS_DIR)
    except (FileNotFoundError, KeyError, RuntimeError) as exc:
        print(f'[WARN] Skipping spatial prediction maps: {exc}')
        print('       (all-grids cache not available; evaluation figures will still be generated)')

    # --- Evaluation figures (only need metrics/) ---
    print('Rendering evaluation figures...')
    plot_classification_performance.plot_classification_performance(
        METRICS_DIR / 'classification_report.txt',
        METRICS_DIR,
    )
    plot_confusion_matrix.plot_confusion_matrix(METRICS_DIR)

    print('All figures rendered successfully.')


if __name__ == '__main__':
    main()