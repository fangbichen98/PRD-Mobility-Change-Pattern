from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

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


def render_from_existing_predictions(predictions_path: Path, output_dir: Path) -> None:
    pred_df = pd.read_csv(predictions_path)
    pred_df = ensure_english_columns(pred_df)

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

    if args.force_inference or not predictions_path.exists():
        reason = 'forced rerun' if args.force_inference else 'missing prediction CSV'
        print(f'Running full prediction pipeline due to {reason}...')
        predict_with_cache.main()
    else:
        print(f'Reusing existing predictions: {predictions_path}')
        render_from_existing_predictions(predictions_path, MODEL_PREDICTIONS_DIR)

    print('Rendering evaluation figures...')
    plot_classification_performance.plot_classification_performance(
        METRICS_DIR / 'classification_report.txt',
        METRICS_DIR,
    )
    plot_confusion_matrix.plot_confusion_matrix(METRICS_DIR)

    print('All figures rendered successfully.')


if __name__ == '__main__':
    main()