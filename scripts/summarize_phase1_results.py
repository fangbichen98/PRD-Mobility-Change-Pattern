from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None


def collect_phase1_runs(outputs_dir: Path, run_tag_keyword: str = 'phase1_small_') -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for run_dir in sorted(outputs_dir.glob('multiscale_temporal_*')):
        if run_tag_keyword not in run_dir.name:
            continue

        metrics_dir = run_dir / 'metrics'
        test_json = metrics_dir / 'test_results.json'
        timing_json = metrics_dir / 'timing_info.json'

        test_data = read_json(test_json)
        if test_data is None:
            continue

        timing_data = read_json(timing_json) or {}
        training_cfg = test_data.get('training_config', {})
        data_info = test_data.get('data_info', {})
        spatial_cfg = test_data.get('model_architecture', {}).get('spatial_branch', {})

        rows.append(
            {
                'run_dir': run_dir.name,
                'label_file': data_info.get('label_file', ''),
                'samples_per_class': _extract_samples_per_class(run_dir.name),
                'spatial_model': str(spatial_cfg.get('type', '')),
                'test_accuracy': test_data.get('test_accuracy', None),
                'test_f1': test_data.get('test_f1', None),
                'train_samples': data_info.get('train_samples', None),
                'val_samples': data_info.get('val_samples', None),
                'test_samples': data_info.get('test_samples', None),
                'batch_size': training_cfg.get('batch_size', None),
                'grad_accum': training_cfg.get('gradient_accumulation', None),
                'learning_rate': training_cfg.get('learning_rate', None),
                'num_epochs_cfg': training_cfg.get('num_epochs', None),
                'early_stop_cfg': training_cfg.get('early_stopping_patience', None),
                'hours': timing_data.get('total_time_hours', None),
                'minutes': timing_data.get('total_time_minutes', None),
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values(['test_f1', 'test_accuracy'], ascending=False, na_position='last').reset_index(drop=True)
    df = df.drop_duplicates(subset=['label_file', 'samples_per_class', 'spatial_model'], keep='first').reset_index(drop=True)
    return df


def _extract_samples_per_class(run_name: str) -> int | None:
    # Script run tag currently includes phase1_small_<label>_<model>; sampling is fixed at 50.
    # Keep this helper in case naming rule is later expanded.
    return 50 if 'phase1_small_' in run_name else None


def build_markdown_summary(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        out_path.write_text('# Phase1 Summary\n\nNo completed runs found.\n', encoding='utf-8')
        return

    lines = ['# Phase1 Screening Summary', '']
    lines.append(f'Total completed runs: {len(df)}')
    lines.append('')
    lines.append('| Rank | Run | Label | Spatial Model | Test Acc (%) | Test F1 | Train/Val/Test | Time (min) |')
    lines.append('|---|---|---|---|---:|---:|---|---:|')

    for idx, row in df.iterrows():
        lines.append(
            '| {rank} | {run} | {label} | {model} | {acc:.2f} | {f1:.4f} | {tr}/{va}/{te} | {mins:.2f} |'.format(
                rank=idx + 1,
                run=row['run_dir'],
                label=row.get('label_file', ''),
                model=row.get('spatial_model', ''),
                acc=float(row['test_accuracy']) if pd.notna(row['test_accuracy']) else float('nan'),
                f1=float(row['test_f1']) if pd.notna(row['test_f1']) else float('nan'),
                tr=int(row['train_samples']) if pd.notna(row['train_samples']) else -1,
                va=int(row['val_samples']) if pd.notna(row['val_samples']) else -1,
                te=int(row['test_samples']) if pd.notna(row['test_samples']) else -1,
                mins=float(row['minutes']) if pd.notna(row['minutes']) else float('nan'),
            )
        )

    lines.append('')
    top_n = min(2, len(df))
    lines.append(f'## Recommended Candidates (Top {top_n})')
    lines.append('')
    for idx in range(top_n):
        row = df.iloc[idx]
        lines.append(
            f"- {idx + 1}. {row['run_dir']} | model={row['spatial_model']} | acc={float(row['test_accuracy']):.2f}% | f1={float(row['test_f1']):.4f}"
        )

    out_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description='Summarize phase1 small-scale experiment results')
    parser.add_argument('--outputs-dir', type=str, default='outputs', help='Outputs root directory')
    parser.add_argument('--csv', type=str, default='outputs/phase1_screening_summary.csv', help='CSV output path')
    parser.add_argument('--md', type=str, default='outputs/phase1_screening_summary.md', help='Markdown output path')
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    csv_path = Path(args.csv)
    md_path = Path(args.md)

    df = collect_phase1_runs(outputs_dir)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    if df.empty:
        pd.DataFrame().to_csv(csv_path, index=False)
        build_markdown_summary(df, md_path)
        print('No completed phase1 runs found yet.')
        print(f'Wrote empty summary: {csv_path}')
        print(f'Wrote markdown summary: {md_path}')
        return

    df.to_csv(csv_path, index=False)
    build_markdown_summary(df, md_path)

    print(f'Wrote summary CSV: {csv_path}')
    print(f'Wrote summary markdown: {md_path}')
    print(df[['run_dir', 'label_file', 'spatial_model', 'test_accuracy', 'test_f1']].to_string(index=False))


if __name__ == '__main__':
    main()
