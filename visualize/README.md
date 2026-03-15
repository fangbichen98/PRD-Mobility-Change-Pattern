# Visualization Workflow

This directory contains the maintained visualization workflow for the SGH mobility pattern experiment.

## 1. Central configuration

Adjust common visualization settings in `visualize/viz_config.py`:

- experiment output directory
- class names
- class colors
- city and district English mappings
- font family
- export DPI
- evaluation figure color settings

## 2. Main scripts

- `visualize/render_all_figures.py`
  - one-click entry point for regenerating all maintained figures
- `visualize/predict_with_cache.py`
  - full-grid prediction and map/distribution plotting
- `visualize/plot_classification_performance.py`
  - category-wise precision / recall / F1 chart
- `visualize/plot_confusion_matrix.py`
  - confusion matrix figure from `confusion_matrix.npy`
- `visualize/figure_data_sources.md`
  - provenance note for each figure

## 3. Recommended usage

Run from the repository root.

Regenerate all figures while reusing existing prediction CSV:

```bash
python visualize/render_all_figures.py
```

Force a full-grid inference rerun before plotting:

```bash
python visualize/render_all_figures.py --force-inference
```

Run individual scripts if only one figure group needs updating:

```bash
python visualize/predict_with_cache.py
python visualize/plot_classification_performance.py
python visualize/plot_confusion_matrix.py
```

## 4. Expected inputs

Map figures require:

- `data/cache/dual_year_data_all_grids.pkl`
- `data/grid_metadata/sgh_grid_metadata.csv`
- `outputs/.../models/best_model.pth`

Evaluation figures require:

- `outputs/.../metrics/classification_report.txt`
- `outputs/.../metrics/confusion_matrix.npy`

All paths above are resolved through `visualize/viz_config.py`.

## 5. Output locations

- spatial prediction figures: `outputs/.../model_predictions/`
- evaluation figures: `outputs/.../metrics/`

## 6. Notes

- The one-click script prefers reusing `all_grids_predictions.csv` to avoid unnecessary inference.
- If an older CSV lacks English columns, they are filled automatically during rendering.
- For publication updates, change the config first, then rerun the one-click script.