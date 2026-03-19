# 图件数据来源说明（Figure Data Sources）

## 1. 说明目的

本文档用于说明当前实验目录下各图件的绘制脚本、直接输入文件、以及上游来源文件，便于论文复现与追溯。

可调图形参数统一配置文件：
`visualize/viz_config.py`

其中集中维护：
- 类别名称
- 类别颜色
- 城市/区县英文映射
- 实验输出目录
- 字体与导出 DPI
- 分类性能图颜色
- 混淆矩阵标签

实验目录：
`outputs/multiscale_temporal_20260313_085258_labels_sgh_0.03_4500_random`

---

## 2. 图件与输入文件对应关系

| 图件文件 | 绘制脚本/方式 | 直接输入文件 | 上游来源 |
|---|---|---|---|
| `model_predictions/full_region_prediction_map.png` / `.pdf` | `predict_with_cache.py` -> `plot_full_region_map()` | `model_predictions/all_grids_predictions.csv` | `models/best_model.pth` + `data/cache/dual_year_data_all_grids.pkl` + `data/grid_metadata/sgh_grid_metadata.csv` |
| `model_predictions/pattern_group_maps.png` / `.pdf` | `predict_with_cache.py` -> `plot_pattern_group_maps()` | `model_predictions/all_grids_predictions.csv` | 同上 |
| `model_predictions/city_map_shenzhen.png` / `.pdf` | `predict_with_cache.py` -> `plot_city_comparison()` | `model_predictions/all_grids_predictions.csv` | 同上 |
| `model_predictions/city_map_dongguan.png` / `.pdf` | `predict_with_cache.py` -> `plot_city_comparison()` | `model_predictions/all_grids_predictions.csv` | 同上 |
| `model_predictions/city_map_huizhou.png` / `.pdf` | `predict_with_cache.py` -> `plot_city_comparison()` | `model_predictions/all_grids_predictions.csv` | 同上 |
| `model_predictions/class_distribution_overall.png` / `.pdf` | `predict_with_cache.py` -> `plot_class_distribution()` | `model_predictions/all_grids_predictions.csv` | 同上 |
| `model_predictions/class_distribution_city_district.png` / `.pdf` | `predict_with_cache.py` -> `plot_class_distribution()` | `model_predictions/all_grids_predictions.csv` | 同上 |
| `metrics/classification_performance_by_category.png` / `.pdf` | `plot_classification_performance.py` | `metrics/classification_report.txt` | 由测试阶段输出的分类报告 |
| `metrics/confusion_matrix_hmp_cd_9class.png` / `.pdf` | Python脚本（基于 `confusion_matrix.npy` 生成） | `metrics/confusion_matrix.npy` | 由测试阶段输出的混淆矩阵 |

---

## 3. 关键数据文件说明

### 3.1 预测与制图主数据

- `model_predictions/all_grids_predictions.csv`
  - 含字段：`grid_id`, `predicted_label`, `lon`, `lat`, `city_name`, `area_name`（以及英文字段）
  - 这是所有地图与分布统计图的直接输入。

- `data/cache/dual_year_data_all_grids.pkl`
  - 全域格网时序特征缓存（用于加速推理）。
  - 由训练 feature cache 和原始 OD 数据加工扩展得到。
  - 必须保留 `train_flow_grid_ids` 字段，但它主要用于实验来源校验，以及恢复 `raw_temporal_mean` 下 spatial raw 节点特征的训练支持集。
  - 当前正确的推理语义不是“所有输入都按训练流量节点掩码置零”，而是：
    - temporal branch 使用全域 cache 中每个被预测节点自己的时序特征；
    - spatial branch 的 raw node feature 默认按 `train_flow_grid_ids` 掩码恢复训练支持集。
  - 如果把 temporal branch 也错误地按 2250 个训练流量节点清零，会导致全图类别分布塌缩。
  - 如果把 spatial raw node feature 错误扩展到 65049 个全图节点，也会改变 GCN 输入分布，明显压低 Decline Static（Class 7）等类别。
  - 对 `raw_temporal_mean` 实验，建议同时包含 `train_flow_label_hash`、`train_flow_total_samples`、`train_flow_cache_file`，用于校验训练流量掩码来源是否与当前实验一致。
  - 若该字段缺失，需要先重新运行 `extract_all_features.py` 重建 cache。

- `models/best_model.pth`
  - 当前实验最优模型权重。

- `data/grid_metadata/sgh_grid_metadata.csv`
  - 格网经纬度与行政区映射（城市、区县）。

### 3.2 评估与性能图主数据

- `metrics/classification_report.txt`
  - 各类别 precision / recall / f1-score / support。
  - `classification_performance_by_category` 图直接解析该文件生成。

- `metrics/confusion_matrix.npy`
  - 9x9 混淆矩阵原始数组。
  - `confusion_matrix_hmp_cd_9class` 图由该数组渲染而成。

---

## 4. 类别命名（当前版本）

当前 1-9 类在图例中的命名如下：

1. Stable Balanced  
2. Stable Aggregation  
3. Stable Diffusion  
4. Growth Balanced  
5. Growth Aggregation  
6. Growth Diffusion  
7. Decline Balanced  
8. Decline Aggregation  
9. Decline Diffusion  

---

## 5. 复现建议（最小流程）

1. 先确保以下文件存在：
   - `models/best_model.pth`
   - `data/cache/dual_year_data_all_grids.pkl`
   - `data/grid_metadata/sgh_grid_metadata.csv`
2. 若 `dual_year_data_all_grids.pkl` 是旧版本，请先运行 `EXPERIMENT_DIR=outputs/当前实验 python extract_all_features.py`，确保其中包含 `train_flow_grid_ids` 及训练流量来源元数据。
3. 对 `raw_temporal_mean` 实验，默认应保持“temporal 全图、spatial raw 按训练支持集”的推理方式，不要把 provenance 校验字段直接当作所有输入的统一掩码。
4. 优先运行 `render_all_figures.py` 一键生成全部维护中的图件。
5. 若只更新空间预测图，可单独运行 `predict_with_cache.py`。
6. 若只更新评估图，可分别运行 `plot_classification_performance.py` 与 `plot_confusion_matrix.py`。

---

## 6. 文档更新时间

- 生成时间：2026-03-13
- 对应实验：`multiscale_temporal_20260313_085258_labels_sgh_0.03_4500_random`
