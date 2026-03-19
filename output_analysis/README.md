# 实验结果分析文档

本目录包含对实验 `multiscale_temporal_20260313_085258_labels_sgh_0.03_4500_random` 的完整分析结果。

## 📁 文件结构

```
output_analysis/
├── README.md                           # 本文件
├── experiment_analysis_report.md       # 完整分析报告
└── (可视化图表位于实验输出目录)

outputs/multiscale_temporal_20260313_085258_labels_sgh_0.03_4500_random/
├── metrics/
│   ├── test_results.json              # 测试结果JSON
│   ├── classification_report.txt      # 分类报告
│   ├── confusion_matrix.npy           # 混淆矩阵
│   └── timing_info.json               # 训练时间统计
├── models/
│   └── best_model.pth                 # 最佳模型权重
├── visualizations/
│   ├── spatial_distribution_map.png   # 空间分布图
│   ├── city_zoom_maps.png             # 城市局部放大图
│   ├── city_class_distribution.png    # 城市类别分布图
│   ├── confusion_matrix_heatmap.png   # 混淆矩阵热图
│   └── city_class_statistics.csv      # 城市统计数据
├── class_performance.png              # 类别性能对比图
└── training.log                       # 训练日志
```

## 📊 快速查看

### 1. 核心图表

**空间分布图** (`spatial_distribution_map.png`)
- 展示深圳-东莞-惠州地区9类移动模式变化的空间分布
- 每个点代表一个500m×500m网格
- 使用色盲友好的9色配色方案

**城市局部放大图** (`city_zoom_maps.png`)
- 4个子图展示典型城市区域的详细模式
- 包括：深圳南山区、深圳福田区、东莞南城街道、惠州惠城区

**城市类别分布图** (`city_class_distribution.png`)
- 堆叠柱状图展示各城市的9类占比
- 清晰对比三个城市的异质性

**混淆矩阵热图** (`confusion_matrix_heatmap.png`)
- 9×9矩阵展示模型预测准确性
- 识别主要混淆模式

**类别性能对比图** (`class_performance.png`)
- 对比各类别的Precision、Recall、F1-Score
- 展示测试集样本分布

### 2. 数据文件

**城市统计数据** (`city_class_statistics.csv`)
```csv
city_name,1,2,3,4,5,6,7,8,9
东莞市,7.78,5.56,28.89,12.22,7.22,7.78,5.56,10.0,15.0
惠州市,9.71,6.04,32.42,11.72,10.26,7.88,4.03,6.59,11.36
深圳市,6.32,5.75,33.33,12.64,6.32,6.90,6.90,8.05,13.79
```

**测试结果** (`test_results.json`)
- 包含完整的模型配置和性能指标
- 数据集信息和类别分布

## 🎯 关键发现

### 整体性能
- **测试准确率**: 58.78%
- **F1分数**: 0.6038
- **最佳验证准确率**: 61.11% (Epoch 21)

### 空间模式
1. **增长主导**: 类别3（增加+平衡）占31.55%
2. **空间平衡化**: 平衡类别合计占比52.45%
3. **城市差异**: 深圳增长最显著，东莞稳定区域最多

### 模型表现
- ✅ **最佳类别**: 类别1（Precision 1.00）、类别6（F1 0.84）、类别8（F1 0.81）
- ⚠️ **需改进**: 类别2（Recall 0.31）、类别7（Recall 0.23）

## 📖 使用指南

### 查看完整分析报告
```bash
# 使用Markdown阅读器
cat experiment_analysis_report.md

# 或在IDE中打开
code experiment_analysis_report.md
```

### 查看可视化图表
```bash
# 进入可视化目录
cd ../outputs/multiscale_temporal_20260313_085258_labels_sgh_0.03_4500_random/visualizations/

# 使用图片查看器
eog spatial_distribution_map.png  # Linux
open spatial_distribution_map.png  # macOS
```

### 重新生成图表
```bash
# 运行可视化脚本
python3 visualize/predict_with_cache.py
python3 visualize/plot_classification_performance.py
python3 visualize/plot_confusion_matrix.py
```

### 加载模型进行预测
```python
import torch
from src.models.enhanced_dual_branch_model import EnhancedDualBranchModel

# 加载模型
model = EnhancedDualBranchModel(...)
model.load_state_dict(torch.load('outputs/.../models/best_model.pth'))
model.eval()

# 进行预测
with torch.no_grad():
    predictions = model(x_2021, x_2024, graphs_2021, graphs_2024, ...)
```

## 🔍 深入分析

### 1. 混淆模式分析

**主要混淆对**:
- 类别2 → 类别3 (71个样本): "增加+向内" vs "增加+平衡"
- 类别7 → 类别9 (47个样本): "稳定+向外" vs "稳定+平衡"
- 类别4 → 类别5 (23个样本): "减少+向外" vs "减少+向内"

**原因分析**:
- 方向性特征（向内/向外/平衡）的边界模糊
- 流量变化阈值敏感性
- 时空特征融合不足

### 2. 城市对比分析

| 指标 | 深圳市 | 东莞市 | 惠州市 |
|------|--------|--------|--------|
| 测试样本数 | 174 | 180 | 546 |
| 主导模式 | 类别3 (33.33%) | 类别3 (28.89%) | 类别3 (32.42%) |
| 稳定占比 | 28.74% | 30.56% | 22.00% |
| 增长占比 | 45.40% | 42.23% | 48.17% |
| 减少占比 | 25.86% | 27.22% | 29.85% |

**洞察**:
- 深圳增长最强劲，减少占比最低
- 东莞稳定性最高，发展趋于成熟
- 惠州样本量最大，模式最多样

### 3. 时空特征重要性

**时序特征**:
- 小时尺度: 捕捉日内变化模式
- 日尺度: 捕捉周内变化趋势
- 周尺度: 捕捉整体变化特征

**空间特征**:
- 图卷积: 捕捉邻域流动关系
- 边权重: 反映流量强度
- 多跳传播: 捕捉远程依赖

## 🚀 改进建议

### 1. 特征工程
- [ ] 增加POI（兴趣点）特征
- [ ] 引入人口密度、土地利用等外部数据
- [ ] 设计更明确的方向性特征
- [ ] 添加时间窗口变化率特征

### 2. 模型优化
- [ ] 尝试图注意力网络（GAT）替代GCN
- [ ] 引入时空注意力机制
- [ ] 调整类别权重平衡precision和recall
- [ ] 探索集成学习方法

### 3. 数据增强
- [ ] 增加边界样本的标注
- [ ] 使用半监督学习利用未标注数据
- [ ] 时间序列数据增强（抖动、缩放等）
- [ ] 空间数据增强（邻域采样等）

### 4. 评估优化
- [ ] 增加类别级别的详细分析
- [ ] 添加空间自相关分析
- [ ] 进行敏感性分析
- [ ] 交叉验证评估稳定性

## 📚 相关文档

- [CLAUDE.md](../CLAUDE.md) - 项目完整文档
- [config.py](../config.py) - 模型配置参数
- [train_multiscale_temporal.py](../train_multiscale_temporal.py) - 训练脚本
- [visualize/predict_with_cache.py](../visualize/predict_with_cache.py) - 地图与分布图主脚本
- [visualize/plot_classification_performance.py](../visualize/plot_classification_performance.py) - 分类性能图脚本
- [visualize/plot_confusion_matrix.py](../visualize/plot_confusion_matrix.py) - 混淆矩阵图脚本

## 📧 联系方式

如有问题或建议，请联系项目维护者。

---

**生成时间**: 2026-03-13
**实验ID**: multiscale_temporal_20260313_085258_labels_sgh_0.03_4500_random
**版本**: 1.0
