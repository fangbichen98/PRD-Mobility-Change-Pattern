# Phase23 Transformer-Full 结果诊断（2026-03-18）

## 结论速览
在固定对照协议（GCN + raw_temporal_mean + topk20 + static + seed202 + spc250 + e300/p30）下，`TRANSFORMER_FULL` 明显落后于轻量 `TRANSFORMER`：

- TRANSFORMER: test acc 74.89%, test F1 0.7434
- TRANSFORMER_FULL: test acc 69.33%, test F1 0.6798
- 差值：acc -5.56pp，F1 -0.0636

## 核心证据

### 1) 泛化结果确实更差
- 轻量 Transformer 测试结果见 [outputs/multiscale_temporal_20260318_173138_label_sgh_phase23_rawflow_gcn_topk_static_transformer_spc250_seed202_e300p30/metrics/test_results.json](outputs/multiscale_temporal_20260318_173138_label_sgh_phase23_rawflow_gcn_topk_static_transformer_spc250_seed202_e300p30/metrics/test_results.json)
- Full Transformer 测试结果见 [outputs/multiscale_temporal_20260318_183428_label_sgh_phase23_rawflow_gcn_topk_static_transformer_full_spc250_seed202_e300p30/metrics/test_results.json](outputs/multiscale_temporal_20260318_183428_label_sgh_phase23_rawflow_gcn_topk_static_transformer_full_spc250_seed202_e300p30/metrics/test_results.json)

### 2) Full 参数量更大、耗时更高，但验证上限更低
- 参数量：
  - 轻量：4,773,967（见 [training.log](outputs/multiscale_temporal_20260318_173138_label_sgh_phase23_rawflow_gcn_topk_static_transformer_spc250_seed202_e300p30/training.log#L35)）
  - Full：7,719,695（见 [training.log](outputs/multiscale_temporal_20260318_183428_label_sgh_phase23_rawflow_gcn_topk_static_transformer_full_spc250_seed202_e300p30/training.log#L35)）
- 训练时长：
  - 轻量：22m06s（见 [timing_info.json](outputs/multiscale_temporal_20260318_173138_label_sgh_phase23_rawflow_gcn_topk_static_transformer_spc250_seed202_e300p30/metrics/timing_info.json)）
  - Full：29m58s（见 [timing_info.json](outputs/multiscale_temporal_20260318_183428_label_sgh_phase23_rawflow_gcn_topk_static_transformer_full_spc250_seed202_e300p30/metrics/timing_info.json)）

### 3) Full 在验证集平台期更早、更低
- 轻量 best val acc = 71.11%，早停 epoch 69（见 [training.log](outputs/multiscale_temporal_20260318_173138_label_sgh_phase23_rawflow_gcn_topk_static_transformer_spc250_seed202_e300p30/training.log#L1516)、[training.log](outputs/multiscale_temporal_20260318_173138_label_sgh_phase23_rawflow_gcn_topk_static_transformer_spc250_seed202_e300p30/training.log#L1519)）
- Full best val acc = 70.22%，早停 epoch 87（见 [training.log](outputs/multiscale_temporal_20260318_183428_label_sgh_phase23_rawflow_gcn_topk_static_transformer_full_spc250_seed202_e300p30/training.log#L1900)、[training.log](outputs/multiscale_temporal_20260318_183428_label_sgh_phase23_rawflow_gcn_topk_static_transformer_full_spc250_seed202_e300p30/training.log#L1903)）

## 可能原因（按优先级）
1. 容量-数据规模不匹配：Full 的时序分支更深更宽（更多层、更大 FFN、CLS 读出），在当前标注规模（spc250）下增加了优化难度，并未转化为更高验证上限。
2. 超参数沿用轻量配置：Full 仍继承相同全局 dropout/LR/调度策略，未做专门 warmup 与正则匹配，容易出现“训练可继续、验证不再提升”的平台。
3. 读出方式变化引入额外不稳定性：轻量版用 mean pooling，Full 改为 CLS token 读出；在短序列（7 天）+ 小样本设置下，CLS 可能不如均值池化稳健。
4. 计算预算成本上升：单次迭代更慢，虽然总 epoch 一样，但有效调参迭代次数更少，且 Full 对学习率与衰减策略更敏感。

## 已发现的流程问题
- `scripts/run_phase23_transformer_full_tuned_e300p30.sh` 当前调用了不被 CLI 支持的参数 `--weight-decay`，导致任务未启动。
- 证据见 [outputs/phase23_gcn_temporal_logs/phase23_rawflow_gcn_topk_static_transformer_full_tuned_spc250_seed202_e300p30.log](outputs/phase23_gcn_temporal_logs/phase23_rawflow_gcn_topk_static_transformer_full_tuned_spc250_seed202_e300p30.log)。

## 下一步建议（最小变更优先）
1. 修复 tuned 脚本参数错误后再跑一次 Full-tuned 对照（只改 run-time config，不改主干代码）。
2. Full 优先尝试：
   - 降模型容量（model_dim 256 -> 192；hourly/daily 层数各减 1）
   - 降 dropout（当前继承 0.4，先试 0.3）
   - 更小初始 LR（如 5e-5）+ warmup（前 5~10 epoch）
3. 若仍不如轻量 Transformer，建议将轻量 Transformer 作为 phase23 结论模型，Full 仅保留为负结果与分析说明。
