# Phase Experiment Tracker

## Purpose

This file is the single maintenance entry for phase-based experiments in this repository.
It is intended to reduce context loss across long conversations and provide a stable place to append new phase results.

## Maintenance Rules

1. Every new phase must be recorded in this file. Do not start or finish a phase without updating this tracker.
2. When a new phase starts, add one row under "Current Progress" first.
3. After a phase finishes, move its final result into the corresponding section below.
4. If the phase is only partially completed, update the current state and next action before ending the working session.
5. Prefer recording results from metrics/test_results.json or phase-level results.csv, not from ad hoc log snippets.
6. For each phase, keep four items when possible: goal, fixed protocol, best result, source files.
7. If a run was started but later abandoned by user decision, keep it in notes as "obsolete" instead of deleting history.

## Current Progress

| Phase | Status | Current State | Next Action | Source |
|---|---|---|---|---|
| phase29 | completed | Transformer+GCN on three 4500-label edge variants finished. `half_edge` = 0.5533 / 0.5650, `max_edge` = 0.5086 / 0.5035, `no_edge` = 0.6724 / 0.6722. All three current test_results now explicitly record `topk_enabled=true`, `graph_topk_out=20`, `graph_topk_in=20`. | Reuse this metadata convention in all future phases. | `scripts/run_phase29_transformer_gcn_4500_edge_variants_e300p30.sh`, `outputs/phase29_transformer_gcn_4500_edge_logs/phase29_results.csv` |
| phase30 | completed | Controlled GINE comparison finished under `label_sgh + spc250 + seed202 + LSTM + topk20/topk20 + static + e300/p30`. `flow_only` reached 68.22 / 0.6744. `flow_distance_direction` reached 68.00 / 0.6757. Both beat the earlier phase24 LSTM+GINE `ones` baseline (66.67 / 0.6619), but neither beat the current LSTM+GCN baseline (71.78 / 0.7000). The new 4-d edge feature slightly improved macro F1 over flow-only (+0.0013) but slightly reduced accuracy (-0.22 pt). | Keep `flow_only` as the stronger overall phase30 GINE setting for now; if continuing this direction, next work should focus on stronger edge feature scaling/selection or a higher-capacity GINE design rather than treating the current 4-d edge feature as a win. | `outputs/phase30_gine_edge_compare_logs/phase30_results.csv`, `outputs/phase30_gine_edge_compare_logs/phase30_summary.md`, `outputs/multiscale_temporal_20260323_172440_label_sgh_phase30_gine_flow_only_spc250_seed202_e300p30/metrics/test_results.json`, `outputs/multiscale_temporal_20260323_175753_label_sgh_phase30_gine_flowdistdir_spc250_seed202_e300p30/metrics/test_results.json` |
| phase31 | completed | Controlled Transformer+GINE follow-up finished under the same phase30 protocol except `LSTM -> TRANSFORMER`. `flow_only` reached 73.11 / 0.7288. `flow_distance_direction` reached 73.56 / 0.7304. Unlike phase30, the 4-d edge feature now wins on both accuracy (+0.45 pt) and macro F1 (+0.0016) over `flow_only`, and both phase31 variants clearly outperform the phase30 LSTM+GINE runs. The best phase31 row also beats the phase24 light Transformer+GCN matrix row (71.78 / 0.7050). | Treat `TRANSFORMER + GINE + flow_distance_direction` as the current best validated edge-feature line. If continuing, the next highest-value work is a focused robustness check across another seed or label subset before broadening the edge feature set again. | `scripts/run_phase31_transformer_gine_edge_followup_e300p30.sh`, `outputs/phase31_transformer_gine_edge_logs/phase31_results.csv`, `outputs/phase31_transformer_gine_edge_logs/phase31_results_table.md`, `outputs/multiscale_temporal_20260323_183751_label_sgh_phase31_transformer_gine_flow_only_spc250_seed202_e300p30/metrics/test_results.json`, `outputs/multiscale_temporal_20260323_192012_label_sgh_phase31_transformer_gine_flowdistdir_spc250_seed202_e300p30/metrics/test_results.json` |
| phase32 | completed | Controlled frozen-sample/frozen-split comparison finished on `data/sampled_labels_spc250_seed202_reconstructed.csv` plus `data/splits/sampled_labels_spc250_seed202_reconstructed_seed42_split.json`. Two valid reruns were collected for each side. Transformer+GCN scored `70.22 / 0.6913` and `70.89 / 0.6966`; Transformer+GINE+flow_distance_direction scored `71.11 / 0.7119` and `70.00 / 0.7028`. Mean accuracy tied at `70.56`, while GINE held the higher mean macro F1 (`0.7073` vs `0.6939`). This means the earlier phase24/phase31 headline gap was materially inflated by sample/split differences; under controlled sample/split, there is no clear accuracy winner yet. | Treat phase32 as the new canonical controlled comparison. If selecting a model by accuracy, do not claim GINE is better yet; first enforce deterministic training or expand to a 3-seed frozen protocol. If selecting by macro F1, GINE currently has the slight edge. | `outputs/phase32_sample_control_design.md`, `outputs/phase32_frozen_pairwise_results.md`, `outputs/multiscale_temporal_20260325_152118_sampled_labels_spc250_seed202_reconstructed_phase32_transformer_gcn_frozen_sample_split_e300p30/metrics/test_results.json`, `outputs/multiscale_temporal_20260325_152422_sampled_labels_spc250_seed202_reconstructed_phase32_transformer_gcn_frozen_sample_split_e300p30/metrics/test_results.json`, `outputs/multiscale_temporal_20260325_161305_sampled_labels_spc250_seed202_reconstructed_phase32_transformer_gine_flowdistdir_frozen_sample_split_e300p30/metrics/test_results.json`, `outputs/multiscale_temporal_20260325_163605_sampled_labels_spc250_seed202_reconstructed_phase32_transformer_gine_flowdistdir_frozen_sample_split_rerun_e300p30/metrics/test_results.json` |
| future phases | pending | User indicated more phase experiments will continue on top of this baseline. | Append new rows before execution. | this file |

## Phase Timeline Overview

### Phase0-Phase13: Early exploration and protocol shaping

These phases established the later standardized protocol, but results are spread across multiple summary files.

| Phase Range | Main Goal | Key Outcome | Primary Summary Files |
|---|---|---|---|
| phase0-1 | Smoke tests and small screening | Early small-scale screening established GCN over SAGE on `label_sgh` small split. | `outputs/phase1_screening_summary.md`, `outputs/smoke_experiments_summary.md` |
| phase2-4 | Random-label direction search, topk/full graph exploration | Identified promising random-label directions and candidate graph settings. | `outputs/dir12_random_fast_summary.md`, `outputs/dir34_random_fast_summary.md`, `outputs/random_next_stage_100x3_summary.md`, `outputs/phase3_winner_decision_summary.md` |
| phase5-6 | Loss and topk ablations | Built early evidence for stable topk settings and later all-sample runs. | `outputs/loss_ablation_fast_summary.md`, `outputs/topk_ablation_summary.md` |
| phase7-10 | Spatial model comparison, full-4500 runs, label confidence comparison, alternative label generation | Narrowed practical model family and label strategy before moving to `label_sgh` standard protocol. | `outputs/full4500_model_results_table.md`, `outputs/label_confidence_compare_results.md`, `outputs/graph_model_small_scale_topk_comparison.md` |
| phase11-13 | Balanced label setup and transition toward standard `label_sgh + spc250` protocol | Prepared later standardized comparison phases. | `outputs/phase14_after_phase13_results.md`, related `multiscale_temporal_*phase11*`, `*phase12*`, `*phase13*` run dirs |

### Phase14-Phase20: Standardized protocol on label_sgh

Shared context for most of these phases:

- Label file: `data/label_sgh.csv`
- Common protocol: topk graph comparison around `topk20`
- Standard comparison seed: mostly `202`
- Standard long run: 300 epochs with early stopping patience 30

| Phase | Goal | Best / Key Result | Notes | Source |
|---|---|---|---|---|
| phase14 | Compare full/topk graph and node feature settings for GCN/SAGE | Best recorded F1 in summary: GCN topk + ones = Acc 67.78%, F1 0.6769 | Built the immediate pre-ablation baseline. | `outputs/phase14_20_results_summary.md` |
| phase15 | Short ablation on GCN branch/fusion settings | Full > temporal_only >> spatial_only; mean fusion collapsed badly | Short-run ablation, later validated by phase17 long run. | `outputs/phase15_branch_ablation_summary.md`, `outputs/phase14_20_results_summary.md` |
| phase16 | Short ablation on SAGE side and concat variants | SAGE temporal_only remained competitive; concat underperformed | Transitional ablation phase before long-run validation. | `outputs/phase14_20_results_summary.md` |
| phase17 | Long-run branch/fusion ablation | GCN full: 70.22 / 0.6996; GCN temporal_only: 69.78 / 0.6928; SAGE temporal_only: 70.44 / 0.6983 | Confirmed temporal branch carries most signal; spatial-only is weak; concat/mean fusion are not preferred. | `outputs/phase14_20_results_summary.md`, `outputs/phase17_branch_ablation_logs/` |
| phase18 | Validate and long-run EvolveGCN | Validate run: 65.56 / 0.6561; full long run: 64.44 / 0.6458 | EvolveGCN remained weaker than GCN baseline. | `outputs/phase14_20_results_summary.md`, `outputs/phase18_evolvegcn_logs/` |
| phase19 | Short raw-flow comparison | GCN raw flow: 70.89 / 0.7041; SAGE: 67.78 / 0.6760; EvolveGCN daily: 66.44 / 0.6486 | Raw-flow node features were promising even in short runs. | `outputs/phase14_20_results_summary.md`, `outputs/phase19_rawflow_logs/` |
| phase20 | Long raw-flow comparison | GCN raw flow: 71.33 / 0.7068; SAGE raw flow: 70.44 / 0.7012 | Established the strong raw-flow GCN reference used by later temporal-model comparisons. | `outputs/phase14_20_results_summary.md`, `outputs/phase20_rawflow_logs/` |

### Phase21-Phase23: Architecture extension phase

| Phase | Goal | Key Result | Notes | Source |
|---|---|---|---|---|
| phase21 | Fill remaining long-run supplements and extend model family | Added WGCN, GAT, GINE, GRU support; EvolveGCN raw-flow daily on GPU was memory-sensitive | Also produced the fixed-GCN GRU comparison basis used in phase22. | `scripts/run_phase21_longrun_supplements.sh`, `outputs/phase21_longrun_supplement_logs/`, `/memories/repo/notes.md` |
| phase22 | Fixed GCN temporal comparison: LSTM vs GRU | LSTM: 71.33 / 0.7068; GRU: 70.00 / 0.6935 | Under fixed GCN, LSTM outperformed GRU on both Acc and F1. | `outputs/phase22_gcn_temporal_compare.md` |
| phase23 | Expand temporal family under fixed GCN + raw_temporal_mean + topk20 + static | TCN: 73.56 / 0.7294; light Transformer: 74.89 / 0.7434; BiGRU: 70.44 / 0.6973; Transformer_FULL: 69.33 / 0.6798 | Light Transformer became the best result in this family. Tuned Transformer_FULL script once failed due to unsupported CLI arg, then was repaired. | `scripts/run_phase23_gcn_temporal_triple_e300p30.sh`, `scripts/run_phase23_transformer_full_e300p30.sh`, `outputs/multiscale_temporal_20260318_165500_label_sgh_phase23_rawflow_gcn_topk_static_tcn_spc250_seed202_e300p30/metrics/test_results.json`, `outputs/multiscale_temporal_20260318_173138_label_sgh_phase23_rawflow_gcn_topk_static_transformer_spc250_seed202_e300p30/metrics/test_results.json`, `outputs/multiscale_temporal_20260318_175354_label_sgh_phase23_rawflow_gcn_topk_static_bigru_spc250_seed202_e300p30/metrics/test_results.json`, `outputs/multiscale_temporal_20260318_183428_label_sgh_phase23_rawflow_gcn_topk_static_transformer_full_spc250_seed202_e300p30/metrics/test_results.json` |

### Phase24-Phase31: Unified comparison matrix and follow-up experiments

| Phase | Goal | Key Result | Notes | Source |
|---|---|---|---|---|
| phase24 | Build the main 250x9 comparison matrix under a unified protocol | LSTM+GCN: 0.7178 / 0.7000; light Transformer+GCN: 0.7178 / 0.7050; TCN+GCN: 0.7178 / 0.7004; LSTM+SAGE: 0.7022 / 0.6989 | Main matrix for later selection and reruns. By F1, light Transformer+GCN is the best row in the GCN temporal group. | `scripts/run_phase24_250x9_matrix_e300p30.sh`, `outputs/phase24_250x9_logs/phase24_results.csv` |
| phase25 | Re-run phase24 top-3 candidates for seed sensitivity | Seed3407: TCN+GCN best at 0.7156 / 0.7199. Seed202 rerun: light Transformer+GCN best at 0.7289 / 0.7206. | Confirms ranking is seed-sensitive, but light Transformer+GCN stays highly competitive. | `scripts/run_phase25_top3_labelsgh_seed3407_e300p30.sh`, `outputs/phase25_top3_labelsgh_seed3407_logs/phase25_results.csv`, `outputs/phase25_top3_labelsgh_seed202_logs/phase25_results.csv` |
| phase26 | Transformer+GCN ablation | Transformer only: 0.7222 / 0.7190; GCN only: 0.4133 / 0.3823; concat: 0.7000 / 0.6839; gated fusion: 0.7222 / 0.7103 | In this setup, temporal branch alone was already very strong; gated fusion remained much better than concat. | `scripts/run_phase26_transformer_gcn_ablation_e300p30.sh`, `outputs/phase26_transformer_gcn_ablation_seed202_logs/phase26_results.csv` |
| phase27 | Ensure 2-layer hourly light Transformer is directly comparable to prior LSTM runs | Transformer(2 layers)+GCN: 71.33% / 0.7038 | This phase was enabled by adding `--temporal-layers` runtime override to the training entry. | `outputs/multiscale_temporal_20260319_133258_label_sgh_phase27_transformer2_gcn_spc250_seed202_e300p30/metrics/test_results.json` |
| phase28 | Follow-up on 3-layer temporal settings | LSTM(3)+GraphSAGE: 0.6667 / 0.6612; LSTM(3)+GCN: 0.7089 / 0.7016 | User later decided not to continue the already-started Transformer(3)+GraphSAGE item, so only the last two rows remain canonical. | `scripts/run_phase28_temporal3_graphsage_gcn_e300p30.sh`, `outputs/phase28_temporal3_graphsage_gcn_logs/phase28_results.csv` |
| phase29 | Compare three 4500-label edge variants using light Transformer+GCN | `half_edge` = 0.5533 / 0.5650; `max_edge` = 0.5086 / 0.5035; `no_edge` = 0.6724 / 0.6722 | All three items finished. Current canonical result files also include `topk_enabled`, `graph_topk_out`, `graph_topk_in`. | `scripts/run_phase29_transformer_gcn_4500_edge_variants_e300p30.sh`, `outputs/phase29_transformer_gcn_4500_edge_logs/phase29_results.csv` |
| phase30 | Compare GINE `flow_only` vs `flow_distance_direction` edge features under the phase24 protocol | `flow_only` = 68.22 / 0.6744; `flow_distance_direction` = 68.00 / 0.6757 | The new 4-d edge feature did not beat flow-only overall. Both improved over the earlier phase24 LSTM+GINE `ones` baseline (66.67 / 0.6619), but both remained below LSTM+GCN (71.78 / 0.7000). | `outputs/phase30_gine_edge_compare_logs/phase30_results.csv`, `outputs/phase30_gine_edge_compare_logs/phase30_summary.md` |
| phase31 | Repeat the phase30 GINE edge comparison with the temporal branch switched from LSTM to Transformer | `flow_distance_direction` = 73.56 / 0.7304; `flow_only` = 73.11 / 0.7288 | This is the first controlled result showing the current 4-d edge feature can become a net positive once the temporal branch is stronger. The best phase31 row beats both phase30 GINE variants and the phase24 light Transformer+GCN matrix row. | `scripts/run_phase31_transformer_gine_edge_followup_e300p30.sh`, `outputs/phase31_transformer_gine_edge_logs/phase31_results.csv`, `outputs/phase31_transformer_gine_edge_logs/phase31_results_table.md` |

### Planned Next Step: Direction/Distance Graph Enhancement

- User first proposed adding direction and distance to the GCN branch to improve recall and make the graph branch matter more.
- Intermediate discussion considered compact node statistics, but the latest agreed direction is to favor edge-level information because each OD edge has its own flow, distance, and orientation; averaging to node-level means risks washing out multi-modal destination structure.
- Current implementation constraint: `src/models/spatial_branch_pure_graph.py` still assumes scalar `edge_weight`, so the existing GCN branch is not the right place for multi-dimensional edge attributes.
- Therefore the current preferred implementation path is:
	- keep the current GCN line as baseline,
	- add a new GINE-based experimental line,
	- use minimal 4-dim edge features `[flow, normalized_distance, cos(theta), sin(theta)]` and let GINE log-transform only the flow channel,
	- keep the feature behind `GINE_EDGE_FEATURE_MODE=flow_distance_direction` / `--gine-edge-feature-mode flow_distance_direction`,
	- record the chosen graph edge feature mode in result metadata,
	- run smoke validation first before any full comparison.
- Current recovery point: implementation is finished and smoke-tested; the next missing step is a controlled experiment comparison.

### Next Step After Phase31

- Phase31 answers the main interaction question: under `TRANSFORMER + GINE`, the current 4-d edge feature is now a small but consistent win over `flow_only`.
- The next most defensible follow-up is therefore a robustness check, not immediate feature expansion:
	- keep `TRANSFORMER + GINE + flow_distance_direction`, `raw_temporal_mean`, `topk20/topk20`, `static`, `spc250`, `e300/p30` fixed,
	- rerun on at least one additional seed or one nearby label subset,
	- verify that the phase31 gain is stable before adding more edge channels or a larger edge encoder.
- Reason: the current gain is real but still narrow, so the first priority is to separate true signal from seed sensitivity.

## Current Best Answers By Question

| Question | Current Answer | Evidence |
|---|---|---|
| Best temporal model under fixed GCN raw-flow protocol? | Light Transformer | phase23: 74.89 / 0.7434 |
| Best row in the main phase24 unified matrix by F1? | Light Transformer+GCN | phase24: 0.7178 / 0.7050 |
| Does GRU beat LSTM under fixed GCN? | No | phase22 |
| Is gated fusion preferable to concat for Transformer+GCN? | Yes | phase26 |
| Are 3-layer LSTM follow-ups stronger than phase24 light Transformer+GCN? | No clear improvement | phase28 vs phase24 |
| Does direction+distance help under GINE? | Under LSTM, not clearly. Under Transformer, yes by a small but consistent margin. | phase30 vs phase31 |

## Files To Update First In Future Work

When new phase work is done, update in this order:

1. phase-level results CSV/MD in `outputs/...`
2. this tracker file
3. repository memory note under `/memories/repo/`

## Append Template

Copy this block for each new phase:

```md
| phaseXX | goal summary | best result or current status | short note on decisions / protocol deltas | script path, results path |
```