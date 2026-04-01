#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="outputs/phase33_gine_edge_distribution_logs"
mkdir -p "$LOG_DIR"

RESULTS_CSV="$LOG_DIR/phase33_results.csv"
RESULTS_MD="$LOG_DIR/phase33_results_table.md"

FROZEN_LABEL="data/sampled_labels_spc250_seed202_reconstructed.csv"
FROZEN_SPLIT="data/splits/sampled_labels_spc250_seed202_reconstructed_seed42_split.json"

echo "variant,temporal_model,spatial_model,edge_feature_mode,acc,f1,run_dir,status" > "$RESULTS_CSV"

run_one() {
  local variant="$1"
  local edge_mode="$2"
  local run_tag="$3"

  local log_file="$LOG_DIR/${run_tag}.log"

  echo "[PHASE33] start variant=${variant} edge_mode=${edge_mode}"

  set +e
  python train_multiscale_temporal.py \
    --label-path "$FROZEN_LABEL" \
    --split-manifest "$FROZEN_SPLIT" \
    --num-epochs 300 \
    --early-stopping-patience 30 \
    --spatial-model GINE \
    --temporal-model TRANSFORMER \
    --spatial-node-feature-mode raw_temporal_mean \
    --graph-topk-out 20 \
    --graph-topk-in 20 \
    --graph-temporal-mode static \
    --gine-edge-feature-mode "$edge_mode" \
    --run-tag "$run_tag" \
    > "$log_file" 2>&1
  local exit_code=$?
  set -e

  local run_dir
  run_dir="$(ls -dt outputs/multiscale_temporal_*_${run_tag} 2>/dev/null | head -n 1 || true)"

  if [[ $exit_code -ne 0 || -z "$run_dir" || ! -f "$run_dir/metrics/test_results.json" ]]; then
    echo "[PHASE33] FAILED variant=${variant} exit=${exit_code} run_dir=${run_dir:-N/A}"
    echo "${variant},TRANSFORMER,GINE,${edge_mode},NA,NA,${run_dir:-N/A},FAILED" >> "$RESULTS_CSV"
    return
  fi

  local acc f1
  acc="$(python - <<PY
import json
with open('${run_dir}/metrics/test_results.json') as f:
    d = json.load(f)
print(f"{float(d['test_accuracy'])/100.0:.4f}")
PY
)"
  f1="$(python - <<PY
import json
with open('${run_dir}/metrics/test_results.json') as f:
    d = json.load(f)
print(f"{float(d['test_f1']):.4f}")
PY
)"

  echo "[PHASE33] done variant=${variant} acc=${acc} f1=${f1} run_dir=${run_dir}"
  echo "${variant},TRANSFORMER,GINE,${edge_mode},${acc},${f1},${run_dir},OK" >> "$RESULTS_CSV"
}

# Run 1: new flow_distribution edge feature
run_one "gine_flow_distribution" "flow_distribution" \
  "phase33_transformer_gine_flow_distribution_frozen_split_e300p30"

# Run 2: existing flow_distance_direction (baseline for comparison)
run_one "gine_flow_distance_direction" "flow_distance_direction" \
  "phase33_transformer_gine_flowdistdir_frozen_split_e300p30"

# Generate markdown summary
python - <<'PY'
import csv
from pathlib import Path

csv_path = Path("outputs/phase33_gine_edge_distribution_logs/phase33_results.csv")
md_path  = Path("outputs/phase33_gine_edge_distribution_logs/phase33_results_table.md")

with csv_path.open() as f:
    rows = list(csv.DictReader(f))

ok_rows = [r for r in rows if r["status"] == "OK"]
best_acc = max((float(r["acc"]) for r in ok_rows), default=None)
best_f1  = max((float(r["f1"])  for r in ok_rows), default=None)

lines = [
    "# Phase33 GINE: flow_distribution vs flow_distance_direction",
    "",
    "Protocol: frozen sample + frozen split (spc250 seed202), Transformer temporal, topk20/20 static",
    "",
    "| Variant | Edge Feature | Acc | F1 | Run Dir | Status |",
    "|---|---|---:|---:|---|---|",
]
for r in rows:
    acc, f1 = r["acc"], r["f1"]
    if r["status"] == "OK":
        if best_acc is not None and abs(float(acc) - best_acc) < 1e-9:
            acc = f"**{acc}**"
        if best_f1 is not None and abs(float(r["f1"]) - best_f1) < 1e-9:
            f1 = f"**{f1}**"
    lines.append(f"| {r['variant']} | {r['edge_feature_mode']} | {acc} | {f1} | {r['run_dir']} | {r['status']} |")

md_path.write_text("\n".join(lines) + "\n")
print(f"Wrote {md_path}")
PY

echo "[PHASE33] all runs completed. CSV: ${RESULTS_CSV}"
