#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="outputs/phase34_gine_larger_capacity_logs"
mkdir -p "$LOG_DIR"

RESULTS_CSV="$LOG_DIR/phase34_results.csv"
RESULTS_MD="$LOG_DIR/phase34_results_table.md"

FROZEN_LABEL="data/sampled_labels_spc250_seed202_reconstructed.csv"
FROZEN_SPLIT="data/splits/sampled_labels_spc250_seed202_reconstructed_seed42_split.json"

echo "variant,temporal_model,spatial_model,edge_feature_mode,spatial_layers,spatial_hidden,acc,f1,run_dir,status" > "$RESULTS_CSV"

run_one() {
  local variant="$1"
  local spatial_layers="$2"
  local spatial_hidden="$3"
  local run_tag="$4"

  local log_file="$LOG_DIR/${run_tag}.log"

  echo "[PHASE34] start variant=${variant} layers=${spatial_layers} hidden=${spatial_hidden}"

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
    --gine-edge-feature-mode flow_distribution \
    --spatial-layers "$spatial_layers" \
    --spatial-hidden-size "$spatial_hidden" \
    --run-tag "$run_tag" \
    > "$log_file" 2>&1
  local exit_code=$?
  local exit_code=$?
  set -e

  local run_dir
  run_dir="$(ls -dt outputs/multiscale_temporal_*_${run_tag} 2>/dev/null | head -n 1 || true)"

  if [[ $exit_code -ne 0 || -z "$run_dir" || ! -f "$run_dir/metrics/test_results.json" ]]; then
    echo "[PHASE34] FAILED variant=${variant} exit=${exit_code} run_dir=${run_dir:-N/A}"
    echo "${variant},TRANSFORMER,GINE,flow_distribution,${spatial_layers},${spatial_hidden},NA,NA,${run_dir:-N/A},FAILED" >> "$RESULTS_CSV"
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

  echo "[PHASE34] done variant=${variant} acc=${acc} f1=${f1} run_dir=${run_dir}"
  echo "${variant},TRANSFORMER,GINE,flow_distribution,${spatial_layers},${spatial_hidden},${acc},${f1},${run_dir},OK" >> "$RESULTS_CSV"
}

# Run 1: larger capacity — 3 layers, hidden 256
run_one "gine_3layer_256hidden" 3 256 \
  "phase34_transformer_gine_flow_dist_3layer_256h_frozen_split_e300p30"

# Generate markdown summary
python - <<'PY'
import csv
from pathlib import Path

csv_path = Path("outputs/phase34_gine_larger_capacity_logs/phase34_results.csv")
md_path  = Path("outputs/phase34_gine_larger_capacity_logs/phase34_results_table.md")

with csv_path.open() as f:
    rows = list(csv.DictReader(f))

ok_rows = [r for r in rows if r["status"] == "OK"]
best_acc = max((float(r["acc"]) for r in ok_rows), default=None)
best_f1  = max((float(r["f1"])  for r in ok_rows), default=None)

lines = [
    "# Phase34 GINE Larger Capacity: layers=3, hidden=256",
    "",
    "Protocol: frozen sample + frozen split (spc250 seed202), Transformer temporal, topk20/20 static, flow_distribution edge feature",
    "Baseline (phase33): GINE layers=2 hidden=128 → acc=0.7156 / F1=0.7185",
    "",
    "| Variant | Layers | Hidden | Acc | F1 | Run Dir | Status |",
    "|---|---:|---:|---:|---:|---|---|",
]
# Add phase33 baseline row for reference
lines.append("| gine_2layer_128hidden (phase33 baseline) | 2 | 128 | 0.7156 | 0.7185 | — | REF |")
for r in rows:
    acc, f1 = r["acc"], r["f1"]
    if r["status"] == "OK":
        if best_acc is not None and abs(float(acc) - best_acc) < 1e-9:
            acc = f"**{acc}**"
        if best_f1 is not None and abs(float(r["f1"]) - best_f1) < 1e-9:
            f1 = f"**{f1}**"
    lines.append(f"| {r['variant']} | {r['spatial_layers']} | {r['spatial_hidden']} | {acc} | {f1} | {r['run_dir']} | {r['status']} |")

md_path.write_text("\n".join(lines) + "\n")
print(f"Wrote {md_path}")
PY

echo "[PHASE34] all runs completed. CSV: ${RESULTS_CSV}"
