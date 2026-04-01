from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def parse_mode_seed(run_dir_name: str):
    mode = "unknown"
    if "_topk_" in run_dir_name:
        mode = "topk"
    elif "_full_" in run_dir_name:
        mode = "full"

    seed = None
    token = "_seed"
    if token in run_dir_name:
        try:
            seed = int(run_dir_name.split(token)[-1])
        except Exception:
            seed = None

    return mode, seed


def collect(outputs_dir: Path) -> pd.DataFrame:
    rows = []
    for run_dir in sorted(outputs_dir.glob("multiscale_temporal_*_phase2_dir12_random_*")):
        test_json = run_dir / "metrics" / "test_results.json"
        data = read_json(test_json)
        if not data:
            continue

        mode, seed = parse_mode_seed(run_dir.name)
        rows.append(
            {
                "run_dir": run_dir.name,
                "mode": mode,
                "seed": seed,
                "test_accuracy": data.get("test_accuracy"),
                "test_f1": data.get("test_f1"),
                "graph_2021_edges": data.get("data_info", {}).get("graph_2021_edges"),
                "graph_2024_edges": data.get("data_info", {}).get("graph_2024_edges"),
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(["mode", "seed"]).reset_index(drop=True)


def main():
    outputs_dir = Path("outputs")
    out_csv = outputs_dir / "dir12_random_fast_summary.csv"
    out_md = outputs_dir / "dir12_random_fast_summary.md"

    df = collect(outputs_dir)
    if df.empty:
        out_csv.write_text("", encoding="utf-8")
        out_md.write_text("# Direction1/2 Random Fast Summary\n\nNo completed runs found.\n", encoding="utf-8")
        print("No completed runs found.")
        return

    df.to_csv(out_csv, index=False)

    lines = ["# Direction1/2 Random Fast Summary", ""]
    lines.append("| Run | Mode | Seed | Test Acc (%) | Test F1 | Edges(2021/2024) |")
    lines.append("|---|---|---:|---:|---:|---|")
    for _, r in df.iterrows():
        lines.append(
            f"| {r['run_dir']} | {r['mode']} | {int(r['seed']) if pd.notna(r['seed']) else -1} | "
            f"{float(r['test_accuracy']):.2f} | {float(r['test_f1']):.4f} | "
            f"{int(r['graph_2021_edges'])}/{int(r['graph_2024_edges'])} |"
        )

    agg = df.groupby("mode")[["test_accuracy", "test_f1"]].mean().reset_index()
    lines.append("")
    lines.append("## Mean by Mode")
    lines.append("")
    for _, r in agg.iterrows():
        lines.append(
            f"- {r['mode']}: mean_acc={float(r['test_accuracy']):.2f}%, mean_f1={float(r['test_f1']):.4f}"
        )

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_md}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
