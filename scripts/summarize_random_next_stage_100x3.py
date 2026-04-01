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


def parse_candidate_seed(run_dir_name: str):
    candidate = "unknown"
    if "_d2_topk_ones_" in run_dir_name:
        candidate = "d2_topk_ones"
    elif "_d4_topk_temporal_mean_" in run_dir_name:
        candidate = "d4_topk_temporal_mean"

    seed = None
    token = "_seed"
    if token in run_dir_name:
        try:
            seed = int(run_dir_name.split(token)[-1])
        except Exception:
            seed = None

    return candidate, seed


def collect(outputs_dir: Path) -> pd.DataFrame:
    rows = []
    for run_dir in sorted(outputs_dir.glob("multiscale_temporal_*_phase3_random_100x3_*")):
        test_json = run_dir / "metrics" / "test_results.json"
        data = read_json(test_json)
        if not data:
            continue

        candidate, seed = parse_candidate_seed(run_dir.name)
        rows.append(
            {
                "run_dir": run_dir.name,
                "candidate": candidate,
                "seed": seed,
                "test_accuracy": data.get("test_accuracy"),
                "test_f1": data.get("test_f1"),
                "graph_2021_edges": data.get("data_info", {}).get("graph_2021_edges"),
                "graph_2024_edges": data.get("data_info", {}).get("graph_2024_edges"),
                "node_feature_mode": data.get("model_architecture", {}).get("spatial_branch", {}).get("node_feature_mode"),
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(["candidate", "seed"]).reset_index(drop=True)


def winner_line(agg: pd.DataFrame) -> str:
    if agg.empty:
        return "No winner: empty summary."
    # Rank by mean F1 first, then mean accuracy.
    ranked = agg.sort_values(["test_f1_mean", "test_accuracy_mean"], ascending=False).reset_index(drop=True)
    top = ranked.iloc[0]
    return (
        f"Winner by mean F1: {top['candidate']} "
        f"(acc={float(top['test_accuracy_mean']):.2f}%+-{float(top['test_accuracy_std']):.2f}, "
        f"f1={float(top['test_f1_mean']):.4f}+-{float(top['test_f1_std']):.4f})"
    )


def main():
    outputs_dir = Path("outputs")
    out_csv = outputs_dir / "random_next_stage_100x3_summary.csv"
    out_md = outputs_dir / "random_next_stage_100x3_summary.md"

    df = collect(outputs_dir)
    if df.empty:
        out_csv.write_text("", encoding="utf-8")
        out_md.write_text("# Random Next-Stage 100x3 Summary\n\nNo completed runs found.\n", encoding="utf-8")
        print("No completed runs found.")
        return

    df.to_csv(out_csv, index=False)

    agg = (
        df.groupby("candidate")[["test_accuracy", "test_f1"]]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg.columns = [
        "candidate",
        "test_accuracy_mean",
        "test_accuracy_std",
        "test_accuracy_count",
        "test_f1_mean",
        "test_f1_std",
        "test_f1_count",
    ]

    lines = ["# Random Next-Stage 100x3 Summary", ""]
    lines.append("| Run | Candidate | Seed | Node Feature | Test Acc (%) | Test F1 | Edges(2021/2024) |")
    lines.append("|---|---|---:|---|---:|---:|---|")
    for _, r in df.iterrows():
        lines.append(
            f"| {r['run_dir']} | {r['candidate']} | {int(r['seed']) if pd.notna(r['seed']) else -1} | {r['node_feature_mode']} | "
            f"{float(r['test_accuracy']):.2f} | {float(r['test_f1']):.4f} | "
            f"{int(r['graph_2021_edges'])}/{int(r['graph_2024_edges'])} |"
        )

    lines.append("")
    lines.append("## Aggregate by Candidate (mean +- std)")
    lines.append("")
    for _, r in agg.iterrows():
        lines.append(
            f"- {r['candidate']}: "
            f"acc={float(r['test_accuracy_mean']):.2f}%+-{float(r['test_accuracy_std']):.2f} "
            f"(n={int(r['test_accuracy_count'])}), "
            f"f1={float(r['test_f1_mean']):.4f}+-{float(r['test_f1_std']):.4f} "
            f"(n={int(r['test_f1_count'])})"
        )

    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(f"- {winner_line(agg)}")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_md}")
    print(df.to_string(index=False))
    print("\n" + agg.to_string(index=False))
    print("\n" + winner_line(agg))


if __name__ == "__main__":
    main()
