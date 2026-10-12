#!/usr/bin/env python3

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def load_condition(result_dir):
    """Load summary and per-request data from a condition's result directory."""
    summary_file = os.path.join(result_dir, "summary.json")

    with open(summary_file) as f:
        summary = json.load(f)

    all_requests = []
    worker_files = sorted(f for f in os.listdir(result_dir) if f.startswith("worker_") and f.endswith(".json"))
    for wf in worker_files:
        with open(os.path.join(result_dir, wf)) as f:
            w = json.load(f)
            for r in w.get("requests", []):
                r["worker_id"] = w["worker_id"]
                all_requests.append(r)

    summary["all_requests"] = all_requests

    gpu_csv = os.path.join(result_dir, "gpu_metrics.csv")
    gpu_util = None

    df = pd.read_csv(gpu_csv)
    df.columns = df.columns.str.strip()
    util_col = [c for c in df.columns if "utilization" in c.lower() and "memory" not in c.lower()]
    if util_col:
        gpu_util = pd.to_numeric(df[util_col[0]], errors="coerce").mean()

    summary["mean_gpu_utilization_pct"] = gpu_util
    return summary


def get_mean_latency(data):
    lats = [r["latency_sec"] for r in data["all_requests"]]
    return sum(lats) / len(lats) if lats else 0


def print_comparison(conditions):
    names = list(conditions.keys())
    data = list(conditions.values())

    col_width = 14
    # Build header with change columns for each non-baseline condition
    header = f"  {'Metric':<30}" + "".join(f"{n:>{col_width}}" for n in names)
    for n in names[1:]:
        short = n[:4]
        header += f"  {short + ' vs Base':>12}"

    print(f"\n{'=' * len(header)}")
    print(f"  MoE MPS Experiment — Comparison")
    print(f"{'=' * len(header)}")
    print(header)
    print(f"  {'-' * (len(header) - 2)}")

    def print_metric_row(label, values, fmt=".4f"):
        row = f"  {label:<30}" + "".join(f"{v:>{col_width}{fmt}}" for v in values)
        for v in values[1:]:
            if values[0] > 0:
                row += f"  {(v-values[0])/values[0]*100:>+11.1f}%"
        print(row)

    # Mean latency
    lats = [get_mean_latency(d) for d in data]
    print_metric_row("Mean latency (sec)", lats)

    # Throughput
    thrs = [d["combined_throughput_rps"] for d in data]
    print_metric_row("Combined throughput (req/s)", thrs, ".2f")

    # GPU utilization
    gpus = [d.get("mean_gpu_utilization_pct") for d in data]
    if all(g is not None for g in gpus):
        row = f"  {'Mean GPU utilization (%)':<30}" + "".join(f"{g:>{col_width-1}.1f}%" for g in gpus)
        for g in gpus[1:]:
            row += f"  {g-gpus[0]:>+11.1f}%"
        print(row)

    print()



def make_chart(conditions, output_dir):
    names = list(conditions.keys())
    data = list(conditions.values())

    colors = ["#4878cf", "#e8793a", "#9b59b6", "#6ab04c", "#e74c3c"][:len(names)]

    lats = [get_mean_latency(d) for d in data]
    thrs = [d["combined_throughput_rps"] for d in data]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(names, lats, color=colors)
    axes[0].set_title("Mean Request Latency")
    axes[0].set_ylabel("Seconds")

    axes[1].bar(names, thrs, color=colors)
    axes[1].set_title("Combined Throughput")
    axes[1].set_ylabel("Requests/sec")

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "comparison.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Analysis] Chart saved to {chart_path}")


def main():
    p = argparse.ArgumentParser(description="Compare experiment conditions")
    p.add_argument("--baseline", type=str, default=None)
    p.add_argument("--mps", type=str, default=None)
    p.add_argument("--spatial", type=str, default=None)
    p.add_argument("--results-dir", type=str, default="results")
    args = p.parse_args()

    conditions = {}

    baseline_dir = args.baseline or os.path.join(args.results_dir, "baseline")
    mps_dir = args.mps or os.path.join(args.results_dir, "mps")
    spatial_dir = args.spatial or os.path.join(args.results_dir, "spatial")

    if os.path.isdir(baseline_dir):
        conditions["Baseline"] = load_condition(baseline_dir)

    if os.path.isdir(mps_dir):
        conditions["MPS"] = load_condition(mps_dir)

    if os.path.isdir(spatial_dir):
        conditions["Spatial"] = load_condition(spatial_dir)

    print_comparison(conditions)
    make_chart(conditions, args.results_dir)


if __name__ == "__main__":
    main()