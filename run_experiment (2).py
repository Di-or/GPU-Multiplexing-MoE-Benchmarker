#!/usr/bin/env python3


import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description="MoE MPS Experiment — run one condition")
    p.add_argument("--condition", required=True, choices=["baseline", "mps", "spatial"])
    p.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--num-requests", type=int, default=20)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--input-seq-len", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--warmup-runs", type=int, default=3)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--profile", action="store_true",
                    help="Run workers under nsys for kernel-level profiling")
    return p.parse_args()


def manage_mps(action):
    script = PROJECT_ROOT / "setup_mps.sh"
    r = subprocess.run(["bash", str(script), action])
    time.sleep(2)


def start_gpu_monitor(output_csv):
    proc = subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu",
            "--format=csv,nounits",
            "--loop-ms=100",
        ],
        stdout=open(output_csv, "w"),
        stderr=subprocess.DEVNULL,
    )
    return proc


def stop_gpu_monitor(proc):
    if proc and proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def clean_sync_files(go_file, ready_dir):
    if os.path.exists(go_file):
        os.remove(go_file)
    if os.path.exists(ready_dir):
        shutil.rmtree(ready_dir)


def run_multiprocess(args, output_dir, gpu_monitor):
    condition = args.condition
    go_file = "/tmp/moe_go_signal"
    ready_dir = "/tmp/moe_ready"
    clean_sync_files(go_file, ready_dir)

    wall_start = time.time()

    # Launch workers
    worker_procs = []
    for wid in range(args.num_workers):
        worker_output = os.path.join(output_dir, f"worker_{wid}.json")

        cmd = [
            sys.executable, str(PROJECT_ROOT / "worker.py"),
            "--worker-id", str(wid),
            "--model", args.model,
            "--dtype", args.dtype,
            "--num-requests", str(args.num_requests),
            "--input-seq-len", str(args.input_seq_len),
            "--max-new-tokens", str(args.max_new_tokens),
            "--warmup-runs", str(args.warmup_runs),
            "--condition", condition,
            "--output-file", worker_output,
            "--go-file", go_file,
            "--ready-dir", ready_dir,
        ]

        # Optional nsys profiling
        if args.profile:
            nsys_output = os.path.join(output_dir, f"worker_{wid}_profile")
            cmd = [
                "nsys", "profile",
                "--output", nsys_output,
                "--force-overwrite", "true",
                "--trace", "cuda,nvtx",
            ] + cmd

        env = os.environ.copy()
        if condition in ("mps", "spatial"):
            env["CUDA_MPS_PIPE_DIRECTORY"] = "/tmp/nvidia-mps"
            env["CUDA_MPS_LOG_DIRECTORY"] = "/tmp/nvidia-mps-log"
        if condition == "spatial":
            env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = "50"

        print(f"[Orchestrator] Launching worker {wid}...")
        proc = subprocess.Popen(cmd, env=env, cwd=str(PROJECT_ROOT))
        worker_procs.append(proc)

    # Wait for all workers to signal ready
    print("[Orchestrator] Waiting for workers to be ready...")
    os.makedirs(ready_dir, exist_ok=True)
    deadline = time.time() + 120
    while True:
        ready_files = [f for f in os.listdir(ready_dir) if f.endswith(".ready")] if os.path.exists(ready_dir) else []
        if len(ready_files) >= args.num_workers:
            break
        for i, proc in enumerate(worker_procs):
            if proc.poll() is not None and proc.returncode != 0:
                print(f"ERROR: Worker {i} crashed during model loading (exit code {proc.returncode})")
                stop_gpu_monitor(gpu_monitor)
                if condition == "mps":
                    manage_mps("stop")
                sys.exit(1)
        if time.time() > deadline:
            print("ERROR: Timed out waiting for workers to load model.")
            for proc in worker_procs:
                proc.kill()
            stop_gpu_monitor(gpu_monitor)
            if condition == "mps":
                manage_mps("stop")
            sys.exit(1)
        time.sleep(0.5)

    # Send go-signal
    print("[Orchestrator] All workers ready. Sending go-signal.")
    open(go_file, "w").close()

    # Wait for workers to finish
    for i, proc in enumerate(worker_procs):
        retcode = proc.wait()
        if retcode != 0:
            print(f"WARNING: Worker {i} exited with code {retcode}")

    wall_end = time.time()
    clean_sync_files(go_file, ready_dir)
    return wall_start, wall_end


def run_condition(args):
    condition = args.condition
    output_dir = os.path.join(args.results_dir, condition)
    os.makedirs(output_dir, exist_ok=True)

    # MPS management
    if condition in ("mps", "spatial"):
        manage_mps("start")
    elif condition == "baseline":
        manage_mps("stop")

    # Start GPU monitoring
    gpu_csv = os.path.join(output_dir, "gpu_metrics.csv")
    gpu_monitor = start_gpu_monitor(gpu_csv)

    # Run the condition
    wall_start, wall_end = run_multiprocess(args, output_dir, gpu_monitor)

    stop_gpu_monitor(gpu_monitor)

    if condition in ("mps", "spatial"):
        manage_mps("stop")

    # Aggregate results
    all_requests = []
    worker_summaries = []
    for wid in range(args.num_workers):
        worker_file = os.path.join(output_dir, f"worker_{wid}.json")
        if os.path.exists(worker_file):
            with open(worker_file) as f:
                w = json.load(f)
                worker_summaries.append({k: v for k, v in w.items() if k != "requests"})
                all_requests.extend(w.get("requests", []))

    combined_time = 0
    combined_throughput = 0
    if all_requests:
        earliest = min(r["start_time"] for r in all_requests)
        latest = max(r["end_time"] for r in all_requests)
        combined_time = latest - earliest
        combined_throughput = len(all_requests) / combined_time if combined_time > 0 else 0

    summary = {
        "condition": condition,
        "model": args.model,
        "num_workers": args.num_workers,
        "num_requests_per_worker": args.num_requests,
        "wall_clock_sec": wall_end - wall_start,
        "combined_total_time_sec": combined_time,
        "combined_throughput_rps": combined_throughput,
        "workers": worker_summaries,
    }

    summary_file = os.path.join(output_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    args = parse_args()
    run_condition(args)