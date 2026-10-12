#!/usr/bin/env python3
import argparse
import json
import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
 
PROMPTS_PER_WORKER = {
    0: [
        "Explain the concept of mixture of experts in machine learning.",
        "What are the main differences between transformers and recurrent neural networks?",
        "Describe how GPU memory management works for large language models.",
        "Write a short summary of the history of artificial intelligence.",
    ],
    1: [
        "How does photosynthesis work in plants and why is it important?",
        "Describe the process of nuclear fusion inside stars.",
        "What were the major causes and consequences of World War One?",
        "Explain how the human immune system responds to viral infections.",
    ],
}
 
 
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--worker-id", type=int, required=True)
    p.add_argument("--model", type=str, default="allenai/OLMoE-1B-7B-0924")
    p.add_argument("--dtype", type=str, default="float16")
    p.add_argument("--num-requests", type=int, default=20)
    p.add_argument("--input-seq-len", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--warmup-runs", type=int, default=3)
    p.add_argument("--condition", type=str, required=True, choices=["baseline", "mps", "spatial"])
    p.add_argument("--output-file", type=str, required=True)
    p.add_argument("--go-file", type=str, default="/tmp/moe_go_signal")
    p.add_argument("--ready-dir", type=str, default="/tmp/moe_ready")
    return p.parse_args()
 
 
def load_model(model_name, dtype_str, device="cuda:0"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
 
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=getattr(torch, dtype_str),
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer
 


def tokenize_prompts(tokenizer, num_requests, input_seq_len, worker_id=0):
    prompts = PROMPTS_PER_WORKER.get(worker_id, PROMPTS_PER_WORKER[0])
    inputs = []
    for i in range(num_requests):
        prompt = prompts[i % len(prompts)]
        tokens = tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=input_seq_len,
            truncation=True,
        )
        inputs.append(tokens)
    return inputs


def wait_for_go(go_file, ready_dir, worker_id):
    os.makedirs(ready_dir, exist_ok=True)
    open(os.path.join(ready_dir, f"worker_{worker_id}.ready"), "w").close()
 
    while not os.path.exists(go_file):
        time.sleep(0.05)
 
 
def run_inference(args):
    device = "cuda:0"
    model, tokenizer = load_model(args.model, args.dtype, device)
    inputs = tokenize_prompts(tokenizer, args.num_requests + args.warmup_runs,
                              args.input_seq_len, args.worker_id)
 
    wait_for_go(args.go_file, args.ready_dir, args.worker_id)
 
    # Warmup
    for i in range(args.warmup_runs):
        inp = {k: v.to(device) for k, v in inputs[i].items()}
        with torch.no_grad():
            model.generate(**inp, max_new_tokens=args.max_new_tokens, do_sample=False)
    torch.cuda.synchronize()
 
    # Timed inference
    requests = []
    for i in range(args.num_requests):
        inp = {k: v.to(device) for k, v in inputs[args.warmup_runs + i].items()}
 
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        with torch.no_grad():
            model.generate(**inp, max_new_tokens=args.max_new_tokens, do_sample=False)
        torch.cuda.synchronize()
        t_end = time.perf_counter()
 
        requests.append({
            "request_id": i,
            "latency_sec": t_end - t_start,
            "start_time": t_start,
            "end_time": t_end,
        })
 
    latencies = [r["latency_sec"] for r in requests]
    earliest = min(r["start_time"] for r in requests)
    latest = max(r["end_time"] for r in requests)
    total_time = latest - earliest
 
    result = {
        "worker_id": args.worker_id,
        "condition": args.condition,
        "num_requests": args.num_requests,
        "mean_latency_sec": sum(latencies) / len(latencies),
        "total_time_sec": total_time,
        "throughput_rps": len(requests) / total_time if total_time > 0 else 0,
        "requests": requests,
    }
 
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(result, f, indent=2)
 
    print(f"[Worker {args.worker_id}] Done")
 
 
if __name__ == "__main__":
    run_inference(parse_args())