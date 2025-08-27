#!/usr/bin/env python3
import argparse, json, subprocess, sys, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--implementations", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(__file__))
    out_dir = os.path.join(root, "results")
    os.makedirs(out_dir, exist_ok=True)
    run_id = "phase4"

    # LLM inference (distilgpt2 small)
    subprocess.run([
        sys.executable,
        os.path.join(root, "src/python/llm/transformer_benchmark.py"),
        "--mode", "inference",
        "--model-name", "distilgpt2",
        "--task", "text-generation",
        "--hyperparams", json.dumps({"batch_sizes": [1], "sequence_lengths": [64], "prompts": ["Hello world!"]}),
        "--run-id", run_id,
        "--output-dir", out_dir
    ], check=True)

    results = {"llm": sorted([
        os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith("_results.json")
    ])}
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()

