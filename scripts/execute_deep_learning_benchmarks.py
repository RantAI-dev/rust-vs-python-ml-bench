#!/usr/bin/env python3
import argparse, json, subprocess, sys, os

def run(cmd):
    subprocess.run(cmd, check=True)

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

    # CNN (synthetic)
    run([
        sys.executable,
        os.path.join(root, "src/python/deep_learning/cnn_benchmark.py"),
        "--mode", "training",
        "--dataset", "synthetic",
        "--architecture", "simple_cnn",
        "--hyperparams", json.dumps({"epochs": 1, "batch_size": 16}),
        "--run-id", run_id,
        "--output-dir", out_dir
    ])

    # RNN (synthetic)
    run([
        sys.executable,
        os.path.join(root, "src/python/deep_learning/rnn_benchmark.py"),
        "--mode", "training",
        "--dataset", "synthetic",
        "--architecture", "gru",
        "--seq-len", "32",
        "--input-size", "16",
        "--hidden-size", "32",
        "--num-layers", "1",
        "--num-classes", "4",
        "--epochs", "1",
        "--batch-size", "32",
        "--run-id", run_id,
        "--output-dir", out_dir
    ])

    results = {"deep_learning": sorted([
        os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith("_results.json")
    ])}
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()

