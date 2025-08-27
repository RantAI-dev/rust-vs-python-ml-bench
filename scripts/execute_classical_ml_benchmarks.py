#!/usr/bin/env python3
import argparse, json, subprocess, sys, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--implementations", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    # Minimal: run Python regression and SVM benchmarks with small configs
    root = os.path.dirname(os.path.dirname(__file__))
    run_id = "phase4"
    out_dir = os.path.join(root, "results")
    os.makedirs(out_dir, exist_ok=True)
    # Run regression
    subprocess.run([
        sys.executable,
        os.path.join(root, "src/python/classical_ml/regression_benchmark.py"),
        "--mode", "training",
        "--dataset", "synthetic_linear",
        "--algorithm", "linear",
        "--hyperparams", "{}",
        "--run-id", run_id,
        "--output-dir", out_dir
    ], check=True)
    # Run SVM
    subprocess.run([
        sys.executable,
        os.path.join(root, "src/python/classical_ml/svm_benchmark.py"),
        "--mode", "training",
        "--dataset", "iris",
        "--algorithm", "svc",
        "--hyperparams", "{\"C\":1.0,\"kernel\":\"linear\"}",
        "--run-id", run_id,
        "--output-dir", out_dir
    ], check=True)
    # Also run Rust regression benchmark
    try:
        # Build and run rust regression_benchmark
        subprocess.run(["cargo", "build", "-p", "regression_benchmark", "--quiet"], cwd=root, check=True)
        rust_bin = os.path.join(root, "target", "debug", "regression_benchmark")
        subprocess.run([
            rust_bin,
            "--mode", "training",
            "--dataset", "synthetic_linear",
            "--algorithm", "linear",
            "--hyperparams", "{}",
            "--run-id", run_id,
            "--output-dir", out_dir
        ], check=True)
    except Exception:
        pass

    # Collect file paths only
    results = {"classical_ml": sorted([
        os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith("_results.json")
    ])}
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()

