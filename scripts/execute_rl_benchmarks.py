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

    # RL DQN (CartPole)
    subprocess.run([
        sys.executable,
        os.path.join(root, "src/python/reinforcement_learning/dqn_benchmark.py"),
        "--mode", "training",
        "--environment", "CartPole-v1",
        "--algorithm", "dqn",
        "--hyperparams", json.dumps({"total_timesteps": 1000}),
        "--run-id", run_id,
        "--output-dir", out_dir
    ], check=True)

    results = {"rl": sorted([
        os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith("_results.json")
    ])}
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()

