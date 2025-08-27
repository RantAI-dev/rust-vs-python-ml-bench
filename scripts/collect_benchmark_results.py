#!/usr/bin/env python3
import argparse, json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classical-ml", required=True)
    ap.add_argument("--deep-learning", required=True)
    ap.add_argument("--rl", required=True)
    ap.add_argument("--llm", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    with open(args.classical_ml, "r") as f:
        classical_ml = json.load(f)
    with open(args.deep_learning, "r") as f:
        deep_learning = json.load(f)
    with open(args.rl, "r") as f:
        rl = json.load(f)
    with open(args.llm, "r") as f:
        llm = json.load(f)

    all_results = {
        "classical_ml": classical_ml.get("classical_ml", []),
        "deep_learning": deep_learning.get("deep_learning", []),
        "rl": rl.get("rl", []),
        "llm": llm.get("llm", []),
    }
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)

if __name__ == "__main__":
    main()

