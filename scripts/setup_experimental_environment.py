#!/usr/bin/env python3
import argparse, json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--implementations", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    # Minimal env config
    env = {"use_gpu": False, "batch_sizes": [1, 10, 100]}
    with open(args.output, "w") as f:
        json.dump(env, f, indent=2)

if __name__ == "__main__":
    main()

