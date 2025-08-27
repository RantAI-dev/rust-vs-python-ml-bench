#!/usr/bin/env python3
import argparse, json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--implementations", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    with open(args.implementations, "r") as f:
        impl = json.load(f)
    lines = ["# Implementation Report", "", json.dumps(impl, indent=2)]
    with open(args.output, "w") as f:
        f.write("\n".join(lines))

if __name__ == "__main__":
    main()

