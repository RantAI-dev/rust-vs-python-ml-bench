#!/usr/bin/env python3
import argparse, json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    with open(args.results, "r") as f:
        res = json.load(f)
    with open(args.output, "w") as f:
        f.write("# Pilot Report\n\n")
        f.write(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()

