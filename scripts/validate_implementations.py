#!/usr/bin/env python3
import argparse, json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", required=True)
    ap.add_argument("--rust", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    with open(args.python, "r") as f:
        py = json.load(f)
    with open(args.rust, "r") as f:
        rs = json.load(f)
    validated = {"python": py.get("python", {}), "rust": rs.get("rust", {})}
    with open(args.output, "w") as f:
        json.dump(validated, f, indent=2)

if __name__ == "__main__":
    main()

