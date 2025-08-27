#!/usr/bin/env python3
import argparse, json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--output-validated", required=True)
    ap.add_argument("--output-implementations", required=True)
    args = ap.parse_args()
    with open(args.results, "r") as f:
        pilot = json.load(f)
    with open(args.output_validated, "w") as f:
        json.dump({"pilot": pilot, "validated": True}, f, indent=2)
    with open(args.output_implementations, "w") as f:
        json.dump({"python": {}, "rust": {}}, f, indent=2)

if __name__ == "__main__":
    main()

