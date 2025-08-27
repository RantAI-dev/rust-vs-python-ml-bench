#!/usr/bin/env python3
import argparse, json, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    with open(args.configs, "r") as f:
        cfg = json.load(f)

    available = {"python": {"classical_ml": {}, "deep_learning": {}, "llm": {}},
                 "rust": {"classical_ml": {}, "deep_learning": {}, "llm": {}}}

    # Accept validator output shape: cfg["python"]["valid"] is a list of {category, framework, config}
    if isinstance(cfg.get("python"), dict) and isinstance(cfg["python"].get("valid"), list):
        for item in cfg["python"]["valid"]:
            cat = item.get("category")
            fw = item.get("framework")
            if cat in available["python"] and fw:
                available["python"][cat][fw] = {"available": True, "functional": True, "version": "n/a", "config": item.get("config", {})}
    if isinstance(cfg.get("rust"), dict) and isinstance(cfg["rust"].get("valid"), list):
        for item in cfg["rust"]["valid"]:
            cat = item.get("category")
            fw = item.get("framework")
            if cat in available["rust"] and fw:
                available["rust"][cat][fw] = {"available": True, "functional": True, "version": "n/a", "config": item.get("config", {})}

    with open(args.output, "w") as f:
        json.dump(available, f, indent=2)

    # Always succeed in local profile
    sys.exit(0)

if __name__ == "__main__":
    main()