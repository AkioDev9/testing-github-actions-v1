#!/usr/bin/env python3
"""
generate_report_tiny.py
Demo ultra-rápida: genera SOLO stats (sin lista de eventos).

Uso:
  python generate_report_tiny.py
  python generate_report_tiny.py --out data/report.json --events 5000 --seed 123
"""

import argparse, json, os, random
from collections import Counter
from datetime import datetime, timezone

CATEGORIES = ["auth", "payments", "search", "profile", "orders", "support"]
REGIONS = ["NA", "SA", "EU", "AS", "AF", "OC"]

def utc_now_iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def ensure_parent_dir(path: str):
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/report.json")
    p.add_argument("--events", type=int, default=2000)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()

def main():
    args = parse_args()
    seed = args.seed if args.seed is not None else random.randint(1, 10_000_000)
    rng = random.Random(seed)

    status = Counter()
    cats = Counter()
    regs = Counter()

    total_dur = 0
    min_dur = 10**9
    max_dur = 0

    # Demo: duración rápida (sin gauss, sin outliers complejos)
    for _ in range(args.events):
        cat = rng.choice(CATEGORIES)
        reg = rng.choice(REGIONS)

        dur = 50 + rng.randint(0, 400)  # simple
        err = (rng.random() < (0.06 if cat in ("payments", "orders") else 0.03))

        st = "error" if err else "ok"
        status[st] += 1
        cats[cat] += 1
        regs[reg] += 1

        total_dur += dur
        if dur < min_dur: min_dur = dur
        if dur > max_dur: max_dur = dur

    n = args.events
    errors = status["error"]
    payload = {
        "meta": {"generated_at_utc": utc_now_iso(), "seed": seed, "events": n},
        "stats": {
            "counts": {"ok": status["ok"], "error": errors, "error_rate": round(errors / n, 4)},
            "duration_ms": {"avg": round(total_dur / n, 2), "min": min_dur, "max": max_dur},
            "top": {"categories": cats.most_common(3), "regions": regs.most_common(3)},
        },
    }

    ensure_parent_dir(args.out)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"OK: events={n} error_rate={payload['stats']['counts']['error_rate']} avg_ms={payload['stats']['duration_ms']['avg']}")
    print(f"JSON written to: {args.out}")

if __name__ == "__main__":
    main()
