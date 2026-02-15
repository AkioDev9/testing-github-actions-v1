#!/usr/bin/env python3
"""
generate_report.py
Genera un JSON con datos sintéticos + estadísticas relevantes.

Uso:
  python generate_report.py
  python generate_report.py --out data/report.json --events 500 --seed 123
  python generate_report.py --out data/report.json --append-history data/history.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


# ----------------------------- Datos -----------------------------

CATEGORIES = ["auth", "payments", "search", "profile", "orders", "support"]
REGIONS = ["NA", "SA", "EU", "AS", "AF", "OC"]
USER_PREFIXES = ["u", "dev", "qa", "ops", "bot"]


@dataclass
class Event:
    id: str
    ts: str                 # ISO 8601 UTC
    user: str
    category: str
    region: str
    duration_ms: int
    status: str             # "ok" | "error"
    bytes_out: int


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def random_user(rng: random.Random) -> str:
    return f"{rng.choice(USER_PREFIXES)}-{rng.randint(1, 500):03d}"


def generate_events(n: int, rng: random.Random) -> List[Event]:
    events: List[Event] = []
    base_ts = datetime.now(timezone.utc)

    # Pesos para que algunas categorías aparezcan más
    cat_weights = [22, 12, 28, 14, 18, 6]  # suma no importa, son pesos
    region_weights = [18, 22, 20, 25, 7, 8]

    for i in range(n):
        # Spread temporal (últimas ~24h) para que se vea “real”
        seconds_ago = rng.randint(0, 24 * 60 * 60)
        ts = (base_ts.replace(microsecond=0) - timedelta_seconds(seconds_ago)).isoformat().replace("+00:00", "Z")

        category = rng.choices(CATEGORIES, weights=cat_weights, k=1)[0]
        region = rng.choices(REGIONS, weights=region_weights, k=1)[0]

        # Duración: base por categoría + ruido, con cola larga ocasional (outliers)
        base = {
            "auth": 120,
            "payments": 260,
            "search": 90,
            "profile": 140,
            "orders": 210,
            "support": 160,
        }[category]

        jitter = int(abs(rng.gauss(0, 60)))
        duration_ms = max(10, base + jitter)

        # Outliers (simula picos)
        if rng.random() < 0.03:
            duration_ms *= rng.randint(3, 10)

        # Errores (un poco más en payments/orders)
        err_prob = 0.03
        if category in ("payments", "orders"):
            err_prob = 0.06
        status = "error" if rng.random() < err_prob else "ok"

        # bytes_out (aprox correlacionado con duración, con ruido)
        bytes_out = max(100, int(duration_ms * rng.uniform(8, 30)))

        ev = Event(
            id=f"evt-{i+1:06d}",
            ts=ts,
            user=random_user(rng),
            category=category,
            region=region,
            duration_ms=duration_ms,
            status=status,
            bytes_out=bytes_out,
        )
        events.append(ev)

    return events


def timedelta_seconds(seconds: int):
    # helper para no importar timedelta explícitamente arriba
    from datetime import timedelta
    return timedelta(seconds=seconds)


# ----------------------------- Stats -----------------------------

def percentile(sorted_vals: List[int], p: float) -> Optional[float]:
    """
    Percentil por interpolación lineal (p en [0,100]).
    Retorna None si lista vacía.
    """
    if not sorted_vals:
        return None
    if p <= 0:
        return float(sorted_vals[0])
    if p >= 100:
        return float(sorted_vals[-1])

    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_vals[int(k)])
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return float(d0 + d1)


def compute_stats(events: List[Event]) -> Dict[str, Any]:
    n = len(events)
    durations = [e.duration_ms for e in events]
    durations_sorted = sorted(durations)
    bytes_out = [e.bytes_out for e in events]
    statuses = Counter(e.status for e in events)
    cats = Counter(e.category for e in events)
    regions = Counter(e.region for e in events)

    errors = statuses.get("error", 0)
    ok = statuses.get("ok", 0)
    error_rate = (errors / n) if n else 0.0

    # stats por categoría
    by_cat: Dict[str, Dict[str, Any]] = {}
    bucket: Dict[str, List[int]] = defaultdict(list)
    bucket_err: Dict[str, int] = defaultdict(int)

    for e in events:
        bucket[e.category].append(e.duration_ms)
        if e.status == "error":
            bucket_err[e.category] += 1

    for cat, vals in bucket.items():
        vals_sorted = sorted(vals)
        by_cat[cat] = {
            "count": len(vals),
            "avg_duration_ms": round(sum(vals) / len(vals), 2),
            "p95_duration_ms": percentile(vals_sorted, 95),
            "errors": bucket_err.get(cat, 0),
            "error_rate": round(bucket_err.get(cat, 0) / len(vals), 4),
        }

    stats = {
        "counts": {
            "events": n,
            "ok": ok,
            "error": errors,
            "error_rate": round(error_rate, 4),
        },
        "duration_ms": {
            "min": min(durations) if durations else None,
            "max": max(durations) if durations else None,
            "avg": round(sum(durations) / n, 2) if n else None,
            "median": statistics.median(durations) if n else None,
            "stdev": round(statistics.pstdev(durations), 2) if n else None,
            "p90": percentile(durations_sorted, 90),
            "p95": percentile(durations_sorted, 95),
            "p99": percentile(durations_sorted, 99),
        },
        "bytes_out": {
            "total": sum(bytes_out) if bytes_out else 0,
            "avg": round(sum(bytes_out) / n, 2) if n else None,
            "min": min(bytes_out) if bytes_out else None,
            "max": max(bytes_out) if bytes_out else None,
        },
        "top": {
            "categories": cats.most_common(3),
            "regions": regions.most_common(3),
        },
        "breakdown": {
            "categories": dict(cats),
            "regions": dict(regions),
        },
        "by_category": by_cat,
    }
    return stats


# ----------------------------- IO -----------------------------

def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def write_json(path: str, payload: Dict[str, Any]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def append_jsonl(path: str, payload: Dict[str, Any]) -> None:
    ensure_parent_dir(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def print_summary(payload: Dict[str, Any]) -> None:
    stats = payload["stats"]
    counts = stats["counts"]
    dur = stats["duration_ms"]
    top = stats["top"]
    print("=== Report Summary ===")
    print(f"generated_at: {payload['meta']['generated_at_utc']}")
    print(f"events: {counts['events']} | ok: {counts['ok']} | error: {counts['error']} | error_rate: {counts['error_rate']}")
    print(f"duration(ms): avg={dur['avg']} median={dur['median']} p95={dur['p95']} p99={dur['p99']} min={dur['min']} max={dur['max']}")
    print(f"bytes_out: total={stats['bytes_out']['total']} avg={stats['bytes_out']['avg']}")
    print(f"top categories: {top['categories']}")
    print(f"top regions: {top['regions']}")


# ----------------------------- Main -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Genera JSON con eventos sintéticos y estadísticas.")
    p.add_argument("--out", default="data/report.json", help="Ruta del JSON de salida.")
    p.add_argument("--events", type=int, default=300, help="Cantidad de eventos a generar.")
    p.add_argument("--seed", type=int, default=None, help="Seed para reproducibilidad (si no, se randomiza).")
    p.add_argument("--append-history", default=None, help="Si lo defines, guarda también una línea JSON por corrida (JSONL).")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    seed = args.seed if args.seed is not None else random.randint(1, 10_000_000)
    rng = random.Random(seed)

    events = generate_events(args.events, rng)
    stats = compute_stats(events)

    payload = {
        "meta": {
            "generated_at_utc": utc_now_iso(),
            "seed": seed,
            "events_requested": args.events,
            "events_generated": len(events),
        },
        "stats": stats,
        "events": [asdict(e) for e in events],
    }

    write_json(args.out, payload)
    if args.append_history:
        # guardamos solo meta+stats en histórico para que no crezca demasiado
        append_jsonl(args.append_history, {"meta": payload["meta"], "stats": payload["stats"]})

    print_summary(payload)
    print(f"\nJSON written to: {args.out}")
    if args.append_history:
        print(f"History appended to: {args.append_history}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
