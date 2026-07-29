#!/usr/bin/env python3
"""
simulate_forgetting.py — Apply the Ebbinghaus memory curve to the knowledge graph
and show how entity strength decays over time.

Uses memory.py:
    stability S = base_stability_days * (1 + boost * ln(1 + reinforcement))
    retention  R = exp(-elapsed_days / S)

Assumes all entities were last seen at day 0; reinforcement = mention_count.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path

from memory import (
    DORMANT,
    LONG_TERM,
    MemoryParams,
    SHORT_TERM,
    classify,
    strength,
)


def load_entities(db_path: Path) -> list[dict]:
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute("SELECT node_json FROM entities").fetchall()
    finally:
        con.close()
    return [json.loads(r[0]) for r in rows]


def simulate(
    nodes: list[dict],
    params: MemoryParams,
    max_days: int = 365,
    step: int = 5,
) -> dict:
    days = list(range(0, max_days + 1, step))
    for m in (0, 7, 30, 90, 180, max_days):
        if m not in days:
            days.append(m)
    days = sorted(set(d for d in days if 0 <= d <= max_days))

    mentions = Counter(max(1, int(n.get("mention_count") or 1)) for n in nodes)
    series = {
        "avg_strength": [],
        "long_term_pct": [],
        "short_term_pct": [],
        "dormant_pct": [],
        "retained_pct": [],
    }
    classic = {1: [], 3: [], 10: [], 50: []}

    for d in days:
        strengths: list[float] = []
        states: Counter[str] = Counter()
        for n in nodes:
            r = max(1, int(n.get("mention_count") or 1))
            s = strength(r, float(d), params)
            strengths.append(s)
            states[classify(s, params)] += 1
        ntot = len(strengths) or 1
        series["avg_strength"].append(round(sum(strengths) / ntot, 4))
        series["long_term_pct"].append(round(100 * states[LONG_TERM] / ntot, 2))
        series["short_term_pct"].append(round(100 * states[SHORT_TERM] / ntot, 2))
        series["dormant_pct"].append(round(100 * states[DORMANT] / ntot, 2))
        series["retained_pct"].append(
            round(100 * (states[LONG_TERM] + states[SHORT_TERM]) / ntot, 2)
        )
        for r in classic:
            classic[r].append(round(strength(r, float(d), params), 4))

    milestones = []
    for d in sorted({0, 7, 30, 90, 180, max_days}):
        if d not in days:
            continue
        i = days.index(d)
        milestones.append(
            {
                "day": d,
                "avg_strength": series["avg_strength"][i],
                "long_term_pct": series["long_term_pct"][i],
                "short_term_pct": series["short_term_pct"][i],
                "dormant_pct": series["dormant_pct"][i],
                "retained_pct": series["retained_pct"][i],
            }
        )

    examples = []
    for n in nodes:
        r = max(1, int(n.get("mention_count") or 1))
        examples.append(
            {
                "name": n.get("name"),
                "type": n.get("entity_type"),
                "reinforcement": r,
                "day30": round(strength(r, 30, params), 3),
                "day90": round(strength(r, 90, params), 3),
                "day365": round(strength(r, float(max_days), params), 3),
                "state_day90": classify(strength(r, 90, params), params),
                "state_day365": classify(strength(r, float(max_days), params), params),
            }
        )

    by_r: dict[int, dict] = {}
    for e in examples:
        by_r.setdefault(e["reinforcement"], e)
    picked = [by_r[r] for r in sorted(by_r) if r in (1, 2, 3, 5, 10, 12)][:6]
    top = sorted(examples, key=lambda e: (-e["reinforcement"], e["name"] or ""))[:8]

    return {
        "params": {
            "base_stability_days": params.base_stability_days,
            "boost": params.boost,
            "promote_threshold": params.promote_threshold,
            "demote_threshold": params.demote_threshold,
            "formula": (
                "R = exp(-t / S), "
                f"S = {params.base_stability_days} * "
                f"(1 + {params.boost} * ln(1 + reinforcement))"
            ),
        },
        "assumption": (
            "All entities last_seen at day 0; reinforcement = mention_count "
            "from the graph. No re-exposure during the window."
        ),
        "days": days,
        "graph_entity_count": len(nodes),
        "mention_distribution": {str(k): v for k, v in sorted(mentions.items())},
        "series": series,
        "classic_by_reinforcement": {str(k): v for k, v in classic.items()},
        "milestones": milestones,
        "example_entities": picked,
        "most_reinforced": top,
    }


def run_simulation(
    graph_db: Path,
    out: Path,
    max_days: int = 365,
    step: int = 5,
    base_stability: float = 30.0,
    boost: float = 1.5,
) -> int:
    """Run forgetting simulation and write JSON. Returns process exit code."""
    db = Path(graph_db)
    if not db.exists():
        print(f"✗ Graph DB not found: {db}")
        return 1

    params = MemoryParams(
        base_stability_days=base_stability,
        boost=boost,
    )
    nodes = load_entities(db)
    if not nodes:
        print("✗ No entities in graph")
        return 1

    result = simulate(nodes, params, max_days=max_days, step=step)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))

    print("=" * 72)
    print("Ebbinghaus forgetting simulation")
    print("=" * 72)
    print(f"Entities: {result['graph_entity_count']}")
    print(f"Formula:  {result['params']['formula']}")
    print(f"Assumption: {result['assumption']}")
    print()
    print(f"{'Day':>6}  {'Avg R':>7}  {'Long%':>7}  {'Short%':>7}  {'Dormant%':>9}")
    for m in result["milestones"]:
        print(
            f"{m['day']:6d}  {m['avg_strength']:7.3f}  "
            f"{m['long_term_pct']:6.1f}%  {m['short_term_pct']:6.1f}%  "
            f"{m['dormant_pct']:8.1f}%"
        )
    print()
    print(f"✓ Wrote {out_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Simulate Ebbinghaus forgetting on a graph")
    parser.add_argument(
        "--graph-db",
        default="./data/store/corpus_graph_improved.db",
        help="Path to graph SQLite DB",
    )
    parser.add_argument(
        "--out",
        default="./data/forgetting_simulation.json",
        help="Output JSON path",
    )
    parser.add_argument("--max-days", type=int, default=365)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--base-stability", type=float, default=30.0)
    parser.add_argument("--boost", type=float, default=1.5)
    args = parser.parse_args(argv)

    return run_simulation(
        graph_db=Path(args.graph_db),
        out=Path(args.out),
        max_days=args.max_days,
        step=args.step,
        base_stability=args.base_stability,
        boost=args.boost,
    )


if __name__ == "__main__":
    raise SystemExit(main())
