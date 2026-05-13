"""Print a leaderboard of every experiment that has produced a result.json.

Run from the repo root:
    python experiments/summarise.py

Looks for experiments/<name>/result.json files, normalises their fields,
and prints a sorted table with the verdict against the locked baseline.
"""

from __future__ import annotations

import json
from pathlib import Path


def _extract_acc(blob: dict) -> float | None:
    """Pull the best test accuracy from any of the field naming variants."""
    candidates = [
        "ensembled_test_acc_pct",
        "ema_q15_test_acc_pct",
        "q15_test_acc_pct",
    ]
    for key in candidates:
        if key in blob and isinstance(blob[key], (int, float)):
            return float(blob[key])
    # tta_search stores a sorted list of configs - take the top entry
    ranked = blob.get("ranked")
    if isinstance(ranked, list) and ranked:
        return float(ranked[0].get("test_acc_pct", 0.0))
    return None


def main() -> None:
    root = Path(__file__).resolve().parent
    rows: list[tuple[str, float, float]] = []
    for result_file in sorted(root.glob("*/result.json")):
        try:
            blob = json.loads(result_file.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"!! could not parse {result_file}: {exc}")
            continue
        acc = _extract_acc(blob)
        if acc is None:
            continue
        baseline = float(blob.get("baseline_test_acc_pct", 0.0))
        rows.append((result_file.parent.name, acc, baseline))

    if not rows:
        print("No experiment results yet.")
        return

    rows.sort(key=lambda r: r[1], reverse=True)
    print(f"{'experiment':<22s}  {'test acc':>9s}  {'baseline':>9s}  {'delta':>7s}")
    print("-" * 56)
    for name, acc, baseline in rows:
        delta = acc - baseline
        print(f"{name:<22s}  {acc:>7.2f} %  {baseline:>7.2f} %  {delta:>+7.2f}")


if __name__ == "__main__":
    main()
