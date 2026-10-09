# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
summarize_ablation.py -- success-rate table of LIBERO evaluations, grouped by camera setting.

Reads every ``eval_metrics.json`` (written by ``eval.py``) under the given files or folders and
groups the runs by (suite, camera keys). Per group:

    * success rate: mean +/- sample standard deviation over runs (one run = one trained checkpoint),
      and the pooled rate over all episodes with a Wilson 95% interval (episode-level: it ignores the
      variation between training seeds, so report it next to the run-level spread, not instead of it);
    * per-task success (mean over runs), the premature task-complete rate of Eq. 9a, the executed
      horizon H_t and the decision-step latency recorded by eval.py.

If a suite was evaluated with exactly one one-camera setting and one two-camera setting, the
difference (2 cameras - 1 camera) is reported, paired by training seed when the seeds match (the seed
is read from the run's ``config.json`` next to the checkpoint). With more settings per suite (e.g. two
different single cameras) no difference is computed, since it would be ambiguous.

Usage:
    python scripts/summarize_ablation.py eval_results/camera_ablation
    python scripts/summarize_ablation.py eval_results/a eval_results/b --json summary.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple


def wilson(successes: int, trials: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval of a binomial proportion."""
    if trials == 0:
        return (float("nan"), float("nan"))
    p = successes / trials
    denom = 1.0 + z * z / trials
    center = (p + z * z / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return (center - half, center + half)


def _mean(values: Sequence[float]) -> Optional[float]:
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


def _std(values: Sequence[float]) -> Optional[float]:
    return statistics.stdev(values) if len(values) > 1 else None


def find_results(paths: Sequence[str]) -> List[str]:
    files: List[str] = []
    for path in paths:
        if os.path.isdir(path):
            files += glob.glob(os.path.join(path, "**", "eval_metrics.json"), recursive=True)
        elif os.path.isfile(path):
            files.append(path)
        else:
            raise SystemExit(f"not found: {path}")
    return sorted(set(files))


def training_seed(checkpoint: str) -> Optional[int]:
    """Seed of the training run that wrote ``checkpoint`` (from its config.json), if available."""
    config = os.path.join(os.path.dirname(checkpoint), "config.json")
    try:
        with open(config, encoding="utf-8") as f:
            return int(json.load(f)["args"]["seed"])
    except (OSError, KeyError, TypeError, ValueError):
        return None


def load_run(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        res = json.load(f)
    cameras = res.get("camera_keys") or [res.get("camera_key", "agentview_rgb")]
    episodes = [ep for task in res["tasks"] for ep in task["episodes"]]
    return {
        "file": path,
        "status": res.get("status"),
        "suite": res.get("suite"),
        "cameras": tuple(cameras),
        "checkpoint": res.get("checkpoint", ""),
        "seed": training_seed(res.get("checkpoint", "")),
        "successes": int(res.get("successes", 0)),
        "trials": int(res.get("trials", 0)),
        "success_rate": float(res.get("success_rate", 0.0)),
        "tasks": {int(t["task_id"]): (t.get("instruction", ""), float(t["success_rate"])) for t in res["tasks"]},
        "premature": _mean([float(bool(ep.get("premature_task_complete"))) for ep in episodes]),
        "horizon": _mean([ep.get("executed_horizon_mean") for ep in episodes]),
        "latency_ms": _mean([ep.get("step_latency_ms_mean") for ep in episodes]),
    }


def _fmt(value: Optional[float], pct: bool = True, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{100 * value:.{digits}f}" if pct else f"{value:.{digits}f}"


def summarize(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    groups: Dict[Tuple[str, Tuple[str, ...]], List[Dict[str, Any]]] = defaultdict(list)
    for run in runs:
        groups[(run["suite"], run["cameras"])].append(run)
    summary: Dict[str, Any] = {"groups": [], "differences": []}
    for (suite, cameras), members in sorted(groups.items(), key=lambda kv: (kv[0][0], len(kv[0][1]), kv[0][1])):
        rates = [r["success_rate"] for r in members]
        succ, trials = sum(r["successes"] for r in members), sum(r["trials"] for r in members)
        task_ids = sorted({t for r in members for t in r["tasks"]})
        summary["groups"].append({
            "suite": suite,
            "cameras": list(cameras),
            "num_views": len(cameras),
            "runs": len(members),
            "seeds": [r["seed"] for r in members],
            "success_mean": statistics.fmean(rates),
            "success_std": _std(rates),
            "pooled_successes": succ,
            "pooled_trials": trials,
            "pooled_wilson95": list(wilson(succ, trials)),
            "per_task": {str(t): {"instruction": next(r["tasks"][t][0] for r in members if t in r["tasks"]),
                                  "success_mean": statistics.fmean([r["tasks"][t][1] for r in members
                                                                     if t in r["tasks"]])}
                         for t in task_ids},
            "premature_rate": _mean([r["premature"] for r in members]),
            "executed_horizon": _mean([r["horizon"] for r in members]),
            "step_latency_ms": _mean([r["latency_ms"] for r in members]),
            "files": [r["file"] for r in members],
            "incomplete": [r["file"] for r in members if r["status"] != "complete"],
        })
    by_suite: Dict[str, Dict[int, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for g in summary["groups"]:
        by_suite[g["suite"]][g["num_views"]].append(g)
    for suite, views in sorted(by_suite.items()):
        if len(views.get(1, [])) != 1 or len(views.get(2, [])) != 1:
            continue
        one, two = views[1][0], views[2][0]
        diff: Dict[str, Any] = {"suite": suite, "one_camera": one["cameras"], "two_cameras": two["cameras"],
                                "mean_difference": two["success_mean"] - one["success_mean"]}
        one_runs = {r["seed"]: r for r in runs
                    if r["suite"] == suite and list(r["cameras"]) == one["cameras"] and r["seed"] is not None}
        two_runs = {r["seed"]: r for r in runs
                    if r["suite"] == suite and list(r["cameras"]) == two["cameras"] and r["seed"] is not None}
        paired = sorted(set(one_runs) & set(two_runs))
        if paired:
            d = [two_runs[s]["success_rate"] - one_runs[s]["success_rate"] for s in paired]
            diff.update(paired_seeds=paired, paired_differences=d, paired_mean=statistics.fmean(d),
                        paired_std=_std(d))
        summary["differences"].append(diff)
    return summary


def print_markdown(summary: Dict[str, Any], out: Any = sys.stdout) -> None:
    w = out.write
    w("## Success rate by camera setting\n\n")
    w("| suite | cameras | runs | success % (mean ± std over runs) | pooled % [Wilson 95%] | episodes "
      "| premature Eq. 9a % | mean H_t | step latency ms |\n")
    w("|---|---|---|---|---|---|---|---|---|\n")
    for g in summary["groups"]:
        lo, hi = g["pooled_wilson95"]
        std = f" ± {_fmt(g['success_std'])}" if g["success_std"] is not None else ""
        pooled = g["pooled_successes"] / max(1, g["pooled_trials"])
        w(f"| {g['suite']} | {' + '.join(g['cameras'])} | {g['runs']} | {_fmt(g['success_mean'])}{std} "
          f"| {_fmt(pooled)} [{_fmt(lo)}, {_fmt(hi)}] | {g['pooled_trials']} | {_fmt(g['premature_rate'])} "
          f"| {_fmt(g['executed_horizon'], pct=False)} | {_fmt(g['step_latency_ms'], pct=False, digits=2)} |\n")
    for d in summary["differences"]:
        w(f"\n**{d['suite']}: two cameras − one camera** = {100 * d['mean_difference']:+.1f} points (difference of means)")
        if "paired_mean" in d:
            std = f" ± {100 * d['paired_std']:.1f}" if d["paired_std"] is not None else ""
            w(f"; paired by seed {d['paired_seeds']}: {100 * d['paired_mean']:+.1f}{std} points")
        w("\n")
    for g in summary["groups"]:
        w(f"\n### {g['suite']}, {' + '.join(g['cameras'])}: per task (mean success % over {g['runs']} run(s))\n\n")
        w("| task | instruction | success % |\n|---|---|---|\n")
        for t, info in g["per_task"].items():
            w(f"| {t} | {info['instruction']} | {_fmt(info['success_mean'])} |\n")
        if g["incomplete"]:
            w(f"\nWarning: not complete: {', '.join(g['incomplete'])}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="eval_metrics.json files or folders containing them")
    parser.add_argument("--json", help="also write the summary as JSON to this file")
    args = parser.parse_args()
    files = find_results(args.paths)
    if not files:
        raise SystemExit("no eval_metrics.json found")
    summary = summarize([load_run(f) for f in files])
    print_markdown(summary)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
