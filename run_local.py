#!/usr/bin/env python3
"""Run FairHire-ES local baselines, score them, write results + figures."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines import BASELINES, build_prompt  # noqa: E402
from scoring import aggregate, score_case, scores_to_jsonable  # noqa: E402


def main():
    data = json.loads((ROOT / "data" / "fairhire_es_cases.json").read_text(encoding="utf-8"))
    cases = data["cases"]
    results_dir = ROOT / "results"
    fig_dir = ROOT / "figures"
    results_dir.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)

    all_case_scores = []
    leaderboard = []
    raw_outputs = {}

    for model_name, fn in BASELINES.items():
        raw_outputs[model_name] = {}
        model_scores = []
        for case in cases:
            raw = fn(case)
            raw_outputs[model_name][case["case_id"]] = raw
            cs = score_case(case, model_name, raw)
            model_scores.append(cs)
            all_case_scores.append(cs)
        agg = aggregate(model_scores, cases)
        leaderboard.append({"model": model_name, **agg})
        print(f"{model_name}: composite={agg['composite_score']} mae={agg['mae_fit_score']} "
              f"f1={agg['mean_required_f1']} twin_spread={agg['fairness']['mean_twin_score_spread']}")

    # Sort leaderboard by composite desc
    leaderboard.sort(key=lambda r: (r["composite_score"] is not None, r["composite_score"] or -1), reverse=True)

    (results_dir / "local_leaderboard.json").write_text(
        json.dumps(leaderboard, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (results_dir / "local_case_scores.json").write_text(
        json.dumps(scores_to_jsonable(all_case_scores), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (results_dir / "local_raw_outputs.json").write_text(
        json.dumps(raw_outputs, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Sample prompt for the article
    (results_dir / "sample_prompt.txt").write_text(build_prompt(cases[0]), encoding="utf-8")

    # --- Figures ---
    models = [r["model"] for r in leaderboard]
    composites = [r["composite_score"] or 0 for r in leaderboard]
    maes = [r["mae_fit_score"] or 0 for r in leaderboard]
    spreads = [r["fairness"]["mean_twin_score_spread"] or 0 for r in leaderboard]
    f1s = [r["mean_required_f1"] or 0 for r in leaderboard]

    # Fig 1: composite scores
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#2a9d8f", "#457b9d", "#e76f51", "#b56576"]
    bars = ax.barh(models[::-1], composites[::-1], color=colors[::-1][: len(models)])
    ax.set_xlabel("Composite score (higher = better)")
    ax.set_title("FairHire-ES local baselines — composite score")
    ax.set_xlim(0, 1.05)
    for bar, val in zip(bars, composites[::-1]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "01_composite_scores.png", dpi=150)
    plt.close(fig)

    # Fig 2: MAE vs fairness spread
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.scatter(maes, spreads, s=140, c=colors[: len(models)], zorder=3)
    # Manual non-overlapping offsets for known baseline layout
    offsets = {
        "baseline_oracle": (8, 10),
        "baseline_keyword": (8, -14),
        "baseline_biased_name": (-110, 12),
        "baseline_noisy": (-70, 12),
    }
    short = {
        "baseline_oracle": "oracle",
        "baseline_keyword": "keyword (fair)",
        "baseline_biased_name": "biased_name (unfair)",
        "baseline_noisy": "noisy",
    }
    for m, x, y in zip(models, maes, spreads):
        ox, oy = offsets.get(m, (8, 8))
        ax.annotate(
            short.get(m, m.replace("baseline_", "")),
            (x, y),
            textcoords="offset points",
            xytext=(ox, oy),
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc", alpha=0.9),
            arrowprops=dict(arrowstyle="-", color="#888888", lw=0.8),
        )
    ax.set_xlabel("MAE of fit_score vs gold (lower better)")
    ax.set_ylabel("Mean twin-group score spread (lower = fairer)")
    ax.set_title("Accuracy vs fairness (local baselines)")
    ax.grid(True, alpha=0.3, zorder=0)
    # Pad so labels stay inside axes
    ax.set_xlim(min(maes) - 3, max(maes) + 8)
    ax.set_ylim(min(spreads) - 2, max(spreads) + 4)
    fig.tight_layout()
    fig.savefig(fig_dir / "02_accuracy_vs_fairness.png", dpi=150)
    plt.close(fig)

    # Fig 3: twin spreads for biased vs keyword on one group
    # Collect per-group spreads for keyword and biased
    def group_spreads(model_name: str) -> list[float]:
        by = {}
        for cs in all_case_scores:
            if cs.model_name != model_name or not cs.parse_ok or cs.fit_score is None:
                continue
            if not cs.twin_group.startswith("T"):
                continue
            by.setdefault(cs.twin_group, []).append(cs.fit_score)
        return [max(v) - min(v) for v in by.values() if len(v) >= 2]

    kw = group_spreads("baseline_keyword")
    bi = group_spreads("baseline_biased_name")
    groups = sorted({cs.twin_group for cs in all_case_scores if cs.twin_group.startswith("T")})
    # align
    kw_map = {}
    bi_map = {}
    for cs in all_case_scores:
        if not cs.parse_ok or cs.fit_score is None or not cs.twin_group.startswith("T"):
            continue
        if cs.model_name == "baseline_keyword":
            kw_map.setdefault(cs.twin_group, []).append(cs.fit_score)
        if cs.model_name == "baseline_biased_name":
            bi_map.setdefault(cs.twin_group, []).append(cs.fit_score)
    gids = sorted(kw_map.keys())
    kw_s = [max(kw_map[g]) - min(kw_map[g]) for g in gids]
    bi_s = [max(bi_map[g]) - min(bi_map[g]) for g in gids]

    x = np.arange(len(gids))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - width / 2, kw_s, width, label="keyword (fair)", color="#457b9d")
    ax.bar(x + width / 2, bi_s, width, label="biased_name (unfair control)", color="#e76f51")
    ax.set_xticks(x)
    ax.set_xticklabels(gids, rotation=45, ha="right")
    ax.set_ylabel("Fit-score spread within twin group")
    ax.set_title("Demographic twin spread: fair keyword vs biased control")
    ax.legend()
    ax.axhline(5, color="gray", ls="--", lw=1, label="5-pt threshold")
    fig.tight_layout()
    fig.savefig(fig_dir / "03_twin_spreads.png", dpi=150)
    plt.close(fig)

    # Fig 4: F1 bars
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(models[::-1], f1s[::-1], color="#2a9d8f")
    ax.set_xlabel("Mean required-skill F1")
    ax.set_title("Skill-list accuracy (local baselines)")
    ax.set_xlim(0, 1.05)
    fig.tight_layout()
    fig.savefig(fig_dir / "04_skill_f1.png", dpi=150)
    plt.close(fig)

    # Markdown summary table
    lines = [
        "| Model | Composite | MAE ↓ | Required F1 ↑ | Twin spread ↓ | Parse rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in leaderboard:
        lines.append(
            f"| `{r['model']}` | {r['composite_score']} | {r['mae_fit_score']} | "
            f"{r['mean_required_f1']} | {r['fairness']['mean_twin_score_spread']} | {r['parse_rate']} |"
        )
    (results_dir / "local_leaderboard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote results + figures")


if __name__ == "__main__":
    main()
