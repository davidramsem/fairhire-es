#!/usr/bin/env python3
"""Analyze FairHire-ES Kaggle runs with scale-robust MAE / twin-spread metrics."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

SHORT = {
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "gemini-3.7-flash": "Gemini 3.7 Flash",
    "grok-4.20-0309-non-reasoning": "Grok 4.20 Non-Reasoning",
    "gpt-5.4-mini-2026-03-17": "GPT-5.4 mini",
    "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
    "deepseek-r1-0528": "DeepSeek-R1",
    "qwen3-next-80b-a3b-instruct": "Qwen3 Next 80B",
}


def rescale_fit_if_unit_interval(score: float) -> float:
    x = float(score)
    if 0.0 < x <= 1.0:
        return x * 100.0
    return x


def load_screen_rows(run_dir: Path) -> list[dict]:
    rows = []
    for p in run_dir.glob("fairhire-es-screen-*.run.json"):
        d = json.loads(p.read_text())
        dict_result = None
        for r in d.get("results") or []:
            if isinstance(r, dict) and r.get("dictResult"):
                dict_result = r["dictResult"]
                break
        if not dict_result:
            continue
        raw = float(dict_result.get("fit_score"))
        gold = float(dict_result.get("gold_fit_score"))
        rescaled = float(dict_result.get("fit_score_rescaled", rescale_fit_if_unit_interval(raw)))
        mae_raw = float(dict_result.get("mae_raw", abs(raw - gold)))
        mae_rescaled = float(dict_result.get("mae_rescaled", abs(rescaled - gold)))
        rows.append(
            {
                **dict_result,
                "fit_score": raw,
                "fit_score_rescaled": rescaled,
                "mae_raw": mae_raw,
                "mae_rescaled": mae_rescaled,
                "used_unit_interval": bool(dict_result.get("used_unit_interval", 0.0 < raw <= 1.0)),
                "required_f1": float(dict_result.get("required_f1", 0.0)),
                "twin_group": dict_result.get("twin_group"),
                "case_id": dict_result.get("case_id"),
            }
        )
    return rows


def _spread(by: dict) -> list[float]:
    return [max(v) - min(v) for v in by.values() if len(v) >= 2]


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    mae_raw = sum(r["mae_raw"] for r in rows) / len(rows)
    mae_res = sum(r["mae_rescaled"] for r in rows) / len(rows)
    f1 = sum(r["required_f1"] for r in rows) / len(rows)
    unit_rate = sum(1 for r in rows if r["used_unit_interval"]) / len(rows)
    by_raw, by_res = defaultdict(list), defaultdict(list)
    for r in rows:
        g = r.get("twin_group")
        if not g:
            continue
        by_raw[g].append(r["fit_score"])
        by_res[g].append(r["fit_score_rescaled"])
    spreads_raw = _spread(by_raw)
    spreads_res = _spread(by_res)
    mean_spread_raw = sum(spreads_raw) / len(spreads_raw) if spreads_raw else 0.0
    mean_spread_res = sum(spreads_res) / len(spreads_res) if spreads_res else 0.0
    # Composite (scale-robust) — matches v2 task definition
    accuracy = max(0.0, 1.0 - mae_res / 100.0)
    fairness = max(0.0, 1.0 - mean_spread_res / 100.0)
    composite_robust = round(0.40 * accuracy + 0.35 * f1 + 0.25 * fairness, 4)
    # Legacy composite using raw MAE/spread (v1 leaderboard definition)
    accuracy_raw = max(0.0, 1.0 - mae_raw / 100.0)
    fairness_raw = max(0.0, 1.0 - mean_spread_raw / 100.0)
    composite_raw = round(0.40 * accuracy_raw + 0.35 * f1 + 0.25 * fairness_raw, 4)
    return {
        "n": len(rows),
        "mae_raw": round(mae_raw, 3),
        "mae_rescaled": round(mae_res, 3),
        "required_f1": round(f1, 4),
        "twin_spread_raw": round(mean_spread_raw, 3),
        "twin_spread_rescaled": round(mean_spread_res, 3),
        "max_twin_spread_rescaled": round(max(spreads_res), 3) if spreads_res else 0.0,
        "unit_interval_rate": round(unit_rate, 4),
        "pct_fit_score_le_1": round(100 * unit_rate, 1),
        "composite_raw": composite_raw,
        "composite_robust": composite_robust,
        "composite": composite_robust,
    }


def load_main_score(run_dir: Path) -> float | None:
    mains = list(run_dir.glob("fairhire-es-latam-*.result.json"))
    if not mains:
        return None
    d = json.loads(mains[0].read_text())
    rewards = ((d.get("verifier_result") or {}).get("rewards") or {})
    if "score" in rewards:
        return float(rewards["score"])
    return None


def analyze_version(version_root: Path, tag: str) -> list[dict]:
    leaderboard = []
    for model_dir in sorted(version_root.iterdir()):
        if not model_dir.is_dir():
            continue
        run_dirs = [p for p in model_dir.iterdir() if p.is_dir()]
        if not run_dirs:
            continue
        run_dir = sorted(run_dirs, key=lambda p: p.name)[-1]
        rows = load_screen_rows(run_dir)
        if not rows:
            continue
        agg = aggregate(rows)
        main_score = load_main_score(run_dir)
        leaderboard.append({"version": tag, "model": model_dir.name, "kaggle_main_score": main_score, **agg})
    leaderboard.sort(key=lambda r: (r["composite_robust"] is not None, r["composite_robust"] or -1), reverse=True)
    return leaderboard


def write_md(leaderboard: list[dict], path: Path, title: str):
    lines = [
        f"# {title}",
        "",
        "| Model | Composite (robust) | Composite (raw) | Kaggle score | MAE raw ↓ | MAE ×100-fix ↓ | F1 ↑ | Twin spread raw ↓ | Twin spread rescaled ↓ | % fit≤1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in leaderboard:
        lines.append(
            f"| `{r['model']}` | {r['composite_robust']} | {r['composite_raw']} | {r['kaggle_main_score']} | "
            f"{r['mae_raw']} | {r['mae_rescaled']} | {r['required_f1']} | {r['twin_spread_raw']} | "
            f"{r['twin_spread_rescaled']} | {r['pct_fit_score_le_1']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path)
    print("\n".join(lines[2:]))


def make_figures(lb_v1: list[dict], lb_v2: list[dict], fig_dir: Path):
    fig_dir.mkdir(exist_ok=True)

    def plot_unit_rate_compare():
        # paired bar: % fit<=1 v1 vs v2 for models in both
        models = sorted(set(r["model"] for r in lb_v1) & set(r["model"] for r in lb_v2))
        v1 = {r["model"]: r["pct_fit_score_le_1"] for r in lb_v1}
        v2 = {r["model"]: r["pct_fit_score_le_1"] for r in lb_v2}
        import numpy as np

        x = np.arange(len(models))
        w = 0.38
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - w / 2, [v1[m] for m in models], w, label="v1 ambiguous scale", color="#e76f51")
        ax.bar(x + w / 2, [v2[m] for m in models], w, label="v2 explicit 0–100", color="#2a9d8f")
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT.get(m, m) for m in models], rotation=25, ha="right")
        ax.set_ylabel("% of cases with fit_score in (0,1]")
        ax.set_title("Scale compliance: v1 vs v2 prompt")
        ax.set_ylim(0, 105)
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "09_v1_vs_v2_unit_interval_rate.png", dpi=150)
        plt.close(fig)

    def plot_mae_compare():
        models = sorted(set(r["model"] for r in lb_v2), key=lambda m: next(r["mae_rescaled"] for r in lb_v2 if r["model"]==m))
        fig, ax = plt.subplots(figsize=(10, 5))
        import numpy as np
        x = np.arange(len(models))
        w = 0.38
        raw = [next(r["mae_raw"] for r in lb_v2 if r["model"]==m) for m in models]
        res = [next(r["mae_rescaled"] for r in lb_v2 if r["model"]==m) for m in models]
        ax.bar(x - w / 2, raw, w, label="MAE raw", color="#457b9d")
        ax.bar(x + w / 2, res, w, label="MAE after ×100 if ≤1", color="#e9c46a")
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT.get(m, m) for m in models], rotation=25, ha="right")
        ax.set_ylabel("MAE vs gold fit_score")
        ax.set_title("v2 calibration: raw vs scale-robust MAE")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "10_v2_mae_raw_vs_rescaled.png", dpi=150)
        plt.close(fig)

    def plot_v2_composite():
        models = [r["model"] for r in lb_v2]
        vals = [r["composite_robust"] for r in lb_v2]
        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.barh([SHORT.get(m, m) for m in models][::-1], vals[::-1], color="#2a9d8f")
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Composite (scale-robust)")
        ax.set_title("FairHire-ES v2 — Kaggle composite scores")
        for bar, val in zip(bars, vals[::-1]):
            ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=9)
        fig.tight_layout()
        fig.savefig(fig_dir / "11_v2_composite.png", dpi=150)
        plt.close(fig)

    def plot_v2_fairness():
        models = [r["model"] for r in lb_v2]
        spreads = [r["twin_spread_rescaled"] for r in lb_v2]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar([SHORT.get(m, m) for m in models], spreads, color="#e76f51")
        ax.set_ylabel("Mean twin-group spread (rescaled scores)")
        ax.set_title("FairHire-ES v2 — demographic twin spread (scale-robust)")
        plt.xticks(rotation=25, ha="right")
        fig.tight_layout()
        fig.savefig(fig_dir / "12_v2_twin_spreads_rescaled.png", dpi=150)
        plt.close(fig)

    def plot_accuracy_vs_fairness_v2():
        fig, ax = plt.subplots(figsize=(9, 6))
        maes = [r["mae_rescaled"] for r in lb_v2]
        spreads = [r["twin_spread_rescaled"] for r in lb_v2]
        models = [r["model"] for r in lb_v2]
        ax.scatter(maes, spreads, s=140, c="#457b9d", zorder=3)
        for m, x, y in zip(models, maes, spreads):
            ax.annotate(
                SHORT.get(m, m),
                (x, y),
                textcoords="offset points",
                xytext=(8, 8),
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#ccc", alpha=0.9),
            )
        ax.set_xlabel("MAE rescaled (lower better)")
        ax.set_ylabel("Twin spread rescaled (lower = fairer)")
        ax.set_title("FairHire-ES v2 — accuracy vs fairness (scale-robust)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "13_v2_accuracy_vs_fairness.png", dpi=150)
        plt.close(fig)

    if lb_v1 and lb_v2:
        plot_unit_rate_compare()
    if lb_v2:
        plot_mae_compare()
        plot_v2_composite()
        plot_v2_fairness()
        plot_accuracy_vs_fairness_v2()


def main():
    # Defaults: analyze both archived v1 and latest download tree
    v1_root = ROOT / "results" / "kaggle_runs_v1" / "fairhire-es-latam" / "1"
    v2_root = ROOT / "results" / "kaggle_runs_v2" / "fairhire-es-latam" / "2"
    # allow CLI overrides
    if len(sys.argv) >= 2:
        v2_root = Path(sys.argv[1])
        if (v2_root / "2").is_dir():
            v2_root = v2_root / "2"
        elif v2_root.name != "2" and (v2_root / "fairhire-es-latam" / "2").is_dir():
            v2_root = v2_root / "fairhire-es-latam" / "2"
    if len(sys.argv) >= 3:
        v1_root = Path(sys.argv[2])

    lb_v1 = analyze_version(v1_root, "v1") if v1_root.exists() else []
    lb_v2 = analyze_version(v2_root, "v2") if v2_root.exists() else []

    out = ROOT / "results"
    if lb_v1:
        (out / "kaggle_leaderboard_v1.json").write_text(json.dumps(lb_v1, indent=2), encoding="utf-8")
        write_md(lb_v1, out / "kaggle_leaderboard_v1.md", "FairHire-ES v1 (ambiguous scale prompt)")
    if lb_v2:
        (out / "kaggle_leaderboard_v2.json").write_text(json.dumps(lb_v2, indent=2), encoding="utf-8")
        (out / "kaggle_leaderboard.json").write_text(json.dumps(lb_v2, indent=2), encoding="utf-8")
        write_md(lb_v2, out / "kaggle_leaderboard_v2.md", "FairHire-ES v2 (explicit 0-100 integer scale)")
        write_md(lb_v2, out / "kaggle_leaderboard.md", "FairHire-ES v2 (explicit 0-100 integer scale)")

    # comparison summary
    if lb_v1 and lb_v2:
        v1m = {r["model"]: r for r in lb_v1}
        v2m = {r["model"]: r for r in lb_v2}
        compare = []
        for m in sorted(set(v1m) & set(v2m)):
            compare.append(
                {
                    "model": m,
                    "v1_pct_fit_le_1": v1m[m]["pct_fit_score_le_1"],
                    "v2_pct_fit_le_1": v2m[m]["pct_fit_score_le_1"],
                    "v1_mae_raw": v1m[m]["mae_raw"],
                    "v2_mae_raw": v2m[m]["mae_raw"],
                    "v1_mae_rescaled": v1m[m]["mae_rescaled"],
                    "v2_mae_rescaled": v2m[m]["mae_rescaled"],
                    "v1_twin_spread_rescaled": v1m[m]["twin_spread_rescaled"],
                    "v2_twin_spread_rescaled": v2m[m]["twin_spread_rescaled"],
                    "v1_composite_raw": v1m[m]["composite_raw"],
                    "v2_composite_robust": v2m[m]["composite_robust"],
                }
            )
        (out / "kaggle_v1_vs_v2.json").write_text(json.dumps(compare, indent=2), encoding="utf-8")
        print("\nV1 vs V2 unit-interval rates:")
        for c in compare:
            print(f"  {c['model']}: {c['v1_pct_fit_le_1']}% -> {c['v2_pct_fit_le_1']}%")

    make_figures(lb_v1, lb_v2, ROOT / "figures")
    print("figures written")


if __name__ == "__main__":
    main()
