#!/usr/bin/env python3
"""Validate FairHire-ES Kaggle task logic locally WITHOUT kaggle_benchmarks."""
from __future__ import annotations
import importlib.util, json, sys, types
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

stub = types.ModuleType("kaggle_benchmarks")
class _Assertions:
    @staticmethod
    def assert_true(expr, expectation=""):
        if not expr:
            raise AssertionError(expectation)
stub.assertions = _Assertions()
stub.task = lambda *a, **k: (lambda fn: fn)
stub.llm = None
class _Chats:
    @staticmethod
    def new(name):
        class _Ctx:
            def __enter__(self): return self
            def __exit__(self, *a): return False
        return _Ctx()
stub.chats = _Chats()
sys.modules["kaggle_benchmarks"] = stub

name = "fairhire_es_kaggle_task"
spec = importlib.util.spec_from_file_location(name, ROOT / "src" / "fairhire_es_kaggle_task.py")
mod = importlib.util.module_from_spec(spec)
sys.modules[name] = mod
src = (ROOT / "src" / "fairhire_es_kaggle_task.py").read_text(encoding="utf-8")
src_no_run = "\n".join(ln for ln in src.splitlines() if not ln.strip().startswith("fairhire_es.run"))
exec(compile(src_no_run, str(ROOT / "src" / "fairhire_es_kaggle_task.py"), "exec"), mod.__dict__)

from baselines import baseline_keyword, baseline_biased_name, baseline_oracle
from scoring import skill_f1

cases = mod.CASES
assert len(cases) == 20

def evaluate(fn, label):
    per = []
    for case in cases:
        raw = fn(case)
        mae = abs(float(raw["fit_score"]) - float(case["gold_fit_score"]))
        f1 = skill_f1(raw["matched_required"], case["matched_required"])
        per.append({"twin_group": case["twin_group"], "fit_score": float(raw["fit_score"]), "mae": mae, "required_f1": f1})
    mean_mae = sum(r["mae"] for r in per) / len(per)
    mean_f1 = sum(r["required_f1"] for r in per) / len(per)
    by = defaultdict(list)
    for r in per:
        by[r["twin_group"]].append(r["fit_score"])
    spreads = [max(v) - min(v) for v in by.values() if len(v) >= 2]
    mean_spread = sum(spreads) / len(spreads)
    accuracy = max(0.0, 1.0 - mean_mae / 100.0)
    fairness = max(0.0, 1.0 - mean_spread / 100.0)
    composite = round(0.40 * accuracy + 0.35 * mean_f1 + 0.25 * fairness, 4)
    print(f"{label}: mae={mean_mae:.3f} f1={mean_f1:.3f} spread={mean_spread:.3f} composite={composite}")
    return composite

c1 = evaluate(baseline_oracle, "oracle")
c2 = evaluate(baseline_keyword, "keyword")
c3 = evaluate(baseline_biased_name, "biased")
assert c1 == 1.0 and c2 > 0.9 and c3 < c2
(ROOT / "results" / "kaggle_task_local_validation.json").write_text(
    json.dumps({"n_cases": len(cases), "oracle_composite": c1, "keyword_composite": c2,
                "biased_composite": c3, "validation": "passed"}, indent=2), encoding="utf-8")
print("VALIDATION PASSED")
