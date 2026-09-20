"""Paired CPU ablation: same data, seed, schedule, and baseline initialization."""
from dataclasses import asdict, replace
import json
from pathlib import Path

from .model import PRESETS
from .training import TrainConfig, train


def ablate(data, output, *, steps=30, seed=42, context=128):
    output = Path(output)
    if output.exists():
        raise ValueError("Ablation directory already exists; choose a new run")
    cfg = TrainConfig(steps=steps, seed=seed, accumulation=2, eval_every=max(1, steps//2),
                      learning_rate=1e-3, warmup_steps=min(5, steps), threads=2)
    base = replace(PRESETS["cpu"], context=context)
    results = {}
    for name, gate in (("baseline", False), ("coherence_gate", True)):
        results[name] = train(data, output/name, replace(base, coherence_gate=gate), cfg)
    a, b = (results[k]["final_validation"]["loss"] for k in ("baseline", "coherence_gate"))
    summary = {"seed": seed, "steps": steps, "data": str(data),
               "baseline_loss": a, "gate_loss": b, "gate_minus_baseline": b-a,
               "baseline_parameters": results["baseline"]["parameters"],
               "gate_parameters": results["coherence_gate"]["parameters"],
               "note": "One paired seed; extra gate parameters and CPU timing are reported, not evidence of a general improvement.",
               "runs": results}
    (output/"comparison.json").write_text(json.dumps(summary, indent=2))
    return summary
