"""Use the user's local VibeCoder without copying or modifying its source."""
import importlib
import json
from pathlib import Path
import re
import sys


def connect(root):
    root = Path(root).resolve(strict=True)
    if not (root/"vibecoder"/"runner.py").is_file():
        raise ValueError("Not a VibeCoder checkout")
    # Explicit trusted local plugin path. Bytecode stays out of the source repo.
    existing = sys.modules.get("vibecoder")
    if existing and not Path(existing.__file__).resolve().is_relative_to(root):
        raise ValueError("A different VibeCoder checkout is already imported")
    sys.dont_write_bytecode = True
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return importlib.import_module("vibecoder.levels"), importlib.import_module("vibecoder.runner")


def list_tasks(root):
    levels, _ = connect(root)
    return [{"id": x.id, "title": x.title, "tags": x.tags} for x in levels.all_levels()]


def check_code(root, task, code, *, seed=42):
    levels, runner = connect(root)
    from vibecoder.models import Source
    level = levels.get_level(task)
    # Generated code always goes to an isolating backend. No host fallback.
    result = runner.run_code(code, level.func_name, level.tests_for(seed), source=Source.THIRD_PARTY)
    return {"task": task, "seed": seed, "passed": result.passed_count,
            "total": result.total_count, "all_passed": result.all_passed,
            "result": result.to_json()}


def export_curriculum(root, output, task_ids):
    if not task_ids:
        raise ValueError("Select training task IDs explicitly; keep evaluation families separate")
    levels, _ = connect(root)
    output = Path(output)
    records = []
    for task in task_ids:
        level = levels.get_level(task)
        records.append({"prompt": level.brief+"\n\n"+level.starter,
                        "completion": level.reference, "group": level.id,
                        "source": f"VibeCoder:{level.id}", "purpose": "supervised_reference"})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record)+"\n")
    return {"records": len(records), "output": str(output), "training_tasks": task_ids}


def extract_code(text):
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, flags=re.S)
    return match.group(1) if match else text


def task_prompt(level, context, *, code=None, feedback=None):
    from .tokenizer import ByteTokenizer
    tok = ByteTokenizer()
    signature = next((line for line in level.starter.splitlines() if line.startswith("def ")), level.func_name)
    base = level.brief + "\n" + signature + "\nReturn only Python code."
    budget = context - 2 - len(tok.encode(base))
    if budget < 0:
        raise ValueError("Task exceeds checkpoint context; use a larger context model")
    if feedback is not None:
        labels = "\nFailure excerpt: \nPrevious code excerpt: "
        remaining = budget - len(tok.encode(labels))
        if remaining < 16:
            raise ValueError("Context has no room for repair feedback; use a larger context model")
        error_budget = min(remaining//2, len(tok.encode(feedback)))
        error = feedback.encode()[:error_budget].decode("utf-8", errors="ignore")
        previous = code.encode()[:remaining-error_budget].decode("utf-8", errors="ignore")
        base += f"\nFailure excerpt: {error}\nPrevious code excerpt: {previous}"
    return base


def repair_episode(root, task, checkpoint, output, *, attempts=2, seed=42, max_new_tokens=128):
    from .training import load_model
    from .tokenizer import ByteTokenizer
    if not 1 <= attempts <= 5:
        raise ValueError("Choose 1–5 attempts")
    levels, _ = connect(root)
    level = levels.get_level(task)
    model, metadata = load_model(checkpoint)
    # Fail before generating if this task was included in training.
    # Caller still owns broader semantic/task-family contamination review.
    if task in metadata.get("training_groups", []):
        raise ValueError("This task was used for training; select a held-out task family")
    tok = ByteTokenizer()
    prompt = task_prompt(level, model.config.context)
    history = []
    for attempt in range(attempts):
        budget = model.config.context-2
        if len(tok.encode(prompt)) > budget:
            raise ValueError("Task/repair feedback exceeds checkpoint context; use a larger context model")
        ids = [tok.bos_id]+tok.encode(prompt)+[tok.sep_id]
        code = extract_code(tok.decode(model.generate(ids, max_new_tokens=max_new_tokens, seed=seed+attempt)))
        outcome = check_code(root, task, code, seed=seed)
        history.append({"attempt": attempt+1, "prompt": prompt, "code": code, "evaluation": outcome})
        if outcome["all_passed"]:
            break
        failures = [r for r in outcome["result"]["outcomes"] if not r["passed"]]
        feedback = outcome["result"]["error"] or json.dumps(failures[:1])
        if attempt+1 < attempts:
            prompt = task_prompt(level, model.config.context, code=code, feedback=feedback)
    verification = check_code(root, task, history[-1]["code"], seed=seed+1) if history[-1]["evaluation"]["all_passed"] else None
    record = {"task": task, "seed": seed, "checkpoint_step": metadata["step"],
              "data_signature": metadata["data_signature"], "attempts": history,
              "verification": verification,
              "passed": bool(verification and verification["all_passed"]),
              "training_eligible": False, "note": "Evaluation trace; not automatically admitted to training."}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as f:
        json.dump(record, f, indent=2)
    return record
