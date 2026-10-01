"""One entry point for the new, independent AURIC development path."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sqlite3
import sys


def emit(value):
    print(json.dumps(value, indent=2, ensure_ascii=False))


def parser():
    p = argparse.ArgumentParser(prog="auric", description="Small from-scratch models and local coding tools")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Check local dependencies and hardware without downloading")
    plan = sub.add_parser("plan", help="Show parameter and persistent training-state estimates")
    plan.add_argument("--preset", choices=("cpu", "2gb", "4gb"), default="2gb")
    plan.add_argument("--coherence-gate", action="store_true")
    bench = sub.add_parser("benchmark", help="Measure a few synthetic training steps on CPU or GPU")
    bench.add_argument("--preset", choices=("cpu", "2gb", "4gb"), default="2gb")
    bench.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    bench.add_argument("--context", type=int)
    bench.add_argument("--steps", type=int, default=3)
    bench.add_argument("--batch-size", type=int, default=1)
    bench.add_argument("--output")
    prep = sub.add_parser("prepare", help="Deduplicate JSONL and create grouped train/validation splits")
    prep.add_argument("inputs", nargs="+")
    prep.add_argument("--output", required=True)
    prep.add_argument("--context", type=int, default=128)
    prep.add_argument("--validation-fraction", type=float, default=0.1)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--max-input-bytes", type=int, default=256_000_000)
    export = sub.add_parser("export-sources", help="Create text training records from selected source roots")
    export.add_argument("roots", nargs="+")
    export.add_argument("--output", required=True)
    export.add_argument("--include-tests", action="store_true")
    ab = sub.add_parser("ablate", help="Paired CPU baseline versus experimental gate")
    ab.add_argument("--data", required=True)
    ab.add_argument("--output", required=True)
    ab.add_argument("--steps", type=int, default=30)
    ab.add_argument("--context", type=int, default=128)
    ab.add_argument("--seed", type=int, default=42)
    tr = sub.add_parser("train", help="Train random weights, or resume an AURIC checkpoint")
    tr.add_argument("--data", required=True)
    tr.add_argument("--output", required=True)
    tr.add_argument("--preset", choices=("cpu", "2gb", "4gb"), default="cpu")
    tr.add_argument("--context", type=int)
    tr.add_argument("--coherence-gate", action="store_true")
    tr.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    tr.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto")
    for name, default in (("steps", 100), ("batch-size", 1), ("accumulation", 8), ("warmup-steps", 10),
                          ("eval-every", 25), ("eval-batches", 16), ("seed", 42), ("threads", 2)):
        tr.add_argument("--"+name, type=int, default=default)
    tr.add_argument("--learning-rate", type=float, default=3e-4)
    tr.add_argument("--resume")
    tr.add_argument("--init-from", help="Start a new optimizer/data run from your own trained checkpoint")
    tr.add_argument("--stop-after", type=int, help="Save after N additional steps without changing the LR schedule")
    ev = sub.add_parser("evaluate")
    ev.add_argument("--checkpoint", required=True)
    ev.add_argument("--data", required=True)
    ev.add_argument("--max-batches", type=int, default=1000000)
    gen = sub.add_parser("generate")
    gen.add_argument("--checkpoint", required=True)
    gen.add_argument("prompt")
    gen.add_argument("--max-new-tokens", type=int, default=128)
    gen.add_argument("--temperature", type=float, default=0.0)
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--instruction", action="store_true", help="Use prompt/answer separator matching supervised data")
    gen.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    idx = sub.add_parser("index", help="Index explicitly selected code/document roots locally")
    idx.add_argument("roots", nargs="+")
    idx.add_argument("--db", default="artifacts/knowledge.sqlite")
    idx.add_argument("--max-files", type=int, default=3000)
    idx.add_argument("--max-bytes", type=int, default=32_000_000)
    for name in ("search", "ask"):
        s = sub.add_parser(name)
        s.add_argument("query")
        s.add_argument("--db", default="artifacts/knowledge.sqlite")
        if name == "ask":
            s.add_argument("--checkpoint")
            s.add_argument("--max-new-tokens", type=int, default=128)
    mem = sub.add_parser("remember")
    mem.add_argument("key")
    mem.add_argument("value")
    mem.add_argument("--evidence", required=True)
    mem.add_argument("--db", default="artifacts/knowledge.sqlite")
    mem = sub.add_parser("memories")
    mem.add_argument("--history", action="store_true")
    mem.add_argument("--db", default="artifacts/knowledge.sqlite")
    mem = sub.add_parser("forget")
    mem.add_argument("key")
    mem.add_argument("--db", default="artifacts/knowledge.sqlite")
    read = sub.add_parser("read")
    read.add_argument("root")
    read.add_argument("path")
    read.add_argument("--start", type=int, default=1)
    read.add_argument("--end", type=int, default=120)
    patch = sub.add_parser("propose-patch", help="Print a reviewable diff without modifying the target")
    patch.add_argument("root")
    patch.add_argument("path")
    patch.add_argument("--replacement", required=True)
    for name in ("tasks", "check-code", "curriculum", "repair"):
        task = sub.add_parser(name)
        task.add_argument("--vibecoder", required=True, help="Trusted local VibeCoder checkout")
        if name == "tasks":
            continue
        if name == "curriculum":
            task.add_argument("--tasks", nargs="+", required=True)
        else:
            task.add_argument("--task", required=True)
            task.add_argument("--seed", type=int, default=42)
        if name == "check-code":
            task.add_argument("--code", required=True)
        else:
            task.add_argument("--output", required=True)
        if name == "repair":
            task.add_argument("--checkpoint", required=True)
            task.add_argument("--attempts", type=int, default=2)
            task.add_argument("--max-new-tokens", type=int, default=128)
    hub = sub.add_parser("hub-catalog", help="Read public Hub metadata; no weights or remote code")
    hub.add_argument("--author", default="9x25dillon")
    hub.add_argument("--output")
    hub = sub.add_parser("hub-fetch", help="Fetch one revision-pinned small data file")
    hub.add_argument("repo")
    hub.add_argument("filename")
    hub.add_argument("--revision", required=True)
    hub.add_argument("--kind", choices=("datasets", "models"), default="datasets")
    hub.add_argument("--output", required=True)
    mcp = sub.add_parser("mcp", help="MCP stdio server for Claude Code and Codex sessions (Porter + local index)")
    mcp.add_argument("--porter-db", help="Shared ledger (default $AURIC_PORTER_DB or ~/.local/share/auric/porter.sqlite)")
    mcp.add_argument("--knowledge-db", help="AURIC knowledge index (default $AURIC_KNOWLEDGE_DB or artifacts/knowledge.sqlite)")
    porter = sub.add_parser("porter", help="See and steer the agent sessions working on your projects")
    porter.add_argument("--db", help="Shared ledger (default $AURIC_PORTER_DB or ~/.local/share/auric/porter.sqlite)")
    porter.add_argument("--project", help="Project root (default: Git root of the current directory)")
    psub = porter.add_subparsers(dest="porter_command", required=True)
    st = psub.add_parser("status", help="Directives, questions waiting on you, live sessions, recent notes")
    st.add_argument("--json", action="store_true")
    st = psub.add_parser("steer", help="Add a directive every session sees in its brief")
    st.add_argument("text")
    st.add_argument("--kind", choices=("priority", "constraint", "preference", "scope", "protect"), default="priority")
    st.add_argument("--paths", nargs="+", default=[], help="scope/protect path patterns, relative to the project")
    st.add_argument("--rank", type=int, help="Order within its kind (default: after existing ones)")
    st.add_argument("--global", dest="global_", action="store_true", help="Apply to every project")
    st = psub.add_parser("retire", help="Retire a directive")
    st.add_argument("id", type=int)
    st = psub.add_parser("answer", help="Answer a question from a session; every session is notified")
    st.add_argument("id", type=int)
    st.add_argument("text")
    for name, verb in (("confirm", "Put a change proposed from an agent's shell into effect (your terminal only)"),
                       ("reject", "Discard a change proposed from an agent's shell")):
        st = psub.add_parser(name, help=verb)
        st.add_argument("kind", choices=("directive", "retire", "answer"))
        st.add_argument("id", type=int)
    st = psub.add_parser("say", help="Message the sessions (all by default)")
    st.add_argument("text")
    st.add_argument("--to", default="*", help="Session id or agent name (claude-code, codex)")
    st = psub.add_parser("end", help="End a session and release its claims")
    st.add_argument("session")
    st.add_argument("--reason", default="ended by the user")
    st = psub.add_parser("history", help="Search coordination history")
    st.add_argument("query", nargs="?")
    st.add_argument("--kinds", nargs="+")
    st.add_argument("--limit", type=int, default=30)
    st = psub.add_parser("watch", help="Follow new events as they happen (Ctrl-C to stop)")
    st.add_argument("--interval", type=float, default=2.0)
    return p


def porter_command(args):
    import time
    from .porter import GLOBAL, Porter, find_project, origin, parse, render_status
    project = str(Path(args.project).resolve()) if args.project else find_project()
    cmd, via = args.porter_command, origin()
    with Porter(args.db) as porter:
        if cmd == "status":
            status = porter.status(project)
            if args.json:
                emit(status)
            else:
                print(render_status(status))
        elif cmd == "steer":
            target = GLOBAL if args.global_ else project
            directive = porter.steer(target, args.kind, args.text, paths=args.paths, rank=args.rank, via=via)
            emit({"directive": directive, "project": target, "via": via})
        elif cmd == "retire":
            porter.retire(args.id, via=via)
            emit({"retired": args.id})
        elif cmd == "answer":
            emit(porter.answer(args.id, args.text, via=via))
        elif cmd in ("confirm", "reject"):
            # Like approve-push: the user's own interactive terminal, never an agent's shell or a pipe.
            if via != "terminal" or not sys.stdin.isatty():
                raise SystemExit(f"auric porter {cmd} runs in your own interactive terminal; agents cannot {cmd} proposals")
            match = [p for p in porter.proposals(project) if p["kind"] == args.kind and p["id"] == args.id]
            if not match:
                raise SystemExit(f"No proposed {args.kind} #{args.id} in {project}; see auric porter status")
            print(f"{args.kind} #{args.id}: {match[0]['text']}\n  proposed from {match[0]['from']}")
            if cmd == "confirm" and input("Type confirm to put it into effect: ").strip() != "confirm":
                print("Nothing changed.")
                return 1
            emit(getattr(porter, cmd)(args.kind, args.id, via=via))
        elif cmd == "say":
            emit({"event": porter.say(project, args.text, to=args.to, via=via), "to": args.to})
        elif cmd == "end":
            emit(porter.end(args.session, args.reason + ("" if via == "terminal" else f" [via {via}]"),
                            actor="user" if via == "terminal" else via))
        elif cmd == "history":
            emit(porter.history(project, query=args.query, kinds=args.kinds, limit=args.limit))
        elif cmd == "watch":
            last = porter.history(project, limit=1)
            last = last[-1]["id"] if last else 0
            print(f"Watching {project} (Ctrl-C to stop)", flush=True)
            try:
                while True:
                    for e in porter.history(project, limit=200):
                        if e["id"] > last:
                            last = e["id"]
                            when = parse(e["created"]).astimezone().strftime("%H:%M:%S")
                            to = f" -> {e['to']}" if e["to"] else ""
                            paths = f" [{', '.join(e['paths'][:6])}]" if e["paths"] else ""
                            print(f"{when} {e['actor']}{to} {e['kind']}: {e['message']}{paths}", flush=True)
                    time.sleep(args.interval)
            except KeyboardInterrupt:
                pass
    return 0


def run(args):
    cmd = args.command
    if cmd == "doctor":
        import importlib.util
        import shutil
        import sqlite3
        import torch
        emit({"python": sys.version.split()[0], "torch": str(torch.__version__),
              "cuda_available": torch.cuda.is_available(), "cuda_build": torch.version.cuda,
              "gpus": [{"name": torch.cuda.get_device_name(i),
                         "total_vram_mib": torch.cuda.get_device_properties(i).total_memory/2**20}
                        for i in range(torch.cuda.device_count())],
              "sqlite": sqlite3.sqlite_version, "bubblewrap_installed": bool(shutil.which("bwrap")),
              "docker_installed": bool(shutil.which("docker")),
              "note": "Installed sandbox binaries are not proof that isolation is usable."})
    elif cmd == "plan":
        from .model import PRESETS, memory_report
        emit(memory_report(replace(PRESETS[args.preset], coherence_gate=args.coherence_gate)))
    elif cmd == "benchmark":
        from .benchmark import benchmark
        result = benchmark(args.preset, device=args.device, steps=args.steps,
                           batch_size=args.batch_size, context=args.context)
        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x") as f:
                json.dump(result, f, indent=2)
        emit(result)
    elif cmd == "prepare":
        from .data import prepare
        result = prepare(args.inputs, args.output, context=args.context,
                         validation_fraction=args.validation_fraction, seed=args.seed,
                         max_input_bytes=args.max_input_bytes)
        emit({k:v for k,v in result.items() if k != "records"})
    elif cmd == "train":
        from .model import PRESETS
        from .training import train, TrainConfig
        model_cfg = replace(PRESETS[args.preset], coherence_gate=args.coherence_gate)
        if args.context:
            model_cfg = replace(model_cfg, context=args.context)
        cfg = TrainConfig(**{k: getattr(args, k) for k in TrainConfig.__dataclass_fields__ if hasattr(args, k)})
        emit(train(args.data, args.output, model_cfg, cfg, resume=args.resume,
                   stop_after=args.stop_after, init_from=args.init_from))
    elif cmd == "ablate":
        from .experiments import ablate
        emit(ablate(args.data, args.output, steps=args.steps, context=args.context, seed=args.seed))
    elif cmd == "export-sources":
        from .workspace import export_sources
        emit(export_sources(args.roots, args.output, include_tests=args.include_tests))
    elif cmd == "evaluate":
        from .data import TokenDataset, file_hash
        from .training import evaluate, load_model
        model, state = load_model(args.checkpoint)
        if args.max_batches < 1:
            raise ValueError("max-batches must be positive")
        data = TokenDataset(args.data, "val")
        if data.manifest["context"] != model.config.context:
            raise ValueError("Dataset context mismatch")
        emit(evaluate(model, data, max_batches=args.max_batches))
    elif cmd == "generate":
        from .training import load_model
        from .tokenizer import ByteTokenizer
        model, _ = load_model(args.checkpoint, args.device)
        tok = ByteTokenizer()
        ids = [tok.bos_id]+tok.encode(args.prompt)+([tok.sep_id] if args.instruction else [])
        if len(ids) > model.config.context:
            raise ValueError("Prompt exceeds model context")
        print(tok.decode(model.generate(ids, max_new_tokens=args.max_new_tokens,
                                       temperature=args.temperature, seed=args.seed)))
    elif cmd in ("index", "search", "remember", "memories", "forget"):
        from .memory import KnowledgeStore
        with KnowledgeStore(args.db) as store:
            if cmd == "index":
                emit([store.index(root, max_files=args.max_files, max_bytes=args.max_bytes) for root in args.roots])
            elif cmd == "search":
                emit(store.search(args.query))
            elif cmd == "remember":
                emit({"revision": store.remember(args.key, args.value, args.evidence)})
            elif cmd == "memories":
                emit(store.memories(history=args.history))
            else:
                store.forget(args.key)
                emit({"forgotten": args.key, "note": "Revision history is retained."})
    elif cmd == "ask":
        from .assistant import ask
        emit(ask(args.db, args.query, checkpoint=args.checkpoint, max_new_tokens=args.max_new_tokens))
    elif cmd == "read":
        from .workspace import read_file
        emit(read_file(args.root, args.path, start=args.start, end=args.end))
    elif cmd == "propose-patch":
        from .assistant import propose_patch
        print(propose_patch(args.root, args.path, Path(args.replacement).read_text()))
    elif cmd in ("tasks", "check-code", "curriculum", "repair"):
        from .vibecoder_bridge import list_tasks, check_code, export_curriculum, repair_episode
        if cmd == "tasks":
            emit(list_tasks(args.vibecoder))
        elif cmd == "check-code":
            result = check_code(args.vibecoder, args.task, Path(args.code).read_text(), seed=args.seed)
            emit(result)
            return 0 if result["all_passed"] else 1
        elif cmd == "curriculum":
            emit(export_curriculum(args.vibecoder, args.output, args.tasks))
        else:
            result = repair_episode(args.vibecoder, args.task, args.checkpoint, args.output,
                                    attempts=args.attempts, seed=args.seed, max_new_tokens=args.max_new_tokens)
            emit(result)
            return 0 if result["passed"] else 1
    elif cmd == "hub-catalog":
        from .hub import catalog
        result = catalog(args.author)
        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x") as f:
                json.dump(result, f, indent=2)
        emit(result)
    elif cmd == "hub-fetch":
        from .hub import fetch_data
        emit(fetch_data(args.repo, args.filename, args.revision, args.output, kind=args.kind))
    elif cmd == "mcp":
        from .mcp_server import main as serve
        return serve(porter_db=args.porter_db, knowledge_db=args.knowledge_db)
    elif cmd == "porter":
        return porter_command(args)
    return 0


def main():
    args = parser().parse_args()
    try:
        if args.command in ("evaluate", "generate", "repair") or (args.command == "ask" and args.checkpoint):
            import torch
            torch.set_num_threads(2)
        code = run(args)
    except (ValueError, OSError, KeyError, RuntimeError, sqlite3.Error) as exc:
        print(f"auric: {exc}", file=sys.stderr)
        code = 2
    raise SystemExit(code or 0)
