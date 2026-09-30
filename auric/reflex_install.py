"""Opt-in project-local hook installation, preserving unrelated configuration."""
import json
import os
from pathlib import Path
import shlex
import stat
import sys
import tempfile

from .push_consent import git, repository
from .reflex import hook_config

MARKER = "# porter-managed-pre-push-v1"


def atomic_write(path, text, mode=0o600):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".porter-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb" if isinstance(text, bytes) else "w") as out:
            out.write(text)
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def merged_config(path, config):
    if path.is_symlink():
        raise ValueError(f"Refusing to replace a symlinked config: {path}")
    original = path.read_bytes() if path.exists() else None
    obj = json.loads(original) if original is not None else {}
    if not isinstance(obj, dict) or not isinstance(obj.get("hooks", {}), dict):
        raise ValueError(f"Invalid hook configuration: {path}")
    hooks = obj.setdefault("hooks", {})
    for event, groups in config["hooks"].items():
        existing = hooks.setdefault(event, [])
        if not isinstance(existing, list):
            raise ValueError(f"Invalid hook groups for {event}: {path}")
        # Reinstall updates only this installer's own handlers.
        runner = str(Path(__file__).resolve().parents[1] / "scripts" / "porter_hook.py")
        retained = []
        for group in existing:
            if not isinstance(group, dict) or not isinstance(group.get("hooks"), list):
                raise ValueError(f"Invalid handler group in {path}")
            kept = []
            for handler in group["hooks"]:
                if not isinstance(handler, dict):
                    raise ValueError(f"Invalid hook handler in {path}")
                try:
                    args = shlex.split(handler.get("command", ""))
                except ValueError:
                    args = []
                if runner not in args:
                    kept.append(handler)
            if kept:
                retained.append({**group, "hooks": kept})
        hooks[event] = retained + groups
    return original, json.dumps(obj, indent=2) + "\n"


def install(repo, *, db=None, capacity=None):
    root, identity = repository(repo)
    common = Path(identity)
    configured = git(root, "config", "--get", "core.hooksPath", allowed=(0, 1)).stdout.strip()
    hook_dir = (root / configured).resolve() if configured else common / "hooks"
    if configured and not hook_dir.is_relative_to(common):
        raise ValueError("An external/shared core.hooksPath is configured. Chain Porter manually there; do not replace it.")
    pre_push = hook_dir / "pre-push"
    backup = hook_dir / "pre-push.before-porter"
    old_hook = pre_push.read_bytes() if pre_push.exists() else None
    if pre_push.is_symlink():
        raise ValueError("Existing pre-push is a symlink; chain it manually")
    managed = old_hook is not None and old_hook.startswith(("#!/bin/sh\n" + MARKER + "\n").encode())
    if old_hook is not None and not managed and backup.exists():
        raise ValueError("A prior hook backup exists; inspect it before installing")
    plans = []
    for relative, client in ((".claude/settings.local.json", "claude-code"), (".codex/hooks.json", "codex")):
        path = root / relative
        old, new = merged_config(path, hook_config(client, db=db, capacity=capacity))
        plans.append((path, old, new))
    runner = Path(__file__).resolve().parents[1] / "scripts" / "porter_hook.py"
    command = [sys.executable, str(runner)]
    if db:
        command += ["--db", str(Path(db).resolve())]
    command += ["pre-push"]
    script = ("#!/bin/sh\n" + MARKER + "\nset -eu\n"
              "porter_input=$(mktemp)\n"
              "trap 'rm -f \"$porter_input\"' EXIT HUP INT TERM\n"
              "cat > \"$porter_input\"\n"
              f"if [ -x {shlex.quote(str(backup))} ]; then\n"
              f"  {shlex.quote(str(backup))} \"$@\" < \"$porter_input\"\nfi\n"
              f"{shlex.join(command)} \"$@\" < \"$porter_input\"\n")
    # Complete preflight before touching any existing files.
    for path, old, _ in plans:
        saved = path.with_name(path.name + ".before-porter")
        if old is not None and saved.exists() and saved.is_symlink():
            raise ValueError(f"Refusing a symlinked backup: {saved}")
    for path, old, new in plans:
        current = path.read_bytes() if path.exists() else None
        if current != old:
            raise ValueError(f"Configuration changed during installation: {path}")
        if old is not None:
            saved = path.with_name(path.name + ".before-porter")
            if not saved.exists():
                atomic_write(saved, old.decode())
        atomic_write(path, new)
    hook_dir.mkdir(parents=True, exist_ok=True)
    current_hook = pre_push.read_bytes() if pre_push.exists() else None
    if current_hook != old_hook:
        raise ValueError("pre-push changed during installation; configurations were saved, but the gate was not replaced")
    if old_hook is not None and not managed:
        atomic_write(backup, old_hook, stat.S_IMODE(pre_push.stat().st_mode))
    atomic_write(pre_push, script, 0o700)
    # These contain installation-specific absolute paths; keep them local.
    exclude = common / "info" / "exclude"
    old_exclude = exclude.read_text() if exclude.exists() else ""
    entries = ("/.claude/settings.local.json", "/.claude/settings.local.json.before-porter",
               "/.codex/hooks.json", "/.codex/hooks.json.before-porter")
    additions = [line for line in entries if line not in old_exclude.splitlines()]
    if additions:
        atomic_write(exclude, old_exclude.rstrip("\n") + "\n" + "\n".join(additions) + "\n")
    return {"project": str(root), "configs": [str(p) for p, _, _ in plans], "pre_push": str(pre_push),
            "next": "Review and trust the Codex hooks in the client; restart clients if required. Global notify was not changed."}
