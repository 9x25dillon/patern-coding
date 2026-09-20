"""Explicit-root source access. Code fences and exact line references survive."""
from collections import Counter
import hashlib
import os
import json
from pathlib import Path
import re

EXTENSIONS = {".py", ".md", ".txt", ".js", ".ts", ".tsx", ".jsx", ".rs", ".go",
              ".jl", ".java", ".c", ".h", ".cpp", ".swift", ".toml", ".yaml", ".yml", ".sh"}
SKIP_DIRS = {"node_modules", "venv", "env", "__pycache__", "site-packages", "target",
             "dist", "build", "runs", "artifacts", "auric_data", "numbskull_env"}
SECRET = re.compile(r"-----BEGIN (?:\w+ )?PRIVATE KEY-----|\bhf_[A-Za-z0-9]{20,}|\bgh[pousr]_[A-Za-z0-9]{20,}|\bsk-[A-Za-z0-9_-]{24,}")


def permitted_name(path):
    return (path.suffix.lower() in EXTENSIONS and not any(p.startswith(".") for p in path.parts)
            and not any(word in path.name.lower() for word in ("credential", "secret", "token.cd")))


def read_file(root, relative, *, start=1, end=120, max_bytes=1_000_000):
    root = Path(root).resolve(strict=True)
    requested = Path(relative)
    if requested.is_absolute() or ".." in requested.parts or not permitted_name(requested):
        raise ValueError("Choose an allowed source file relative to the workspace root")
    candidate = root/requested
    if any((root/Path(*requested.parts[:i])).is_symlink() for i in range(1, len(requested.parts)+1)):
        raise ValueError("Symlink paths are not readable through workspace tools")
    path = candidate.resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size > max_bytes:
        raise ValueError("File is outside the workspace or exceeds the read budget")
    if start < 1 or end < start or end-start >= 500:
        raise ValueError("Read between 1 and 500 lines")
    text = path.read_text(encoding="utf-8")
    if "\x00" in text or SECRET.search(text):
        raise ValueError("Binary or credential-bearing source is excluded")
    return {"path": str(path), "start": start, "end": min(end, len(text.splitlines())),
            "text": "\n".join(text.splitlines()[start-1:end])}


def scan(root, *, max_files=3000, max_bytes=32_000_000, chunk_lines=60):
    root = Path(root).resolve(strict=True)
    if not root.is_dir() or min(max_files, max_bytes, chunk_lines) < 1:
        raise ValueError("Choose a directory and positive ingestion budgets")
    chunks, stats, total = [], Counter(), 0
    for folder, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d not in SKIP_DIRS
                         and not (Path(folder)/d).is_symlink())
        for name in sorted(files):
            path = Path(folder)/name
            relative = path.relative_to(root)
            if path.is_symlink() or not permitted_name(relative):
                stats["skipped"] += 1
                continue
            size = path.stat().st_size
            if size > 1_000_000:
                stats["oversized"] += 1
                continue
            total += size
            stats["files_considered"] += 1
            if stats["files_considered"] > max_files or total > max_bytes:
                raise ValueError("Ingestion budget exceeded; choose a narrower root or increase explicit limits")
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeError:
                stats["non_utf8"] += 1
                continue
            if "\x00" in text or SECRET.search(text):
                stats["excluded_content"] += 1
                continue
            stats["files_indexed"] += 1
            lines = text.splitlines()
            digest = hashlib.sha256(text.encode()).hexdigest()
            for i in range(0, len(lines), chunk_lines):
                content = "\n".join(lines[i:i+chunk_lines])
                if content.strip():
                    chunks.append({"root": str(root), "path": str(relative), "start": i+1,
                                   "end": min(i+chunk_lines, len(lines)), "content": content,
                                   "hash": digest})
    return chunks, dict(stats)


def export_sources(roots, output, *, include_tests=False):
    output = Path(output)
    if output.exists():
        raise ValueError("Corpus output already exists")
    records, seen, excluded = [], set(), 0
    for root in roots:
        chunks, _ = scan(root)
        for c in chunks:
            path = Path(c["path"])
            # VibeCoder challenge implementations contain reference answers.
            # Do not accidentally train on the evaluation suite while reading code.
            if not include_tests and ("tests" in path.parts or "levels" in path.parts or
                                      path.name.startswith("test_") or path.name.endswith("_test.py")):
                excluded += 1
                continue
            digest = hashlib.sha256(c["content"].encode()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            records.append({"text": c["content"], "source": str(Path(c["root"])/c["path"]),
                            "group": str(Path(c["root"])/c["path"]), "file_sha256": c["hash"],
                            "start_line": c["start"], "end_line": c["end"]})
    if not records:
        raise ValueError("No source records found")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False)+"\n")
    return {"output": str(output), "records": len(records), "evaluation_chunks_excluded": excluded,
            "note": "Exact chunks deduplicated. Review provenance, near-duplicates and task-family leakage before training."}
