"""Read public Hugging Face metadata and explicitly selected small data files."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from urllib.parse import quote
from urllib.request import Request, urlopen


def request_json(url):
    with urlopen(Request(url, headers={"User-Agent": "auric-local/0.1"}), timeout=30) as f:
        raw = f.read(4_000_001)
    if len(raw) > 4_000_000:
        raise ValueError("Hub metadata exceeds 4 MB limit")
    return json.loads(raw)


def catalog(author="9x25dillon"):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", author):
        raise ValueError("Invalid Hub author")
    result = {"author": author, "repositories": []}
    for kind in ("models", "datasets"):
        items = request_json(f"https://huggingface.co/api/{kind}?author={quote(author)}&limit=100")
        for item in items:
            info = request_json(f'https://huggingface.co/api/{kind}/{quote(item["id"], safe="/")}')
            result["repositories"].append({"id": info.get("id", item["id"]), "kind": kind,
                "revision": info.get("sha"), "gated": info.get("gated", False),
                "license": (info.get("cardData") or {}).get("license"),
                "files": [s["rfilename"] for s in info.get("siblings", [])]})
    return result


def fetch_data(repo, filename, revision, output, *, kind="datasets", max_bytes=8_000_000):
    if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+", repo):
        raise ValueError("Expected owner/repository")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Use an immutable 40-character commit revision from hub-catalog")
    name = PurePosixPath(filename)
    if name.is_absolute() or ".." in name.parts or name.suffix not in (".jsonl", ".json", ".txt", ".md"):
        raise ValueError("Choose a relative JSONL, JSON, text or Markdown data file")
    if kind not in ("models", "datasets") or max_bytes < 1:
        raise ValueError("Invalid repository kind or size budget")
    output = Path(output)
    if output.exists() or output.with_suffix(output.suffix+".source.json").exists():
        raise ValueError("Download destination already exists")
    prefix = "datasets/" if kind == "datasets" else ""
    url = f"https://huggingface.co/{prefix}{repo}/resolve/{revision}/{quote(filename, safe='/')}"
    with urlopen(Request(url, headers={"User-Agent": "auric-local/0.1"}), timeout=30) as f:
        body = f.read(max_bytes+1)
    if len(body) > max_bytes:
        raise ValueError("Selected file exceeds download budget")
    body.decode("utf-8")
    if body.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise ValueError("Server returned an LFS pointer, not usable data")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as f:
        f.write(body)
    provenance = {"repository": repo, "kind": kind, "revision": revision, "filename": filename,
                  "url": url, "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body),
                  "admitted_to_training": False}
    output.with_suffix(output.suffix+".source.json").write_text(json.dumps(provenance, indent=2))
    return provenance
