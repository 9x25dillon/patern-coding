"""Document-grouped splits and disk-backed, next-byte prediction batches."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
from torch.utils.data import Dataset
import torch

from .tokenizer import ByteTokenizer


def fingerprint(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_record(raw: dict) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("Each JSONL record must be an object")
    if "prompt" in raw and "completion" in raw:
        result = {k: raw[k] for k in ("prompt", "completion")}
    else:
        text = next((raw[k] for k in ("text", "content", "processed_text") if raw.get(k)), None)
        result = {"text": text}
    if not all(isinstance(v, str) for v in result.values()) or not any(v.strip() for v in result.values()):
        raise ValueError("Each record needs nonempty text, or string prompt/completion")
    # Preserve code whitespace; only normalize line endings and outer whitespace
    # for duplicate detection. Metadata never enters the language objective.
    canonical = json.dumps({k: v.replace("\r\n", "\n").strip() for k, v in result.items()}, sort_keys=True)
    result["hash"] = fingerprint(canonical)
    result["group"] = str(raw.get("group") or raw.get("source_file") or result["hash"])
    result["source"] = str(raw.get("source", raw.get("source_file", "unspecified")))
    return result


def prepare(inputs, output, *, context=128, validation_fraction=0.1, seed=42, max_input_bytes=256_000_000):
    if context < 2 or not 0 < validation_fraction < 1:
        raise ValueError("context >= 2 and validation_fraction in (0, 1) required")
    output = Path(output)
    if output.exists():
        raise ValueError("Dataset output already exists; choose a fresh directory")
    if sum(Path(p).stat().st_size for p in inputs) > max_input_bytes:
        raise ValueError("Corpus exceeds preprocessing budget; split/reduce it or increase max_input_bytes explicitly")
    rows, seen, duplicates = [], set(), 0
    for name in inputs:
        with Path(name).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = normalize_record(json.loads(line))
                except (ValueError, TypeError, KeyError) as exc:
                    raise ValueError(f"{name}:{line_number}: {exc}") from exc
                if row["hash"] in seen:
                    duplicates += 1
                    continue
                seen.add(row["hash"])
                rows.append(row)
    groups = sorted({r["group"] for r in rows}, key=lambda g: fingerprint(f"{seed}:{g}"))
    if len(groups) < 2:
        raise ValueError("Need at least two independent document/task groups for a held-out split")
    val_groups = set(groups[:min(len(groups)-1, max(1, math.ceil(len(groups)*validation_fraction)))])
    output.mkdir(parents=True)
    tok = ByteTokenizer()
    counts = {}
    try:
        for split in ("train", "val"):
            stats = Counter()
            with (output / f"{split}.x.bin").open("wb") as xf, (output / f"{split}.y.bin").open("wb") as yf:
                for row in rows:
                    if (row["group"] in val_groups) != (split == "val"):
                        continue
                    ids, mask = tok.example(row)
                    stats["records"] += 1
                    for start in range(0, len(ids)-1, context):
                        x = ids[start:start+context]
                        y = [v if keep else -100 for v, keep in zip(
                            ids[start+1:start+context+1], mask[start+1:start+context+1])]
                        # Final input must have a corresponding target.
                        x = x[:len(y)]
                        if not y or all(t == -100 for t in y):
                            continue
                        stats["target_tokens"] += sum(t != -100 for t in y)
                        np.array(x + [tok.pad_id]*(context-len(x)), dtype="<u2").tofile(xf)
                        np.array(y + [-100]*(context-len(y)), dtype="<i4").tofile(yf)
                        stats["blocks"] += 1
            if not stats["blocks"]:
                raise ValueError(f"No trainable blocks in {split}")
            counts[split] = dict(stats)
        manifest = {"format": 1, "tokenizer": tok.version, "context": context, "seed": seed,
                    "duplicates_removed": duplicates, "splits": counts,
                    "records": [{k: r[k] for k in ("hash", "group", "source")} |
                                {"split": "val" if r["group"] in val_groups else "train"} for r in rows]}
        manifest["files"] = {p.name: file_hash(p) for p in output.glob("*.bin")}
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    except Exception:
        # Leave an explicit incomplete directory rather than a valid manifest.
        raise
    return manifest


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


class TokenDataset(Dataset):
    def __init__(self, path, split):
        root = Path(path)
        self.manifest = json.loads((root / "manifest.json").read_text())
        if self.manifest["tokenizer"] != ByteTokenizer.version:
            raise ValueError("Tokenizer version mismatch")
        shape = (self.manifest["splits"][split]["blocks"], self.manifest["context"])
        for suffix in ("x", "y"):
            name = f"{split}.{suffix}.bin"
            if file_hash(root/name) != self.manifest["files"][name]:
                raise ValueError(f"Dataset checksum mismatch: {name}")
        self.x = np.memmap(root/f"{split}.x.bin", mode="r", dtype="<u2", shape=shape)
        self.y = np.memmap(root/f"{split}.y.bin", mode="r", dtype="<i4", shape=shape)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return torch.from_numpy(self.x[index].astype(np.int64)), torch.from_numpy(self.y[index].astype(np.int64))
