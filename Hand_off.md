# AURIC handoff — next session

## Mission

Continue building a small personal coding assistant around a language model trained
from random weights. Develop on CPU now. Target a future GPU with 2–4 GB VRAM. The
assistant should use explicitly indexed project documents, revisioned memory, and
tested development tools.

## Repository and current commit

- GitHub: `https://github.com/9x25dillon/patern-coding`
- Current branch: `main`
- Current remote commit: `82c770f Merge patern-coding and preserve AURIC workbench`
- Previous AURIC implementation commit: `d5fe451`
- The remote already contained `MessageVectorizer`; preserve it.
- The broader workspace contains many untracked legacy projects and artifacts. Do
  not stage them unless the user explicitly selects them.

## Read first

1. [docs/AURIC.md](docs/AURIC.md) — commands, model presets, data workflow, and
   operating limits.
2. [docs/AURIC_ARCHITECTURE.md](docs/AURIC_ARCHITECTURE.md) — contracts and the
   experimental gate.
3. [docs/AURIC_VERIFICATION.md](docs/AURIC_VERIFICATION.md) — evidence and known
   limitations.
4. [docs/AURIC_HUB.md](docs/AURIC_HUB.md) — Hugging Face inventory and provenance.
5. [docs/REVIEW_2026-09-20.md](docs/REVIEW_2026-09-20.md) — decisions and unresolved
   assumptions from the previous session.

## Current implementation

The primary code lives in `auric/`:

- `model.py`: byte-level causal decoder, tied embeddings, optional coherence gate.
- `tokenizer.py`: UTF-8 byte tokenizer, vocabulary size 260.
- `data.py`: normalized JSONL, deduplication, grouped split, checksummed memmaps.
- `training.py`: CPU/GPU training, validation, checkpoint/resume, deterministic RNG.
- `workspace.py` and `memory.py`: bounded source indexing and SQLite FTS5 memory.
- `assistant.py`: bounded context assembly, retrieval-only answer mode, patch diff.
- `vibecoder_bridge.py`: isolated code evaluation and bounded repair episodes.
- `hub.py`: public Hugging Face catalog and pinned small-file fetch.
- `benchmark.py` and `experiments.py`: hardware checks and paired ablation.

## Verified state

Run:

```bash
python -m unittest discover -s tests -p 'test_auric.py' -v
```

The focused suite passed 31 tests at the end of the session. The package also built
as a wheel. Existing smoke artifacts are ignored by Git:

- `artifacts/runs/cpu-ablation/`
- `artifacts/runs/assistant-smoke/`
- `artifacts/data/`
- `artifacts/knowledge.sqlite`
- `artifacts/huggingface/catalog.json`

The smoke model learned on a tiny authored corpus but generated whitespace in the
greeting repair episode. Treat that as a correct failure record, not a model success.
The hand-authored greeting passed 8/8 VibeCoder tests through the isolated bridge.

## First commands next session

```bash
cd /home/kill/AURIC_OCTITRICE
git status --short --branch
python -m auric doctor
python -m unittest discover -s tests -p 'test_auric.py' -v
python -m auric ask "How does checkpoint resume work?"
python -m auric plan --preset 2gb
```

If the user has a GPU by then:

```bash
python -m auric doctor
python -m auric benchmark --preset 2gb --device cuda --steps 3 --output artifacts/benchmarks/gpu-2gb.json
```

Do not claim a preset fits until that benchmark runs on the actual card. Start with
batch size 1 and adjust accumulation before increasing microbatch size.

## Recommended next milestone

Curate a larger, licensed, provenance-tracked corpus from selected source files and
VibeCoder-style tasks. Keep training and evaluation task families separate. Prepare
it with:

```bash
python -m auric export-sources <selected-root> --output artifacts/corpora/source.jsonl
python -m auric prepare artifacts/corpora/source.jsonl --output artifacts/data/source --context 512
```

Then run a real baseline training job with a fixed seed and record held-out loss,
target tokens, elapsed time, and checkpoint metadata. Only after that should the
experimental gate, a subword tokenizer, longer context, embeddings, or a UI be
advanced.

## Safety and quality boundaries

- Never execute shell commands or patches emitted by the model automatically.
- Generated code must use VibeCoder’s isolated backend; fail closed if unavailable.
- Do not index credentials, wallet files, private keys, tokens, or unrelated personal
  data. Review public Hugging Face repositories before importing them.
- Do not use reference answers or test files in training when the corresponding task
  family is used for evaluation.
- Keep model output, retrieved evidence, tool results, and user-confirmed memory
  visibly distinct.
- Preserve failed experiments and their metrics. A failed ablation is evidence.

## How to prompt the next session

Start with the priority and acceptance test. Example:

> “Priority: build a clean 100k–1M-token coding corpus from selected local sources.
> Preserve provenance and exclude tests/reference answers. Add leakage checks and
> tests. Do not change the model architecture. Report the corpus counts and run the
> full focused suite.”

If asking for an architectural experiment, specify the baseline, seed count, held-out
split, and metric before implementation. If asking for a repository mutation, specify
the allowed paths and whether to commit or push.

## Open questions

- Which exact GPU model and VRAM will be available?
- Is the first useful product a local retrieval/tool assistant, a trained small model,
  or both with separate milestones?
- Which local documents are licensed and appropriate for training versus retrieval?
- Which coding task families should remain permanently held out?
- Should the next tokenizer remain byte-level for comparability, or should we begin a
  versioned BPE experiment after the baseline corpus is ready?
