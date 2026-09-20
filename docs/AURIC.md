# AURIC development guide

Build target: a personal coding assistant whose generative model starts from
random weights, with CPU development now and a future 2–4 GB VRAM GPU. The current
deliverable is a tested development workbench, not a pretrained coding expert.

Run commands from `/home/kill/AURIC_OCTITRICE`. Python 3.10+, PyTorch 2.4+, and
NumPy 1.24+ are declared dependencies; this build was tested with Python 3.14.7
and CPU PyTorch 2.14.0. The existing environment already has the required packages.
No install or network access is needed for the core commands. An optional editable
install of this project exposes the same CLI as `auric`.

## What works now

- A causal transformer with tied input/output embeddings, correctly shifted
  targets, UTF-8 byte tokenization, and optional experimental residual gates.
- Pretraining text and prompt/completion supervised data; prompt labels are
  masked for supervised examples. Padding never contributes to loss.
- Document/task-grouped train/validation splits, exact deduplication, file hashes,
  and memory-mapped token blocks. Preparation holds normalized records in RAM
  within a configurable 256 MB input limit; training reads blocks from disk.
- Gradient accumulation, clipping, warmup/cosine learning rate, CUDA autocast,
  FP16 gradient scaling, activation checkpointing, evaluation and resumable saves.
- Trailing padding is trimmed per batch before attention and GPU transfer.
- SQLite FTS5 search over explicit code/document roots, exact file/line citations,
  reindexing that removes stale files, and durable memory revisions with evidence.
- Read-only source tools, reviewable patch proposals, and isolated VibeCoder
  function testing. There is no execution of shell commands emitted by a model.
- Paired baseline/gate ablations and a synthetic hardware benchmark.
- Public Hugging Face inventory and bounded, commit-pinned data downloads.

## First CPU run

Use a fresh output directory each time; existing runs are never silently overwritten.

```bash
python -m auric doctor
python -m unittest discover -s tests -p 'test_auric.py' -v
python -m auric prepare examples/auric_smoke.jsonl --output artifacts/data/my-smoke --context 128
python -m auric train --data artifacts/data/my-smoke --output artifacts/runs/my-smoke --steps 40 --accumulation 2 --learning-rate 0.001 --warmup-steps 5 --eval-every 20
python -m auric evaluate --checkpoint artifacts/runs/my-smoke/last.pt --data artifacts/data/my-smoke
python -m auric generate --checkpoint artifacts/runs/my-smoke/last.pt "def add" --max-new-tokens 64
```

The authored 16-record fixture is for testing the machinery. It is far too small
to teach general language or programming. Poor generated text at this stage is
expected; a lower held-out loss is the measurable first milestone.

Already-created artifacts in this workspace:

- `artifacts/runs/cpu-ablation/baseline/last.pt`: 128-byte CPU smoke checkpoint.
- `artifacts/runs/cpu-ablation/coherence_gate/last.pt`: paired experimental checkpoint.
- `artifacts/runs/cpu-ablation/comparison.json`: measured comparison.
- `artifacts/runs/assistant-smoke/last.pt`: CPU model with 512-byte context for
  exercising source-context assembly and the coding episode interface.
- `artifacts/knowledge.sqlite`: persistent local source index and user-stated goals.

These artifacts are ignored by Git. Keep valuable checkpoints and datasets backed up
separately. `last.pt` and `best.pt` contain weights, optimizer state, RNG state,
configuration, tokenizer identity, dataset signature, and training-group history.

## Resume versus a new training phase

Resume keeps the original total step budget and optimizer schedule. For a planned
interruption, use `--stop-after` to stop at an optimizer boundary:

```bash
python -m auric train --data artifacts/data/my-smoke --output artifacts/runs/resumable --steps 100 --stop-after 10
python -m auric train --data artifacts/data/my-smoke --output artifacts/runs/resumable --steps 100 --resume artifacts/runs/resumable/last.pt
```

Repeat the original flags when resuming. Model configuration, data signature and
schedule must match; device, precision and CPU thread count may change. Exact
resume equivalence is tested on CPU, not promised across hardware or PyTorch versions.

For continued training or supervised tuning on a new dataset, use `--init-from`
with your own checkpoint, a fresh output directory, and the same architecture.
It starts a new optimizer/schedule and preserves inherited training-group history.
No pretrained external model is involved.

## Documents, code, and project memory

```bash
python -m auric index auric docs /home/kill/VIbe_coder_9xk1ll /home/kill/The_St
python -m auric search "gradient accumulation checkpoint"
python -m auric ask "How does VibeCoder isolate code?"
python -m auric remember editor "Prefer small reviewable patches" --evidence "User preference"
python -m auric memories
python -m auric memories --history
python -m auric forget editor
python -m auric read /home/kill/VIbe_coder_9xk1ll vibecoder/runner.py --start 49 --end 105
```

Search uses BM25 lexical ranking, so it works immediately without an embedding
model. It preserves code fences and indentation. Re-run `index` after source edits.
The index is a snapshot; citations do not certify that a generated claim is true.

`forget` hides the current memory value and appends a tombstone; history remains
available. Memory is never silently updated from model guesses. The SQLite database
stores readable local text and is not encrypted.

Generate an experimental answer with a checkpoint:

```bash
python -m auric ask "checkpoint resume" --checkpoint artifacts/runs/assistant-smoke/last.pt --max-new-tokens 64
```

Prompt construction respects the checkpoint's byte-token context, preserves the
question, and reports which source excerpts fit. A 512-byte model has very little
room for retrieved code. The tool layer can search larger documents, but the model
can only see excerpts within its context. Longer context and a future trained
subword tokenizer are separate, measured upgrades.

Patch proposals never modify the target:

```bash
python -m auric propose-patch /path/to/project relative/file.py --replacement /path/to/replacement.py
```

The first version supports complete-file proposals up to 500 lines. Review and
apply the diff in your normal development workflow.

## VibeCoder exercises and repair episodes

The bridge imports the explicitly supplied **trusted local** VibeCoder checkout.
It does not copy or alter it. Generated submissions always use `Source.THIRD_PARTY`,
which requires an isolating backend; unavailable isolation produces an error.

```bash
python -m auric tasks --vibecoder /home/kill/VIbe_coder_9xk1ll
python -m auric check-code --vibecoder /home/kill/VIbe_coder_9xk1ll --task w1-l1-greet --code examples/greet_solution.py
python -m auric repair --vibecoder /home/kill/VIbe_coder_9xk1ll --task w1-l1-greet --checkpoint artifacts/runs/assistant-smoke/last.pt --attempts 2 --max-new-tokens 64 --output artifacts/episodes/greet.json
```

The smoke model will usually fail the task. The episode should accurately capture
that failure and a bounded repair attempt. If a proposal passes the feedback cases,
the bridge checks a fresh seed before reporting success. A new seed is an additional
check, not an independent task family. Saved episodes are evaluation records and are
not automatically admitted to training.
`repair` and `check-code` exit with status 1 for a completed but unsuccessful attempt,
and status 2 for a configuration or execution error.

On this machine the isolated greeting fixture passed outside the managed development
sandbox; nested isolation was unavailable inside it. On your normal terminal the
installed bubblewrap backend should be tried first. No Docker image is downloaded
by AURIC, and host subprocess execution is never substituted for generated code.

Export explicitly selected reference tasks for supervised data:

```bash
python -m auric curriculum --vibecoder /home/kill/VIbe_coder_9xk1ll --tasks w1-l1-greet w1-l2-bigger w1-l3-count --output artifacts/corpora/vibecoder-train.jsonl
```

Reserve other **task families** for evaluation. Checkpoints record training groups;
`repair` rejects a task ID already recorded as trained. This catches exact IDs, not
semantic copies in prose, source code or older datasets.

## Prepare your own corpus

Accepted JSONL forms:

```json
{"text":"A document or code snippet", "group":"document-123", "source":"notes/design.md"}
{"prompt":"Write a function...", "completion":"def ...", "group":"task-family-7", "source":"reviewed examples"}
```

`content` and `processed_text` are accepted text aliases. Metadata is not serialized
into the training text. For code, exact whitespace is retained. Assign all chunks of
a document, repository duplicate, or related task to a shared `group`. If absent,
`source_file` or a content hash is used. The seed deterministically partitions the
set of groups; adding groups can change which groups are held out, so retain manifests.

```bash
python -m auric export-sources auric --output artifacts/corpora/auric-source.jsonl
python -m auric prepare artifacts/corpora/auric-source.jsonl --output artifacts/data/auric-source --context 512
```

Source export excludes `tests/`, `levels/`, and common test filenames by default;
`--include-tests` is explicit. Docs may still contain reference answers. Exact
deduplication is implemented; near-duplicate and semantic contamination review
remain your responsibility. Separate personal retrieval material from deliberately
curated training data. The scan budget fails before replacing an existing index.

Prepared blocks do not cross document boundaries. Long documents are split into
context windows; a long instruction prefix can exceed the available context, so
review/shorten supervised examples rather than assuming the full prompt was seen.

## GPU arrival

```bash
python -m auric plan --preset 2gb
python -m auric doctor
python -m auric benchmark --preset 2gb --device cuda --steps 3 --output artifacts/benchmarks/my-gpu.json
```

First install a PyTorch build appropriate to the actual GPU. This environment's
PyTorch is CPU-only. CUDA mode is implemented; ROCm/other GPU execution is not verified.

The `2gb` preset has 10.93M parameters and 512-byte context; `4gb` has 25.86M and
1,024-byte context. Both use activation checkpointing. FP32 weights + gradients +
Adam moments alone use about 167 MiB and 395 MiB respectively. These figures exclude
activations, CUDA/runtime memory, optimizer temporaries and allocator reservations.
They are not guarantees that a specific card will fit.

Start at batch size 1 and adjust only after the benchmark. If memory is tight,
reduce context and prepare matching data. Increase accumulation instead of microbatch
size. Older cards may run FP16 slowly or use unfused attention; measure throughput.

```bash
python -m auric prepare /path/to/reviewed.jsonl --output artifacts/data/gpu --context 512
python -m auric train --preset 2gb --device cuda --data artifacts/data/gpu --output artifacts/runs/gpu-first --steps 1000 --batch-size 1 --accumulation 16 --eval-every 100
```

There is no quantized frozen base: all model weights train from scratch. LoRA is
not used to freeze a randomly initialized model. The GPU benchmark records measured
allocated/reserved memory; GPU capacity remains unverified until it runs there.

## Architectural experiments

```bash
python -m auric ablate --data artifacts/data/my-smoke --output artifacts/runs/my-ablation --steps 40
```

This pairs identical baseline initialization, data, sampling seed and schedule with
and without a per-token residual gate. It is inspired by the earlier control/coherence
ideas but does not claim to implement or prove the earlier theoretical framework.
One seed on a tiny fixture cannot establish a general gain. Run multiple seeds and
larger held-out corpora before changing defaults.

## Remaining research milestones

1. Curate a substantially larger language/code corpus with known provenance.
2. Measure GPU fit and throughput, then choose context and training-token budget.
3. Train a usable small base model; evaluate against held-out document/task families.
4. Supervise retrieval-aware answers and tool decisions with verified examples.
5. Consider a trained byte-level BPE tokenizer and longer context. Both require a
   versioned tokenizer/checkpoint migration, not an in-place vocabulary swap.
6. Evaluate architectural changes on multiple seeds, correctness, latency and memory.

Automatic self-training, open-ended shell autonomy, package installation by model
output, persistent semantic embedding search, and a production chat UI are not
implemented. The existing CLI is the development interface.
