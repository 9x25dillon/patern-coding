# Build verification — 2026-09-20

Environment: Python 3.14.7, PyTorch 2.14.0 CPU build, NumPy installed, SQLite 3.53.4
with FTS5. No CUDA GPU is visible. Existing companion repositories were read and
used through their public Python interfaces; their source files were not modified.

## Correctness suite

`python -m unittest discover -s tests -p 'test_auric.py' -v`

**31 tests passed.** An installable wheel also built successfully with
`python -m pip wheel --no-deps --no-build-isolation --wheel-dir /tmp/auric-wheel .`.
No global Python installation was changed.

The suite covers UTF-8/code round-trip, supervised label masking, future-token
invariance, padding-independent loss, activation-checkpoint gradient equivalence,
paired gate initialization, deterministic sampling, parameter counting, grouped
splits, exact deduplication, next-token labels, corrupted-data rejection, actual
learning, checkpoint reload, exact CPU resume, schedule mismatch rejection,
batch trimming, source citations, code fences, stale-file removal, memory revisions,
path boundaries, failed-scan atomicity, patch proposal non-mutation, corpus export,
isolated-run selection, no unsafe fallback, repair feedback and task-group exclusion.

Tests for repair success use controlled mock model outputs; they verify orchestration,
not the trained smoke model's programming ability.

## Actual model runs

16 authored fixture records, grouped before splitting: 14 training records, 2
validation records, 283 held-out byte/EOS targets. All weights initialized randomly.

| Run | Parameters | Steps | Initial held-out loss | Final held-out loss |
| --- | ---: | ---: | ---: | ---: |
| CPU baseline, context 128 | 445,440 | 40 | 5.51815 | 3.23435 |
| Experimental gate, context 128 | 478,464 | 40 | 5.51815 | 3.23819 |
| CPU assistant-interface smoke, context 512 | 494,592 | 40 | 5.52443 | 3.19403 |

Artifacts: `artifacts/runs/cpu-ablation/comparison.json` and
`artifacts/runs/assistant-smoke/summary.json`. These initial runs preceded the
trailing-padding trim optimization; saved checkpoints remain compatible.

The gate was slightly worse in the paired run, so it remains disabled by default.
These very small, single-seed experiments prove learning machinery, not general
language understanding, robust code generation, or an architectural advantage.

The real 512-context smoke checkpoint was used for source-conditioned generation
and a two-attempt VibeCoder episode. It generated whitespace rather than valid code.
Both attempts correctly failed with `MissingFunction`; the complete failure record
is `artifacts/episodes/greet-smoke.json`. No successful model-generated coding
capability is claimed.

## Actual source/tool integration

- Indexed AURIC source/docs, VibeCoder, The Saint, and
  `Consciousness_as_Topological_Holography` into `artifacts/knowledge.sqlite`.
- Queried checkpoint/resume documentation and verified file/line citations.
- Stored the user's stated hardware target and assistant purpose as explicit
  evidence-backed memory entries.
- Exported AURIC source records and prepared a separate 512-context grouped dataset.
- A hand-authored greeting fixture passed **8/8** VibeCoder cases under an isolating
  backend outside the managed development sandbox. This proves the bridge, not
  model ability. Nested isolation was unavailable inside the development sandbox
  and correctly failed closed.

## Hardware preparation

Both final GPU-target architectures completed synthetic forward/backward/Adam
steps on CPU with activation checkpointing:

| Target | Parameters | Context | Persistent FP32 weights/gradients/Adam estimate |
| --- | ---: | ---: | ---: |
| 2 GB target | 10,934,784 | 512 | 166.9 MiB |
| 4 GB target | 25,861,120 | 1,024 | 394.6 MiB |

Recorded in `artifacts/benchmarks/2gb-512-cpu.json` and
`artifacts/benchmarks/4gb-1024-cpu.json`. These CPU benchmarks ran concurrently
with other checks; their throughput numbers are not isolated hardware comparisons.
GPU activation/runtime allocation, CUDA precision behavior and actual VRAM fit are
unverified. The `benchmark --device cuda` command is ready for arrival-day testing.

## Hugging Face

Recorded live public metadata, downloaded one small commit-pinned JSONL file,
and verified it exactly duplicates the existing local prompts. No pretrained weights
were loaded, no remote Python was executed, and nothing was published. Details and
immutable provenance are in [the Hub findings](AURIC_HUB.md).
