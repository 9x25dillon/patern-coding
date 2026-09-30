# AURIC: local model and coding assistant development

The new development path is **`auric/`**: a small language model trained from
random weights, a persistent document/code index, explicit project memory, and
a VibeCoder bridge for isolated coding exercises. Existing research projects
below remain available as source material and experiments.

**Start here: [AURIC development guide](docs/AURIC.md).**

```bash
python -m auric doctor
python -m unittest discover -s tests -p 'test_auric.py' -v
python -m auric ask "How does checkpoint resume work?"
```

`ask` without a checkpoint returns local source excerpts, not a generated answer.
The local knowledge index is created with `python -m auric index <chosen-root>`.
After initial setup in this workspace it includes AURIC, VibeCoder, The Saint,
and the topological-holography source directory.

| Preset | Parameters | Context, UTF-8 bytes | Purpose |
| --- | ---: | ---: | --- |
| `cpu` | 445,440 | 128 | Fast correctness and learning experiments |
| `2gb` | 10,934,784 | 512 | Conservative future small-GPU training target |
| `4gb` | 25,861,120 | 1,024 | Larger future small-GPU training target |

GPU fit is **not yet measured**. The included CPU checkpoints prove the pipeline
learns and reloads; they are not capable general coding assistants. Model weights
start random and there are no automatic model downloads or external inference calls.

See [build verification](docs/AURIC_VERIFICATION.md),
[architecture and experiments](docs/AURIC_ARCHITECTURE.md), and
[Hugging Face inventory findings](docs/AURIC_HUB.md).

## Porter: steer concurrent Claude Code and Codex sessions

Run Claude Code and Codex on the same project at once without them colliding or
drifting. Both CLIs call the `auric` MCP server (`python -m auric mcp`). Through it
they share one local ledger: who is working on what, which files each has claimed, your
directives, questions waiting on you, and handoffs to the next session.

```bash
auric porter status                                   # every session, claims, questions for you
auric porter steer "Priority: corpus pipeline first"  # every session sees it in its brief
auric porter steer "Hands off" --kind protect --paths MessageVectorizer
auric porter answer 3 "Yes, hold out week 3"          # broadcast to all sessions
```

Agents use `porter_checkin`, `porter_claim`, `porter_note`, `porter_ask_user`, and
`porter_checkout`; `/porter` (Claude Code) or `$porter` (Codex) runs a sync. Setup,
semantics and limits: **[docs/PORTER.md](docs/PORTER.md)**.

The [Porter reflexes](docs/PORTER_REFLEXES.md) add automatic lifecycle checkpoints,
expiring edit leases, and exact one-attempt push approvals. Their repo-local
installer preserves existing settings and Git hooks; live activation is a separate
step. See [Hand_off.md](Hand_off.md) for the current milestone and next checks.

## Earlier research service notes: ChaosRAGJulia

A compact Julia service that unifies a **KFP chaos router**, **HHT/EEMD** time–frequency analytics, and **OpenAI-based RAG** for crypto research.

**License: Apache 2.0** (full text below).

## Install & Run

```bash
export DATABASE_URL=postgres://user:pass@localhost:5432/chaos
# optional
export OPENAI_API_KEY=sk-...

julia --project -e 'using Pkg; Pkg.add.(["HTTP","JSON3","LibPQ","DSP","UUIDs","Interpolations"])'
julia server.jl
```

The server bootstraps the schema and tries to enable `pgvector`. If extensions can’t be installed by your DB role, pre-install them or ignore the warning; tables still create.

## Endpoints

- `POST /chaos/rag/index` — index docs `{docs:[{source,kind,content,meta}]}`
- `POST /chaos/telemetry` — push `asset, realized_vol, entropy, mod_intensity_grad`
- `POST /chaos/hht/ingest` — EEMD + Hilbert on window `{asset, ts[], x[], fs}`
- `POST /chaos/graph/entangle` — upsert edges `{pairs:[[src,dst],...], weight?, nesting_level?, attrs?}`
- `GET  /chaos/graph/:uuid` — fetch node + edges
- `POST /chaos/rag/query` — chaos-routed mixed retrieval + LLM answer `{q, k?}`

## Router (KFP-inspired)
`stress = σ(1.8·vol + 1.5·entropy + 0.8·|grad|)`\
`mix = { vector, graph, hht }` increase HHT/graph under stress, shift back to vector when calm. `top_k` shrinks as stress rises.

## HHT/EEMD
CPU-only minimalist EEMD (ensemble, noise_std, max_imfs). Hilbert features: instantaneous frequency & amplitude with burst flag by amplitude percentile threshold.

## Apache License 2.0
Copyright 2025 Your Name

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

## MessageVectorizer

The repository also contains the existing Julia project in
[`MessageVectorizer/`](MessageVectorizer/), including its Docker setup, demo,
and test runner.
