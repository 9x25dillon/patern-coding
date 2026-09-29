# Agent guide: patern-coding

This repository holds AURIC: a small language model trained from random weights, a local
coding workbench, and **Porter**, which coordinates concurrent Claude Code and Codex
sessions for the user. Read [Hand_off.md](Hand_off.md) for current state and
[docs/PORTER.md](docs/PORTER.md) for how Porter works.

## Working alongside other sessions

The user often runs Claude Code and Codex at the same time. When the `auric` MCP tools
(`porter_*`) are available:

1. Start with `porter_checkin`: your task and absolute working directory.
2. Follow `user_directives`; they outrank your own plan. `scope` and `protect`
   directives are the user's path boundaries.
3. `porter_claim` files before editing them. If a claim conflicts or is denied, do not
   edit those paths.
4. Record decisions and blockers with `porter_note`. Put decisions that belong to the
   user in `porter_ask_user`, then continue with independent work; answers arrive in
   `porter_inbox`.
5. Finish with `porter_checkout` (summary and next steps). It releases your claims.

`auric porter steer|answer|retire|end` are the user's commands; do not run them. The
read-only `auric porter status` and `auric porter history` are fine. If the tools are
not available, say so; do not imitate them.

## Commands

```bash
python -m unittest discover -s tests -p 'test_*.py'   # full suite; test_porter.py needs no torch
python -m auric porter status                         # what every session is doing
python -m auric doctor
```

## Boundaries

- Never execute shell commands or patches emitted by the local model automatically.
- Generated code runs only through VibeCoder's isolated backend; fail closed without it.
- Do not index or train on credentials, wallet files, private keys, tokens, or unrelated
  personal data.
- Do not train on tests or reference answers of task families used for evaluation.
- Keep model output, retrieved evidence, tool results, and user-confirmed memory visibly
  distinct. Preserve failed experiments and their metrics.
- Do not stage untracked legacy projects or artifacts unless the user selects them.
