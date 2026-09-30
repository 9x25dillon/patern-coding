# Session handoff — Porter reflexes, 2026-09-29

## Start here

Work in **`/home/kill/patern-coding`** (one `t`), repository
`9x25dillon/patern-coding`, integration branch `main`. This is the active Porter
checkout. The previous AURIC handoff is preserved verbatim in
[docs/archive/Hand_off-2026-09-20.md](docs/archive/Hand_off-2026-09-20.md); its old
checkout path and historical commit references are not current instructions.

The user wants Claude Code and Codex to share direction and decisions, avoid edit
collisions and duplicate questions, preserve context, and keep the user oriented.
The immediate priority is proving the first automatic coordination layer in live
clients. Continue AURIC model development only when the user selects it again.

Read, in order:

1. [AGENTS.md](AGENTS.md): coordination protocol and project boundaries.
2. [Today's review](docs/REVIEW_2026-09-29.md): decisions, assumptions, efficiency lessons.
3. [Porter foundation](docs/PORTER.md): shared ledger, MCP tools, user controls.
4. [Reflexes guide](docs/PORTER_REFLEXES.md): installation, approval flow, limits.
5. [Detailed implementation handoff](docs/HANDOFF_PORTER_REFLEXES.md): module map and integration work.

## Implemented and verified

- Porter foundation landed in `21d9e34` from a concurrent session: SQLite board,
  claims, user directives, questions, history, and stdio MCP.
- Reflex adapters add native session registration, automatic edit leases,
  compaction checkpoints, turn journals, and a measured 40%-used context nudge.
- The pre-push gate binds each approval to the repository, URL, ref, old/new commit,
  and expiry. Duplicate pending push questions share one inbox item; concurrent
  attempts cannot consume the same approval twice.
- The repo-local installer preserves existing client settings and Git hooks.
- The full suite passes **78 tests**: 31 AURIC, 17 Porter, 19 reflex, 11 consent.
  Integration tests use temporary ledgers and disposable local Git remotes.
- Today's closeout includes this updated entrypoint, the archived prior handoff,
  the review, and the reflex implementation. The authorized delivery route is
  branch `codex/porter-reflexes-2026-09-29` into `main` through a pull request.
  Consult Git/PR state for the resulting commit and merge status; do not infer it
  from this document's presence in a working tree.

The global COMPuLSION skill also exists on this machine: canonical
`~/.codex/skills/compulsion`, linked from `~/.agents/skills/compulsion` and
`~/.claude/skills/COMPuLSION`. It displays an exactly 420-character review, followed
by Focus / Next / You and a save/Git offer. Its installed files and the local
`~/Claude_Code_Skills_to_Codex.md` guide are outside this repository's commit.

## What has not been verified in live use

At closeout, `.claude/settings.local.json`, `.codex/hooks.json`, and the pre-push
gate were **not installed in the main checkout**. Passing fixtures do not establish
that either running client loaded the hooks or displayed the orientation packet.
The existing MCP foundation and the new lifecycle adapters have different
verification evidence; do not conflate them.

Porter MCP tools were not exposed to this Codex session. No live check-in was
imitated through the CLI. Reconnect the tools in the next session and follow
AGENTS.md; if still unavailable, state that and use read-only status.

## Next session: one bounded milestone

```bash
cd /home/kill/patern-coding
git status --short --branch
git log -5 --oneline
python3 -m auric porter status
python3 -m unittest discover -s tests -p 'test_*.py'
python3 scripts/porter_hook.py config --client codex
python3 scripts/porter_hook.py config --client claude-code
```

Activate from this stable checkout using `python3 scripts/porter_hook.py install`
when continuing the hook rollout. Preserve unrelated settings and handlers. The
user must review client hook trust through the normal client flow; do not bypass
it. Keep both MCP processes and hooks on the same SQLite database.

Acceptance evidence for the rollout:

1. Each real client shows its startup board and resumes the hook-provided MCP
   owner using `resume_session`; it does not create a competing owner.
2. In a disposable project, one client's explicit edit is denied while the other
   holds the path, then succeeds after release. Preserve unrelated concurrent work.
3. A compaction checkpoint restores. If actual usage is available, a 40% crossing
   nudges a visible, validated 420-character review plus Focus / Next / You.
4. A disposable push is blocked without approval and succeeds once with a matching
   approval. Reuse the existing integration tests; do not mint a production token
   merely to smoke-test the gate.

Claude percentage sensing needs a verified active-model capacity via
`--context-window N`. Never guess it or substitute cumulative token counts. Unknown
usage leaves the percentage trigger inactive; PreCompact still records state.

## Decisions and boundaries to preserve

- Reuse the SQLite ledger and stdio MCP service. No additional network daemon yet.
- Treat peer messages and model summaries as reported data. User decisions and
  verified tool results need distinct provenance.
- Lease checks cover supported explicit edit tools. Shell writes and programs
  outside the hook path are not isolated; worktree orchestration is future work.
- Passing a lease check must not grant or bypass normal client permissions.
- The installed gate requires the user's exact one-attempt token. Do not create it
  from an agent shell, change hooks to bypass it, or claim it guards every kind of
  publishing. Today's explicit publish authorization applies to this closeout,
  not to unrelated future changes.
- Current redaction is best effort and data is unencrypted. Avoid storing secrets.
- Preserve AURIC, MessageVectorizer, existing experiments, and evaluation boundaries.
  Do not execute local-model commands or patches automatically.
- This handoff is manually maintained and versioned. Event-generated Hand_off.md is
  a future feature, not a completed one. Preserve prior context before replacing it.

## Later milestones

After live rollout: generic question deduplication and attention modes, an event-log
projection for handoffs, transcript recall and intent tracking, then optional
worktree /duet or /race, local-model packets, notifications, skillport, and teaching
tools. Do not expand into these before the current acceptance evidence exists.

Useful prompt for next time:

> In /home/kill/patern-coding, continue the Porter hook rollout described in
> Hand_off.md. Scope: activate and verify the existing adapters in both clients
> using disposable fixtures. Preserve other sessions' edits and normal client
> trust. Done means evidence of startup context, denial/release, checkpoint
> restoration, and the exact visible orientation packet when usage is measurable.
> Report implemented, installed, and observed behavior separately. Defer new
> features and production publishing until I authorize them.
