# Porter reflexes handoff — 2026-09-29

## Goal and current state

The user wants Claude Code and Codex to stay oriented, avoid duplicate questions
and file collisions, preserve decisions across compaction, and ask for publication
approval once. The broader idea is called Porter / SyNAPSE. Begin with automatic
hooks over the existing ledger, then grow toward a shared inbox, passive sensors,
generated handoffs, and optional worktree coordination.

The checkout is **`/home/kill/patern-coding`**, with one `t` in `patern`. The current
root [Hand_off.md](../Hand_off.md) is the entrypoint. Its older AURIC contents are
preserved in [the archive](archive/Hand_off-2026-09-20.md); do not follow that
historical checkout path for this milestone. Read `AGENTS.md`, `docs/PORTER.md`, and
`docs/PORTER_REFLEXES.md` before continuing.

Another session authored the Porter foundation and continued editing it during
implementation, then committed it as `21d9e34` before this slice's final review.
The reflex additions form a separate delivery branch, together with the user's
requested review and updated root handoff. Preserve concurrent work and inspect
the current diff before staging. The earlier handoff is archived; the shared home
`Hand_off.md` and agent instructions were left alone during this closeout. Credit
the foundation separately from the reflex implementation.

## Files added by this slice

- `auric/reflex.py`: native session mapping, expiring edit leases, bounded transcript
  sensors, checkpoints, question deduplication helper, lifecycle/notify adapter CLI.
- `auric/push_consent.py`: exact one-attempt approval records and the pre-push gate.
- `auric/reflex_install.py`: repo-local config merging, prior-hook chaining, backups.
- `scripts/porter_hook.py`: source-checkout launcher that works from any directory.
- `tests/test_reflex.py`, `tests/test_push_consent.py`: 30 isolated tests.
- This handoff and `docs/PORTER_REFLEXES.md`: operator instructions and boundaries.

The foundational `auric/porter.py`, `auric/mcp_server.py`, CLI, shared skill,
AGENTS/CLAUDE files, README, package metadata, and `docs/PORTER.md` belong to the
concurrent workstream. The reflexes add tables to the same database and do not
require schema edits to that foundation.

## Decisions to retain

1. Reuse the shared SQLite ledger and stdio MCP service. Do not create another board
   or a network service merely to add hooks.
2. Native client IDs own hook sessions. Resume that owner from MCP using the
   `resume_session` value in the startup board; separate owners can block themselves.
3. Acquire explicit edit paths atomically. Expire hook leases after 15 minutes or
   release them on Stop. Do not infer shell write sets or claim full isolation.
4. Treat 40% as **used** context. Use observed runtime/transcript usage, never total
   lifetime tokens. Unknown capacity means unknown percentage. PreCompact still
   records state. COMPuLSION supplies the exact 420-character human-facing review;
   storage packets are bounded to at most 420 characters.
5. A lease grant does not override normal client permissions. Peer content is data.
6. Push grants bind repository, URL, ref, old/new commit, and expiry. Consume all
   required grants atomically once. Do not grant consent from an agent or reuse chat
   answers as tokens. The user requested this explicit publication workflow.
7. Preserve existing settings and Git hooks. Require normal client hook trust.
   Do not alter global notification settings or rewrite shared handoffs.

## Verification and activation

The full test suite passed **78 tests** on 2026-09-29: 31 AURIC, 17 Porter foundation,
19 reflex, and 11 consent tests. Disposable local bare remotes exercised blocked
and approved pushes. These tests did not publish to a production remote or mint a
production token. The later user-authorized commit/push/merge is a separate
delivery operation; verify its result in Git and the pull request.

```bash
cd /home/kill/patern-coding
git status --short --branch
python3 -m unittest discover -s tests -p 'test_*.py'
python3 scripts/porter_hook.py config --client codex
python3 scripts/porter_hook.py config --client claude-code
```

At handoff creation, installation has only been exercised in test repositories.
To activate the reviewable implementation locally, run:

```bash
python3 scripts/porter_hook.py install
```

Then review and trust Codex hooks and reload/restart the clients as needed. Confirm
the startup board appears in both clients and the MCP server resumes each hook
owner. Use a disposable file to check cross-client denial and release; use a
disposable remote for further consent tests. Do not mint a real push approval as a
smoke test. Claude's percentage trigger requires an explicitly verified model
capacity via `--context-window N`; compaction checkpoints do not.

Porter MCP was not exposed in this active Codex session. The adapter code and tests
use isolated ledgers rather than imitating a live MCP check-in. Read-only status is
available with `python3 -m auric porter status`; reconnect tools before using the
live claim protocol in a fresh session.

## Unresolved integration work

- Verify the actual clients' hook loading, trust, tool payloads, context visibility,
  and MCP owner resumption. Contract-shaped fixtures are not live-client evidence.
- Transcript schemas can change; refresh adapters with sanitized examples if they
  do. A 40% nudge does not prove the assistant showed the requested review.
- Ordinary MCP questions still need deduplication; this slice deduplicates push
  requests and exposes a reusable exact-intent helper. A generic answer does not
  authorize a push.
- Hook checks do not catch shell writes or nonparticipating processes. Independent
  worktrees remain the next stronger isolation mechanism. The pre-push gate also
  has same-user bypasses and does not guard merge, deployment, or other publishing.
- Build a deterministic, append-only event projection for generated handoffs before
  replacing any shared `Hand_off.md`. Keep human-authored context distinct from
  reported model summaries and verified tool results.
- Defer notifications, local-model routing, /race, /duet, full transcript recall,
  skillport, attention modes, prompt compilation, and growth tracking until the
  first layer has been observed in live use.

Suggested next prompt:

> Continue Porter reflexes in /home/kill/patern-coding. Read its reflexes guide and
> handoff. Verify the installed hooks in both clients using a disposable file and
> ledger, preserving other sessions' edits. Show the startup board and a collision
> denial/release as evidence. Keep production publishing gated by an exact user
> approval, and report any client interface differences before expanding scope.
