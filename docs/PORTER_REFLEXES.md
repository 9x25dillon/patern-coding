# Porter reflexes: automatic coordination checkpoints

This adds lifecycle hooks and a Git push gate to the shared SQLite ledger in
[PORTER.md](PORTER.md). It is the first implementation of the user's SyNAPSE / Porter
brainstorm. The existing MCP server remains the agent interface; the new hooks run
without the model remembering to check in. No daemon, network listener, local model,
or additional Python dependency is required.

## What works

| Trigger | Result |
| --- | --- |
| SessionStart | Register the native session and inject the board, pending questions, decisions, and its saved checkpoint. |
| UserPromptSubmit | Record a redacted task packet. The prompt is not rewritten. |
| PreToolUse | Expire stale hook leases, check explicit edits, and inspect available context usage. Conflicting edits are denied. |
| Observed context use reaches 40% | Ask COMPuLSION to show its 420-character review plus Focus / Next / You and the save/Git offer. Repeat only after usage has been observed below 40%. |
| PreCompact | Persist the last reported state; restore it through SessionStart. |
| Stop | Append a reported turn result and release that session's hook leases. |
| SessionEnd | Release leases and close the session. |
| Codex notify, optional fallback | Record an `agent-turn-complete` result using the same adapter. |
| Git pre-push | Require a matching, unexpired, unused approval for every branch update. Queue a deduplicated question if one is missing. |

Hook ownership uses the client's native session ID, never the temporary hook
process's PID. Hook claims are visible to Porter MCP. On startup, the board tells
the agent to call `porter_checkin` with its `resume_session` value so the hook and MCP
process use the same owner. Starting another MCP owner can cause a self-conflict.

Leases default to 15 minutes, renew when the same file is edited again, and expire
on the next hook invocation. Claim batches are atomic. Move patches claim both
paths. Physical paths are compared across project roots, so nested workspaces and
symlink aliases do not silently avoid a conflict. Edits outside the current project
are denied; coordinate the destination as a separate workspace.

## Install in this checkout

Use Python 3.10 or newer. The launcher works without installing Torch or the AURIC
package. Keep this checkout at the same location after installation: generated
commands contain its absolute path and the selected Python interpreter.

```bash
cd /home/kill/patern-coding
python3 scripts/porter_hook.py config --client claude-code
python3 scripts/porter_hook.py config --client codex
python3 scripts/porter_hook.py install
```

The installer merges `.claude/settings.local.json` and `.codex/hooks.json`, preserves
unrelated settings and handlers, and installs `.git/hooks/pre-push`. An existing
pre-push hook is retained as `pre-push.before-porter` and runs first with the same
input. Its rejection does not consume an approval. First-install config backups use
the suffix `.before-porter`; subsequent installations update Porter's handlers.
Installation paths are excluded through `.git/info/exclude`.

Review and trust the Codex hooks in the client, and restart clients when needed.
The installer does not change the client's hook trust policy. An external or shared
`core.hooksPath` is refused; manually chain the `pre-push` command there after
reviewing that hook's existing workflow.

The shared ledger defaults to `~/.local/share/auric/porter.sqlite`. Set
`AURIC_PORTER_DB` consistently for both clients, MCP servers, and Git, or bake an
absolute database path into the generated configuration:

```bash
python3 scripts/porter_hook.py --db /absolute/path/porter.sqlite install
```

The base MCP server's `--porter-db` must point to the same database. Otherwise the
clients will see different boards.

These adapters follow the current [Codex hook interface](https://learn.chatgpt.com/docs/hooks)
and [Claude Code hook interface](https://code.claude.com/docs/en/hooks). Their
installation has been tested in disposable repositories. Live client loading and
trust must still be checked after activation.

### Context measurements and the human-facing checkpoint

The hook stores packets of **at most** 420 normalized Unicode characters. It does
not pretend that truncating an assistant message produces a complete work review.
COMPuLSION performs the review in the conversation, validates its **exactly**
420-character summary, and shows Focus / Next / You separately. That is the layer
that keeps the user oriented. A hook nudge requests this behavior; it cannot prove
the model displayed or followed it.

Usage comes from an explicit `context_window.used_percentage` when a caller
supplies it, or from a bounded tail of the current transcript. The transcript
adapters are fallbacks rather than stable public APIs:

- Codex: use the latest `last_token_usage` with `model_context_window`, ignoring
  cumulative token totals.
- Claude Code: use assistant input, cache, and output usage only when the operator
  provides the active model's capacity through `--context-window N` at installation.
  Verify that capacity for the actual model. No default capacity is guessed.
- Unknown or malformed usage: skip the percentage trigger. PreCompact still saves
  reported state, and COMPuLSION remains manually callable.

Only the launcher-supplied transcript is read, at most its last 2 MiB. There is no
scan of all sessions, transcript archive, or search index in this milestone.
Claude discards contextual output from PreCompact, so the checkpoint is stored
first and injected on the next SessionStart.

### Optional Codex notify compatibility

Prefer the installed Stop hook for current clients. For a client using `notify`,
the equivalent user-level Codex `config.toml` setting is:

```toml
notify = ["python3", "/home/kill/patern-coding/scripts/porter_hook.py", "notify"]
```

Codex appends the JSON event argument. This setting is user-level, as described in
the [advanced configuration documentation](https://learn.chatgpt.com/docs/config-file/config-advanced).
The installer leaves existing global notification programs alone. Use either Stop
logging or this notify adapter to avoid double-recording turns. Notify alone does
not provide edit gates, startup context, or pre-compaction checkpoints.

## Review and approve a push

After installation, run these from your own terminal when a concrete commit is
ready. Replace `main` and `HEAD` with the destination branch and reviewed commit if
different:

```bash
cd /home/kill/patern-coding
git status --short --branch
git log -1 --oneline
python3 scripts/porter_hook.py approve-push --remote origin --ref refs/heads/main --commit HEAD
git push origin HEAD:refs/heads/main
```

The approval command reads the remote's advertised branch, displays the resolved
commit and destination, and requires the literal response `approve`. It refuses
noninteractive input. Agents must not run it or simulate the user's terminal.
There is no approval-granting MCP tool.

An approval binds the Git common directory, push URL hash, destination ref, current
remote commit, proposed commit, and expiry. Default expiry is five minutes;
`--ttl` accepts 1–3600 seconds. Worktrees share the repository identity. A changed
commit, destination, or remote branch needs a fresh approval. Reapproving the same
update replaces the previous attempt rather than stacking tokens.

Every ref must be approved before any token is consumed. One concurrent attempt
wins. A token authorizes an **attempt**, so a later network failure, server-side
rejection, or dry run can still consume it. Delivery must be verified separately.
Already-up-to-date pushes consume nothing. This version supports branch creation
and fast-forward branch updates; it refuses deletion and tag approval, and does not
offer force-push approval. It requires exactly one configured push URL.

Rejected attempts appear in `python3 -m auric porter status`. Identical pending
requests share a question, which closes when the required approvals exist. A replay
after consumption opens a new request. Ordinary `porter_ask_user` questions are
unchanged; semantic deduplication of every agent question is still future work.
A chat answer or `auric porter answer` records a decision but does not mint a push
token. Merge and other publishing operations are outside this gate.

## Boundaries

- Hook checks coordinate cooperating clients; they are not operating-system file
  locks. Shell writes, tools outside the supported edit shapes, disabled hooks,
  timeouts, and nonparticipating programs can bypass them. Worktree isolation is a
  separate planned feature.
- Successful lease checks return no permission override. The client's own tool
  approvals and sandbox still apply. Malformed supported edits or ledger errors
  return a blocking hook error. SessionEnd cleanup is best effort; lease expiry
  handles abandoned sessions.
- The board labels peer text as data, not instructions or publication authority.
  No peer-message forwarding loop or automatic command execution is introduced.
- Hook packets mask common credentials and URL userinfo before storage. This is
  best-effort redaction, not complete secret detection. The database and its current
  SQLite sidecars are set to mode 0600; the data is local and unencrypted. Existing
  Porter records are not retroactively scrubbed, and raw MCP writes retain the
  foundation's own behavior.
- The pre-push gate is a workflow check. A process with the same user's access can
  change the database, replace hooks, or bypass them. It is not authenticated human
  consent against a malicious local process.

To undo an installation, remove only commands pointing to `porter_hook.py` from
the two client configs. Restore `pre-push.before-porter` if one exists; otherwise
remove the managed pre-push hook after checking its marker. Config backups are a
reference: restoring them wholesale could erase later settings. Keep the ledger
unless its history is intentionally being discarded.

## Verification and next milestones

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

On 2026-09-29 the full suite passed **78 tests**, including 30 new reflex/consent
tests. Coverage includes cross-client collisions, concurrent leases, scope and
protected paths, traversal and symlink escapes, expiry, checkpoint restoration,
malformed input, nonblocking transcript handling, redaction, approval replay and
expiry, atomic multi-ref approval, and real pushes to disposable local bare Git
repositories. Installation tests preserve existing hooks/settings, repeat the
installation, and check Git worktree identity. No production remote was pushed and
no production approval was minted by these tests.

| Milestone | State |
| --- | --- |
| Shared board and MCP tools | Existing Porter foundation; reused here. |
| Lifecycle adapters, edit leases, context nudge, pre-push consent | Implemented; activate and verify with the actual clients. |
| One inbox for arbitrary duplicate questions, attention modes | Future integration over the existing questions/events tables. |
| Local-model packets, tmux cockpit, ntfy, cross-client delegation | Future; no model or external notification calls added. |
| Generated Hand_off.md, transcript recall, INTENT.md drift detection | Future; this slice records checkpoints without rewriting shared handoffs. |
| /duet, /race, worktree orchestration, trailers/rerere/range-diff | Future; no extra agents or worktrees launched. |
| skillport, prompt compiler, growth tracker | Future; mixed-case Claude skill invocations remain supported. |

Continue with [HANDOFF_PORTER_REFLEXES.md](HANDOFF_PORTER_REFLEXES.md).
