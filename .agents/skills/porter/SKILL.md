---
name: porter
description: "Porter: sync this session with the user's other Claude Code and Codex sessions. Check in, show the user what every session is doing, surface decisions waiting on them, and realign the current plan with their directives. Use on /porter or $porter, when starting work while another agent session may be running, or when the user asks what the other sessions are doing."
---

# /porter

Porter is a shared ledger behind the `auric` MCP server. Every Claude Code and Codex session on this
machine talks to it, so each can see the others and the user's directives. Your job with this skill:
bring this session, the other sessions, and the project back in line with what the user wants.

## If the tools are missing

Porter tools appear as `porter_*` (Claude Code: `mcp__auric__porter_*`). If none are available, tell the
user Porter is not connected in this CLI and point them to `docs/PORTER.md` (setup section) in the
patern-coding repository. Do not imitate the tools or invent other sessions.

## Steps

1. **Check in.** Call `porter_checkin` with your current task in one or two sentences and your absolute
   working directory as `cwd`. If you already checked in this session, call `porter_brief` instead.
2. **Show the user the picture**, briefly and in this order:
   - **Your directives:** each active directive (`#id kind: text`), flagging new ones.
   - **Waiting on you:** each open question with the exact reply command:
     `auric porter answer <id> "..."`. Mark which session asked.
   - **Sessions:** for each other live session: agent, task, and claimed paths. Mention stale sessions only
     if they hold claims the user may want cleared (`auric porter end <session>`).
   - **Since last time:** user decisions, handoff summaries, and inbox items that change what should happen.
   Keep it scannable; skip empty sections.
3. **Realign.** Compare your current plan with the directives and user decisions. If they diverge, say
   exactly where and propose the correction. Do not silently switch tasks, and do not decide open
   questions yourself.
4. **Coordinate.** Before editing, `porter_claim` the paths. On conflict, work elsewhere or
   `porter_message` the holding session; on denial, stop and `porter_ask_user`.
5. **Carry the user's words.** If the user states a priority or decision in this chat, relay it to the
   other sessions with `porter_message` to `*`, quoting them exactly and saying it came from the user in
   this session. Suggest the durable form for them to run from their terminal:
   `auric porter steer "..."` (add `--kind protect --paths <dir>` for boundaries). Never run
   `auric porter steer|answer|retire|end` yourself; those are the user's commands.

## Ending work

Before finishing, call `porter_checkout` with what you changed, what you verified, and the next step.
The next session, Claude or Codex, receives this as a handoff. When COMPuLSION or a handoff skill runs,
use `porter_brief` as the source for other active sessions and name each by its session id.
