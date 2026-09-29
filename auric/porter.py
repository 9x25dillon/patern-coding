"""Porter: a shared ledger that keeps concurrent agent sessions pointed at the user.

Every Claude Code or Codex session starts its own `auric mcp` process. Those
processes share one SQLite file, so each session can see what the others are
doing, avoid editing the same files, carry notes forward to the next session,
and queue questions that only the user can answer.

Directives (priorities, constraints, scope and protected paths) come from the
user through `auric porter ...`; the MCP server gives agents no tool to change
them. Claims are advisory: they coordinate cooperating agents, they are not a
filesystem lock or a security boundary.
"""
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import fnmatch
import json
import os
from pathlib import Path
import re
import secrets
import socket
import sqlite3

DIRECTIVE_KINDS = ("priority", "constraint", "preference", "scope", "protect")
PATH_KINDS = ("scope", "protect")
NOTE_KINDS = ("progress", "decision", "finding", "blocker", "test", "handoff")
NOTABLE = ("decision", "finding", "blocker", "handoff", "checkout", "ended", "answer")
MAX_TEXT = 4000
MAX_PATHS = 64

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY, agent TEXT NOT NULL, project TEXT NOT NULL, task TEXT NOT NULL,
    host TEXT, pid INTEGER, started TEXT NOT NULL, last_seen TEXT NOT NULL,
    ended TEXT, summary TEXT, cursor INTEGER NOT NULL DEFAULT 0,
    seen_directive INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS directives (
    id INTEGER PRIMARY KEY, project TEXT NOT NULL, kind TEXT NOT NULL, text TEXT NOT NULL,
    paths TEXT NOT NULL DEFAULT '[]', rank INTEGER NOT NULL DEFAULT 100,
    via TEXT NOT NULL, created TEXT NOT NULL, retired TEXT);
CREATE TABLE IF NOT EXISTS claims (
    session TEXT NOT NULL, project TEXT NOT NULL, path TEXT NOT NULL, intent TEXT NOT NULL,
    created TEXT NOT NULL, PRIMARY KEY (session, path));
CREATE TABLE IF NOT EXISTS questions (
    id INTEGER PRIMARY KEY, project TEXT NOT NULL, session TEXT NOT NULL, question TEXT NOT NULL,
    options TEXT NOT NULL DEFAULT '[]', context TEXT NOT NULL DEFAULT '', created TEXT NOT NULL,
    answer TEXT, answered TEXT, via TEXT);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY, project TEXT NOT NULL, actor TEXT NOT NULL, kind TEXT NOT NULL,
    message TEXT NOT NULL, paths TEXT NOT NULL DEFAULT '[]', recipient TEXT, ref INTEGER,
    created TEXT NOT NULL, delivered_to TEXT);
CREATE INDEX IF NOT EXISTS events_project ON events(project, id);
CREATE INDEX IF NOT EXISTS claims_project ON claims(project);
"""

# "*" is the project for global directives and user broadcasts.
GLOBAL = "*"


def default_db():
    if os.environ.get("AURIC_PORTER_DB"):
        return Path(os.environ["AURIC_PORTER_DB"])
    base = os.environ.get("XDG_DATA_HOME") or Path.home()/".local"/"share"
    return Path(base)/"auric"/"porter.sqlite"


def find_project(start=None):
    """Nearest enclosing Git work tree, else the directory itself."""
    here = Path(start or os.getcwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate/".git").exists():
            return str(candidate)
    return str(here)


def origin():
    """Best-effort label for where a user-only command was run from.

    An agent can run the CLI through its shell. That is not prevented; it is
    labelled, so the user and other sessions can see it did not come from the
    user's own terminal.
    """
    if os.environ.get("CLAUDECODE"):
        return "claude-code shell"
    if any(os.environ.get(v) for v in ("CODEX_SANDBOX", "CODEX_SANDBOX_NETWORK_DISABLED")):
        return "codex shell"
    return "terminal"


def agent_name(client):
    """Stable short agent label; it prefixes session ids."""
    client = (client or "").lower()
    if "claude" in client:
        return "claude-code"
    if "codex" in client:
        return "codex"
    return re.sub(r"[^a-z0-9_-]+", "-", client).strip("-")[:32] or "agent"


def now():
    return datetime.now(timezone.utc)


def stamp(moment=None):
    return (moment or now()).isoformat(timespec="seconds")


def parse(value):
    return datetime.fromisoformat(value)


def ago(value, reference=None):
    seconds = int(((reference or now()) - parse(value)).total_seconds())
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if seconds >= size:
            return f"{seconds//size}{unit} ago"
    return "just now"


def text_arg(value, name, *, required=True):
    value = (value or "").strip()
    if required and not value:
        raise ValueError(f"{name} is required")
    if len(value) > MAX_TEXT:
        raise ValueError(f"{name} exceeds {MAX_TEXT} characters; summarize it")
    return value


def relative_path(project, path):
    """Project-relative POSIX path. Claims name concrete files or directories."""
    if not isinstance(path, str) or not path.strip():
        raise ValueError("Paths must be non-empty strings")
    if any(ch in path for ch in "*?["):
        raise ValueError(f"Claim concrete files or directories, not glob patterns: {path}")
    candidate = Path(path.strip())
    if candidate.is_absolute():
        try:
            candidate = candidate.resolve().relative_to(project)
        except ValueError:
            raise ValueError(f"{path} is outside project {project}") from None
    candidate = Path(os.path.normpath(candidate))
    if candidate.parts[:1] == ("..",):
        raise ValueError(f"{path} is outside project {project}")
    text = candidate.as_posix()
    return "." if text in ("", ".") else text


def overlaps(a, b):
    """True when one path is the other or contains it."""
    return "." in (a, b) or a == b or a.startswith(b+"/") or b.startswith(a+"/")


def covered(path, pattern):
    """True when a directive pattern applies to all of `path`."""
    pattern = pattern.strip().rstrip("/") or "."
    return pattern == "." or path == pattern or path.startswith(pattern+"/") or fnmatch.fnmatchcase(path, pattern)


def touches(path, pattern):
    """True when claiming `path` would include anything matched by `pattern`."""
    if covered(path, pattern):
        return True
    # Fixed directory part of the pattern ("keys/*.pem" -> "keys"). A claim on,
    # inside, or above it may include matching files. A pattern with no fixed
    # directory ("*.lock") only blocks whole-project claims and matching files.
    static = pattern.strip().rstrip("/")
    for i, ch in enumerate(static):
        if ch in "*?[":
            static = static[:i].rsplit("/", 1)[0] if "/" in static[:i] else ""
            break
    return path == "." or (static != "" and overlaps(path, static))


def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class Porter:
    def __init__(self, path=None, *, ttl_hours=None):
        self.path = Path(path) if path else default_db()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours if ttl_hours is not None
                             else float(os.environ.get("AURIC_PORTER_TTL_HOURS", 12)))
        self.host = socket.gethostname()
        # Autocommit mode; writes go through _tx() so check-then-insert is atomic
        # across the separate server processes of concurrent sessions.
        self.db = sqlite3.connect(self.path, timeout=15, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA busy_timeout=15000")
        with self._tx():
            for statement in filter(str.strip, SCHEMA.split(";")):
                self.db.execute(statement)

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @contextmanager
    def _tx(self):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        self.db.execute("COMMIT")

    def _event(self, project, actor, kind, message, *, paths=(), recipient=None, ref=None):
        cursor = self.db.execute(
            "INSERT INTO events(project,actor,kind,message,paths,recipient,ref,created) VALUES(?,?,?,?,?,?,?,?)",
            (project, actor, kind, message, json.dumps(list(paths)), recipient, ref, stamp()))
        return cursor.lastrowid

    def _max_event(self):
        return self.db.execute("SELECT COALESCE(MAX(id),0) FROM events").fetchone()[0]

    # Sessions ---------------------------------------------------------------

    def checkin(self, agent, project, task, *, pid=None, resume=None):
        task = text_arg(task, "task")
        project = str(Path(project).resolve())
        with self._tx():
            if resume:
                row = self._session_row(resume)
                if row["ended"]:
                    raise ValueError(f"Session {resume} has ended; check in without resume")
                self.db.execute("UPDATE sessions SET task=?, pid=?, host=?, last_seen=? WHERE id=?",
                                (task, pid, self.host, stamp(), resume))
                self._event(row["project"], resume, "checkin", f"resumed: {task}")
                return self.session(resume)
            session_id = f"{agent}-{secrets.token_hex(2)}"
            while self.db.execute("SELECT 1 FROM sessions WHERE id=?", (session_id,)).fetchone():
                session_id = f"{agent}-{secrets.token_hex(3)}"
            # Start after existing broadcasts; messages queued for this agent
            # name are still delivered because they are matched separately.
            self.db.execute("INSERT INTO sessions(id,agent,project,task,host,pid,started,last_seen,cursor) "
                            "VALUES(?,?,?,?,?,?,?,?,?)",
                            (session_id, agent, project, task, self.host, pid, stamp(), stamp(), self._max_event()))
            self._event(project, session_id, "checkin", task)
        return self.session(session_id)

    def _session_row(self, session_id):
        row = self.db.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        if row is None:
            raise ValueError(f"Unknown session {session_id}")
        return row

    def session(self, session_id):
        return self._describe(self._session_row(session_id))

    def update_task(self, session_id, task):
        task = text_arg(task, "task")
        with self._tx():
            row = self._live_row(session_id)
            self.db.execute("UPDATE sessions SET task=?, last_seen=? WHERE id=?", (task, stamp(), session_id))
            self._event(row["project"], session_id, "checkin", f"task: {task}")
        return self.session(session_id)

    def _live_row(self, session_id):
        row = self._session_row(session_id)
        if row["ended"]:
            raise ValueError(f"Session {session_id} has ended; check in again to continue")
        return row

    def touch(self, session_id):
        self.db.execute("UPDATE sessions SET last_seen=? WHERE id=? AND ended IS NULL", (stamp(), session_id))

    def alive(self, row, reference=None):
        if row["ended"] or (reference or now()) - parse(row["last_seen"]) > self.ttl:
            return False
        if row["pid"] and row["host"] == self.host:
            return pid_alive(row["pid"])
        return True

    def _describe(self, row):
        item = {k: row[k] for k in ("id", "agent", "project", "task", "started", "last_seen", "ended", "summary")}
        item["alive"] = self.alive(row)
        item["claims"] = [dict(c) for c in self.db.execute(
            "SELECT path, intent, created FROM claims WHERE session=? ORDER BY path", (row["id"],))]
        return item

    def sessions(self, project, *, include_ended=False):
        rows = self.db.execute("SELECT * FROM sessions WHERE project=? " +
                               ("" if include_ended else "AND ended IS NULL ") +
                               "ORDER BY last_seen DESC", (project,))
        return [self._describe(r) for r in rows]

    def checkout(self, session_id, summary, next_steps=""):
        summary = text_arg(summary, "summary")
        next_steps = text_arg(next_steps, "next_steps", required=False)
        return self._end(session_id, summary + (f"\nNext: {next_steps}" if next_steps else ""),
                         actor=session_id, kind="checkout")

    def end(self, session_id, reason, *, actor):
        return self._end(session_id, text_arg(reason, "reason"), actor=actor, kind="ended")

    def _end(self, session_id, summary, *, actor, kind):
        with self._tx():
            row = self._session_row(session_id)
            if row["ended"]:
                return {"session": session_id, "already_ended": row["ended"]}
            released = [r["path"] for r in self.db.execute("SELECT path FROM claims WHERE session=?", (session_id,))]
            self.db.execute("DELETE FROM claims WHERE session=?", (session_id,))
            self.db.execute("UPDATE sessions SET ended=?, summary=?, last_seen=? WHERE id=?",
                            (stamp(), summary, stamp(), session_id))
            self._event(row["project"], actor, kind, f"{session_id}: {summary}", paths=released)
        return {"session": session_id, "ended": True, "released": released}

    # User directives -------------------------------------------------------

    def steer(self, project, kind, text, *, paths=(), rank=None, via="terminal"):
        if kind not in DIRECTIVE_KINDS:
            raise ValueError(f"Directive kind must be one of {', '.join(DIRECTIVE_KINDS)}")
        text = text_arg(text, "directive")
        project = project if project == GLOBAL else str(Path(project).resolve())
        paths = [self._pattern(project, p) for p in paths if p.strip()]
        if kind in PATH_KINDS and not paths:
            raise ValueError(f"A {kind} directive needs at least one --paths pattern")
        if kind not in PATH_KINDS and paths:
            raise ValueError("Only scope and protect directives take paths")
        with self._tx():
            if rank is None:
                rank = self.db.execute("SELECT COALESCE(MAX(rank),0)+1 FROM directives "
                                       "WHERE project=? AND kind=? AND retired IS NULL", (project, kind)).fetchone()[0]
            cursor = self.db.execute("INSERT INTO directives(project,kind,text,paths,rank,via,created) "
                                     "VALUES(?,?,?,?,?,?,?)", (project, kind, text, json.dumps(paths), rank, via, stamp()))
            directive_id = cursor.lastrowid
            self._event(project, "user", "directive", f"#{directive_id} {kind}: {text}",
                        paths=paths, recipient="*", ref=directive_id)
        return directive_id

    @staticmethod
    def _pattern(project, pattern):
        """Project-relative directive pattern: "./auric/" -> "auric"."""
        candidate = Path(pattern.strip())
        if candidate.is_absolute():
            if project == GLOBAL:
                raise ValueError("Global directives take project-relative patterns")
            try:
                candidate = candidate.relative_to(project)
            except ValueError:
                raise ValueError(f"{pattern} is outside project {project}") from None
        text = os.path.normpath(candidate).replace(os.sep, "/")
        if text == ".." or text.startswith("../"):
            raise ValueError(f"{pattern} is outside the project")
        return text

    def retire(self, directive_id, *, via="terminal"):
        with self._tx():
            row = self.db.execute("SELECT * FROM directives WHERE id=?", (directive_id,)).fetchone()
            if row is None or row["retired"]:
                raise ValueError(f"No active directive #{directive_id}")
            self.db.execute("UPDATE directives SET retired=? WHERE id=?", (stamp(), directive_id))
            self._event(row["project"], "user", "retire", f"#{directive_id} retired ({via}): {row['text']}",
                        recipient="*", ref=directive_id)

    def directives(self, project):
        rows = self.db.execute("SELECT * FROM directives WHERE project IN (?,?) AND retired IS NULL "
                               "ORDER BY CASE kind WHEN 'priority' THEN 0 WHEN 'scope' THEN 1 WHEN 'protect' THEN 2 "
                               "WHEN 'constraint' THEN 3 ELSE 4 END, rank, id", (project, GLOBAL))
        return [{"id": r["id"], "kind": r["kind"], "text": r["text"], "paths": json.loads(r["paths"]),
                 "rank": r["rank"], "global": r["project"] == GLOBAL, "via": r["via"], "created": r["created"]}
                for r in rows]

    # Claims ----------------------------------------------------------------

    def claim(self, session_id, paths, intent):
        intent = text_arg(intent, "intent")
        if not paths or len(paths) > MAX_PATHS:
            raise ValueError(f"Claim between 1 and {MAX_PATHS} paths")
        with self._tx():
            me = self._live_row(session_id)
            project = me["project"]
            wanted = list(dict.fromkeys(relative_path(project, p) for p in paths))
            denied, conflicts = [], []
            directives = self.directives(project)
            scopes = [d for d in directives if d["kind"] == "scope"]
            for path in wanted:
                for d in directives:
                    if d["kind"] == "protect" and any(touches(path, p) for p in d["paths"]):
                        denied.append({"path": path, "directive": d["id"], "reason": f"protected by the user: {d['text']}"})
                if scopes and not any(covered(path, p) for d in scopes for p in d["paths"]):
                    denied.append({"path": path, "directive": [d["id"] for d in scopes],
                                   "reason": "outside the user's scope: " + "; ".join(d["text"] for d in scopes)})
            others = self.db.execute("SELECT c.*, s.agent, s.task FROM claims c JOIN sessions s ON s.id=c.session "
                                     "WHERE c.project=? AND c.session<>?", (project, session_id)).fetchall()
            sessions = {}
            for c in others:
                if c["session"] not in sessions:
                    sessions[c["session"]] = self.alive(self._session_row(c["session"]))
                if not sessions[c["session"]]:
                    continue
                for path in wanted:
                    if overlaps(path, c["path"]):
                        conflicts.append({"path": path, "held": c["path"], "by": c["session"], "agent": c["agent"],
                                          "their_intent": c["intent"], "their_task": c["task"]})
            granted = not denied and not conflicts
            if granted:
                self.db.executemany("INSERT OR REPLACE INTO claims(session,project,path,intent,created) VALUES(?,?,?,?,?)",
                                    [(session_id, project, p, intent, stamp()) for p in wanted])
                self._event(project, session_id, "claim", intent, paths=wanted)
            self.touch(session_id)
        result = {"granted": granted, "paths": wanted, "denied": denied, "conflicts": conflicts}
        if denied:
            result["next"] = "Do not edit denied paths. If the work needs them, porter_ask_user."
        elif conflicts:
            result["next"] = ("Another live session holds these paths. Work elsewhere, porter_message the holder, "
                              "or porter_ask_user to decide who proceeds.")
        return result

    def release(self, session_id, paths=None):
        with self._tx():
            row = self._session_row(session_id)
            if paths:
                rel = [relative_path(row["project"], p) for p in paths]
                self.db.executemany("DELETE FROM claims WHERE session=? AND path=?", [(session_id, p) for p in rel])
            else:
                rel = [r["path"] for r in self.db.execute("SELECT path FROM claims WHERE session=?", (session_id,))]
                self.db.execute("DELETE FROM claims WHERE session=?", (session_id,))
            if rel:
                self._event(row["project"], session_id, "release", "released", paths=rel)
            self.touch(session_id)
        return {"released": rel}

    # Notes, messages, questions -------------------------------------------

    def note(self, session_id, kind, message, *, paths=(), evidence=""):
        if kind not in NOTE_KINDS:
            raise ValueError(f"Note kind must be one of {', '.join(NOTE_KINDS)}")
        message = text_arg(message, "message")
        evidence = text_arg(evidence, "evidence", required=False)
        with self._tx():
            row = self._live_row(session_id)
            rel = [relative_path(row["project"], p) for p in paths]
            event = self._event(row["project"], session_id, kind,
                                message + (f"\nEvidence: {evidence}" if evidence else ""), paths=rel)
            self.touch(session_id)
        return {"event": event}

    def message(self, session_id, to, body):
        body = text_arg(body, "message")
        to = text_arg(to, "to")
        with self._tx():
            row = self._live_row(session_id)
            known = {r["id"] for r in self.db.execute("SELECT id FROM sessions WHERE project=?", (row["project"],))}
            agents = {r["agent"] for r in self.db.execute("SELECT agent FROM sessions WHERE project=?", (row["project"],))}
            if to not in known | agents | {"*", "user"}:
                raise ValueError(f"Unknown recipient {to!r}. Use a session id, an agent name "
                                 f"({', '.join(sorted(agents))}), '*' for all sessions, or 'user'.")
            if to == session_id:
                raise ValueError("A session cannot message itself")
            event = self._event(row["project"], session_id, "message", body, recipient=to)
            self.touch(session_id)
        return {"event": event, "to": to,
                "delivery": "queued for the next session of that agent" if to in agents - known else "posted"}

    def say(self, project, body, *, to="*", via="terminal"):
        """User message to agent sessions."""
        body = text_arg(body, "message")
        with self._tx():
            return self._event(project, "user", "message", body + ("" if via == "terminal" else f" [via {via}]"),
                               recipient=to)

    def ask_user(self, session_id, question, *, options=(), context=""):
        question = text_arg(question, "question")
        context = text_arg(context, "context", required=False)
        options = [text_arg(o, "option") for o in options][:8]
        with self._tx():
            row = self._live_row(session_id)
            cursor = self.db.execute("INSERT INTO questions(project,session,question,options,context,created) "
                                     "VALUES(?,?,?,?,?,?)",
                                     (row["project"], session_id, question, json.dumps(options), context, stamp()))
            qid = cursor.lastrowid
            self._event(row["project"], session_id, "question", question, recipient="user", ref=qid)
            self.touch(session_id)
        return {"question": qid, "status": "waiting for the user",
                "next": "Continue with work that does not depend on this answer; porter_inbox shows the reply."}

    def answer(self, question_id, answer, *, via="terminal"):
        answer = text_arg(answer, "answer")
        with self._tx():
            row = self.db.execute("SELECT * FROM questions WHERE id=?", (question_id,)).fetchone()
            if row is None:
                raise ValueError(f"No question #{question_id}")
            if row["answered"]:
                raise ValueError(f"Question #{question_id} was already answered: {row['answer']}")
            self.db.execute("UPDATE questions SET answer=?, answered=?, via=? WHERE id=?",
                            (answer, stamp(), via, question_id))
            # Broadcast: a user decision binds every session on the project.
            self._event(row["project"], "user", "answer", f"Q{question_id} {row['question']}\nAnswer: {answer}"
                        + ("" if via == "terminal" else f" [via {via}]"), recipient="*", ref=question_id)
        return {"question": question_id, "answered": True}

    def questions(self, project, *, open_only=True, limit=20):
        rows = self.db.execute("SELECT * FROM questions WHERE project=? " +
                               ("AND answered IS NULL " if open_only else "AND answered IS NOT NULL ") +
                               "ORDER BY id " + ("" if open_only else "DESC ") + "LIMIT ?", (project, limit))
        return [{"id": r["id"], "session": r["session"], "question": r["question"],
                 "options": json.loads(r["options"]), "context": r["context"], "created": r["created"],
                 "answer": r["answer"], "answered": r["answered"], "via": r["via"]} for r in rows]

    # Views -----------------------------------------------------------------

    def inbox(self, session_id):
        with self._tx():
            me = self._live_row(session_id)
            rows = self.db.execute(
                "SELECT * FROM events WHERE project IN (?,?) AND actor<>? AND ("
                "(recipient IN (?,'*') AND id>?) OR (recipient=? AND delivered_to IS NULL)) ORDER BY id",
                (me["project"], GLOBAL, session_id, session_id, me["cursor"], me["agent"])).fetchall()
            queued = [r["id"] for r in rows if r["recipient"] == me["agent"]]
            self.db.executemany("UPDATE events SET delivered_to=? WHERE id=?", [(session_id, i) for i in queued])
            self.db.execute("UPDATE sessions SET cursor=?, last_seen=? WHERE id=?",
                            (self._max_event(), stamp(), session_id))
        return [{"id": r["id"], "from": r["actor"], "kind": r["kind"], "to": r["recipient"],
                 "message": r["message"], "created": r["created"]} for r in rows]

    def history(self, project, *, query=None, kinds=None, limit=20):
        if not 1 <= limit <= 200:
            raise ValueError("History limit must be between 1 and 200")
        sql, params = "SELECT * FROM events WHERE project IN (?,?)", [project, GLOBAL]
        if kinds:
            sql += " AND kind IN (%s)" % ",".join("?"*len(kinds))
            params += list(kinds)
        for word in (query or "").split()[:8]:
            sql += " AND (message LIKE ? ESCAPE '\\' OR paths LIKE ? ESCAPE '\\')"
            like = "%" + word.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
            params += [like, like]
        rows = self.db.execute(sql + " ORDER BY id DESC LIMIT ?", params + [limit]).fetchall()
        return [{"id": r["id"], "actor": r["actor"], "kind": r["kind"], "message": r["message"],
                 "paths": json.loads(r["paths"]), "to": r["recipient"], "created": r["created"]}
                for r in reversed(rows)]

    def brief(self, session_id):
        """Everything a session needs to realign with the user and the other sessions."""
        me = self._live_row(session_id)
        project = me["project"]
        directives = self.directives(project)
        for d in directives:
            d["new"] = d["id"] > me["seen_directive"]
        everyone = self.sessions(project)
        others = [s for s in everyone if s["id"] != session_id and s["alive"]]
        stale = [s for s in everyone if s["id"] != session_id and not s["alive"]]
        waiting = self.questions(project)
        for q in waiting:
            q["mine"] = q["session"] == session_id
        decided = self.questions(project, open_only=False, limit=5)
        handoffs = [s for s in self.sessions(project, include_ended=True) if s["ended"]][:3]
        recent = self.history(project, kinds=NOTABLE, limit=8)
        inbox = self.inbox(session_id)
        max_directive = max([d["id"] for d in directives], default=me["seen_directive"])
        self.db.execute("UPDATE sessions SET seen_directive=MAX(seen_directive,?) WHERE id=?", (max_directive, session_id))

        guidance = []
        if not directives:
            guidance.append("The user has recorded no directives for this project. Confirm the goal with the user "
                            "before broad changes; do not invent priorities.")
        if any(d["new"] for d in directives):
            guidance.append("New user directives (marked new) since your last brief: re-check your plan against them.")
        mine = [q for q in waiting if q["mine"]]
        if mine:
            guidance.append(f"{len(mine)} of your questions await the user; do not decide them yourself.")
        if [q for q in waiting if not q["mine"]]:
            guidance.append("Other sessions are waiting on user answers; avoid making those decisions implicitly.")
        for s in others:
            if s["claims"]:
                guidance.append(f"{s['id']} ({s['agent']}) holds " + ", ".join(c["path"] for c in s["claims"]) +
                                " - do not edit those; message it if you need them.")
        if not self.session(session_id)["claims"]:
            guidance.append("Claim files with porter_claim before editing them.")
        if me["task"].startswith("(task not stated"):
            guidance.append("State your task with porter_checkin so the user and other sessions can see it.")

        def short(text, size=500):
            return text if len(text) <= size else text[:size] + "..."

        brief = {
            "you": {"session": session_id, "agent": me["agent"], "task": me["task"], "project": project,
                    "claims": self.session(session_id)["claims"]},
            "user_directives": [{k: v for k, v in d.items() if k in ("id", "kind", "text") or
                                 (k == "paths" and v) or (k in ("new", "global") and v) or
                                 (k == "via" and v != "terminal")} for d in directives],
            "waiting_on_user": waiting,
            "user_decisions": [{"id": q["id"], "question": short(q["question"], 300), "answer": q["answer"],
                                "answered": q["answered"]} for q in decided],
            "other_sessions": [{"id": s["id"], "agent": s["agent"], "task": short(s["task"], 300),
                                "last_seen": ago(s["last_seen"]), "claims": s["claims"]} for s in others],
            "stale_sessions": [{"id": s["id"], "agent": s["agent"], "last_seen": ago(s["last_seen"]),
                                "claims_not_blocking": [c["path"] for c in s["claims"]]} for s in stale],
            "recent_handoffs": [{"id": s["id"], "agent": s["agent"], "ended": s["ended"],
                                 "summary": short(s["summary"] or "")} for s in handoffs],
            "recent_notes": [{"id": e["id"], "by": e["actor"], "kind": e["kind"], "message": short(e["message"]),
                              "paths": e["paths"]} for e in recent],
            "inbox": inbox,
            "guidance": guidance,
        }
        # Empty sections cost tokens on every call; directives and guidance stay visible even when empty.
        return {k: v for k, v in brief.items() if v or k in ("user_directives", "guidance")}

    def status(self, project):
        """The user's view of one project."""
        everyone = self.sessions(project)
        for_user = [e for e in self.history(project, limit=200) if e["to"] == "user" and e["kind"] == "message"][-10:]
        return {"project": project, "directives": self.directives(project),
                "waiting_on_you": self.questions(project),
                "sessions": [s for s in everyone if s["alive"]],
                "stale_sessions": [s for s in everyone if not s["alive"]],
                "messages_for_you": for_user,
                "recent": self.history(project, kinds=NOTABLE + ("claim", "checkin"), limit=12)}


def render_status(status, reference=None):
    """Plain-text dashboard for the terminal."""
    out = [f"Porter: {status['project']}", ""]
    out.append("Your directives")
    if not status["directives"]:
        out.append('  (none) - set one: auric porter steer "Priority: ..."')
    for d in status["directives"]:
        paths = f"  [{', '.join(d['paths'])}]" if d["paths"] else ""
        flags = (" global" if d["global"] else "") + ("" if d["via"] == "terminal" else f"  (set from {d['via']})")
        out.append(f"  #{d['id']} {d['kind']:<10} {d['text']}{paths}{flags}")
    out += ["", f"Waiting on you ({len(status['waiting_on_you'])})"]
    for q in status["waiting_on_you"]:
        out.append(f"  Q{q['id']} from {q['session']} ({ago(q['created'], reference)}): {q['question']}")
        if q["options"]:
            out.append("      options: " + " | ".join(q["options"]))
        if q["context"]:
            out.append("      context: " + q["context"][:300])
    if status["waiting_on_you"]:
        out.append('  reply: auric porter answer <id> "..."')
    for m in status["messages_for_you"]:
        out.append(f"  FYI {m['actor']}: {m['message'][:300]}")
    out += ["", f"Live sessions ({len(status['sessions'])})"]
    for s in status["sessions"]:
        out.append(f"  {s['id']:<18} {s['task'][:100]}  (seen {ago(s['last_seen'], reference)})")
        for c in s["claims"]:
            out.append(f"      claims {c['path']}: {c['intent'][:80]}")
    if status["stale_sessions"]:
        out.append("  stale (not blocking): " + ", ".join(s["id"] for s in status["stale_sessions"])
                   + "  - clear with: auric porter end <session>")
    out += ["", "Recent"]
    for e in status["recent"]:
        when = parse(e["created"]).astimezone().strftime("%m-%d %H:%M")
        paths = f" [{', '.join(e['paths'][:4])}]" if e["paths"] else ""
        out.append(f"  {when} {e['actor']:<18} {e['kind']:<9} {e['message'].splitlines()[0][:110]}{paths}")
    return "\n".join(out)
