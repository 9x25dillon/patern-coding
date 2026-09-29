"""MCP stdio server: Porter coordination plus AURIC's local index for agent CLIs.

Claude Code and Codex each launch one server process per session and speak
newline-delimited JSON-RPC 2.0 over stdin/stdout. The process is the session:
it registers on first use and checks out when the client disconnects, which
releases its claims. Nothing here executes commands or edits project files.
Only JSON-RPC goes to stdout; diagnostics go to stderr.
"""
import json
import os
from pathlib import Path
import signal
import sys

from . import __version__
from .porter import NOTE_KINDS, Porter, agent_name, find_project

PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
UNSTATED = "(task not stated - call porter_checkin)"

INSTRUCTIONS = """\
Porter connects this session to the user's other Claude Code and Codex sessions and keeps the work pointed at what the user asked for.
- Start with porter_checkin (your task and working directory). Treat user_directives as outranking your own plan.
- porter_claim files before editing them. If a claim is denied or conflicts, do not edit those paths: work elsewhere, porter_message the holder, or porter_ask_user.
- Record decisions, findings and blockers with porter_note so other sessions and the user can see why.
- When a choice belongs to the user, porter_ask_user and continue independent work; answers arrive in porter_inbox.
- Check porter_inbox between steps. Finish with porter_checkout (summary and next steps); it releases your claims.
Directives can only be changed by the user from their terminal (auric porter steer)."""


def default_knowledge_db():
    if os.environ.get("AURIC_KNOWLEDGE_DB"):
        return Path(os.environ["AURIC_KNOWLEDGE_DB"])
    return Path(__file__).resolve().parents[1]/"artifacts"/"knowledge.sqlite"


def schema(properties, required=()):
    return {"type": "object", "properties": properties, "required": list(required), "additionalProperties": False}


STR = {"type": "string"}
PATHS = {"type": "array", "items": STR, "description": "Files or directories, relative to the project root or absolute inside it"}
READ_ONLY = {"readOnlyHint": True, "openWorldHint": False}
WRITES = {"readOnlyHint": False, "destructiveHint": False, "openWorldHint": False}

TOOLS = [
    ("porter_checkin", "Register this session (or update its task) and get the brief: user directives, "
     "other sessions and their claimed files, questions waiting on the user, recent decisions and handoffs.",
     schema({"task": {**STR, "description": "What you are working on, in one or two sentences"},
             "cwd": {**STR, "description": "Your absolute working directory; selects the project"},
             "agent": {**STR, "description": "Override the detected agent name (claude-code, codex)"},
             "resume_session": {**STR, "description": "Reattach to your earlier session id after a reconnect"}},
            ["task"]), WRITES),
    ("porter_brief", "Current picture for this project: user directives (new ones flagged), who is working on "
     "what, claims, questions waiting on the user, user decisions, handoffs, your inbox and guidance.",
     schema({}), READ_ONLY),
    ("porter_inbox", "New messages for this session: user answers, directive changes, messages from other sessions.",
     schema({}), READ_ONLY),
    ("porter_claim", "Claim files or directories before editing. All-or-nothing: returns granted=false with "
     "conflicts (another live session holds an overlapping path) or denials (user scope/protect directives).",
     schema({"paths": {**PATHS, "minItems": 1}, "intent": {**STR, "description": "What you will change"}},
            ["paths", "intent"]), WRITES),
    ("porter_release", "Release claimed paths (all of yours when paths is omitted).",
     schema({"paths": PATHS}), WRITES),
    ("porter_note", "Record progress, a decision, finding, blocker, test result or handoff note for the user "
     "and other sessions.",
     schema({"kind": {"type": "string", "enum": list(NOTE_KINDS)}, "message": STR,
             "paths": PATHS, "evidence": {**STR, "description": "Command output, file:line, or test name"}},
            ["kind", "message"]), WRITES),
    ("porter_message", "Send a message to another session id, an agent name (queued for that agent's next "
     "session to read it), '*' for every session on the project, or 'user'.",
     schema({"to": STR, "message": STR}, ["to", "message"]), WRITES),
    ("porter_ask_user", "Queue a decision for the user instead of guessing. The user answers from their "
     "terminal; the answer reaches every session through porter_inbox.",
     schema({"question": STR, "options": {"type": "array", "items": STR, "maxItems": 8},
             "context": {**STR, "description": "What depends on the answer and what you recommend"}},
            ["question"]), WRITES),
    ("porter_checkout", "End this session with a summary and next steps for the user and the next session. "
     "Releases your claims.",
     schema({"summary": STR, "next_steps": STR}, ["summary"]), WRITES),
    ("porter_history", "Search this project's coordination history (notes, decisions, messages, answers).",
     schema({"query": STR, "kinds": {"type": "array", "items": STR},
             "limit": {"type": "integer", "minimum": 1, "maximum": 200}}), READ_ONLY),
    ("auric_search", "Search the user's local AURIC knowledge index (explicitly indexed code and documents) "
     "with BM25. Returns cited excerpts; citations are not proof of a claim.",
     schema({"query": STR, "limit": {"type": "integer", "minimum": 1, "maximum": 20}}, ["query"]), READ_ONLY),
    ("auric_memories", "User-confirmed project memories (key, value, evidence) from the AURIC knowledge store.",
     schema({}), READ_ONLY),
]
SPECS = {name: {"name": name, "description": desc, "inputSchema": params, "annotations": notes}
         for name, desc, params, notes in TOOLS}
TYPES = {"string": str, "array": list, "integer": int, "object": dict}


class RpcError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def check_arguments(name, arguments):
    params = SPECS[name]["inputSchema"]
    if not isinstance(arguments, dict):
        raise ValueError("Arguments must be an object")
    unknown = set(arguments) - set(params["properties"])
    if unknown:
        raise ValueError(f"Unexpected argument(s) for {name}: {', '.join(sorted(unknown))}")
    for key in params["required"]:
        if key not in arguments:
            raise ValueError(f"{name} requires '{key}'")
    for key, value in arguments.items():
        expected = params["properties"][key]["type"]
        if not isinstance(value, TYPES[expected]) or (expected == "integer" and isinstance(value, bool)):
            raise ValueError(f"'{key}' must be {expected}")
        if expected == "array" and not all(isinstance(v, str) for v in value):
            raise ValueError(f"'{key}' must contain strings")


class Server:
    def __init__(self, *, porter_db=None, knowledge_db=None, cwd=None):
        self.porter = Porter(porter_db)
        self.knowledge_db = Path(knowledge_db) if knowledge_db else default_knowledge_db()
        self.cwd = cwd or os.getcwd()
        self.client = None
        self.agent_override = None
        self.session_id = None

    @property
    def agent(self):
        return self.agent_override or agent_name(self.client)

    def session(self):
        if self.session_id is None:
            self.session_id = self.porter.checkin(self.agent, find_project(self.cwd), UNSTATED, pid=os.getpid())["id"]
        else:
            self.porter.touch(self.session_id)
        return self.session_id

    def close(self):
        if self.session_id:
            try:
                self.porter.end(self.session_id, "client disconnected without porter_checkout", actor=self.session_id)
            except Exception as exc:  # never mask the shutdown reason
                print(f"auric mcp: cleanup failed: {exc}", file=sys.stderr)
            self.session_id = None
        self.porter.close()

    # Tools -------------------------------------------------------------------

    def porter_checkin(self, task, cwd=None, agent=None, resume_session=None):
        if cwd:
            if not Path(cwd).is_absolute() or not Path(cwd).is_dir():
                raise ValueError("cwd must be an existing absolute directory")
            self.cwd = cwd
        if agent:
            self.agent_override = agent_name(agent)
        if self.session_id and self.porter.session(self.session_id)["ended"]:
            self.session_id = None  # ended from the user's terminal; start fresh
        project = find_project(self.cwd)
        if resume_session:
            session = self.porter.checkin(self.agent, project, task, pid=os.getpid(), resume=resume_session)
            if self.session_id and self.session_id != resume_session:
                self.porter.end(self.session_id, f"replaced by resumed session {resume_session}", actor=self.session_id)
            self.session_id = session["id"]
        elif self.session_id and self.porter.session(self.session_id)["project"] == project:
            self.porter.update_task(self.session_id, task)
        else:
            if self.session_id:
                self.porter.end(self.session_id, f"moved to {project}", actor=self.session_id)
            self.session_id = self.porter.checkin(self.agent, project, task, pid=os.getpid())["id"]
        return {"session": self.session_id, "brief": self.porter.brief(self.session_id)}

    def porter_brief(self):
        return self.porter.brief(self.session())

    def porter_inbox(self):
        session = self.session()
        waiting = [q["id"] for q in self.porter.questions(self.porter.session(session)["project"])
                   if q["session"] == session]
        return {"inbox": self.porter.inbox(session), "your_questions_waiting": waiting}

    def porter_claim(self, paths, intent):
        return self.porter.claim(self.session(), paths, intent)

    def porter_release(self, paths=None):
        return self.porter.release(self.session(), paths)

    def porter_note(self, kind, message, paths=(), evidence=""):
        return self.porter.note(self.session(), kind, message, paths=paths, evidence=evidence)

    def porter_message(self, to, message):
        return self.porter.message(self.session(), to, message)

    def porter_ask_user(self, question, options=(), context=""):
        return self.porter.ask_user(self.session(), question, options=options, context=context)

    def porter_checkout(self, summary, next_steps=""):
        result = self.porter.checkout(self.session(), summary, next_steps)
        self.session_id = None
        return result

    def porter_history(self, query=None, kinds=None, limit=20):
        return self.porter.history(self.porter.session(self.session())["project"], query=query, kinds=kinds, limit=limit)

    def _knowledge(self):
        if not self.knowledge_db.is_file():
            raise ValueError(f"No AURIC knowledge index at {self.knowledge_db}. The user can build one with: "
                             "python -m auric index <roots>")
        from .memory import KnowledgeStore
        return KnowledgeStore(self.knowledge_db)

    def auric_search(self, query, limit=5):
        with self._knowledge() as store:
            hits = store.search(query, limit=limit)
        return {"mode": "retrieval_only", "results": [{"citation": h["citation"], "text": h["content"][:3000]} for h in hits]}

    def auric_memories(self):
        with self._knowledge() as store:
            return [{k: m[k] for k in ("key", "value", "evidence", "created")} for m in store.memories()]

    # JSON-RPC ----------------------------------------------------------------

    def initialize(self, params):
        self.client = (params.get("clientInfo") or {}).get("name")
        requested = params.get("protocolVersion")
        return {"protocolVersion": requested if requested in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0],
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "auric-porter", "version": __version__},
                "instructions": INSTRUCTIONS}

    def call(self, params):
        name = params.get("name")
        if name not in SPECS:
            raise RpcError(-32602, f"Unknown tool: {name}")
        arguments = params.get("arguments") or {}
        try:
            check_arguments(name, arguments)
            result = getattr(self, name)(**arguments)
        except (ValueError, KeyError, OSError) as exc:
            return {"content": [{"type": "text", "text": f"{name} failed: {exc}"}], "isError": True}
        return {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False, indent=1)}],
                "isError": False}

    def handle(self, message):
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            return {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Invalid Request"}}
        method, is_request = message.get("method"), "id" in message
        if method is None:
            return None  # a response to a request we never send
        try:
            params = message.get("params") or {}
            if method == "initialize":
                result = self.initialize(params)
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": list(SPECS.values())}
            elif method == "tools/call":
                result = self.call(params)
            elif method.startswith("notifications/"):
                return None
            else:
                raise RpcError(-32601, f"Method not found: {method}")
        except RpcError as exc:
            return {"jsonrpc": "2.0", "id": message["id"], "error": {"code": exc.code, "message": str(exc)}} if is_request else None
        return {"jsonrpc": "2.0", "id": message["id"], "result": result} if is_request else None

    def serve(self, stdin, stdout):
        while True:
            line = stdin.readline()
            if not line:
                return
            if not line.strip():
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                reply = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}
            else:
                if isinstance(message, list):
                    reply = [r for r in map(self.handle, message) if r] or None
                else:
                    reply = self.handle(message)
            if reply:
                stdout.write(json.dumps(reply, ensure_ascii=False) + "\n")
                stdout.flush()


def main(porter_db=None, knowledge_db=None):
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    # The client stops us with SIGTERM or by closing stdin; both run close().
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    server = Server(porter_db=porter_db, knowledge_db=knowledge_db)
    try:
        server.serve(sys.stdin, sys.stdout)
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
    return 0
