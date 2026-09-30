"""Exact, expiring, single-attempt approvals consumed by Git's pre-push hook.

This is a local workflow gate, not protection against an account owner who can
replace hooks or edit the database. No agent-facing approval tool is exposed.
"""
import json
from pathlib import Path
import re
import secrets
import subprocess
import sys

from .reflex import ReflexLedger, digest, packet, redact
from .porter import stamp

SCHEMA = """
CREATE TABLE IF NOT EXISTS reflex_push_grants (
    id TEXT PRIMARY KEY, repository TEXT NOT NULL, remote_hash TEXT NOT NULL,
    ref TEXT NOT NULL, old_oid TEXT NOT NULL, new_oid TEXT NOT NULL,
    expires REAL NOT NULL, consumed REAL, created REAL NOT NULL);
CREATE INDEX IF NOT EXISTS reflex_push_match ON reflex_push_grants
    (repository,remote_hash,ref,old_oid,new_oid);
CREATE TABLE IF NOT EXISTS reflex_push_requests (
    fingerprint TEXT PRIMARY KEY, question_id INTEGER NOT NULL,
    repository TEXT NOT NULL, remote_hash TEXT NOT NULL, updates TEXT NOT NULL);
"""


def git(repo, *args, allowed=(0,)):
    try:
        result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=20)
    except subprocess.TimeoutExpired:
        raise ValueError("Git verification timed out; no approval was granted") from None
    if result.returncode not in allowed:
        raise ValueError("Git could not verify the push: " + packet(result.stderr or result.stdout, 700))
    return result


def repository(repo):
    root = Path(git(repo, "rev-parse", "--show-toplevel").stdout.strip()).resolve()
    common = Path(git(root, "rev-parse", "--git-common-dir").stdout.strip())
    return root, str((root / common).resolve())


def oid(value):
    return isinstance(value, str) and re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", value) is not None


def parse_updates(text):
    if len(text) > 65536:
        raise ValueError("Push ref list exceeds 64 KiB")
    updates, refs = [], set()
    for line in text.splitlines():
        fields = line.split()
        if len(fields) != 4:
            raise ValueError("Malformed pre-push ref update")
        local_ref, new, remote_ref, old = fields
        if not oid(new) or not oid(old) or len(new) != len(old):
            raise ValueError("Malformed push object id")
        if not remote_ref.startswith("refs/heads/") or remote_ref in refs:
            raise ValueError("This gate accepts distinct branch updates only")
        if set(new) == {"0"}:
            raise ValueError("Branch deletion is outside this approval workflow")
        refs.add(remote_ref)
        updates.append({"ref": remote_ref, "old_oid": old, "new_oid": new})
    if len(updates) > 64:
        raise ValueError("Split pushes into at most 64 refs")
    return updates


def proposal(repo, remote, ref, commit):
    root, identity = repository(repo)
    if not remote or remote.startswith("-") or not ref.startswith("refs/heads/"):
        raise ValueError("Select a configured remote and a destination refs/heads/... branch")
    git(root, "check-ref-format", ref)
    urls = git(root, "remote", "get-url", "--push", "--all", remote).stdout.splitlines()
    if len(urls) != 1:
        raise ValueError("Approve remotes with exactly one push URL")
    url = urls[0]
    new = git(root, "rev-parse", "--verify", "--end-of-options", commit + "^{commit}").stdout.strip()
    if not oid(new):
        raise ValueError("Commit did not resolve to a valid object id")
    advertised = git(root, "ls-remote", "--refs", "--", url, ref).stdout.splitlines()
    rows = [line.split() for line in advertised]
    if any(len(row) != 2 for row in rows):
        raise ValueError("Remote returned an invalid ref listing")
    matches = [row[0] for row in rows if row[1] == ref]
    if len(matches) > 1 or any(not oid(value) for value in matches):
        raise ValueError("Remote branch could not be verified")
    old = matches[0] if matches else "0" * len(new)
    if old == new:
        raise ValueError("The remote already has this commit; no approval is needed")
    if set(old) != {"0"}:
        # Requiring the advertised object locally also forces a reviewable fetch
        # when the branch changed since the user's last inspection.
        git(root, "cat-file", "-e", old + "^{commit}")
        result = git(root, "merge-base", "--is-ancestor", old, new, allowed=(0, 1))
        if result.returncode:
            raise ValueError("Only fast-forward pushes are supported; review the divergence separately")
    return {"project": str(root), "repository": identity, "remote_hash": digest(url),
            "remote_display": packet(redact(url)), "ref": ref, "old_oid": old, "new_oid": new}


class MissingConsent(ValueError):
    pass


class PushLedger(ReflexLedger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with self._tx():
            for statement in filter(str.strip, SCHEMA.split(";")):
                self.db.execute(statement)

    def grant(self, request, ttl=300):
        if not isinstance(ttl, int) or isinstance(ttl, bool) or not 1 <= ttl <= 3600:
            raise ValueError("Approval lifetime must be 1..3600 seconds")
        approval = secrets.token_hex(16)
        with self._tx():
            # Reissuing the same approval replaces, rather than stacks, attempts.
            params = tuple(request[k] for k in ("repository", "remote_hash", "ref", "old_oid", "new_oid"))
            self.db.execute("UPDATE reflex_push_grants SET consumed=? WHERE repository=? AND remote_hash=? "
                            "AND ref=? AND old_oid=? AND new_oid=? AND consumed IS NULL", (self.clock(), *params))
            self.db.execute("INSERT INTO reflex_push_grants VALUES(?,?,?,?,?,?,?,NULL,?)",
                            (approval, *params, self.clock() + ttl, self.clock()))
            self._event(request["project"], "user", "answer",
                        packet(f"Push approved once: {request['new_oid']} -> {request['ref']}; expires in {ttl}s"))
            pending = self.db.execute("SELECT r.* FROM reflex_push_requests r JOIN questions q ON q.id=r.question_id "
                                      "WHERE r.repository=? AND r.remote_hash=? AND q.answered IS NULL",
                                      (request["repository"], request["remote_hash"])).fetchall()
            for row in pending:
                updates = json.loads(row["updates"])
                if all(self._matching(request["repository"], request["remote_hash"], u) for u in updates):
                    answer = "Approved the exact refs for one expiring push attempt."
                    self.db.execute("UPDATE questions SET answer=?,answered=?,via=? WHERE id=?",
                                    (answer, stamp(), "user terminal", row["question_id"]))
                    self._event(request["project"], "user", "answer", answer, recipient="*", ref=row["question_id"])
        return {"approval": approval, "expires": self.clock() + ttl}

    def _matching(self, identity, remote_hash, update):
        return self.db.execute("SELECT id FROM reflex_push_grants WHERE repository=? AND remote_hash=? "
                               "AND ref=? AND old_oid=? AND new_oid=? AND consumed IS NULL AND expires>? "
                               "ORDER BY created DESC LIMIT 1",
                               (identity, remote_hash, update["ref"], update["old_oid"], update["new_oid"], self.clock())).fetchone()

    def consume(self, identity, url, updates, project):
        with self._tx():
            chosen = []
            for update in updates:
                row = self._matching(identity, digest(url), update)
                if row is None:
                    raise MissingConsent(f"No current one-time approval for {update['new_oid'][:12]} -> {update['ref']}")
                chosen.append(row["id"])
            # All-or-nothing: no token is burned by an unapproved second ref.
            self.db.executemany("UPDATE reflex_push_grants SET consumed=? WHERE id=?", [(self.clock(), key) for key in chosen])
            if chosen:
                self._event(project, "git", "progress", f"Consumed {len(chosen)} push approval(s); delivery not yet verified")
            return chosen

    def request_approval(self, project, identity, url, updates):
        session = self.register("git", "pre-push", project)
        updates = sorted(updates, key=lambda u: u["ref"])
        remote_hash = digest(url)
        generation = []
        for u in updates:
            row = self.db.execute("SELECT id FROM reflex_push_grants WHERE repository=? AND remote_hash=? "
                                  "AND ref=? AND old_oid=? AND new_oid=? ORDER BY created DESC,rowid DESC LIMIT 1",
                                  (identity, remote_hash, u["ref"], u["old_oid"], u["new_oid"])).fetchone()
            generation.append(row["id"] if row else None)
        key = digest([identity, remote_hash, updates, generation])
        refs = ", ".join(f"{u['new_oid'][:12]} -> {u['ref']}" for u in updates)
        result = self.ask_once(session, key, "Approve this exact push? " + refs,
                               "Run the user-only approve-push command after review. A chat reply alone does not mint a token.")
        self.db.execute("INSERT OR IGNORE INTO reflex_push_requests VALUES(?,?,?,?,?)",
                        (key, result["id"], identity, remote_hash, json.dumps(updates)))
        return result["id"]


def command(args):
    if args.command == "approve-push":
        # This intentionally is not a noninteractive agent or MCP operation.
        if not sys.stdin.isatty():
            raise ValueError("approve-push requires the user's interactive terminal; agents must not mint approvals")
        request = proposal(Path.cwd(), args.remote, args.ref, args.commit)
        print("One-time push proposal:\n" + json.dumps(request, indent=2))
        if input("Type approve to authorize this exact push attempt: ").strip() != "approve":
            print("No approval created.")
            return 1
        with PushLedger(args.db) as ledger:
            print(json.dumps(ledger.grant(request, args.ttl)))
        return 0
    root, identity = repository(Path.cwd())
    updates = parse_updates(sys.stdin.read(65537))
    if not updates:
        return 0  # Up-to-date pushes publish nothing.
    with PushLedger(args.db) as ledger:
        try:
            ledger.consume(identity, args.url, updates, str(root))
        except MissingConsent as exc:
            qid = ledger.request_approval(str(root), identity, args.url, updates)
            print(f"Porter blocked the push (inbox Q{qid}): {packet(str(exc))}. "
                  "Review and run approve-push from your terminal, then retry.", file=sys.stderr)
            return 1
    return 0
