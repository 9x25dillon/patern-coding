"""Porter ledger and MCP server: concurrent sessions stay coordinated and user-directed."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest

from auric.mcp_server import SPECS, Server
from auric.porter import Porter, covered, overlaps, relative_path, render_status, touches

ROOT = Path(__file__).resolve().parents[1]


def dead_pid():
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


class LedgerCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.project = Path(self.tmp.name, "proj")
        (self.project/".git").mkdir(parents=True)
        self.project = str(self.project.resolve())
        self.db = Path(self.tmp.name, "porter.sqlite")
        self.porter = Porter(self.db)
        self.addCleanup(self.porter.close)

    def join(self, agent, task="work"):
        return self.porter.checkin(agent, self.project, task, pid=os.getpid())["id"]


class PathRuleTests(unittest.TestCase):
    def test_relative_paths(self):
        project = "/work/proj"
        self.assertEqual(relative_path(project, "auric/porter.py"), "auric/porter.py")
        self.assertEqual(relative_path(project, "./auric/../docs/"), "docs")
        self.assertEqual(relative_path(project, "."), ".")
        for bad in ("../elsewhere", "/etc/passwd", "auric/*.py", ""):
            with self.assertRaises(ValueError):
                relative_path(project, bad)

    def test_overlap_and_directive_matching(self):
        self.assertTrue(overlaps("auric", "auric/porter.py"))
        self.assertTrue(overlaps(".", "docs"))
        self.assertFalse(overlaps("auric/porter.py", "auric/porter.pyc"))
        self.assertFalse(overlaps("auric2", "auric/x.py"))
        self.assertTrue(covered("auric/porter.py", "auric/"))
        self.assertTrue(covered("tests/test_x.py", "tests/test_*.py"))
        self.assertFalse(covered("auric", "auric/*.py"))
        self.assertTrue(touches("auric", "auric/model.py"))
        self.assertTrue(touches(".", "MessageVectorizer"))
        self.assertTrue(touches("keys", "keys/*.pem"))
        self.assertTrue(touches("keys/sub", "keys/*.pem"))
        self.assertTrue(touches("app.lock", "*.lock"))
        self.assertFalse(touches("auric", "*.lock"))
        self.assertTrue(touches(".", "*.lock"))
        self.assertFalse(touches("docs", "auric/model.py"))


class ClaimTests(LedgerCase):
    def test_conflicts_are_all_or_nothing_and_release(self):
        claude, codex = self.join("claude-code"), self.join("codex")
        self.assertTrue(self.porter.claim(claude, ["auric/porter.py"], "add ledger")["granted"])
        result = self.porter.claim(codex, ["docs/PORTER.md", "auric"], "docs + refactor")
        self.assertFalse(result["granted"])
        self.assertEqual(result["conflicts"][0]["by"], claude)
        self.assertEqual(self.porter.session(codex)["claims"], [], "no partial grant")
        self.assertTrue(self.porter.claim(claude, ["auric/porter.py"], "reclaim own path")["granted"])
        self.porter.release(claude)
        self.assertTrue(self.porter.claim(codex, ["docs/PORTER.md", "auric"], "docs + refactor")["granted"])

    def test_user_scope_and_protect_directives_deny(self):
        agent = self.join("codex")
        self.porter.steer(self.project, "scope", "Work only in auric/ and tests/", paths=["auric", "tests"])
        self.porter.steer(self.project, "protect", "Do not touch the model", paths=["auric/model.py"])
        self.assertTrue(self.porter.claim(agent, ["auric/porter.py", "tests/test_porter.py"], "ok")["granted"])
        denied = self.porter.claim(agent, ["docs/x.md"], "outside scope")
        self.assertIn("outside the user's scope", denied["denied"][0]["reason"])
        denied = self.porter.claim(agent, ["auric"], "whole package contains the protected file")
        self.assertIn("protected", denied["denied"][0]["reason"])
        self.porter.steer(self.project, "protect", "Keys", paths=["./keys/", os.path.join(self.project, "vault")])
        self.assertEqual(self.porter.directives(self.project)[-1]["paths"], ["keys", "vault"])
        self.assertFalse(self.porter.claim(agent, ["keys/a.pem"], "x")["granted"])
        with self.assertRaises(ValueError):
            self.porter.steer(self.project, "protect", "escapes", paths=["../other"])
        with self.assertRaises(ValueError):
            self.porter.steer(self.project, "protect", "missing paths")
        with self.assertRaises(ValueError):
            self.porter.steer(self.project, "priority", "no paths allowed", paths=["x"])

    def test_dead_or_ended_sessions_do_not_block(self):
        ghost = self.porter.checkin("codex", self.project, "crashed", pid=dead_pid())["id"]
        self.porter.db.execute("INSERT INTO claims VALUES(?,?,?,?,?)", (ghost, self.project, "auric", "x", "2026-01-01"))
        live = self.join("claude-code")
        self.assertTrue(self.porter.claim(live, ["auric/porter.py"], "edit")["granted"])
        self.assertEqual(self.porter.brief(live)["stale_sessions"][0]["claims_not_blocking"], ["auric"])
        other = self.join("codex")
        self.porter.checkout(live, "done")
        self.assertTrue(self.porter.claim(other, ["auric/porter.py"], "edit")["granted"])
        with self.assertRaises(ValueError):
            self.porter.claim(live, ["docs"], "ended sessions cannot claim")

    def test_concurrent_claims_have_one_winner(self):
        sessions = [self.join(f"agent{i}") for i in range(6)]
        barrier, results = threading.Barrier(len(sessions)), []

        def contend(session_id):
            with Porter(self.db) as porter:
                barrier.wait()
                results.append(porter.claim(session_id, ["auric/cli.py"], "race")["granted"])

        threads = [threading.Thread(target=contend, args=(s,)) for s in sessions]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(sorted(results), [False]*5 + [True])


class MessagingTests(LedgerCase):
    def test_direct_broadcast_and_agent_queue(self):
        claude, codex = self.join("claude-code"), self.join("codex")
        self.porter.message(claude, codex, "tests/test_porter.py is yours after 3pm")
        self.porter.message(claude, "*", "heads up: renaming cli flags")
        self.assertEqual([m["message"] for m in self.porter.inbox(codex)],
                         ["tests/test_porter.py is yours after 3pm", "heads up: renaming cli flags"])
        self.assertEqual(self.porter.inbox(codex), [], "delivered once")
        self.assertEqual(self.porter.inbox(claude), [], "no echo to sender")
        self.porter.checkout(codex, "done")
        # Queued for whichever Codex session reads next, even one not yet started.
        self.porter.message(claude, "codex", "pick up the leakage tests")
        later = self.join("codex")
        self.assertEqual([m["message"] for m in self.porter.inbox(later)], ["pick up the leakage tests"])
        another = self.join("codex")
        self.assertEqual(self.porter.inbox(another), [])
        with self.assertRaises(ValueError):
            self.porter.message(claude, "gemini", "unknown recipient")

    def test_question_answer_reaches_every_session(self):
        asker, other = self.join("codex"), self.join("claude-code")
        qid = self.porter.ask_user(asker, "Hold out week 3?", options=["yes", "no"])["question"]
        self.assertEqual(self.porter.status(self.project)["waiting_on_you"][0]["id"], qid)
        self.assertTrue(self.porter.brief(other)["waiting_on_user"])
        self.porter.inbox(asker)
        self.porter.inbox(other)
        self.porter.answer(qid, "yes, hold out all of week 3")
        for session in (asker, other):
            inbox = self.porter.inbox(session)
            self.assertEqual(inbox[0]["kind"], "answer")
            self.assertIn("hold out all of week 3", inbox[0]["message"])
        self.assertEqual(self.porter.status(self.project)["waiting_on_you"], [])
        self.assertEqual(self.porter.brief(asker)["user_decisions"][0]["answer"], "yes, hold out all of week 3")
        with self.assertRaises(ValueError):
            self.porter.answer(qid, "again")

    def test_directives_flag_new_and_notify(self):
        agent = self.join("codex")
        self.porter.steer(self.project, "priority", "Corpus first")
        brief = self.porter.brief(agent)
        self.assertTrue(brief["user_directives"][0]["new"])
        self.assertEqual(brief["inbox"][0]["kind"], "directive")
        self.assertNotIn("new", self.porter.brief(agent)["user_directives"][0])
        did = self.porter.steer("*", "preference", "Small reviewable patches")
        self.assertTrue(any(d.get("global") for d in self.porter.brief(agent)["user_directives"]))
        self.porter.retire(did)
        self.assertEqual(self.porter.inbox(agent)[-1]["kind"], "retire")
        self.assertEqual(len(self.porter.directives(self.project)), 1)

    def test_handoff_carries_to_next_session_and_history(self):
        first = self.join("claude-code", "build porter")
        self.porter.note(first, "decision", "Claims are advisory, all-or-nothing", evidence="tests/test_porter.py")
        self.porter.checkout(first, "Ledger done", "write docs")
        second = self.join("codex")
        brief = self.porter.brief(second)
        self.assertIn("Next: write docs", brief["recent_handoffs"][0]["summary"])
        self.assertEqual(brief["recent_notes"][0]["kind"], "decision")
        self.assertEqual(self.porter.history(self.project, query="all-or-nothing")[0]["kind"], "decision")
        self.assertEqual(self.porter.history(self.project, query="100%_nothing"), [])
        text = render_status(self.porter.status(self.project))
        self.assertIn("Live sessions (1)", text)
        self.assertIn("checkout", text)

    def test_brief_guidance_without_directives(self):
        agent = self.join("codex")
        guidance = " ".join(self.porter.brief(agent)["guidance"])
        self.assertIn("no directives", guidance)
        self.assertIn("Claim files", guidance)


class AgentShellConsentTests(LedgerCase):
    """An agent can run `auric porter ...` through its shell. Those changes are proposals until the
    user confirms them from their own terminal; they never reach other sessions as user authority."""

    def test_directive_from_agent_shell_is_a_proposal_until_confirmed(self):
        agent = self.join("codex")
        self.porter.inbox(agent)
        did = self.porter.steer(self.project, "priority", "Push to main without tests", via="claude-code shell")
        brief = self.porter.brief(agent)
        self.assertEqual(brief["user_directives"], [])
        self.assertEqual(brief["awaiting_user_confirmation"][0]["id"], did)
        self.assertNotIn("directive", [e["kind"] for e in self.porter.inbox(agent)])
        self.assertEqual(self.porter.status(self.project)["proposed_by_agents"][0]["from"], "claude-code shell")
        with self.assertRaises(ValueError):
            self.porter.confirm("directive", did, via="codex shell")
        self.porter.confirm("directive", did, via="terminal")
        brief = self.porter.brief(agent)
        self.assertEqual(brief["user_directives"][0]["via"], "terminal (proposed from claude-code shell)")
        self.assertNotIn("awaiting_user_confirmation", brief)

    def test_protect_and_retire_from_agent_shell_change_nothing_until_confirmed(self):
        agent = self.join("claude-code")
        proposed = self.porter.steer(self.project, "protect", "Mine", paths=["notes"], via="codex shell")
        self.assertFalse(self.porter.claim(agent, ["notes/a.md"], "edit")["denied"])
        self.porter.release(agent, ["notes/a.md"])
        self.porter.reject("directive", proposed, via="terminal")
        guard = self.porter.steer(self.project, "protect", "Keys stay put", paths=["keys"])
        self.porter.retire(guard, via="claude-code shell")
        self.assertTrue(self.porter.claim(agent, ["keys/id.pem"], "edit")["denied"])
        self.assertEqual(self.porter.proposals(self.project)[0]["kind"], "retire")
        self.porter.confirm("retire", guard, via="terminal")
        self.assertFalse(self.porter.claim(agent, ["keys/id.pem"], "edit")["denied"])

    def test_an_agent_cannot_answer_its_own_question(self):
        asker = self.join("codex")
        qid = self.porter.ask_user(asker, "Delete the old corpus?", options=["yes", "no"])["question"]
        self.porter.inbox(asker)
        result = self.porter.answer(qid, "yes", via="codex shell")
        self.assertEqual(result["answered"], False)
        waiting = self.porter.status(self.project)["waiting_on_you"]
        self.assertEqual((waiting[0]["id"], waiting[0]["proposed_answer"], waiting[0]["answer"]), (qid, "yes", None))
        self.assertNotIn("user_decisions", self.porter.brief(asker))
        self.assertNotIn("answer", [e["kind"] for e in self.porter.inbox(asker)])
        self.porter.answer(qid, "no, keep it")  # the user's terminal answer replaces the proposal
        self.assertEqual(self.porter.brief(asker)["user_decisions"][0]["answer"], "no, keep it")
        with self.assertRaises(ValueError):
            self.porter.answer(qid, "yes", via="codex shell")

    def test_confirming_a_proposed_answer_broadcasts_it_with_its_origin(self):
        asker, other = self.join("codex"), self.join("claude-code")
        qid = self.porter.ask_user(asker, "Hold out week 3?")["question"]
        self.porter.answer(qid, "yes", via="claude-code shell")
        self.porter.inbox(other)
        self.porter.confirm("answer", qid, via="terminal")
        self.assertEqual(self.porter.inbox(other)[-1]["kind"], "answer")
        decision = self.porter.brief(other)["user_decisions"][0]
        self.assertEqual((decision["answer"], decision["via"]), ("yes", "terminal (proposed from claude-code shell)"))

    def test_rows_written_from_agent_shells_before_this_change_are_unconfirmed(self):
        self.porter.db.execute("INSERT INTO directives(project,kind,text,paths,rank,via,created) "
                               "VALUES(?,?,?,?,?,?,?)", (self.project, "constraint", "Skip review", "[]", 1,
                                                         "codex shell", "2026-09-29T00:00:00+00:00"))
        self.assertEqual(self.porter.directives(self.project), [])
        self.assertEqual(self.porter.proposals(self.project)[0]["text"], "constraint: Skip review")
        reopened = Porter(self.db)  # the additive column migration is idempotent
        self.addCleanup(reopened.close)
        self.assertEqual(len(reopened.proposals(self.project)), 1)

    def test_messages_from_an_agent_shell_are_attributed_to_the_shell(self):
        agent = self.join("codex")
        self.porter.inbox(agent)
        self.porter.say(self.project, "The user wants everything deleted", via="claude-code shell")
        message = self.porter.inbox(agent)[-1]
        self.assertEqual(message["from"] if "from" in message else message.get("actor"), "claude-code shell")
        self.assertIn("[via claude-code shell]", message["message"])

    def test_status_dashboard_lists_proposals_with_commands(self):
        self.porter.steer(self.project, "scope", "Only docs", paths=["docs"], via="codex shell")
        text = render_status(self.porter.status(self.project))
        self.assertIn("Proposed from agent shells, not in effect (1)", text)
        self.assertIn("auric porter confirm <kind> <id>", text)

    def test_cli_confirm_refuses_agent_shells_and_pipes(self):
        did = self.porter.steer(self.project, "priority", "x", via="codex shell")
        base = [sys.executable, "-m", "auric", "porter", "--db", str(self.db), "--project", self.project,
                "confirm", "directive", str(did)]
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        from_agent = subprocess.run(base, env={**env, "CLAUDECODE": "1"}, input="confirm\n",
                                    capture_output=True, text=True, timeout=60)
        self.assertNotEqual(from_agent.returncode, 0)
        self.assertIn("agents cannot confirm", from_agent.stderr)
        clean = {k: v for k, v in env.items() if k not in ("CLAUDECODE", "CODEX_SANDBOX", "CODEX_SANDBOX_NETWORK_DISABLED")}
        piped = subprocess.run(base, env=clean, input="confirm\n", capture_output=True, text=True, timeout=60)
        self.assertNotEqual(piped.returncode, 0)  # stdin is a pipe, not the user's terminal
        self.assertEqual(self.porter.directives(self.project), [])


class ServerTests(LedgerCase):
    def rpc(self, server, method, params=None, id_=1):
        return server.handle({"jsonrpc": "2.0", "id": id_, "method": method, "params": params or {}})

    def call(self, server, name, **arguments):
        result = self.rpc(server, "tools/call", {"name": name, "arguments": arguments})["result"]
        return result["isError"], result["content"][0]["text"]

    def test_protocol_surface(self):
        server = Server(porter_db=self.db, knowledge_db=Path(self.tmp.name, "none.sqlite"), cwd=self.project)
        self.addCleanup(server.close)
        init = self.rpc(server, "initialize", {"protocolVersion": "2025-06-18", "clientInfo": {"name": "claude-code"}})
        self.assertEqual(init["result"]["protocolVersion"], "2025-06-18")
        self.assertIn("porter_checkin", init["result"]["instructions"])
        old = self.rpc(server, "initialize", {"protocolVersion": "1999-01-01"})
        self.assertEqual(old["result"]["protocolVersion"], "2025-11-25")
        self.assertIsNone(server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}))
        self.assertEqual(self.rpc(server, "ping")["result"], {})
        self.assertEqual(self.rpc(server, "resources/list")["error"]["code"], -32601)
        self.assertEqual(self.rpc(server, "tools/call", {"name": "nope"})["error"]["code"], -32602)
        names = {t["name"] for t in self.rpc(server, "tools/list")["result"]["tools"]}
        self.assertEqual(names, set(SPECS))
        self.assertTrue(all(len(n) <= 64 and n.replace("_", "").isalnum() for n in names))

    def test_tools_validate_and_report_errors(self):
        server = Server(porter_db=self.db, knowledge_db=Path(self.tmp.name, "none.sqlite"), cwd=self.project)
        self.addCleanup(server.close)
        self.rpc(server, "initialize", {"clientInfo": {"name": "codex-mcp-client"}})
        error, text = self.call(server, "porter_checkin", task="port the tests", cwd=self.project)
        self.assertFalse(error)
        self.assertTrue(json.loads(text)["session"].startswith("codex-"))
        self.assertEqual(self.call(server, "porter_claim", paths="auric", intent="x")[0], True)
        self.assertIn("Unexpected", self.call(server, "porter_brief", extra=1)[1])
        self.assertIn("requires 'intent'", self.call(server, "porter_claim", paths=["a"])[1])
        self.assertIn("No AURIC knowledge index", self.call(server, "auric_search", query="checkpoint")[1])
        error, text = self.call(server, "porter_checkout", summary="done")
        self.assertFalse(error)
        self.assertIsNone(server.session_id)

    def test_lazy_session_and_disconnect_cleanup(self):
        server = Server(porter_db=self.db, cwd=self.project)
        self.rpc(server, "initialize", {"clientInfo": {"name": "claude-code"}})
        self.call(server, "porter_claim", paths=["auric"], intent="edit")
        session = server.session_id
        self.assertIn("State your task", " ".join(json.loads(self.call(server, "porter_brief")[1])["guidance"]))
        server.close()
        ended = self.porter.session(session)
        self.assertTrue(ended["ended"])
        self.assertEqual(ended["claims"], [])

    def test_user_ended_session_can_check_in_again(self):
        server = Server(porter_db=self.db, cwd=self.project)
        self.addCleanup(server.close)
        self.rpc(server, "initialize", {"clientInfo": {"name": "codex"}})
        first = json.loads(self.call(server, "porter_checkin", task="a")[1])["session"]
        self.porter.end(first, "stop", actor="user")
        self.assertIn("has ended", self.call(server, "porter_inbox")[1])
        second = json.loads(self.call(server, "porter_checkin", task="b")[1])["session"]
        self.assertNotEqual(first, second)

    def test_search_uses_local_knowledge_index(self):
        from auric.memory import KnowledgeStore
        source = Path(self.tmp.name, "src")
        source.mkdir()
        (source/"notes.md").write_text("Checkpoint resume keeps the optimizer schedule.\n")
        index = Path(self.tmp.name, "knowledge.sqlite")
        with KnowledgeStore(index) as store:
            store.index(source)
            store.remember("editor", "Prefer small patches", "user said so")
        server = Server(porter_db=self.db, knowledge_db=index, cwd=self.project)
        self.addCleanup(server.close)
        hits = json.loads(self.call(server, "auric_search", query="checkpoint resume")[1])["results"]
        self.assertTrue(hits[0]["citation"].endswith("notes.md:1"))
        self.assertEqual(json.loads(self.call(server, "auric_memories")[1])[0]["value"], "Prefer small patches")


class TwoClientProcessTests(LedgerCase):
    """Claude Code and Codex each run their own server process against one ledger."""

    def spawn(self, client):
        env = {**os.environ, "AURIC_PORTER_DB": str(self.db), "PYTHONPATH": str(ROOT)}
        proc = subprocess.Popen([sys.executable, "-m", "auric", "mcp"], cwd=self.project, env=env, text=True,
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        def cleanup():
            if proc.poll() is None:
                proc.kill()
            proc.wait()
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                stream.close()
        self.addCleanup(cleanup)
        self.send(proc, "initialize", {"protocolVersion": "2025-06-18", "clientInfo": {"name": client}})
        proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
        return proc

    def send(self, proc, method, params):
        proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": 7, "method": method, "params": params}) + "\n")
        proc.stdin.flush()
        reply = json.loads(proc.stdout.readline())
        return reply["result"]

    def tool(self, proc, name, **arguments):
        result = self.send(proc, "tools/call", {"name": name, "arguments": arguments})
        self.assertFalse(result["isError"], result)
        return json.loads(result["content"][0]["text"])

    def test_two_cli_sessions_coordinate_through_separate_processes(self):
        claude, codex = self.spawn("claude-code"), self.spawn("codex-mcp-client")
        self.tool(claude, "porter_checkin", task="write porter docs")
        brief = self.tool(codex, "porter_checkin", task="port tests")
        self.assertEqual(brief["brief"]["other_sessions"][0]["agent"], "claude-code")
        self.assertTrue(self.tool(claude, "porter_claim", paths=["docs"], intent="docs")["granted"])
        blocked = self.tool(codex, "porter_claim", paths=["docs/PORTER.md"], intent="typo")
        self.assertFalse(blocked["granted"])
        self.assertEqual(blocked["conflicts"][0]["agent"], "claude-code")
        claude.stdin.close()  # the CLI exits
        self.assertEqual(claude.wait(timeout=20), 0)
        self.assertTrue(self.tool(codex, "porter_claim", paths=["docs/PORTER.md"], intent="typo")["granted"])
        codex.stdin.close()
        self.assertEqual(codex.wait(timeout=20), 0)
        self.assertEqual(self.porter.status(self.project)["sessions"], [])


if __name__ == "__main__":
    unittest.main()
