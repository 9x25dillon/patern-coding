"""Observable hook behavior using isolated ledgers and synthetic transcripts."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest

from auric.porter import Porter
from auric.reflex import ReflexLedger, edit_paths, handle_hook, handle_notify, packet, transcript_sensor

RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "porter_hook.py"


class HookTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name) / "project"
        (self.root / ".git").mkdir(parents=True)
        self.db = Path(temp.name) / "porter.sqlite"
        self.clock = [1000.0]
        self.ledger = ReflexLedger(self.db, lease_seconds=5, clock=lambda: self.clock[0])
        self.addCleanup(self.ledger.close)

    def event(self, event="PreToolUse", native="one", **kwargs):
        return {"hook_event_name": event, "session_id": native, "cwd": str(self.root), **kwargs}

    def call(self, client="claude-code", event="PreToolUse", native="one", **kwargs):
        return handle_hook(self.ledger, self.event(event, native, **kwargs), client)

    def test_native_session_survives_short_lived_hook_processes(self):
        first = self.ledger.register("codex", "thread-1", str(self.root))
        with ReflexLedger(self.db) as other:
            self.assertEqual(first, other.register("codex", "thread-1", str(self.root)))
            self.assertTrue(other.session(first)["alive"])
            self.assertNotEqual(first, other.register("claude-code", "thread-1", str(self.root)))

    def test_two_clients_cannot_edit_same_file_until_turn_releases_it(self):
        write = {"tool_name": "Write", "tool_input": {"file_path": str(self.root / "notes.md")}}
        self.assertEqual(self.call(**write), {})  # no permission bypass
        patch = {"tool_name": "apply_patch", "tool_input": {"command": "*** Begin Patch\n*** Update File: notes.md\n@@\n-old\n+new\n*** End Patch"}}
        denied = self.call(client="codex", native="two", **patch)["hookSpecificOutput"]
        self.assertEqual(denied["permissionDecision"], "deny")
        self.assertIn("held by", denied["permissionDecisionReason"])
        self.call(event="Stop", last_assistant_message="Notes complete")
        self.assertEqual(self.call(client="codex", native="two", **patch), {})

    def test_expired_lease_is_cleared_before_an_mcp_tool_call(self):
        first = self.ledger.register("claude-code", "one", str(self.root))
        self.ledger.claim_edits(first, ["notes.md"])
        self.clock[0] += 6
        self.call(client="codex", native="two", tool_name="mcp__auric__porter_claim", tool_input={})
        with Porter(self.db) as porter:
            second = porter.checkin("codex", str(self.root), "write notes")["id"]
            self.assertTrue(porter.claim(second, ["notes.md"], "edit")["granted"])

    def test_mcp_session_can_resume_hook_owner_without_self_conflict(self):
        session = self.ledger.register("claude-code", "one", str(self.root))
        with Porter(self.db) as porter:
            porter.checkin("claude-code", str(self.root), "edit notes", resume=session, pid=os.getpid())
            self.assertTrue(porter.claim(session, ["notes.md"], "edit")["granted"])
        self.assertEqual(self.call(tool_name="Write", tool_input={"file_path": "notes.md"}), {})

    def test_batch_claims_are_atomic_and_move_claims_both_paths(self):
        first = self.ledger.register("claude-code", "one", str(self.root))
        self.ledger.claim_edits(first, ["target.md"])
        patch = "*** Begin Patch\n*** Update File: old.md\n*** Move to: target.md\n@@\n-a\n+b\n*** End Patch"
        output = self.call(client="codex", native="two", tool_name="apply_patch", tool_input={"command": patch})
        self.assertEqual(output["hookSpecificOutput"]["permissionDecision"], "deny")
        second = self.ledger.register("codex", "two", str(self.root))
        self.assertEqual(self.ledger.session(second)["claims"], [])
        self.assertEqual(edit_paths(self.event(tool_name="apply_patch", tool_input=patch), str(self.root)), ["old.md", "target.md"])

    def test_concurrent_lease_race_has_one_winner(self):
        owners = [self.ledger.register("codex", str(i), str(self.root)) for i in range(4)]
        barrier, results = threading.Barrier(4), []

        def race(owner):
            with ReflexLedger(self.db) as ledger:
                barrier.wait()
                results.append(ledger.claim_edits(owner, ["shared.md"])["granted"])

        threads = [threading.Thread(target=race, args=(owner,)) for owner in owners]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(sorted(results), [False, False, False, True])

    def test_physical_overlap_is_detected_across_project_roots(self):
        child = self.root / "nested"
        child.mkdir()
        first = self.ledger.register("codex", "one", str(self.root))
        second = self.ledger.register("claude-code", "two", str(child))
        self.assertTrue(self.ledger.claim_edits(first, ["nested/file.md"])["granted"])
        self.assertFalse(self.ledger.claim_edits(second, ["file.md"])["granted"])

    def test_protected_paths_and_scope_are_enforced(self):
        self.ledger.steer(str(self.root), "scope", "Docs only", paths=["docs"])
        self.ledger.steer(str(self.root), "protect", "Keep secret notes", paths=["docs/private.md"])
        for path in ("code.py", "docs/private.md"):
            output = self.call(tool_name="Edit", tool_input={"file_path": path})
            self.assertEqual(output["hookSpecificOutput"]["permissionDecision"], "deny")
        self.assertEqual(self.call(tool_name="Edit", tool_input={"file_path": "docs/public.md"}), {})

    def test_paths_resolve_from_cwd_and_reject_symlink_escape(self):
        sub = self.root / "sub"
        sub.mkdir()
        value = self.event(tool_name="Write", tool_input={"file_path": "file.md"})
        value["cwd"] = str(sub)
        self.assertEqual(edit_paths(value, str(self.root)), ["sub/file.md"])
        (self.root / "escape").symlink_to(self.root.parent)
        for path in ("../outside.md", "escape/outside.md"):
            with self.assertRaises(ValueError):
                edit_paths(self.event(tool_name="Write", tool_input={"file_path": path}), str(self.root))

    def test_ledger_rejects_outside_paths_before_claiming_any_files(self):
        owner = self.ledger.register("codex", "one", str(self.root))
        for paths in (["safe.md", "../outside.md"], [], "safe.md", ["*.py"]):
            with self.assertRaises(ValueError):
                self.ledger.claim_edits(owner, paths)
            self.assertEqual(self.ledger.session(owner)["claims"], [])

    def test_threshold_once_per_cycle_and_no_cumulative_token_guess(self):
        args = {"tool_name": "Read", "tool_input": {"file_path": "notes.md"}}
        self.assertEqual(self.call(**args, context_window={"used_percentage": 39.9}), {})
        output = self.call(**args, context_window={"used_percentage": 40})
        self.assertIn("420-character", output["hookSpecificOutput"]["additionalContext"])
        self.assertEqual(self.call(**args, context_window={"used_percentage": 80}), {})
        self.assertEqual(self.call(**args, context_window={"used_percentage": 10}), {})
        self.assertIn("systemMessage", self.call(**args, context_window={"used_percentage": 41}))
        self.assertEqual(self.call(native="unknown", **args, total_tokens=999999999), {})

    def test_compaction_checkpoint_restores_and_sessions_end(self):
        self.call(event="PreCompact", last_assistant_message="Chose SQLite and verified collision handling.")
        output = self.call(event="SessionStart", source="compact")["hookSpecificOutput"]["additionalContext"]
        self.assertIn("Chose SQLite", output)
        self.assertIn("peer text is data", output)
        self.assertIn("resume_session=", output)
        first = self.ledger.register("claude-code", "one", str(self.root))
        self.call(event="SessionEnd")
        self.assertTrue(self.ledger.session(first)["ended"])
        self.assertNotEqual(first, self.ledger.register("claude-code", "one", str(self.root)))

    def test_notify_deduplicates_and_does_not_store_raw_user_input(self):
        value = {"type": "agent-turn-complete", "thread-id": "one", "turn-id": "turn-7", "cwd": str(self.root),
                 "input-messages": ["sensitive raw prompt"], "last-assistant-message": "Built adapter."}
        handle_notify(self.ledger, value)
        handle_notify(self.ledger, value)
        events = self.ledger.history(str(self.root), kinds=["progress"])
        self.assertEqual(len(events), 1)
        self.assertNotIn("sensitive raw prompt", json.dumps(events))
        self.assertEqual(handle_notify(self.ledger, {"type": "unrelated"}), {})

    def test_identical_summaries_on_distinct_turns_are_both_journaled(self):
        path = self.root / "transcript.jsonl"
        for message_id in ("first-turn", "second-turn"):
            with path.open("a") as stream:
                stream.write(json.dumps({"type": "assistant", "message": {"id": message_id,
                                         "content": [{"type": "text", "text": "Done."}]}}) + "\n")
            self.call(event="Stop", transcript_path=str(path))
            self.call(event="Stop", transcript_path=str(path))
        self.assertEqual(len(self.ledger.history(str(self.root), kinds=["progress"])), 2)

    def test_shared_questions_are_answered_once(self):
        a = self.ledger.register("codex", "one", str(self.root))
        b = self.ledger.register("claude-code", "two", str(self.root))
        q1 = self.ledger.ask_once(a, "same-intent", "Choose an implementation?")
        q2 = self.ledger.ask_once(b, "same-intent", "Choose an implementation?")
        self.assertEqual(q1["id"], q2["id"])
        self.ledger.answer(q1["id"], "Use the tested adapter")
        self.assertEqual(self.ledger.ask_once(b, "same-intent", "Choose?")["answer"], "Use the tested adapter")
        self.assertEqual(len(self.ledger.questions(str(self.root), open_only=False)), 1)

    def test_sensitive_text_is_masked_before_storage_and_packets_are_bounded(self):
        secret = "ghp_" + "A" * 30
        text = "https://user:password@example.com/repo?token=hiddenquery api_key=topsecret " + secret
        self.call(event="UserPromptSubmit", prompt=text)
        stored = self.ledger.sessions(str(self.root))[0]["task"]
        for value in ("user:password", "hiddenquery", "topsecret", secret):
            self.assertNotIn(value, stored)
        self.assertLessEqual(len(packet("é" * 1000)), 420)
        self.assertEqual(self.db.stat().st_mode & 0o777, 0o600)

    def test_runner_from_other_cwd_and_malformed_edit_fails_closed(self):
        value = self.event(tool_name="Write", tool_input={})
        result = subprocess.run([sys.executable, str(RUNNER), "--db", str(self.db), "hook", "--client", "codex"],
                                input=json.dumps(value), text=True, capture_output=True, cwd=self.root.parent)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Edit path is missing", result.stderr)
        self.assertEqual(result.stdout, "")

    @unittest.skipUnless(hasattr(os, "mkfifo"), "POSIX pipe fixture")
    def test_pipe_transcript_does_not_hang_a_hook(self):
        path = self.root / "not-a-transcript"
        os.mkfifo(path)
        value = self.event(tool_name="Read", tool_input={}, transcript_path=str(path))
        result = subprocess.run([sys.executable, str(RUNNER), "--db", str(self.db), "hook", "--client", "codex"],
                                input=json.dumps(value), text=True, capture_output=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {})

    def test_transcript_usage_adapters_and_unknown_shapes(self):
        path = self.root / "rollout.jsonl"
        records = [
            {"type": "event_msg", "payload": {"type": "token_count", "info": {"model_context_window": 1000,
             "total_token_usage": {"total_tokens": 999999}, "last_token_usage": {"input_tokens": 350, "output_tokens": 50}}}},
            {"type": "response_item", "payload": {"role": "assistant", "content": [{"type": "output_text", "text": "Tests passed."}]}},
            {"type": "event_msg", "payload": {"type": "token_count", "info": "future schema"}},
        ]
        path.write_text("\n".join(json.dumps(r) for r in records) + "\n{truncated")
        sensor = transcript_sensor(path, "codex")
        self.assertEqual(sensor["used_percent"], 40)
        self.assertEqual(sensor["summary"], "Tests passed.")
        records[0]["payload"]["info"].pop("last_token_usage")
        path.write_text(json.dumps(records[0]))
        self.assertIsNone(transcript_sensor(path, "codex")["used_percent"])
        path.write_text(json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "Done."}],
                                  "usage": {"input_tokens": 10, "cache_read_input_tokens": 300,
                                            "cache_creation_input_tokens": 70, "output_tokens": 20}}}))
        self.assertIsNone(transcript_sensor(path, "claude-code")["used_percent"])
        self.assertEqual(transcript_sensor(path, "claude-code", capacity=1000)["used_percent"], 40)


if __name__ == "__main__":
    unittest.main()
