"""Single-use push consent, including real Git pushes to disposable local remotes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest

from auric.push_consent import MissingConsent, PushLedger, digest, parse_updates, proposal, repository
from auric.reflex_install import install

RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "porter_hook.py"


class ConsentTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name) / "project"
        self.root.mkdir()
        self.db = Path(temp.name) / "porter.sqlite"
        self.clock = [1000.0]
        self.ledger = PushLedger(self.db, clock=lambda: self.clock[0])
        self.addCleanup(self.ledger.close)
        self.url = "https://example.com/org/repo.git"
        self.identity = str(self.root / ".git")
        self.update = {"ref": "refs/heads/main", "old_oid": "0" * 40, "new_oid": "1" * 40}
        self.request = {"project": str(self.root), "repository": self.identity, "remote_hash": digest(self.url), **self.update}

    def test_exact_binding_expiry_and_replay(self):
        self.ledger.grant(self.request, ttl=10)
        for identity, url, update in (("other", self.url, self.update), (self.identity, self.url + "/other", self.update),
                                      (self.identity, self.url, {**self.update, "ref": "refs/heads/other"}),
                                      (self.identity, self.url, {**self.update, "new_oid": "2" * 40}),
                                      (self.identity, self.url, {**self.update, "old_oid": "3" * 40})):
            with self.assertRaises(MissingConsent):
                self.ledger.consume(identity, url, [update], str(self.root))
        self.assertEqual(len(self.ledger.consume(self.identity, self.url, [self.update], str(self.root))), 1)
        with self.assertRaises(MissingConsent):
            self.ledger.consume(self.identity, self.url, [self.update], str(self.root))
        self.ledger.grant(self.request, ttl=5)
        self.clock[0] += 5
        with self.assertRaises(MissingConsent):
            self.ledger.consume(self.identity, self.url, [self.update], str(self.root))

    def test_reissuing_does_not_stack_tokens_and_batch_failure_burns_none(self):
        self.ledger.grant(self.request)
        self.ledger.grant(self.request)
        second = {**self.update, "ref": "refs/heads/other"}
        with self.assertRaises(MissingConsent):
            self.ledger.consume(self.identity, self.url, [self.update, second], str(self.root))
        self.ledger.grant({**self.request, **second})
        self.assertEqual(len(self.ledger.consume(self.identity, self.url, [self.update, second], str(self.root))), 2)
        with self.assertRaises(MissingConsent):
            self.ledger.consume(self.identity, self.url, [self.update], str(self.root))

    def test_concurrent_consumption_has_one_winner(self):
        self.ledger.grant(self.request)
        barrier, results = threading.Barrier(4), []

        def attempt():
            with PushLedger(self.db, clock=lambda: self.clock[0]) as ledger:
                barrier.wait()
                try:
                    ledger.consume(self.identity, self.url, [self.update], str(self.root))
                    results.append(True)
                except MissingConsent:
                    results.append(False)

        threads = [threading.Thread(target=attempt) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(sorted(results), [False, False, False, True])

    def test_requests_deduplicate_close_on_approval_and_reopen_for_replay(self):
        first = self.ledger.request_approval(str(self.root), self.identity, self.url, [self.update])
        self.assertEqual(first, self.ledger.request_approval(str(self.root), self.identity, self.url, [self.update]))
        self.ledger.grant(self.request)
        self.assertEqual(self.ledger.questions(str(self.root)), [])
        self.ledger.consume(self.identity, self.url, [self.update], str(self.root))
        second = self.ledger.request_approval(str(self.root), self.identity, self.url, [self.update])
        self.assertNotEqual(first, second)
        self.assertEqual(len(self.ledger.questions(str(self.root))), 1)

    def test_multiref_question_waits_for_all_approvals(self):
        second = {**self.update, "ref": "refs/heads/other"}
        qid = self.ledger.request_approval(str(self.root), self.identity, self.url, [second, self.update])
        self.assertEqual(qid, self.ledger.request_approval(str(self.root), self.identity, self.url, [self.update, second]))
        self.ledger.grant(self.request)
        self.assertEqual(len(self.ledger.questions(str(self.root))), 1)
        self.ledger.grant({**self.request, **second})
        self.assertEqual(self.ledger.questions(str(self.root)), [])

    def test_protocol_parser_rejects_deletions_duplicates_and_invalid_oids(self):
        line = f"refs/heads/main {'1'*40} refs/heads/main {'0'*40}\n"
        self.assertEqual(parse_updates(line), [self.update])
        self.assertEqual(parse_updates(""), [])
        for value in (line + line, line.replace("1" * 40, "0" * 40), "bad input", line.replace("1" * 40, "not-a-sha")):
            with self.assertRaises(ValueError):
                parse_updates(value)


class GitIntegrationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="porter-git-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name) / "work space"
        self.remote = Path(temp.name) / "remote.git"
        self.db = Path(temp.name) / "porter.sqlite"
        self.git("init", "--bare", str(self.remote), cwd=Path(temp.name))
        self.git("init", "-b", "main", str(self.root), cwd=Path(temp.name))
        self.git("config", "user.name", "Porter Test")
        self.git("config", "user.email", "porter@example.invalid")
        self.git("config", "commit.gpgsign", "false")
        self.git("remote", "add", "origin", str(self.remote))
        (self.root / "notes.md").write_text("Initial\n")
        self.git("add", "notes.md")
        self.git("commit", "-m", "Initial")

    def git(self, *args, cwd=None, expected=0):
        env = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull}
        result = subprocess.run(["git", *args], cwd=cwd or self.root, text=True, capture_output=True, env=env)
        if expected is not None:
            self.assertEqual(result.returncode, expected, result.stderr)
        return result

    def test_real_push_denied_then_exact_approval_succeeds_new_commit_denied(self):
        install(self.root, db=self.db)
        denied = self.git("push", "origin", "main", expected=None)
        self.assertNotEqual(denied.returncode, 0)
        self.assertIn("Porter blocked", denied.stderr)
        request = proposal(self.root, "origin", "refs/heads/main", "HEAD")
        with PushLedger(self.db) as ledger:
            ledger.grant(request)
        self.git("push", "origin", "main")
        head = self.git("rev-parse", "HEAD").stdout.strip()
        remote_head = self.git("--git-dir", str(self.remote), "rev-parse", "refs/heads/main").stdout.strip()
        self.assertEqual(head, remote_head)
        self.git("push", "origin", "main")  # up-to-date needs no additional consent
        (self.root / "notes.md").write_text("Next\n")
        self.git("commit", "-am", "Next")
        self.assertNotEqual(self.git("push", "origin", "main", expected=None).returncode, 0)

    def test_installer_preserves_config_and_previous_hook_and_is_idempotent(self):
        settings = self.root / ".claude" / "settings.local.json"
        settings.parent.mkdir()
        settings.write_text(json.dumps({"permissions": {"deny": ["Bash(rm:*)"]}, "hooks": {"SessionStart": [
            {"hooks": [{"type": "command", "command": "echo existing"}]}]}}))
        pre_push = self.root / ".git" / "hooks" / "pre-push"
        pre_push.write_text("#!/bin/sh\ncat > \"$(git rev-parse --git-dir)/previous-input\"\nexit 1\n")
        pre_push.chmod(0o700)
        original = pre_push.read_bytes()
        install(self.root, db=self.db)
        install(self.root, db=self.db)
        config = json.loads(settings.read_text())
        self.assertEqual(config["permissions"]["deny"], ["Bash(rm:*)"])
        self.assertEqual(len(config["hooks"]["SessionStart"]), 2)
        self.assertEqual((pre_push.parent / "pre-push.before-porter").read_bytes(), original)
        request = proposal(self.root, "origin", "refs/heads/main", "HEAD")
        with PushLedger(self.db) as ledger:
            ledger.grant(request)
        self.assertNotEqual(self.git("push", "origin", "main", expected=None).returncode, 0)
        self.assertIn("refs/heads/main", (pre_push.parent.parent / "previous-input").read_text())
        with PushLedger(self.db) as ledger:
            self.assertEqual(ledger.db.execute("SELECT COUNT(*) FROM reflex_push_grants WHERE consumed IS NULL").fetchone()[0], 1)
        self.assertIn(".codex/hooks.json", self.git("ls-files", "--others", "--ignored", "--exclude-standard").stdout)

    def test_external_shared_hook_path_and_invalid_config_do_not_get_overwritten(self):
        shared = self.root.parent / "shared-hooks"
        shared.mkdir()
        self.git("config", "core.hooksPath", str(shared))
        with self.assertRaisesRegex(ValueError, "external/shared"):
            install(self.root, db=self.db)
        self.assertFalse((self.root / ".claude").exists())
        self.git("config", "--unset", "core.hooksPath")
        path = self.root / ".codex" / "hooks.json"
        path.parent.mkdir()
        path.write_text("invalid JSON")
        with self.assertRaises(ValueError):
            install(self.root, db=self.db)
        self.assertEqual(path.read_text(), "invalid JSON")
        self.assertFalse((self.root / ".claude").exists())

    def test_approval_cli_cannot_mint_token_noninteractively(self):
        result = subprocess.run([sys.executable, str(RUNNER), "--db", str(self.db), "approve-push", "--ref", "refs/heads/main"],
                                cwd=self.root, input="approve\n", capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("interactive terminal", result.stderr)
        self.assertFalse(self.db.exists())

    def test_worktrees_share_repository_binding(self):
        other = self.root.parent / "other-worktree"
        self.git("worktree", "add", "-b", "other", str(other))
        self.assertEqual(repository(self.root)[1], repository(other)[1])


if __name__ == "__main__":
    unittest.main()
