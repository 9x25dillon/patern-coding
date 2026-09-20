"""Behavioral checks for training correctness and the local tool boundary."""
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch

import torch

from auric.assistant import build_prompt, propose_patch, ask
from auric.data import prepare, TokenDataset
from auric.memory import KnowledgeStore
from auric.model import LanguageModel, ModelConfig, memory_report
from auric.tokenizer import ByteTokenizer
from auric.training import train, TrainConfig, load_model, evaluate
from auric.workspace import read_file, scan

torch.set_num_threads(1)


class TokenizerTests(unittest.TestCase):
    def test_unicode_and_code_round_trip(self):
        tok = ByteTokenizer()
        for text in ("", "def f(x):\n    return x + 1\n", "λ = '你好 🚀'", "\x00"):
            self.assertEqual(tok.decode(tok.encode(text)), text)
            self.assertTrue(all(4 <= i < tok.vocab_size for i in tok.encode(text)))

    def test_supervised_mask_and_eos(self):
        tok = ByteTokenizer()
        ids, mask = tok.example({"prompt": "Q", "completion": "A"})
        self.assertEqual(ids, [1, ord("Q")+4, 3, ord("A")+4, 2])
        self.assertEqual(mask, [False, False, False, True, True])


class ModelTests(unittest.TestCase):
    def test_paired_gate_initialization_matches_baseline(self):
        cfg=ModelConfig(dim=16,layers=2,heads=2,context=8)
        torch.manual_seed(6); a=LanguageModel(cfg)
        torch.manual_seed(6); b=LanguageModel(replace(cfg,coherence_gate=True))
        x=torch.tensor([[1,10,11]])
        torch.testing.assert_close(a(x)[0],b(x)[0],rtol=0,atol=0)

    def test_future_tokens_cannot_change_past_logits(self):
        for gate in (False, True):
            torch.manual_seed(7)
            model = LanguageModel(ModelConfig(dim=32, layers=2, heads=4, context=16, coherence_gate=gate)).eval()
            x = torch.tensor([[1, 10, 20, 30, 40, 50]])
            y = x.clone(); y[:, 3:] = torch.tensor([80, 90, 100])
            torch.testing.assert_close(model(x)[0][:, :3], model(y)[0][:, :3], rtol=0, atol=1e-6)

    def test_padding_targets_do_not_change_loss(self):
        model = LanguageModel(ModelConfig(dim=32, layers=1, heads=4, context=8)).eval()
        x = torch.tensor([[1, 20, 30]])
        y = torch.tensor([[20, 30, 2]])
        padded_x = torch.tensor([[1, 20, 30, 0, 0]])
        padded_y = torch.tensor([[20, 30, 2, -100, -100]])
        torch.testing.assert_close(model(x, y)[1], model(padded_x, padded_y)[1])

    def test_checkpointing_preserves_gradients(self):
        cfg = ModelConfig(dim=32, layers=2, heads=4, context=8, coherence_gate=True)
        a = LanguageModel(cfg)
        b = LanguageModel(replace(cfg, gradient_checkpointing=True))
        b.load_state_dict(a.state_dict())
        x = torch.tensor([[1, 10, 11, 12]])
        y = torch.tensor([[10, 11, 12, 2]])
        a(x, y)[1].backward(); b(x, y)[1].backward()
        for pa, pb in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(pa.grad, pb.grad)

    def test_sampling_reproducible_and_bounded(self):
        model = LanguageModel(ModelConfig(dim=16, layers=1, heads=2, context=4))
        a = model.generate([1, 10, 20, 30, 40], max_new_tokens=12, temperature=0.7, seed=9)
        b = model.generate([1, 10, 20, 30, 40], max_new_tokens=12, temperature=0.7, seed=9)
        self.assertEqual(a, b)
        self.assertLessEqual(len(a), 12)
        self.assertTrue(all(i >= 4 for i in a))
        self.assertTrue(model.training)

    def test_memory_estimate_counts_shared_embedding_once(self):
        cfg = ModelConfig(dim=16, layers=1, heads=2, context=8)
        model = LanguageModel(cfg)
        self.assertEqual(memory_report(cfg)["parameters"], sum(p.numel() for p in model.parameters()))


class DataTests(unittest.TestCase):
    def test_group_split_dedup_and_next_token_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = [{"text": "abc", "group": "fileA"}, {"text": "def", "group": "fileA"},
                       {"text": "ghi", "group": "fileB"}, {"text": "abc", "group": "copy"}]
            (root/"data.jsonl").write_text("\n".join(json.dumps(x) for x in records))
            m = prepare([root/"data.jsonl"], root/"out", context=8)
            self.assertEqual(m["duplicates_removed"], 1)
            groups = {s: {r["group"] for r in m["records"] if r["split"] == s} for s in ("train", "val")}
            self.assertFalse(groups["train"] & groups["val"])
            for split in ("train", "val"):
                ds = TokenDataset(root/"out", split)
                for x, y in ds:
                    self.assertEqual(x[0], 1)
                    torch.testing.assert_close(x[1:4], y[:3])
                    self.assertEqual(y[3], 2)
                    self.assertTrue((y[4:] == -100).all())

    def test_supervised_padding_and_prompt_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/"s.jsonl").write_text('\n'.join(json.dumps({"prompt": q, "completion": "A"}) for q in ["Q", "R"]))
            prepare([root/"s.jsonl"], root/"out", context=8)
            x,y = TokenDataset(root/"out", "train")[0]
            self.assertEqual(y.tolist(), [-100, -100, ord("A")+4, 2, -100, -100, -100, -100])

    def test_checksum_detects_modified_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/"s.jsonl").write_text('{"text":"abc"}\n{"text":"def"}\n')
            prepare([root/"s.jsonl"], root/"out", context=8)
            with (root/"out/train.x.bin").open("ab") as f: f.write(b"bad")
            with self.assertRaisesRegex(ValueError, "checksum"):
                TokenDataset(root/"out", "train")

    def test_single_group_and_unknown_schema_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for record in ({"text":"abc"}, {"metadata":{"looks":"like data"}}):
                (root/"s.jsonl").write_text(json.dumps(record))
                with self.assertRaises(ValueError):
                    prepare([root/"s.jsonl"], root/"out")
                self.assertFalse((root/"out").exists())


class TrainingTests(unittest.TestCase):
    def test_batch_trimming_keeps_prompt_and_all_targets(self):
        from auric.training import collate
        x=torch.tensor([1, 10, 11, 3, 12, 0, 0])
        y=torch.tensor([-100, -100, -100, 12, 2, -100, -100])
        a,b=collate([(x,y)],"cpu")
        self.assertEqual(a.tolist(),[[1,10,11,3,12]])
        self.assertEqual(b.tolist(),[[-100,-100,-100,12,2]])

    def fixture(self, root):
        path = root/"input.jsonl"
        path.write_text('\n'.join(json.dumps({"text": "a "*20+str(i), "group": f"g{i}"}) for i in range(8)))
        prepare([path], root/"data", context=16)
        return root/"data"

    def test_learns_and_checkpoint_reloads(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp); data = self.fixture(root)
            cfg = ModelConfig(dim=32, layers=1, heads=4, context=16)
            result = train(data, root/"run", cfg, TrainConfig(steps=30, accumulation=1,
                           learning_rate=0.01, warmup_steps=2, eval_every=15, threads=1))
            self.assertLess(result["final_validation"]["loss"], result["initial_validation"]["loss"]-1)
            model, state = load_model(root/"run/last.pt")
            self.assertEqual(state["step"], 30)
            measured = evaluate(model, TokenDataset(data,"val"))
            self.assertAlmostEqual(measured["loss"], result["final_validation"]["loss"], places=6)
            self.assertTrue(result["complete"])

    def test_resume_matches_uninterrupted_weights(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp); data = self.fixture(root)
            cfg = ModelConfig(dim=16, layers=1, heads=2, context=16, dropout=0.1)
            tc = TrainConfig(steps=6, accumulation=3, eval_every=3, threads=1)
            train(data, root/"full", cfg, tc)
            result = train(data, root/"split", cfg, tc, stop_after=3)
            self.assertFalse(result["complete"])
            train(data, root/"split", cfg, tc, resume=root/"split/last.pt")
            a,_ = load_model(root/"full/last.pt"); b,_ = load_model(root/"split/last.pt")
            for pa,pb in zip(a.parameters(), b.parameters()):
                torch.testing.assert_close(pa, pb, atol=0, rtol=0)

    def test_resume_rejects_changed_schedule(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp); data = self.fixture(root)
            cfg = ModelConfig(dim=16,layers=1,heads=2,context=16)
            tc = TrainConfig(steps=2, accumulation=1, threads=1)
            train(data, root/"run", cfg, tc, stop_after=1)
            with self.assertRaisesRegex(ValueError, "configuration"):
                train(data, root/"run", cfg, replace(tc, steps=4), resume=root/"run/last.pt")


class KnowledgeTests(unittest.TestCase):
    def test_index_fences_citations_and_refresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); docs=root/"docs"; docs.mkdir()
            (docs/"guide.md").write_text("# Usage\n```python\ndef checkpoint_resume():\n    return 1\n```\n")
            with KnowledgeStore(root/"db.sqlite") as store:
                store.index(docs)
                hits=store.search("checkpoint_resume")
                self.assertEqual(len(hits),1)
                self.assertIn("```python",hits[0]["content"])
                self.assertEqual(hits[0]["citation"],str(docs/"guide.md")+":1")
                (docs/"guide.md").unlink()
                store.index(docs)
                self.assertEqual(store.search("checkpoint_resume"),[])

    def test_memory_revisions_survive_reopen_and_forget(self):
        with tempfile.TemporaryDirectory() as tmp:
            db=Path(tmp)/"db.sqlite"
            with KnowledgeStore(db) as s:
                s.remember("language","Python","user")
                s.remember("language","Rust","user correction")
            with KnowledgeStore(db) as s:
                self.assertEqual(s.memories()[0]["value"],"Rust")
                self.assertEqual(len(s.memories(history=True)),2)
                s.forget("language")
                self.assertEqual(s.memories(),[])
                self.assertEqual(len(s.memories(history=True)),3)

    def test_failed_scan_does_not_erase_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); docs=root/"docs";docs.mkdir()
            (docs/"a.py").write_text("checkpoint = 1")
            with KnowledgeStore(root/"db.sqlite") as s:
                s.index(docs)
                (docs/"b.py").write_text("more = 2")
                with self.assertRaises(ValueError): s.index(docs,max_files=1)
                self.assertEqual(len(s.search("checkpoint")),1)

    def test_read_rejects_escape_symlink_and_secrets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);docs=root/"docs";docs.mkdir()
            (root/"outside.py").write_text("outside")
            (docs/"link.py").symlink_to(root/"outside.py")
            (docs/"keys.py").write_text("hf_"+"a"*30)
            for path in ("../outside.py","link.py","keys.py","/etc/passwd"):
                with self.assertRaises(ValueError): read_file(docs,path)
            chunks,_=scan(docs)
            self.assertEqual(chunks,[])

    def test_query_syntax_is_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            with KnowledgeStore(Path(tmp)/"db.sqlite") as s:
                self.assertEqual(s.search('" OR * NEAR(foo) --'),[])

    def test_retrieval_only_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=ask(Path(tmp)/"db.sqlite","hello")
            self.assertEqual(result["mode"],"retrieval_only")
            self.assertNotIn("answer",result)

    def test_context_budget_preserves_question(self):
        tok=ByteTokenizer()
        hits=[{"path":"a.py","start":1,"content":"x = 1\n"*100,"citation":"/a.py:1"}]
        ids, citations=build_prompt("fix x",hits,[],128)
        self.assertLessEqual(len(ids),128)
        self.assertIn("Question: fix x",tok.decode(ids))
        self.assertEqual(citations,["/a.py:1"])
        with self.assertRaises(ValueError): build_prompt("long"*100,hits,[],128)

    def test_patch_proposal_does_not_modify_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/"a.py").write_text("x = 1\n")
            diff=propose_patch(root,"a.py","x = 2\n")
            self.assertIn("+x = 2",diff)
            self.assertEqual((root/"a.py").read_text(),"x = 1\n")

    def test_source_export_excludes_evaluation_and_groups_files(self):
        from auric.workspace import export_sources
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); docs=root/"docs";docs.mkdir()
            (docs/"a.py").write_text("def add(a,b):\n    return a+b\n"*40)
            (docs/"test_a.py").write_text("heldout = 3")
            (docs/"levels").mkdir();(docs/"levels/answer.py").write_text("reference = 42")
            out=root/"corpus.jsonl";export_sources([docs],out)
            rows=[json.loads(line) for line in out.read_text().splitlines()]
            self.assertEqual(len({r["group"] for r in rows}),1)
            self.assertTrue(all("heldout" not in r["text"] and "reference" not in r["text"] for r in rows))


class BridgeTests(unittest.TestCase):
    def test_generated_code_requires_isolation(self):
        import sys
        from auric.vibecoder_bridge import check_code
        models=ModuleType("vibecoder.models")
        sentinel=object();models.Source=SimpleNamespace(THIRD_PARTY=sentinel)
        result=SimpleNamespace(passed_count=1,total_count=1,all_passed=True,to_json=lambda:{"outcomes":[]})
        from unittest.mock import Mock
        runner=SimpleNamespace(run_code=Mock(return_value=result))
        level=SimpleNamespace(func_name="f",tests_for=lambda seed: [])
        levels=SimpleNamespace(get_level=lambda task:level)
        with patch.dict(sys.modules,{"vibecoder.models":models}), patch("auric.vibecoder_bridge.connect",return_value=(levels,runner)):
            check_code("unused","task","def f(): return 1")
        self.assertIs(runner.run_code.call_args.kwargs["source"],sentinel)

    def test_no_fallback_when_isolation_fails(self):
        import sys
        from auric.vibecoder_bridge import check_code
        from unittest.mock import Mock
        models=ModuleType("vibecoder.models");models.Source=SimpleNamespace(THIRD_PARTY=object())
        runner=SimpleNamespace(run_code=Mock(side_effect=RuntimeError("no isolation")))
        levels=SimpleNamespace(get_level=lambda t:SimpleNamespace(func_name="f",tests_for=lambda s:[]))
        with patch.dict(sys.modules,{"vibecoder.models":models}), patch("auric.vibecoder_bridge.connect",return_value=(levels,runner)):
            with self.assertRaisesRegex(RuntimeError,"no isolation"): check_code("unused","t","bad")
        self.assertEqual(runner.run_code.call_count,1)

    def test_repair_loop_records_failure_then_success(self):
        from auric.vibecoder_bridge import repair_episode
        from unittest.mock import Mock
        tok=ByteTokenizer()
        level=SimpleNamespace(brief="Return one.",starter="def f():\n    pass",func_name="f")
        levels=SimpleNamespace(get_level=lambda task:level)
        model=SimpleNamespace(config=SimpleNamespace(context=512),generate=Mock(side_effect=[tok.encode("def f(): return 0"),tok.encode("def f(): return 1")]))
        state={"step":1,"data_signature":"test","training_groups":[]}
        failed={"all_passed":False,"result":{"error":"","outcomes":[{"passed":False,"got":"0","expected":"1"}]}}
        passed={"all_passed":True,"result":{"error":"","outcomes":[{"passed":True}]}}
        with tempfile.TemporaryDirectory() as tmp, patch("auric.vibecoder_bridge.connect",return_value=(levels,None)), patch("auric.training.load_model",return_value=(model,state)), patch("auric.vibecoder_bridge.check_code",side_effect=[failed,passed,passed]):
            out=Path(tmp)/"episode.json"
            result=repair_episode("unused","heldout","unused",out)
            self.assertTrue(result["passed"])
            self.assertEqual(len(result["attempts"]),2)
            self.assertIn("Failure excerpt",result["attempts"][1]["prompt"])
            self.assertFalse(result["training_eligible"])
            self.assertTrue(out.exists())

    def test_repair_rejects_training_task(self):
        from auric.vibecoder_bridge import repair_episode
        with patch("auric.vibecoder_bridge.connect",return_value=(SimpleNamespace(get_level=lambda t:None),None)), patch("auric.training.load_model",return_value=(None,{"training_groups":["seen"]})):
            with self.assertRaisesRegex(ValueError,"used for training"):
                repair_episode("unused","seen","unused","unused.json")

    def test_task_feedback_fits_byte_budget(self):
        from auric.vibecoder_bridge import task_prompt
        level=SimpleNamespace(brief="Return one.",starter="def f():\n    pass",func_name="f")
        text=task_prompt(level,128,code="é"*100,feedback="failure"*100)
        self.assertLessEqual(len(ByteTokenizer().encode(text))+2,128)
        self.assertIn("def f():",text)


class HubTests(unittest.TestCase):
    def test_fetch_requires_pinned_revision_and_data_extension(self):
        from auric.hub import fetch_data
        for revision,name in (("main","data.jsonl"),("a"*40,"remote.py"),("a"*40,"../data.jsonl")):
            with self.assertRaises(ValueError): fetch_data("owner/repo",name,revision,"unused")


if __name__ == "__main__":
    unittest.main()
