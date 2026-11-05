import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

if any(importlib.util.find_spec(name) is None for name in ("numpy", "torch", "tokenizers")):
    raise unittest.SkipTest("Install the dependencies")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

import frame
from model import GPT, GPTConfig
from prepare import BOS, train_tokenizer

TEXTS = ["The cat sat on the mat. Ünïcödé text, with numbers 12345.", "Another document; café au lait."] * 20
HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None


class FrameTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.tokenizer_path = self.directory / "tokenizer.json"
        self.tokenizer = train_tokenizer(iter(TEXTS), 300)
        self.tokenizer.save(str(self.tokenizer_path))
        self.bos = self.tokenizer.token_to_id(BOS)


def scores(qa, math_, code):
    return dict(zip(frame.OLMES_SUITES, (qa, math_, code)))


class DecisionTest(unittest.TestCase):
    def test_adopts_a_candidate_whose_rates_of_improvement_average_above_zero(self):
        self.assertTrue(frame.suites_improve(scores(0.99, 1.0, 1.0), scores(1.0, 1.0, 1.0)))

    def test_counts_each_suite_by_its_rate_rather_than_its_size(self):
        # code falls by 1% of 2.0 and math rises by 1.5% of 1.0: the rates average below zero, the BPB do not
        self.assertFalse(frame.suites_improve(scores(1.0, 1.015, 1.98), scores(1.0, 1.0, 2.0)))

    def test_keeps_the_current_best_when_nothing_changes(self):
        self.assertFalse(frame.suites_improve(scores(1.0, 1.0, 1.0), scores(1.0, 1.0, 1.0)))

    def test_refuses_to_decide_without_every_suite_finite(self):
        with self.assertRaises(ValueError):
            frame.suites_improve(scores(math.nan, 1.0, 1.0), scores(1.0, 1.0, 1.0))


class TokenizerTest(FrameTest):
    @unittest.skipUnless(HAS_TRANSFORMERS, "olmes's wrapper needs transformers")
    def test_puts_the_bos_token_before_each_text_for_olmes_and_pads_with_it(self):
        tokenizer = frame.olmes_tokenizer(self.tokenizer_path)
        ids = self.tokenizer.encode(TEXTS[0], add_special_tokens=False).ids
        self.assertEqual(tokenizer.encode(TEXTS[0]), [self.bos, *ids])
        self.assertEqual(tokenizer.encode(TEXTS[0], add_special_tokens=False), ids)
        self.assertEqual(tokenizer.pad_token_id, self.bos)


class CountTest(FrameTest):
    def write_requests(self, instances):
        directory = self.directory / "requests"
        directory.mkdir()
        with (directory / "piqa:rc:bpb.jsonl").open("w") as f:
            for instance in instances:
                f.write(json.dumps(instance) + "\n")
        return directory

    @unittest.skipUnless(HAS_TRANSFORMERS, "olmes's tokenizer needs transformers")
    def test_counts_the_tokens_the_model_reads_for_each_request_within_its_context(self):
        context, continuation = TEXTS[0], " " + TEXTS[1]
        directory = self.write_requests([
            {"request_type": "loglikelihood", "request": {"context": context, "continuation": continuation}},
            {"request_type": "loglikelihood_rolling", "request": {"context": TEXTS[1]}},
        ])
        whole = len(self.tokenizer.encode(context + continuation, add_special_tokens=False).ids) + 1
        rolling = len(self.tokenizer.encode(TEXTS[1], add_special_tokens=False).ids) + 1
        counts = frame.count_tokens(directory, self.tokenizer_path, context=8)
        self.assertEqual(counts, {"piqa:rc:bpb": {"requests": 2, "tokens": min(whole, 9) - 1 + rolling}})

    @unittest.skipUnless(HAS_TRANSFORMERS, "olmes's tokenizer needs transformers")
    def test_refuses_requests_that_a_bpb_suite_does_not_make(self):
        directory = self.write_requests([{"request_type": "generate_until", "request": {"context": "x"}}])
        with self.assertRaises(ValueError):
            frame.count_tokens(directory, self.tokenizer_path)


def small_model(vocabulary):
    torch.manual_seed(0)
    return GPT(GPTConfig(block_size=8, vocab_size=vocabulary, n_layer=1, n_head=2, n_embd=16, bias=False,
                         rotary=True))


@unittest.skipUnless(HAS_TRANSFORMERS, "olmes's wrapper needs transformers")
class OlmesModelTest(unittest.TestCase):
    PAD = 0

    def setUp(self):
        self.network = small_model(300)  # a context of 8
        self.model = frame.OlmesModel(self.network, self.PAD)

    def alone(self, row):
        row = row.contiguous()
        return self.network(row, row)[0]

    def test_gives_the_logits_of_every_position_as_a_causal_language_model(self):
        tokens = torch.randint(1, 300, (3, 6), generator=torch.Generator().manual_seed(1))
        self.assertTrue(torch.allclose(self.model(tokens).logits, self.alone(tokens)))
        self.assertEqual(self.model.config.model_type, "gpt2")
        self.assertEqual(self.model.device, torch.device("cpu"))

    def test_reads_a_longer_row_in_windows_that_end_at_its_end_and_step_back_by_half(self):
        row = torch.randint(1, 300, (1, 11), generator=torch.Generator().manual_seed(1))
        logits = self.model(row).logits
        self.assertEqual(logits.shape, (1, 11, 300))
        # the last half of a context from the window that ends with the row, the rest from the one at the start
        self.assertTrue(torch.allclose(logits[:, 7:], self.alone(row[:, 3:])[:, 4:], atol=1e-5))
        self.assertTrue(torch.allclose(logits[:, :7], self.alone(row[:, :8])[:, :7], atol=1e-5))

    def test_reads_only_the_windows_that_reach_the_longest_answer(self):
        row = torch.randint(1, 300, (1, 11), generator=torch.Generator().manual_seed(1))
        self.model.answer_tokens = 4
        logits = self.model(row).logits
        self.assertTrue(torch.allclose(logits[:, 7:], self.alone(row[:, 3:])[:, 4:], atol=1e-5))
        self.assertEqual(logits[:, :7].abs().sum().item(), 0.0)

    def test_reads_a_row_by_its_own_length_whatever_rows_share_its_batch(self):
        long_row = torch.randint(1, 300, (12,), generator=torch.Generator().manual_seed(1))
        padded_long = torch.cat([long_row[:10], torch.full((2,), self.PAD)])
        short = torch.cat([long_row[:5], torch.full((7,), self.PAD)])
        logits = self.model(torch.stack([long_row, padded_long, short])).logits
        self.assertTrue(torch.allclose(logits[1, 6:10], self.alone(padded_long[None, 2:10])[0, 4:], atol=1e-5))
        self.assertTrue(torch.allclose(logits[2, :5], self.alone(short[None, :5])[0], atol=1e-5))


class ShareTest(FrameTest):
    def test_splits_the_tasks_into_shares_of_about_equal_requests(self):
        requests = self.directory / "requests"
        requests.mkdir()
        tasks = []
        for alias, size in (("a", 900), ("b", 500), ("c", 400), ("d", 300), ("e", 200)):
            (requests / f"{alias}.jsonl").write_text("x" * size)
            tasks.append(json.dumps({"task_name": alias, "metadata": {"alias": alias}}))
        shares = frame.task_shares(tasks, requests, 2)
        aliases = [sorted(json.loads(task)["metadata"]["alias"] for task in share) for share in shares]
        # largest first, each to the share with the fewest so far: 900+300 and 500+400+200
        self.assertEqual(sorted(aliases), [["a", "d"], ["b", "c", "e"]])


@unittest.skipUnless(importlib.util.find_spec("oe_eval") is not None, "needs olmes")
class AverageTest(unittest.TestCase):
    def task(self, alias, bpb, instances=10):
        return {"task_name": alias, "task_idx": 0, "num_instances": instances, "metrics": {"primary_score": bpb},
                "task_config": {"metadata": {"alias": alias}}, "model_hash": "", "model_config": {},
                "compute_config": {}, "processing_time": 0, "current_date": "", "beaker_info": {}}

    def test_averages_arc_easy_and_challenge_as_one_member_of_the_qa_suite(self):
        from oe_eval.configs.task_suites import TASK_SUITE_CONFIGS
        from oe_eval.tasks.aggregate_tasks import add_aggregate_tasks

        frame.average_arc()
        tasks = [self.task("arc_easy:rc:bpb::olmes:full", 1.0), self.task("arc_challenge:rc:bpb::olmes:full", 2.0)]
        groups = {task["task_name"]: task for task in add_aggregate_tasks(tasks)}
        self.assertAlmostEqual(groups["arc:rc:bpb::olmes:full"]["metrics"]["primary_score"], 1.5)
        self.assertIn("arc:rc:bpb::olmes:full", TASK_SUITE_CONFIGS["olmo3:base_easy:qa_bpb"]["tasks"])


@unittest.skipUnless(importlib.util.find_spec("datasets") is not None, "needs datasets")
class DatasetsTest(unittest.TestCase):
    def test_reads_the_list_features_of_datasets_4_as_sequences(self):
        from datasets.features import features

        frame.read_datasets_4_features()
        parsed = features.generate_from_dict({"_type": "List", "feature": {"_type": "Value", "dtype": "string"}})
        self.assertIsInstance(parsed, (features.Sequence, getattr(features, "List", features.Sequence)))


class ValidationTest(FrameTest):
    def test_measures_the_validation_bpb_as_scoring_does(self):
        from loading import nanogpt_model
        from scoring import validation_bits

        data = self.directory / "data" / "tiny"
        data.mkdir(parents=True)
        tokens = [token for text in TEXTS for token in [self.bos, *self.tokenizer.encode(text, add_special_tokens=False).ids]]
        np.array(tokens, dtype=np.uint16).tofile(data / "val.bin")
        self.tokenizer.save(str(data / "tokenizer.json"))
        run = self.directory / "run"
        run.mkdir()
        network = small_model(300)
        torch.save({"model": network.state_dict(), "model_args": {**vars(network.config)},
                    "config": {"dataset": str(data), "count_table": ""}}, run / "ckpt.pt")
        model = nanogpt_model(run, data / "tokenizer.json", "cpu", "bfloat16", 262144)
        expected = validation_bits(model, data / "val.bin", self.tokenizer.decode)
        self.assertAlmostEqual(frame.validation_bpb(run, "cpu"), expected, places=6)


if __name__ == "__main__":
    unittest.main()
