import functools
import importlib.util
import json
import math
import random
import sys
import tempfile
import types
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
HAS_OLMES = all(importlib.util.find_spec(name) is not None for name in ("oe_eval", "transformers"))


class FrameTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.tokenizer_path = self.directory / "tokenizer.json"
        self.tokenizer = train_tokenizer(iter(TEXTS), 300)
        self.tokenizer.save(str(self.tokenizer_path))
        self.bos = self.tokenizer.token_to_id(BOS)

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False).ids


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


def request(doc_id, idx, label, continuation=" x", context="Q:"):
    return {"doc_id": doc_id, "native_id": doc_id % frame.UNCONDITIONED, "idx": idx, "label": label,
            "request": {"context": context, "continuation": continuation}}


def multiple_choice(labels, choices=3, unconditioned=True):
    """A task's requests as olmes writes them: each question's choices, then the same without the questions."""
    asked = [request(doc_id, idx, label, f" choice {doc_id}-{idx}") for doc_id, label in enumerate(labels)
             for idx in range(choices)]
    unasked = [{**r, "doc_id": r["doc_id"] + frame.UNCONDITIONED,
                "request": {"context": "Answer:", "continuation": r["request"]["continuation"]}}
               for r in asked] if unconditioned else []
    return asked + unasked


class ScoredRequestTest(unittest.TestCase):
    def test_takes_the_choice_at_each_questions_label_and_leaves_out_the_ones_without_it(self):
        scored = frame.scored_requests(multiple_choice([2, 0]))
        self.assertEqual([r["request"]["continuation"] for r in scored], [" choice 0-2", " choice 1-0"])

    def test_takes_the_only_request_of_a_question_whose_label_is_not_a_number(self):
        requests = [request(0, 0, "4"), request(1, 0, None), request(2, 0, ["a", "b"])]
        self.assertEqual(frame.scored_requests(requests), requests)

    def test_refuses_requests_without_their_question_in_another_order(self):
        requests = multiple_choice([1])
        requests[3:] = requests[3:][::-1]
        with self.assertRaises(ValueError):
            frame.scored_requests(requests)


class EncodeTest(FrameTest):
    def test_moves_the_contexts_trailing_spaces_to_the_answer_after_the_bos_token(self):
        context, answer = "The cat sat on the", "mat."
        [(tokens, length)] = frame.encode_answers(self.tokenizer_path, [{"context": context + " ", "continuation": answer}])
        whole = self.encode(context + " " + answer)
        before = self.encode(context)
        self.assertEqual(tokens.tolist(), [self.bos, *before, *whole[len(before):]])
        self.assertEqual(length, len(whole) - len(before))

    def test_keeps_one_token_more_than_the_length_from_the_end_of_a_row(self):
        context, answer = TEXTS[0], " " + TEXTS[1]
        [(tokens, length)] = frame.encode_answers(self.tokenizer_path, [{"context": context, "continuation": answer}],
                                                  length=20)
        before, whole = self.encode(context), self.encode(context + answer)
        self.assertEqual(tokens.tolist(), [self.bos, *before, *whole[len(before):]][-21:])
        self.assertEqual(length, len(whole) - len(before))

    def test_refuses_an_answer_longer_than_the_length(self):
        with self.assertRaises(ValueError):
            frame.encode_answers(self.tokenizer_path, [{"context": "A", "continuation": " " + TEXTS[1]}], length=3)

    @unittest.skipUnless(HAS_OLMES, "needs olmes")
    def test_makes_the_rows_that_olmes_wrapper_makes(self):
        import transformers
        from lm_eval.api.model import TemplateLM
        from oe_eval.models.eleuther_huggingface import HFLM_Verbose

        tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=self.tokenizer, bos_token=BOS)
        wrapper = types.SimpleNamespace(tokenizer=tokenizer, add_bos_token=True,
                                        AUTO_MODEL_CLASS=transformers.AutoModelForCausalLM)
        wrapper.tok_encode = functools.partial(HFLM_Verbose.tok_encode, wrapper)
        requests = [{"context": TEXTS[k] + "  ", "continuation": TEXTS[k + 1][:k + 3]} for k in range(10)]
        requests += [{"context": "Question: what?\nAnswer:", "continuation": " café"}]
        for (tokens, length), r in zip(frame.encode_answers(self.tokenizer_path, requests, length=16), requests):
            context, answer = TemplateLM._encode_pair(wrapper, r["context"], r["continuation"])
            self.assertEqual(tokens.tolist(), (context + answer)[-17:])
            self.assertEqual(length, len(answer))


def small_model(vocabulary, block_size=8):
    torch.manual_seed(0)
    return GPT(GPTConfig(block_size=block_size, vocab_size=vocabulary, n_layer=1, n_head=2, n_embd=16, bias=False,
                         rotary=True)).eval()


class WindowTest(unittest.TestCase):
    def test_reads_a_row_within_the_context_whole(self):
        self.assertEqual(frame.answer_windows(6, 2, 8), [(0, 6, 4)])

    def test_reads_a_longer_row_in_windows_that_end_at_its_end_and_step_back_by_half(self):
        self.assertEqual(frame.answer_windows(11, 11, 8), [(3, 11, 7), (0, 7, 0)])
        self.assertEqual(frame.answer_windows(20, 14, 8), [(12, 20, 16), (8, 16, 12), (4, 12, 8), (0, 8, 6)])

    def test_reads_only_the_windows_that_reach_the_answer(self):
        self.assertEqual(frame.answer_windows(11, 4, 8), [(3, 11, 7)])


class LogLikelihoodTest(unittest.TestCase):
    PAD = 0

    def setUp(self):
        self.network = small_model(300)  # a context of 8

    def expected(self, tokens, answer):
        """The answer's log-likelihood, each prediction read in its window alone."""
        total = 0.0
        for start, end, first in frame.answer_windows(len(tokens) - 1, answer, 8):
            row = tokens[None, start:end]
            logits = self.network(row, row)[0][0].log_softmax(-1)
            total += sum(logits[p - start, tokens[p + 1]].item() for p in range(first, end))
        return total

    def test_sums_the_log_softmax_of_the_answers_tokens(self):
        generator = torch.Generator().manual_seed(1)
        rows = [(torch.randint(1, 300, (n,), generator=generator), answer) for n, answer in ((7, 3), (12, 9), (20, 2))]
        totals = frame.answer_log_likelihoods(self.network, rows, self.PAD)
        for (tokens, answer), total in zip(rows, totals):
            self.assertAlmostEqual(total, self.expected(tokens, answer), places=4)

    def test_gives_each_answer_the_same_whatever_rows_share_its_batch(self):
        generator = torch.Generator().manual_seed(2)
        rows = [(torch.randint(1, 300, (n,), generator=generator), 2) for n in (4, 6, 9, 13)]
        together = frame.answer_log_likelihoods(self.network, rows, self.PAD, batch_tokens=64)
        alone = [frame.answer_log_likelihoods(self.network, [row], self.PAD)[0] for row in rows]
        for a, b in zip(together, alone):
            self.assertAlmostEqual(a, b, places=4)


class ScoreTest(FrameTest):
    def write_answers(self, tasks, groups):
        directory = self.directory / "answers"
        directory.mkdir()
        for name, answers in tasks.items():
            (directory / f"{name}.jsonl").write_text("".join(json.dumps(answer) + "\n" for answer in answers))
        (directory / "suites.json").write_text(json.dumps(groups))
        return directory

    def test_scores_a_task_by_the_mean_bits_per_byte_of_its_answers(self):
        answers = [{"doc_id": 0, "context": TEXTS[0], "continuation": " café"},
                   {"doc_id": 1, "context": TEXTS[1], "continuation": " au lait"}]
        directory = self.write_answers({"task": answers}, {})
        network = small_model(300, block_size=64)
        rows = frame.encode_answers(self.tokenizer_path, answers)
        totals = frame.answer_log_likelihoods(network, rows, self.bos)
        bits = [-frame.LOG_2_OF_E * total / len(a["continuation"].encode("utf-8")) for total, a in zip(totals, answers)]
        scores = frame.task_scores(network, self.tokenizer_path, directory)
        self.assertAlmostEqual(scores["task"], sum(bits) / 2, places=6)

    def test_averages_each_suites_members_a_group_counting_as_one(self):
        qa, math_, code = frame.OLMES_SUITES
        groups = {qa: ["a", "group"], "group": ["b", "c", "d"], math_: ["e"], code: ["f", "g"]}
        directory = self.write_answers({}, groups)
        tasks = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 7.0, "e": 0.5, "f": 1.0, "g": 2.0}
        self.assertEqual(frame.suite_scores(tasks, directory), {qa: 2.5, math_: 0.5, code: 1.5})

    def test_reads_the_rows_kept_for_its_tokenizer_and_encodes_them_for_another(self):
        answers = [{"doc_id": k, "context": TEXTS[k], "continuation": " " + TEXTS[k + 1][:k + 2]} for k in range(4)]
        directory = self.write_answers({"task": answers}, {})
        self.assertEqual(frame.make_rows(directory, self.tokenizer_path), 4)
        other = self.directory / "other.json"
        train_tokenizer(iter(TEXTS[::-1]), 280).save(str(other))
        for tokenizer in (self.tokenizer_path, other):
            rows = frame.answer_rows(directory, tokenizer, answers)
            encoded = frame.encode_answers(tokenizer, answers)
            self.assertEqual([(t.tolist(), a) for t, a in rows], [(t.tolist(), a) for t, a in encoded])

    def test_counts_the_tokens_that_the_model_reads_in_its_windows(self):
        answers = [{"doc_id": 0, "context": TEXTS[0], "continuation": " " + TEXTS[1]}]
        directory = self.write_answers({"task": answers}, {})
        [(tokens, length)] = frame.encode_answers(self.tokenizer_path, answers)
        windows = frame.answer_windows(len(tokens) - 1, length, 8)
        counts = frame.count_tokens(directory, self.tokenizer_path, context=8)
        self.assertEqual(counts, {"task": {"answers": 1, "tokens": sum(end - start for start, end, _ in windows)}})


@unittest.skipUnless(HAS_OLMES, "needs olmes")
class OlmesTest(unittest.TestCase):
    def test_scores_a_task_as_olmes_scores_it_from_every_request(self):
        from oe_eval.metrics.metric import MCAccuracy

        rng = random.Random(0)
        for requests, unconditioned in ((multiple_choice([2, 0, 1, 2]), True),
                                        (multiple_choice([1, 0], choices=2, unconditioned=False), False),
                                        ([request(k, 0, "gold", " ü" * (k + 1)) for k in range(3)], False)):
            results = [{**r, "model_resps": {"sum_logits": -rng.uniform(1, 30), "num_tokens": 3}} for r in requests]
            metric = MCAccuracy(**({"uncond_docid_offset": frame.UNCONDITIONED} if unconditioned else {}))
            metric.compute_for_docs([dict(result, model_resps=dict(result["model_resps"])) for result in results])
            expected = metric.aggregate_to_task("bits_per_byte_corr")["primary_score"]
            logits = {id(r): result["model_resps"]["sum_logits"] for r, result in zip(requests, results)}
            bits = [-frame.LOG_2_OF_E * logits[id(r)] / len(r["request"]["continuation"].encode("utf-8"))
                    for r in frame.scored_requests(requests)]
            self.assertAlmostEqual(sum(bits) / len(bits), expected, places=12)

    def test_averages_the_suites_as_olmes_aggregates_them(self):
        from oe_eval.tasks.aggregate_tasks import add_aggregate_tasks

        groups = frame.suite_groups()
        leaves = sorted({m for members in groups.values() for m in members if m not in groups})
        rng = random.Random(0)
        tasks = {name: rng.uniform(0.5, 3.0) for name in leaves}
        metrics = [{"task_name": name, "task_idx": k, "num_instances": rng.randint(10, 1000),
                    "metrics": {"primary_score": score}, "task_config": {"metadata": {"alias": name}},
                    "model_hash": "", "model_config": {}, "compute_config": {}, "processing_time": 0,
                    "current_date": "", "beaker_info": {}} for k, (name, score) in enumerate(tasks.items())]
        aggregated = {entry["task_name"]: entry["metrics"]["primary_score"] for entry in add_aggregate_tasks(metrics)}
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "suites.json").write_text(json.dumps(groups))
            for suite, score in frame.suite_scores(tasks, directory).items():
                self.assertAlmostEqual(score, aggregated[suite], places=12)
        self.assertEqual(len(leaves), 109)

    def test_averages_arc_easy_and_challenge_as_one_member_of_the_qa_suite(self):
        groups = frame.suite_groups()
        self.assertEqual(groups["arc:rc:bpb::olmes:full"],
                         ["arc_easy:rc:bpb::olmes:full", "arc_challenge:rc:bpb::olmes:full"])
        self.assertIn("arc:rc:bpb::olmes:full", groups["olmo3:base_easy:qa_bpb"])


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
        tokens = [token for text in TEXTS for token in [self.bos, *self.encode(text)]]
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
