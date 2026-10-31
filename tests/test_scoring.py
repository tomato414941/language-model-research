import importlib.util
import math
import sys
import unittest
from pathlib import Path

if any(importlib.util.find_spec(name) is None for name in ("torch", "lm_eval")):
    raise unittest.SkipTest("Install the dependencies")

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from compare import compare
from scoring import ContinuationModel, bits_per_byte, choose_questions, summarize


def counting_model(batch_tokens=4096):
    """Each position's logits favor the token one above the one it reads, among eight tokens."""

    def logits(tokens):
        scores = torch.zeros((*tokens.shape, 8))
        scores.scatter_(-1, ((tokens + 1) % 8)[..., None], 3.0)
        return scores

    return ContinuationModel(
        logits, encode=lambda text: [int(c) for c in text], context=6, prefix=0, batch_tokens=batch_tokens
    )


def request(context, continuation):
    return (None, context, continuation)


def probability(hit):
    return math.log((math.exp(3) if hit else 1) / (math.exp(3) + 7))


class ContinuationModelTest(unittest.TestCase):
    def test_sums_the_log_probabilities_of_the_continuation(self):
        [(loglikelihood, greedy)] = counting_model()._loglikelihood_tokens([request([0, 1], [2, 5])])
        self.assertAlmostEqual(loglikelihood, probability(True) + probability(False), places=5)
        self.assertFalse(greedy)

    def test_reports_a_continuation_the_model_would_have_written(self):
        [(_, greedy)] = counting_model()._loglikelihood_tokens([request([0, 1], [2, 3])])
        self.assertTrue(greedy)

    def test_gives_the_same_scores_in_any_batch(self):
        requests = [request([0, 1, 2], [3]), request([0], [1, 2, 3, 7]), request([4, 5], [6])]
        together = counting_model()._loglikelihood_tokens(requests)
        alone = counting_model(batch_tokens=1)._loglikelihood_tokens(requests)
        expected = [probability(True), 3 * probability(True) + probability(False), probability(True)]
        for (first, _), (second, _), value in zip(together, alone, expected):
            self.assertAlmostEqual(first, value, places=5)
            self.assertAlmostEqual(second, value, places=5)

    def test_keeps_the_end_of_a_context_too_long_for_the_model(self):
        [(short, _)] = counting_model()._loglikelihood_tokens([request([1, 2, 3, 4, 5, 6], [7])])
        [(long, _)] = counting_model()._loglikelihood_tokens([request([7, 7, 7, 1, 2, 3, 4, 5, 6], [7])])
        self.assertAlmostEqual(long, short, places=5)

    def test_refuses_a_continuation_longer_than_the_model_can_read(self):
        with self.assertRaisesRegex(ValueError, "fit within the context"):
            counting_model()._loglikelihood_tokens([request([0], [1] * 7)])

    def test_begins_a_string_with_the_prefix_unless_told_not_to(self):
        model = counting_model()
        self.assertEqual(model.tok_encode("12"), [0, 1, 2])
        self.assertEqual(model.tok_encode("12", add_special_tokens=False), [1, 2])
        self.assertEqual(model.tok_encode("01"), [0, 1])


class BitsPerByteTest(unittest.TestCase):
    def test_counts_a_context_s_trailing_spaces_with_the_continuation(self):
        self.assertAlmostEqual(bits_per_byte("question ", "yes", -math.log(2) * 4), 1.0)

    def test_measures_bytes_rather_than_characters(self):
        self.assertAlmostEqual(bits_per_byte("質問", "はい", -math.log(2) * 6), 1.0)


class ChooseQuestionsTest(unittest.TestCase):
    def test_chooses_the_same_questions_every_time(self):
        self.assertEqual(choose_questions({"sciq": 1000}, 200), choose_questions({"sciq": 1000}, 200))

    def test_asks_as_many_questions_as_it_was_told(self):
        self.assertEqual(len(choose_questions({"sciq": 1000}, 200)["sciq"]), 200)

    def test_shares_the_sample_among_a_benchmark_s_tasks_by_their_sizes(self):
        chosen = choose_questions({"large": 900, "small": 100}, 200)
        self.assertEqual({task: len(ids) for task, ids in chosen.items()}, {"large": 180, "small": 20})

    def test_asks_at_least_one_question_of_every_task(self):
        self.assertEqual(len(choose_questions({"large": 100000, "small": 10}, 10)["small"]), 1)

    def test_asks_no_more_questions_than_a_task_holds(self):
        self.assertEqual(choose_questions({"small": 5}, 200)["small"], [0, 1, 2, 3, 4])


class CompareTest(unittest.TestCase):
    def score(self, **bits):
        return {
            "questions": {name: {"bits": values} for name, values in bits.items()},
            "benchmarks": {name: {"language": "en"} for name in bits},
        }

    def test_measures_the_change_on_each_question(self):
        result = compare(self.score(sciq=[1.0, 2.0, 3.0, 4.0]), self.score(sciq=[0.9, 1.9, 2.8, 3.8]))
        self.assertAlmostEqual(result["benchmarks"]["sciq"]["change"], -0.15)

    def test_is_surer_of_a_steady_change(self):
        steady = compare(self.score(sciq=[1.0, 2.0, 3.0, 4.0]), self.score(sciq=[0.9, 1.9, 2.9, 3.9]))
        unsteady = compare(self.score(sciq=[1.0, 2.0, 3.0, 4.0]), self.score(sciq=[0.5, 2.3, 2.6, 4.2]))
        self.assertLess(steady["benchmarks"]["sciq"]["stderr"], unsteady["benchmarks"]["sciq"]["stderr"])

    def test_averages_the_benchmarks_of_a_language(self):
        result = compare(self.score(sciq=[1.0, 1.0], piqa=[2.0, 2.0]), self.score(sciq=[0.9, 0.9], piqa=[2.3, 2.3]))
        self.assertAlmostEqual(result["languages"]["en"]["change"], 0.1)

    def test_refuses_scores_of_different_questions(self):
        with self.assertRaisesRegex(ValueError, "same questions"):
            compare(self.score(sciq=[1.0, 2.0]), self.score(sciq=[1.0]))


class SummaryTest(unittest.TestCase):
    def test_averages_questions_within_a_benchmark_and_benchmarks_within_a_language(self):
        summary = summarize(
            {
                "arc_easy": [{"accuracy": 1.0, "bits": 1.0}, {"accuracy": 0.0, "bits": 3.0}],
                "sciq": [{"accuracy": 1.0, "bits": 2.0}],
                "ja_leaderboard_jnli": [{"accuracy": 0.0, "bits": 4.0}],
            }
        )
        self.assertEqual(
            summary["benchmarks"]["arc_easy"],
            {"language": "en", "metric": "acc_norm", "items": 2, "accuracy": 0.5, "gold_bits_per_byte": 2.0},
        )
        self.assertEqual(
            summary["languages"],
            {
                "en": {"accuracy": 0.75, "gold_bits_per_byte": 2.0},
                "ja": {"accuracy": 0.0, "gold_bits_per_byte": 4.0},
            },
        )

    def test_keeps_each_question_s_result(self):
        summary = summarize({"sciq": [{"accuracy": 1.0, "bits": 1.5}, {"accuracy": 0.0, "bits": 2.5}]})
        self.assertEqual(summary["questions"]["sciq"], {"correct": [1.0, 0.0], "bits": [1.5, 2.5]})


if __name__ == "__main__":
    unittest.main()
