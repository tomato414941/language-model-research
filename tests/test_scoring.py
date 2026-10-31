import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("Install the dependencies")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from compare import compare
from loading import LanguageModel, encode_pair, loglikelihoods
from scoring import bits_per_byte, score_questions, summarize, validation_bits


def counting_model(batch_tokens=4096, context=6):
    """Each position's logits favor the token one above the one it reads, among eight tokens."""

    def logits(tokens):
        scores = torch.zeros((*tokens.shape, 8))
        scores.scatter_(-1, ((tokens + 1) % 8)[..., None], 3.0)
        return scores

    return LanguageModel(
        logits, encode=lambda text: [int(c) for c in text], context=context, prefix=0, batch_tokens=batch_tokens
    )


def probability(hit):
    return math.log((math.exp(3) if hit else 1) / (math.exp(3) + 7))


class LoglikelihoodsTest(unittest.TestCase):
    def test_sums_the_log_probabilities_of_the_continuation(self):
        [(loglikelihood, greedy)] = loglikelihoods(counting_model(), [([0, 1], [2, 5])])
        self.assertAlmostEqual(loglikelihood, probability(True) + probability(False), places=5)
        self.assertFalse(greedy)

    def test_reports_a_continuation_the_model_would_have_written(self):
        [(_, greedy)] = loglikelihoods(counting_model(), [([0, 1], [2, 3])])
        self.assertTrue(greedy)

    def test_gives_the_same_scores_in_any_batch(self):
        requests = [([0, 1, 2], [3]), ([0], [1, 2, 3, 7]), ([4, 5], [6])]
        together = loglikelihoods(counting_model(), requests)
        alone = loglikelihoods(counting_model(batch_tokens=1), requests)
        expected = [probability(True), 3 * probability(True) + probability(False), probability(True)]
        for (first, _), (second, _), value in zip(together, alone, expected):
            self.assertAlmostEqual(first, value, places=5)
            self.assertAlmostEqual(second, value, places=5)

    def test_keeps_the_end_of_a_context_too_long_for_the_model(self):
        [(short, _)] = loglikelihoods(counting_model(), [([1, 2, 3, 4, 5, 6], [7])])
        [(long, _)] = loglikelihoods(counting_model(), [([7, 7, 7, 1, 2, 3, 4, 5, 6], [7])])
        self.assertAlmostEqual(long, short, places=5)

    def test_refuses_a_continuation_longer_than_the_model_can_read(self):
        with self.assertRaisesRegex(ValueError, "fit within the context"):
            loglikelihoods(counting_model(), [([0], [1] * 7)])


class EncodePairTest(unittest.TestCase):
    def test_begins_the_context_with_the_prefix(self):
        self.assertEqual(encode_pair(counting_model(), "12", "34"), ([0, 1, 2], [3, 4]))

    def test_gives_a_context_s_trailing_spaces_to_the_continuation(self):
        spaces = LanguageModel(None, encode=lambda text: [ord(c) for c in text], context=6, prefix=0)
        self.assertEqual(encode_pair(spaces, "a ", "b"), ([0, ord("a")], [ord(" "), ord("b")]))


class BitsPerByteTest(unittest.TestCase):
    def test_counts_a_context_s_trailing_spaces_with_the_continuation(self):
        self.assertAlmostEqual(bits_per_byte("question ", "yes", -math.log(2) * 4), 1.0)

    def test_measures_bytes_rather_than_characters(self):
        self.assertAlmostEqual(bits_per_byte("質問", "はい", -math.log(2) * 6), 1.0)


class ScoreTest(unittest.TestCase):
    def test_scores_the_correct_answer_of_each_question(self):
        questions = [
            {"benchmark": "sciq", "language": "en", "context": "12", "answer": "3"},
            {"benchmark": "sciq", "language": "en", "context": "12", "answer": "5"},
        ]
        result = score_questions(counting_model(), questions)
        expected = [-probability(True) / math.log(2), -probability(False) / math.log(2)]
        for value, wanted in zip(result["questions"]["sciq"]["bits"], expected):
            self.assertAlmostEqual(value, wanted, places=4)

    def test_averages_questions_within_a_benchmark_and_benchmarks_within_a_language(self):
        questions = [
            {"benchmark": "arc_easy", "language": "en"},
            {"benchmark": "arc_easy", "language": "en"},
            {"benchmark": "sciq", "language": "en"},
            {"benchmark": "ja_leaderboard_jnli", "language": "ja"},
        ]
        summary = summarize(questions, [1.0, 3.0, 2.0, 4.0])
        self.assertEqual(
            summary["benchmarks"]["arc_easy"], {"language": "en", "questions": 2, "gold_bits_per_byte": 2.0}
        )
        self.assertEqual(
            summary["languages"], {"en": {"gold_bits_per_byte": 2.0}, "ja": {"gold_bits_per_byte": 4.0}}
        )

    def test_keeps_each_question_s_result(self):
        summary = summarize([{"benchmark": "sciq", "language": "en"}] * 2, [1.5, 2.5])
        self.assertEqual(summary["questions"]["sciq"], {"bits": [1.5, 2.5]})


class ValidationBitsTest(unittest.TestCase):
    def test_reads_the_text_in_rows_and_divides_by_its_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "val.bin"
            # The model writes each of these tokens itself, and each stands for one byte here.
            np.arange(1, 8, dtype=np.uint16).tofile(path)
            bits = validation_bits(counting_model(context=3), path, lambda tokens: "x" * len(tokens))
        self.assertAlmostEqual(bits, -probability(True) / math.log(2), places=5)


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


if __name__ == "__main__":
    unittest.main()
