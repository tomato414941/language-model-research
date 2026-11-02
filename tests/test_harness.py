import importlib.util
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

if importlib.util.find_spec("lm_eval") is None:
    raise unittest.SkipTest("Install the dependencies")

sys.path.insert(0, str(Path(__file__).parent.parent))

from harness import score_in_full


class ChoosingTask:
    """A task whose documents name their correct choice."""

    multiple_input = False

    def doc_to_target(self, doc):
        return doc["correct"]


def sample(correct, loglikelihoods, accuracy):
    return {
        "doc": {"correct": correct},
        "arguments": [("question ", "yes"), ("question ", "no")],
        "filtered_resps": [(value, False) for value in loglikelihoods],
        "acc": accuracy,
    }


class ScoreInFullTest(unittest.TestCase):
    def score(self):
        bit = -math.log(2)
        samples = [sample(0, (4 * bit, 9 * bit), 1.0), sample(1, (bit, 6 * bit), 0.0)]
        asked = [("sciq", "sciq", ChoosingTask(), row) for row in samples]
        with patch("harness.ask", return_value=asked):
            return score_in_full(SimpleNamespace(device="cpu"), {"sciq": ("en", "acc")})

    def test_keeps_whether_each_question_was_answered_correctly(self):
        self.assertEqual(self.score()["questions"]["sciq"]["correct"], [1.0, 0.0])

    def test_keeps_the_bits_per_byte_of_each_question_s_correct_choice(self):
        # " yes" is four bytes and " no" three, and the context's trailing space counts with them.
        self.assertEqual(self.score()["questions"]["sciq"]["bits"], [1.0, 2.0])

    def test_averages_over_the_questions_of_a_benchmark(self):
        benchmark = self.score()["benchmarks"]["sciq"]
        self.assertEqual(benchmark["questions"], 2)
        self.assertEqual(benchmark["accuracy"], 0.5)
        self.assertEqual(benchmark["gold_bits_per_byte"], 1.5)


if __name__ == "__main__":
    unittest.main()
