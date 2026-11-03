import importlib.util
import sys
import unittest
from pathlib import Path

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("Install the dependencies")

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from averaging import WeightAverage, averaged_updates


class AveragedUpdatesTest(unittest.TestCase):
    def test_keeps_the_weights_of_updates_evenly_apart_up_to_the_last(self):
        self.assertEqual(averaged_updates(1281, 8, 32), {1281, 1249, 1217, 1185, 1153, 1121, 1089, 1057})

    def test_refuses_weights_from_before_the_first_update(self):
        with self.assertRaises(ValueError):
            averaged_updates(25, 8, 32)


def states(dtype=torch.float32):
    """Three states of one run, whose weights go 1, 2, 6 and whose count goes 0, 1, 2."""
    return [{"weight": torch.full((2, 2), value, dtype=dtype), "count": torch.tensor(index)}
            for index, value in enumerate((1.0, 2.0, 6.0))]


def averaged(given):
    average = WeightAverage(["weight"])
    for state in given:
        average.add(state)
    return average


class WeightAverageTest(unittest.TestCase):
    def test_mixes_the_last_weights_with_the_average_of_all_the_weights(self):
        mixed = averaged(states()).mixed(states()[-1], 0.55)
        # The average of 1, 2 and 6 is 3.
        self.assertTrue(torch.allclose(mixed["weight"], torch.full((2, 2), 0.45 * 6 + 0.55 * 3)))

    def test_gives_the_last_weights_when_the_average_has_no_share(self):
        mixed = averaged(states()).mixed(states()[-1], 0.0)
        self.assertTrue(torch.equal(mixed["weight"], states()[-1]["weight"]))

    def test_takes_what_it_does_not_average_from_the_last_state(self):
        mixed = averaged(states()).mixed(states()[-1], 0.55)
        self.assertEqual(mixed["count"].item(), 2)

    def test_keeps_each_tensor_in_its_own_type(self):
        mixed = averaged(states(torch.bfloat16)).mixed(states(torch.bfloat16)[-1], 0.55)
        self.assertEqual(mixed["weight"].dtype, torch.bfloat16)
        self.assertTrue(torch.allclose(mixed["weight"].float(), torch.full((2, 2), 4.35), atol=0.02))

    def test_keeps_the_weights_as_they_were_when_given(self):
        state = {"weight": torch.ones(2)}
        average = averaged([state])
        # Training goes on changing its weights in place.
        state["weight"].add_(10.0)
        self.assertTrue(torch.equal(average.mixed({"weight": torch.zeros(2)}, 1.0)["weight"], torch.ones(2)))


if __name__ == "__main__":
    unittest.main()
