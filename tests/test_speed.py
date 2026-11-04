import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from speed import (PARTS_IN_DOUBT, attention_used, bundle_way, cheapest, cost, same_way, seconds_per_update,
                   stream_in_bfloat16, training_options, updates_for_the_cost)


def log_of(*milliseconds):
    return "\n".join(f"iter {i}: loss 3.1000, time {seconds:.2f}ms, mfu 39.00%" for i, seconds in enumerate(milliseconds))


def costing(seconds, startup=30.0):
    return {"seconds_per_update": seconds, "startup_seconds": startup}


class SpeedTest(unittest.TestCase):
    def test_reads_the_same_tokens_per_update_in_every_way_it_tries(self):
        ways = [bundle_way("sdpa"), bundle_way("fa3", 32, 16), bundle_way("fa3", 64, 8), bundle_way("fa3", 256, 2),
                bundle_way("fa3", 32, 8, block_size=2048), bundle_way("fa2", stream=True),
                *[bundle_way("fa2", part=part) for part in PARTS_IN_DOUBT]]
        for way in ways:
            options = way["options"]
            self.assertEqual(options["batch_size"] * options["gradient_accumulation_steps"] * options["block_size"], 524288)

    def test_gives_each_way_its_own_name(self):
        ways = (bundle_way("fa3"), bundle_way("fa4"), bundle_way("fa3", 64, 8), bundle_way("fa3", compile_model=False),
                bundle_way("fa3", block_size=2048), bundle_way("fa3", stream=True),
                *[bundle_way("fa3", part=part) for part in PARTS_IN_DOUBT])
        self.assertEqual(len({way["name"] for way in ways}), 6 + len(PARTS_IN_DOUBT))

    def test_takes_the_median_after_the_settling_updates(self):
        log = log_of(30000, 900, 900, 900, 900, 500, 600, 700, 800, 900)
        self.assertAlmostEqual(seconds_per_update(log), 0.7)

    def test_ignores_the_other_lines_of_a_log(self):
        log = "step 0: train loss 9.0, val loss 9.0\n" + log_of(*[500] * 10) + "\nsaving checkpoint"
        self.assertAlmostEqual(seconds_per_update(log), 0.5)

    def test_gives_no_time_for_a_run_that_stopped_early(self):
        self.assertIsNone(seconds_per_update(log_of(30000, 900)))
        self.assertIsNone(seconds_per_update("CUDA out of memory"))

    def test_reads_which_attention_ran(self):
        self.assertEqual(attention_used("number of parameters: 1M\nattention: fa3\ncompiling the model"), "fa3")

    def test_counts_the_time_around_the_updates(self):
        measured = cost("attention: fa3\n" + log_of(*[500] * 10), total_seconds=45.0, updates=10, attention="fa3")
        self.assertAlmostEqual(measured["seconds_per_update"], 0.5)
        self.assertAlmostEqual(measured["startup_seconds"], 40.0)

    def test_gives_no_cost_to_a_run_that_fell_back_to_another_attention(self):
        self.assertIsNone(cost("attention: sdpa\n" + log_of(*[500] * 10), total_seconds=45.0, updates=10, attention="fa3"))

    def test_chooses_the_cheapest_way_that_worked(self):
        ways = [bundle_way("sdpa"), bundle_way("fa3"), bundle_way("fa4")]
        costs = {"baseline": costing(0.8), "sdpa-128x4": costing(0.6), "fa3-128x4": None, "fa4-128x4": costing(0.5)}
        self.assertEqual(cheapest(ways, costs, 954)["name"], "fa4-128x4")

    def test_chooses_the_stream_in_bfloat16_when_it_is_cheaper(self):
        ways = [bundle_way("fa2"), bundle_way("fa2", stream=True)]
        costs = {"baseline": costing(0.8), "fa2-128x4": costing(0.6), "fa2-128x4-stream": costing(0.55)}
        self.assertEqual(cheapest(ways, costs, 954)["name"], "fa2-128x4-stream")

    def test_keeps_the_values_as_computed_when_the_stream_in_bfloat16_is_not_cheaper(self):
        ways = [bundle_way("fa2"), bundle_way("fa2", stream=True)]
        costs = {"baseline": costing(0.8), "fa2-128x4": costing(0.55), "fa2-128x4-stream": costing(0.6)}
        self.assertEqual(cheapest(ways, costs, 954)["name"], "fa2-128x4")

    def test_does_not_choose_the_bundle_without_one_of_its_parts(self):
        part = bundle_way("fa2", part="without-count")
        costs = {"baseline": costing(0.8), "fa2-128x4": costing(0.6), part["name"]: costing(0.5)}
        self.assertEqual(cheapest([bundle_way("fa2"), part], costs, 954)["name"], "fa2-128x4")

    def test_turns_each_part_off_in_its_run(self):
        self.assertEqual(bundle_way("fa2", part="without-count")["options"]["count_table"], "")
        self.assertEqual(bundle_way("fa2", part="without-dispersion")["options"]["dispersion"], 0.0)

    def test_reads_whether_the_stream_ran_in_bfloat16(self):
        self.assertTrue(stream_in_bfloat16("attention: fa2\nstream: torch.bfloat16\n"))
        self.assertFalse(stream_in_bfloat16("attention: fa2\nstream: as computed\n"))
        self.assertFalse(stream_in_bfloat16("attention: fa2\n"))

    def test_gives_no_cost_to_a_run_that_did_not_keep_the_stream_as_asked(self):
        log = "attention: fa2\nstream: as computed\n" + log_of(*[500] * 10)
        self.assertIsNone(cost(log, total_seconds=45.0, updates=10, attention="fa2", stream=True))
        self.assertIsNotNone(cost(log, total_seconds=45.0, updates=10, attention="fa2", stream=False))

    def test_chooses_the_shorter_startup_between_equal_updates(self):
        ways = [bundle_way("sdpa"), bundle_way("fa2")]
        costs = {"baseline": costing(0.8), "sdpa-128x4": costing(0.557, 34.0), "fa2-128x4": costing(0.558, 11.0)}
        self.assertEqual(cheapest(ways, costs, 954)["name"], "fa2-128x4")

    def test_keeps_the_faster_update_over_a_short_startup_when_the_update_counts_for_more(self):
        ways = [bundle_way("sdpa"), bundle_way("sdpa", compile_model=False)]
        costs = {"baseline": costing(0.8), "sdpa-128x4": costing(0.557, 34.0), "sdpa-128x4-uncompiled": costing(0.733, 5.0)}
        self.assertEqual(cheapest(ways, costs, 954)["name"], "sdpa-128x4")

    def test_does_not_choose_a_way_that_changes_the_model(self):
        longer = bundle_way("sdpa", 32, 8, block_size=2048)
        costs = {"baseline": costing(0.8), "sdpa-128x4": costing(0.6), longer["name"]: costing(0.3)}
        self.assertEqual(cheapest([bundle_way("sdpa"), longer], costs, 954)["name"], "sdpa-128x4")

    def test_chooses_nothing_when_no_way_worked(self):
        self.assertIsNone(cheapest([bundle_way("sdpa")], {"baseline": costing(0.8), "sdpa-128x4": None}, 954))

    def test_buys_more_updates_with_a_faster_update(self):
        self.assertEqual(updates_for_the_cost(954, costing(0.8), costing(0.5)), 1526)
        self.assertEqual(updates_for_the_cost(954, costing(0.8), costing(0.8)), 954)

    def test_takes_a_longer_startup_out_of_the_budget(self):
        self.assertEqual(updates_for_the_cost(954, costing(0.8, 30.0), costing(0.5, 60.0)), 1466)

    def test_runs_another_configuration_in_the_chosen_way_under_its_own_name(self):
        chosen = bundle_way("fa2", 128, 4, stream=True)
        other = same_way(chosen, "configs/bundle2.py")
        self.assertEqual((other["config"], other["options"]), ("configs/bundle2.py", chosen["options"]))
        self.assertNotEqual(other["name"], chosen["name"])

    def test_gives_options_that_train_for_those_updates_in_that_way(self):
        options = training_options(bundle_way("fa3", 128, 4), 1700)
        for option in ("--attention=fa3", "--batch_size=128", "--gradient_accumulation_steps=4", "--compile=True",
                       "--max_iters=1700", "--lr_decay_iters=1700", "--eval_interval=1700"):
            self.assertIn(option, options)


if __name__ == "__main__":
    unittest.main()
