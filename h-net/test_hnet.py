"""Checks of the H-Net model that run without a GPU, with its causal convolution in place of the outer layers.

Run from the repository's root: python -m unittest discover -s h-net
"""

import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "h-net")]

from hnet_model import BOS, HNet, HNetConfig, byte_language_model, causal_within_rows, smooth  # noqa: E402
from muon import configure_muon  # noqa: E402


def small_model(**fields):
    torch.manual_seed(0)
    config = HNetConfig(block_size=64, vocab_size=BOS + 1, n_layer=2, n_head=2, n_embd=32, bias=False, rotary=True,
                        rms_norm=True, qk_norm=True, relu_squared=True, zero_output=False, separate_embeddings=True,
                        residual_weights=True, logit_softcap=15.0, outer="conv", outer_width=24, encoder_layers=2,
                        decoder_layers=2, compile_main=False, **fields)
    return HNet(config).eval()


class HNetTest(unittest.TestCase):
    def test_predicts_each_byte_from_the_bytes_before_it_only(self):
        model = small_model()
        first = torch.randint(0, 256, (3, 40))
        second = first.clone()
        second[:, 25:] = torch.randint(0, 256, (3, 15))
        with torch.no_grad():
            before, _ = model(first, first)
            after, _ = model(second, second)
        torch.testing.assert_close(before[:, :25], after[:, :25])

    def test_smooths_each_chunk_with_the_ones_before_it_in_its_row(self):
        torch.manual_seed(1)
        count = 150
        values, probability = torch.randn(count, 5), torch.rand(count)
        fresh = torch.zeros(count, dtype=torch.bool)
        fresh[[0, 37, 64, 65, 140]] = True
        expected, state = [], torch.zeros(5)
        for value, p, starts in zip(values, probability.clamp(1e-4, 1 - 1e-4), fresh):
            state = p * value + (0 if starts else (1 - p) * state)
            expected.append(state)
        torch.testing.assert_close(smooth(values, probability, fresh), torch.stack(expected), rtol=1e-5, atol=1e-5)

    def test_attends_within_each_packed_row(self):
        torch.manual_seed(2)
        lengths = [5, 1, 9]
        q, k, v = (torch.randn(sum(lengths), 2, 8) for _ in range(3))
        starts = torch.tensor([0, 5, 6, 15], dtype=torch.int32)
        packed = causal_within_rows(q, k, v, starts, max(lengths))
        for begin, end in zip(starts[:-1].tolist(), starts[1:].tolist()):
            row = [t[begin:end].transpose(0, 1)[None] for t in (q, k, v)]
            alone = F.scaled_dot_product_attention(*row, is_causal=True)[0].transpose(0, 1)
            torch.testing.assert_close(packed[begin:end], alone)

    def test_begins_a_chunk_at_every_fixed_number_of_bytes_when_asked(self):
        model = small_model(fixed_chunks=4).train()
        tokens = torch.randint(0, 256, (2, 30))
        with torch.no_grad():
            model(tokens, tokens)
        self.assertAlmostEqual(model.bytes_per_chunk, 30 / 8, places=5)

    def test_raises_the_ratio_weight_in_training_up_to_one_while_chunks_are_finer_than_the_target(self):
        model = small_model(fixed_chunks=2, ratio_weight_rate=100.0).train()
        tokens = torch.randint(0, 256, (2, 30))
        weights = []
        with torch.no_grad():
            for _ in range(20):
                model(tokens, tokens)
                weights.append(model.ratio_log_weight.exp().item())
        self.assertGreater(weights[0], 0.03)
        self.assertGreater(weights[1], weights[0])
        self.assertAlmostEqual(weights[-1], 1.0, places=6)

    def test_lowers_the_ratio_weight_to_its_start_while_chunks_are_coarser_than_the_target(self):
        model = small_model(fixed_chunks=12, ratio_weight_rate=100.0).train()
        model.ratio_log_weight.fill_(-1.0)
        tokens = torch.randint(0, 256, (2, 30))
        with torch.no_grad():
            model(tokens, tokens)
            lowered = model.ratio_log_weight.exp().item()
            for _ in range(40):
                model(tokens, tokens)
        self.assertLess(lowered, torch.tensor(-1.0).exp().item())
        self.assertAlmostEqual(model.ratio_log_weight.exp().item(), 0.03, places=6)

    def test_keeps_the_ratio_weight_outside_training(self):
        model = small_model(fixed_chunks=2, ratio_weight_rate=100.0).eval()
        tokens = torch.randint(0, 256, (2, 30))
        with torch.no_grad():
            model(tokens, tokens)
        self.assertAlmostEqual(model.ratio_log_weight.exp().item(), 0.03, places=6)

    def test_gives_the_main_networks_matrices_to_muon_and_the_byte_table_its_own_rate(self):
        model = small_model()
        optimizer = configure_muon(model, 0.04, 0.0044, embedding_lr=0.66)
        by_id = {id(p): group for group in optimizer.param_groups for p in group["params"]}
        names = dict(model.named_parameters())
        self.assertEqual(by_id[id(names["transformer.h.0.attn.c_attn.weight"])]["kind"], "muon")
        self.assertEqual(by_id[id(names["transformer.h.1.mlp.c_fc.weight"])]["kind"], "muon")
        self.assertEqual(by_id[id(names["transformer.wte.weight"])]["lr"], 0.66)
        self.assertEqual(by_id[id(names["encoder.0.mixer.in_proj.weight"])]["kind"], "adamw")

    def test_reads_text_as_its_utf8_bytes_after_bos(self):
        language_model = byte_language_model(small_model(), "cpu", 4096)
        self.assertEqual(language_model.encode("aあ"), [97, 227, 129, 130])
        self.assertEqual(language_model.prefix, BOS)
        self.assertEqual(language_model.context, 64)

    def test_learns_to_repeat_a_pattern(self):
        model = small_model().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
        pattern = torch.tensor(list(b"hello world, ") * 5)[:64][None].repeat(4, 1)
        losses = []
        for _ in range(60):
            _, loss = model(pattern[:, :-1], pattern[:, 1:])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        self.assertLess(losses[-1], losses[0] / 2)

    def test_imports_a_package_s_module_without_the_package_s_own_imports(self):
        import importlib
        import sys
        import tempfile
        from pathlib import Path

        from hnet_model import without_package_imports

        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "brokenpackage"
            package.mkdir()
            (package / "__init__.py").write_text("raise ImportError('a kernel this machine lacks')\n")
            (package / "part.py").write_text("VALUE = 7\n")
            sys.path.insert(0, directory)
            try:
                without_package_imports("brokenpackage")
                self.assertEqual(importlib.import_module("brokenpackage.part").VALUE, 7)
            finally:
                sys.path.remove(directory)
                for name in [name for name in sys.modules if name.startswith("brokenpackage")]:
                    del sys.modules[name]

if __name__ == "__main__":
    unittest.main()
