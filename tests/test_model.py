import importlib.util
import sys
import unittest
import unittest.mock
from pathlib import Path

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("Install the dependencies")

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from attention_check import agree
from model import GPT, GPTConfig, choose_attention, has_value_embedding, load_flash_attention
from muon import configure_muon


def small_model(layers=3, value_embeddings=True):
    torch.manual_seed(0)
    return GPT(GPTConfig(block_size=16, vocab_size=32, n_layer=layers, n_head=2, n_embd=32, bias=False,
                         value_embeddings=value_embeddings))


def tokens():
    return torch.randint(0, 32, (4, 17), generator=torch.Generator().manual_seed(1))


def all_logits(model, inputs):
    # A target makes the model return the logits of every position.
    logits, _ = model(inputs, inputs.clone())
    return logits


class ValueEmbeddingsTest(unittest.TestCase):
    def test_puts_tables_on_every_other_layer_and_always_on_the_last(self):
        self.assertEqual([has_value_embedding(layer, 5) for layer in range(5)], [True, False, True, False, True])
        self.assertEqual([has_value_embedding(layer, 4) for layer in range(4)], [False, True, False, True])
        self.assertEqual(sorted(small_model(layers=5).value_embeds), ['0', '2', '4'])

    def test_gives_each_table_a_row_per_token_as_wide_as_the_model(self):
        table = small_model().value_embeds['2']
        self.assertEqual(table.weight.shape, (32, 32))

    def test_adds_no_weights_when_turned_off(self):
        names = small_model(value_embeddings=False).state_dict()
        self.assertFalse([name for name in names if 'value' in name])

    def test_changes_nothing_else_about_the_model_when_turned_off(self):
        with_tables = small_model()
        without = small_model(value_embeddings=False)
        without.load_state_dict({k: v for k, v in with_tables.state_dict().items() if k in without.state_dict()})
        for table in with_tables.value_embeds.values():
            torch.nn.init.zeros_(table.weight)
        inputs = tokens()[:, :-1]
        self.assertTrue(torch.allclose(all_logits(with_tables, inputs), all_logits(without, inputs)))

    def test_affects_the_predictions(self):
        model = small_model()
        inputs = tokens()[:, :-1]
        before = all_logits(model, inputs).detach().clone()
        torch.nn.init.normal_(model.value_embeds['2'].weight)
        self.assertFalse(torch.allclose(all_logits(model, inputs), before))

    def test_keeps_earlier_positions_blind_to_later_tokens(self):
        model = small_model()
        inputs = tokens()[:, :-1]
        changed = inputs.clone()
        changed[:, -1] = (changed[:, -1] + 1) % 32
        self.assertTrue(torch.allclose(all_logits(model, inputs)[:, :-1], all_logits(model, changed)[:, :-1]))

    def test_leaves_the_tables_out_of_the_count_used_for_compute(self):
        model = small_model()
        self.assertEqual(model.get_num_params(), small_model(value_embeddings=False).get_num_params()
                         + sum(p.numel() for name, p in model.named_parameters() if 'value_gate' in name))

    def test_loads_the_weights_of_a_model_saved_before_tables_existed(self):
        saved = small_model(value_embeddings=False).state_dict()
        arguments = dict(block_size=16, vocab_size=32, n_layer=3, n_head=2, n_embd=32, bias=False)
        GPT(GPTConfig(**arguments)).load_state_dict(saved)

    def test_learns_a_fixed_batch_with_muon(self):
        model = small_model()
        optimizer = configure_muon(model, muon_lr=0.04, other_lr=0.004)
        batch = tokens()
        losses = []
        for _ in range(30):
            _, loss = model(batch[:, :-1], batch[:, 1:].contiguous())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(loss.item())
        self.assertLess(losses[-1], losses[0] * 0.8)

    def test_gives_tables_to_adamw_and_gates_to_muon(self):
        model = small_model()
        names = {id(p): name for name, p in model.named_parameters()}
        groups = configure_muon(model, muon_lr=0.04, other_lr=0.004, value_embedding_lr=0.1).param_groups
        tables = groups[-1]
        self.assertEqual((tables['kind'], tables['lr']), ('adamw', 0.1))
        self.assertTrue(all(names[id(p)].startswith('value_embeds.') for p in tables['params']))
        muon = {names[id(p)] for group in groups if group['kind'] == 'muon' for p in group['params']}
        self.assertIn('transformer.h.2.attn.value_gate.weight', muon)


def attention_with_heads_after_time(q, k, v, causal):
    """PyTorch's attention in the shape and call of FlashAttention-3, which only runs on a GPU."""
    swapped = (tensor.transpose(1, 2) for tensor in (q, k, v))
    return torch.nn.functional.scaled_dot_product_attention(*swapped, is_causal=causal).transpose(1, 2)


class FlashAttentionTest(unittest.TestCase):
    def run_on_bf16(self, model):
        with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
            return all_logits(model, tokens()[:, :-1]).float()

    def test_predicts_what_pytorchs_attention_predicts_when_the_function_does_the_same(self):
        model = small_model()
        before = self.run_on_bf16(model)
        model.use_flash_attention(attention_with_heads_after_time)
        self.assertTrue(torch.allclose(self.run_on_bf16(model), before, atol=1e-2))

    def test_calls_the_function_in_its_shape_and_precision_with_a_causal_mask(self):
        calls = []

        def record(q, k, v, causal):
            calls.append((q.shape, k.shape, v.shape, q.dtype, k.dtype, v.dtype, causal))
            return attention_with_heads_after_time(q, k, v, causal)

        model = small_model(layers=3)
        model.use_flash_attention(record)
        self.run_on_bf16(model)
        self.assertEqual(len(calls), 3)
        for call in calls:
            self.assertEqual(call, ((4, 16, 2, 16),) * 3 + (torch.bfloat16,) * 3 + (True,))

    def test_keeps_the_saved_weights_the_same(self):
        model = small_model()
        names = list(model.state_dict())
        model.use_flash_attention(attention_with_heads_after_time)
        self.assertEqual(list(model.state_dict()), names)

    def test_loads_nothing_where_there_is_no_gpu(self):
        with unittest.mock.patch("torch.cuda.is_available", return_value=False):
            self.assertIsNone(load_flash_attention())

    def test_loads_nothing_on_a_gpu_newer_than_its_kernels(self):
        with unittest.mock.patch("torch.cuda.is_available", return_value=True), \
                unittest.mock.patch("torch.cuda.get_device_capability", return_value=(10, 0)):
            self.assertIsNone(load_flash_attention())

    def test_loads_nothing_when_the_kernel_cannot_be_fetched(self):
        with unittest.mock.patch("torch.cuda.is_available", return_value=True), \
                unittest.mock.patch("torch.cuda.get_device_capability", return_value=(9, 0)), \
                unittest.mock.patch.dict(sys.modules, {"kernels": None}):
            self.assertIsNone(load_flash_attention())


class ChooseAttentionTest(unittest.TestCase):
    def test_says_pytorch_chooses_when_flash_attention_3_cannot_load(self):
        model = small_model()
        with unittest.mock.patch("model.load_flash_attention", return_value=None):
            self.assertEqual(choose_attention(model, "fa3"), "sdpa")
        self.assertTrue(all(block.attn.flash_attention is None for block in model.transformer.h))

    def test_sends_attention_through_flash_attention_3_when_it_loads(self):
        model = small_model()
        with unittest.mock.patch("model.load_flash_attention", return_value=attention_with_heads_after_time):
            self.assertEqual(choose_attention(model, "fa3"), "fa3")
        self.assertTrue(all(block.attn.flash_attention is attention_with_heads_after_time for block in model.transformer.h))

    def test_says_pytorch_chooses_when_flash_attention_4_cannot_load(self):
        with unittest.mock.patch("torch.nn.attention.activate_flash_attention_impl", side_effect=ImportError("absent")), \
                unittest.mock.patch("torch.backends.cuda.enable_cudnn_sdp") as turn_off:
            self.assertEqual(choose_attention(small_model(), "fa4"), "sdpa")
        turn_off.assert_not_called()

    def test_turns_cudnn_off_for_the_flash_attention_behind_pytorchs_call(self):
        for attention in ("fa2", "fa4"):
            with unittest.mock.patch("torch.nn.attention.activate_flash_attention_impl"), \
                    unittest.mock.patch("torch.backends.cuda.enable_cudnn_sdp") as turn_off:
                self.assertEqual(choose_attention(small_model(), attention), attention)
            turn_off.assert_called_once_with(False)

    def test_changes_nothing_when_pytorch_chooses(self):
        with unittest.mock.patch("torch.backends.cuda.enable_cudnn_sdp") as turn_off:
            self.assertEqual(choose_attention(small_model(), "sdpa"), "sdpa")
        turn_off.assert_not_called()


class AttentionCheckTest(unittest.TestCase):
    def outputs(self, logits=0.0, gradient=(1.0, 2.0, 3.0)):
        return torch.full((2, 3), logits), {"weight": torch.tensor(gradient)}

    def test_accepts_outputs_that_differ_by_rounding(self):
        self.assertTrue(agree(self.outputs(), self.outputs(logits=0.01, gradient=(1.0, 2.001, 3.0))))

    def test_rejects_different_predictions(self):
        self.assertFalse(agree(self.outputs(), self.outputs(logits=0.5)))

    def test_rejects_different_gradients(self):
        self.assertFalse(agree(self.outputs(), self.outputs(gradient=(3.0, -2.0, 1.0))))


if __name__ == "__main__":
    unittest.main()
