import importlib.util
import sys
import unittest
from pathlib import Path

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("Install the dependencies")

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from model import GPT, GPTConfig, has_value_embedding
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


if __name__ == "__main__":
    unittest.main()
