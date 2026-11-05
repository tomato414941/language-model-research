import importlib.util
import sys
import unittest
import unittest.mock
from pathlib import Path

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("Install the dependencies")

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from model import (GPT, MLP, XIELU, GPTConfig, bigram_rows, document_ids, has_stem_table, segment_bounds,
                   trigram_rows)
from muon import configure_muon

BOS = 31


def model(**options):
    torch.manual_seed(0)
    settings = dict(block_size=32, vocab_size=32, n_layer=4, n_head=2, n_embd=32, bias=False, rotary=True,
                    rms_norm=True, qk_norm=True)
    settings.update(options)
    return GPT(GPTConfig(**settings))


def all_logits(network, inputs):
    # A target makes the model return the logits of every position.
    logits, _ = network(inputs, inputs.clone())
    return logits


def fill_tables(network):
    """Gives the tables that start at zero values of their own, so that they change the predictions."""
    torch.manual_seed(1)
    for name, p in network.named_parameters():
        if 'table' in name:
            torch.nn.init.normal_(p, std=0.5)


class DocumentAttentionTest(unittest.TestCase):
    def test_predicts_a_document_the_same_whether_or_not_another_comes_before_it(self):
        network = model(document_bos=BOS, key_offset=True, bigram_table=True, trigram_table=True,
                        value_embeddings=True, stem=True)
        fill_tables(network)
        first = torch.tensor([[BOS, 3, 4, 5, 6]])
        second = torch.tensor([[BOS, 7, 8, 9, 10, 11]])
        alone = all_logits(network, second)
        after = all_logits(network, torch.cat([first, second], dim=1))[:, first.size(1):]
        self.assertTrue(torch.allclose(alone, after, atol=1e-4))

    def test_lets_a_document_attend_to_the_one_before_it_when_turned_off(self):
        network = model()
        first = torch.tensor([[BOS, 3, 4, 5, 6]])
        second = torch.tensor([[BOS, 7, 8, 9, 10, 11]])
        alone = all_logits(network, second)
        after = all_logits(network, torch.cat([first, second], dim=1))[:, first.size(1):]
        self.assertFalse(torch.allclose(alone, after, atol=1e-4))

    def test_keeps_earlier_positions_blind_to_later_tokens(self):
        network = model(document_bos=BOS)
        tokens = torch.tensor([[BOS, 3, 4, BOS, 5, 6, 7]])
        changed = tokens.clone()
        changed[0, -1] = 9
        self.assertTrue(torch.allclose(all_logits(network, tokens)[:, :-1], all_logits(network, changed)[:, :-1]))

    def test_numbers_each_document_of_a_row_from_its_bos(self):
        tokens = torch.tensor([[3, BOS, 4, BOS, 5], [BOS, 6, 7, 8, BOS]])
        self.assertEqual(document_ids(tokens, BOS).tolist(), [[0, 1, 1, 2, 2], [1, 1, 1, 1, 2]])


class SegmentBoundsTest(unittest.TestCase):
    def test_begins_a_segment_at_each_row_and_document_and_repeats_the_end_in_the_room_left(self):
        tokens = torch.tensor([[BOS, 1, 2, BOS, 3], [4, BOS, 5, 6, 7]])
        bounds = segment_bounds(document_ids(tokens, BOS))
        self.assertEqual(bounds.dtype, torch.int32)
        self.assertEqual(bounds.tolist(), [0, 3, 5, 6] + [10] * 13)

    def test_keeps_every_row_when_a_pass_holds_more_documents_than_its_room(self):
        tokens = torch.tensor([[BOS, BOS, BOS, BOS], [BOS, 1, BOS, 2]])
        with unittest.mock.patch('model.SEGMENTS_PER_ROW', 1):
            bounds = segment_bounds(document_ids(tokens, BOS))
        self.assertEqual(bounds.tolist(), [0, 4, 8])

    def test_keeps_the_earliest_documents_when_some_must_share(self):
        tokens = torch.tensor([[BOS, 1, BOS, 2, BOS, 3, BOS, 4]])
        with unittest.mock.patch('model.SEGMENTS_PER_ROW', 2):
            bounds = segment_bounds(document_ids(tokens, BOS))
        self.assertEqual(bounds.tolist(), [0, 2, 8])


class TokenTupleTablesTest(unittest.TestCase):
    def test_gives_the_head_of_a_document_the_last_row_of_the_table_of_pairs(self):
        tokens = torch.tensor([[BOS, 1, 2, BOS, 3]])
        rows = bigram_rows(tokens, 50, document_ids(tokens, BOS))
        self.assertEqual(rows[0, [0, 3]].tolist(), [49, 49])
        self.assertTrue((rows[0, [1, 2, 4]] < 49).all())

    def test_gives_positions_without_two_tokens_before_them_the_last_row_of_the_table_of_triples(self):
        rows = trigram_rows(torch.tensor([[1, 2, 3, 4]]), 50)
        self.assertEqual(rows[0, :2].tolist(), [49, 49])
        self.assertTrue((rows[0, 2:] < 49).all())

    def test_gives_the_last_row_to_triples_that_cross_the_head_of_a_document(self):
        tokens = torch.tensor([[BOS, 1, 2, BOS, 3, 4]])
        rows = trigram_rows(tokens, 50, document_ids(tokens, BOS))
        self.assertEqual(rows[0, [0, 1, 3, 4]].tolist(), [49, 49, 49, 49])
        self.assertTrue((rows[0, [2, 5]] < 49).all())

    def test_tells_the_order_of_the_three_tokens_apart(self):
        rows = trigram_rows(torch.tensor([[1, 2, 3, 3, 2, 1]]), 10007)
        self.assertNotEqual(rows[0, 2].item(), rows[0, 5].item())

    def test_sizes_both_tables_by_the_rows_per_token(self):
        network = model(bigram_table=True, trigram_table=True, bigram_rows_per_token=3)
        self.assertEqual(network.bigram_table.weight.shape, (96, 32))
        self.assertEqual(network.trigram_table.weight.shape, (96, 32))

    def test_starts_the_table_of_triples_at_nothing_added(self):
        with_triples = model(bigram_table=True, trigram_table=True)
        without = model(bigram_table=True)
        without.load_state_dict({k: v for k, v in with_triples.state_dict().items() if k in without.state_dict()})
        tokens = torch.randint(0, 32, (2, 12), generator=torch.Generator().manual_seed(2))
        self.assertTrue(torch.allclose(all_logits(with_triples, tokens), all_logits(without, tokens)))

    def test_changes_the_predictions_once_the_table_of_triples_has_learned(self):
        network = model(trigram_table=True)
        tokens = torch.randint(0, 32, (2, 12), generator=torch.Generator().manual_seed(2))
        before = all_logits(network, tokens)
        fill_tables(network)
        self.assertFalse(torch.allclose(before, all_logits(network, tokens)))

    def test_leaves_the_table_of_triples_out_of_the_count_used_for_compute(self):
        self.assertEqual(model(trigram_table=True).get_num_params() - 4, model().get_num_params())


class StemTest(unittest.TestCase):
    def test_puts_tables_on_the_layers_without_value_embeddings(self):
        self.assertEqual([has_stem_table(layer, 4) for layer in range(4)], [True, False, True, False])
        self.assertEqual(sorted(model(stem=True).stem_tables), ['0', '2'])

    def test_gives_each_table_a_row_per_token_as_wide_as_the_widened_values(self):
        self.assertEqual(model(stem=True).stem_tables['0'].weight.shape, (32, 128))

    def test_predicts_as_the_model_without_tables_while_the_tables_are_zero(self):
        with_tables = model(stem=True)
        without = model()
        without.load_state_dict({k: v for k, v in with_tables.state_dict().items() if k in without.state_dict()})
        tokens = torch.randint(0, 32, (2, 12), generator=torch.Generator().manual_seed(2))
        self.assertTrue(torch.allclose(all_logits(with_tables, tokens), all_logits(without, tokens)))

    def test_scales_each_widened_value_by_one_plus_the_tokens_number(self):
        torch.manual_seed(0)
        mlp = MLP(GPTConfig(n_embd=8, bias=False, relu_squared=True))
        x = torch.randn(1, 3, 8)
        row = torch.randn(1, 3, 32)
        expected = mlp.c_proj(torch.relu(mlp.c_fc(x)).square() * (1 + row))
        self.assertTrue(torch.allclose(mlp(x, row), expected, atol=1e-6))

    def test_leaves_the_tables_out_of_the_count_used_for_compute(self):
        self.assertEqual(model(stem=True).get_num_params(), model().get_num_params())

    def test_gives_the_tables_to_adamw_at_the_rate_of_the_table_of_pairs(self):
        network = model(stem=True, bigram_table=True, trigram_table=True)
        optimizer = configure_muon(network, 0.04, 0.004, value_embedding_lr=0.3)
        groups = {id(p): group for group in optimizer.param_groups for p in group['params']}
        pairs = groups[id(network.bigram_table.weight)]
        for table in [*network.stem_tables.values(), network.trigram_table]:
            self.assertIs(groups[id(table.weight)], pairs)
        self.assertEqual((pairs['kind'], pairs['lr']), ('adamw', 0.3))
        self.assertIs(groups[id(network.trigram_weights)], groups[id(network.bigram_weights)])


class XIELUTest(unittest.TestCase):
    def test_starts_with_the_papers_shape_above_and_below_zero(self):
        x = torch.tensor([-2.0, -0.5, 0.5, 2.0])
        negative = 0.8 * torch.expm1(x) - 0.8 * x + 0.5 * x
        expected = torch.where(x > 0, 0.8 * x * x + 0.5 * x, negative)
        self.assertTrue(torch.allclose(XIELU()(x), expected, atol=1e-5))

    def test_passes_a_gradient_below_zero(self):
        x = torch.tensor([-1.0], requires_grad=True)
        XIELU()(x).sum().backward()
        self.assertNotEqual(x.grad.item(), 0.0)

    def test_learns_both_of_its_values(self):
        activation = XIELU()
        activation(torch.tensor([-1.0, 1.0])).sum().backward()
        self.assertNotEqual(activation.alpha_p.grad.item(), 0.0)
        self.assertNotEqual(activation.alpha_n.grad.item(), 0.0)

    def test_takes_the_place_of_relu_squared_in_every_mlp(self):
        network = model(xielu=True, relu_squared=True)
        self.assertTrue(all(isinstance(block.mlp.activation, XIELU) for block in network.transformer.h))


if __name__ == "__main__":
    unittest.main()
