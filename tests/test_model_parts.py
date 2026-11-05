import importlib.util
import math
import sys
import unittest
import unittest.mock
from pathlib import Path

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("Install the dependencies")

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from model import (GPT, MLP, XIELU, Experts, GPTConfig, bigram_rows, document_ids, has_stem_table, segment_bounds,
                   trigram_rows, window_segments)
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


class WindowTest(unittest.TestCase):
    def test_cuts_each_document_where_the_rows_positions_cross_a_multiple_of_the_window(self):
        tokens = torch.tensor([[BOS, 1, 2, 3, BOS, 4, 5, 6]])
        segments = window_segments(document_ids(tokens, BOS), torch.tensor(3))
        heads = (segments[0, 1:] != segments[0, :-1]).nonzero().flatten() + 1
        self.assertEqual(heads.tolist(), [3, 4, 6])

    def test_keeps_later_windows_blind_to_the_tokens_of_earlier_ones(self):
        network = model(document_bos=BOS, smallest_window=4, key_offset=True, bigram_table=True, trigram_table=True)
        fill_tables(network)
        tokens = torch.tensor([[BOS, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]])
        changed = tokens.clone()
        changed[0, 1] = 20
        network.attention_window.fill_(4)
        self.assertTrue(torch.allclose(all_logits(network, tokens)[:, 4:], all_logits(network, changed)[:, 4:]))
        network.attention_window.fill_(32)
        self.assertFalse(torch.allclose(all_logits(network, tokens)[:, 4:], all_logits(network, changed)[:, 4:]))

    def test_reads_the_whole_row_unless_a_window_is_set(self):
        tokens = torch.tensor([[BOS, 3, 4, 5, 6, 7, BOS, 9, 10, 11, 12, 13]])
        windowed = all_logits(model(document_bos=BOS, smallest_window=4), tokens)
        self.assertTrue(torch.allclose(windowed, all_logits(model(document_bos=BOS), tokens)))

    def test_begins_a_segment_for_attention_over_variable_lengths_at_every_piece(self):
        tokens = torch.tensor([[BOS] + list(range(1, 16))])
        segments = window_segments(document_ids(tokens, BOS), torch.tensor(4))
        self.assertEqual(segment_bounds(segments, 8 + 16 // 4).tolist(), [0, 4, 8, 12] + [16] * 9)


def experts(count=4, chosen=2, shared=0):
    torch.manual_seed(0)
    config = GPTConfig(block_size=32, vocab_size=32, n_layer=4, n_head=2, n_embd=16, bias=False, experts=count,
                       experts_per_token=chosen, expert_width=8, shared_experts=shared)
    return Experts(config)


class ExpertsTest(unittest.TestCase):
    def test_takes_the_mlps_of_the_layers_without_stems_tables(self):
        network = model(stem=True, experts=4, experts_per_token=2, expert_width=8)
        kinds = [type(block.mlp) for block in network.transformer.h]
        self.assertEqual(kinds, [MLP, Experts, MLP, Experts])
        self.assertEqual(kinds, [MLP if has_stem_table(layer, 4) else Experts for layer in range(4)])

    def test_routes_by_affinity_and_bias_and_weighs_by_affinity_alone(self):
        layer = experts()
        torch.nn.init.zeros_(layer.router.weight)
        layer.balance.copy_(torch.tensor([0.0, 0.0, 1.0, 2.0]))
        chosen, weights = layer.route(torch.randn(3, 16))
        self.assertEqual(sorted(chosen[0].tolist()), [2, 3])
        self.assertTrue(torch.allclose(weights, torch.full((3, 2), 0.5)))

    def test_gives_in_training_what_it_gives_outside_while_no_expert_overflows(self):
        layer = experts(shared=1)
        torch.nn.init.normal_(layer.w_out, std=0.1)
        x = torch.randn(2, 6, 16)
        with unittest.mock.patch('model.EXPERT_CAPACITY', 100.0):
            trained = layer.train()(x)
        self.assertTrue(torch.allclose(trained, layer.eval()(x), atol=1e-5))

    def test_drops_the_routes_past_an_experts_capacity_the_later_tokens_first(self):
        layer = experts(count=2, chosen=1)
        torch.nn.init.normal_(layer.w_out, std=0.1)
        layer.balance.copy_(torch.tensor([10.0, 0.0])) # every token to the first expert
        x = torch.randn(1, 8, 16)
        trained, whole = layer.train()(x), layer.eval()(x)
        kept = math.ceil(8 / 2 * 1.25)
        self.assertTrue(torch.allclose(trained[0, :kept], whole[0, :kept], atol=1e-5))
        self.assertEqual(trained[0, kept:].abs().sum().item(), 0.0)

    def test_counts_the_load_in_training_and_moves_each_bias_toward_the_mean(self):
        layer = experts(count=4, chosen=1)
        layer.balance.copy_(torch.tensor([10.0, 0.0, 0.0, 0.0]))
        layer.eval()(torch.randn(1, 5, 16))
        self.assertEqual(layer.load.sum().item(), 0.0)
        layer.train()(torch.randn(1, 5, 16))
        self.assertEqual(layer.load.tolist(), [5.0, 0.0, 0.0, 0.0])
        layer.rebalance(0.1)
        self.assertTrue(torch.allclose(layer.balance, torch.tensor([9.9, 0.1, 0.1, 0.1])))
        self.assertEqual(layer.load.sum().item(), 0.0)

    def test_starts_adding_nothing_when_the_outputs_start_at_zero(self):
        network = model(experts=4, experts_per_token=2, expert_width=8, shared_experts=1, zero_output=True)
        layer = network.transformer.h[0].mlp
        self.assertEqual(layer.train()(torch.randn(1, 4, 32)).abs().sum().item(), 0.0)


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
