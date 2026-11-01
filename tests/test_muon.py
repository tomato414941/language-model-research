import importlib.util
import sys
import unittest
from pathlib import Path

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("Install the dependencies")

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

import muon as module
from model import GPT, GPTConfig


def small_model():
    torch.manual_seed(0)
    return GPT(GPTConfig(block_size=16, vocab_size=32, n_layer=2, n_head=2, n_embd=32, bias=False))


def configure(model, **settings):
    return module.configure_muon(model, muon_lr=0.04, other_lr=0.004, **settings)


def gradients(model):
    tokens = torch.randint(0, 32, (4, 17), generator=torch.Generator().manual_seed(1))
    _, loss = model(tokens[:, :-1], tokens[:, 1:].contiguous())
    loss.backward()
    return loss.item()


class OrthogonalizeTest(unittest.TestCase):
    def singular_values(self, shape, coefficients):
        matrices = torch.randn(shape, generator=torch.Generator().manual_seed(2))
        return torch.linalg.svdvals(module.orthogonalize(matrices, coefficients))

    def test_polar_express_brings_singular_values_near_one(self):
        for shape in ((3, 128, 32), (2, 32, 128)):
            values = self.singular_values(shape, module.POLAR_EXPRESS)
            self.assertTrue(((values > 0.8) & (values < 1.2)).all(), f"{shape}: {values}")

    def test_newton_schulz_brings_singular_values_into_its_known_range(self):
        values = self.singular_values((1, 128, 32), module.NEWTON_SCHULZ)
        self.assertTrue(((values > 0.5) & (values < 1.5)).all(), values)


class NormalizeNeuronsTest(unittest.TestCase):
    def test_evens_out_the_rows_and_keeps_the_size_of_the_matrix(self):
        update = torch.randn(1, 8, 4, generator=torch.Generator().manual_seed(3))
        update[0, 0] *= 10
        second_moment = torch.zeros(1, 8, 1)
        result = module.normalize_neurons(update, second_moment, beta2=0.0)
        self.assertAlmostEqual(result.norm().item(), update.norm().item(), places=3)
        rows = result.norm(dim=-1)
        self.assertLess((rows.max() / rows.min()).item(), 1.001)


class ConfigureTest(unittest.TestCase):
    def setUp(self):
        self.model = small_model()
        self.names = {id(p): name for name, p in self.model.named_parameters()}
        self.groups = configure(self.model).param_groups

    def names_of(self, group):
        return {self.names[id(p)] for p in group['params']}

    def test_gives_every_weight_to_exactly_one_group(self):
        given = [name for group in self.groups for name in self.names_of(group)]
        self.assertEqual(sorted(given), sorted(self.names.values()))

    def test_gives_the_shared_embeddings_positions_and_norms_to_adamw(self):
        others = self.names_of(self.groups[0])
        self.assertEqual(self.groups[0]['kind'], 'adamw')
        self.assertIn('transformer.wte.weight', others)
        self.assertIn('transformer.wpe.weight', others)
        self.assertIn('transformer.ln_f.weight', others)
        self.assertIn('transformer.h.0.ln_1.weight', others)

    def test_gives_the_matrices_of_the_blocks_to_muon(self):
        self.assertEqual(self.groups[1]['kind'], 'muon')
        self.assertIn('transformer.h.1.mlp.c_fc.weight', self.names_of(self.groups[1]))
        self.assertIn('transformer.h.1.attn.c_proj.weight', self.names_of(self.groups[1]))

    def test_treats_the_query_key_value_matrix_as_three(self):
        self.assertEqual(self.groups[2]['chunks'], 3)
        self.assertEqual(self.names_of(self.groups[2]),
                         {'transformer.h.0.attn.c_attn.weight', 'transformer.h.1.attn.c_attn.weight'})


class SeparateEmbeddingsGroupsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = GPT(GPTConfig(block_size=16, vocab_size=32, n_layer=2, n_head=2, n_embd=32, bias=False,
                                   separate_embeddings=True))
        self.groups = configure(self.model, embedding_lr=0.66).param_groups

    def rate_of(self, weight):
        return next(group['lr'] for group in self.groups if any(p is weight for p in group['params']))

    def test_gives_the_reading_table_its_own_rate(self):
        self.assertEqual(self.rate_of(self.model.transformer.wte.weight), 0.66)

    def test_keeps_the_predicting_table_at_the_rate_of_other_weights(self):
        self.assertEqual(self.rate_of(self.model.lm_head.weight), 0.004)

    def test_gives_every_weight_to_exactly_one_group(self):
        given = [id(p) for group in self.groups for p in group['params']]
        self.assertEqual(sorted(given), sorted(id(p) for p in self.model.parameters()))

    def test_keeps_a_shared_table_at_the_rate_of_other_weights(self):
        model = small_model()
        groups = configure(model, embedding_lr=0.66).param_groups
        rate = next(group['lr'] for group in groups if any(p is model.lm_head.weight for p in group['params']))
        self.assertEqual(rate, 0.004)


class StepTest(unittest.TestCase):
    def step_with_random_gradient(self, name, weight_decay=0.0, **settings):
        model = small_model()
        optimizer = configure(model, neuron_normalization=False, **settings)
        for group in optimizer.param_groups:
            if group['kind'] == 'muon':
                group['weight_decay'] = weight_decay
        weight = dict(model.named_parameters())[name]
        weight.grad = torch.randn(weight.shape, generator=torch.Generator().manual_seed(4))
        before = weight.detach().clone()
        optimizer.step()
        return before, weight.detach().clone()

    def test_learns_a_fixed_batch(self):
        model = small_model()
        optimizer = configure(model)
        first = gradients(model)
        for _ in range(30):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            last = gradients(model)
        self.assertLess(last, first * 0.8)

    def test_orthogonalizes_each_projection_of_the_query_key_value_matrix(self):
        before, after = self.step_with_random_gradient('transformer.h.0.attn.c_attn.weight')
        for change in (before - after).chunk(3, dim=0):
            values = torch.linalg.svdvals(change)
            self.assertLess((values.max() / values.min()).item(), 2.0)

    def test_scales_a_tall_matrix_so_that_its_updates_match_a_square_one(self):
        tall_before, tall_after = self.step_with_random_gradient('transformer.h.0.mlp.c_fc.weight')
        before, after = self.step_with_random_gradient('transformer.h.0.attn.c_proj.weight')
        tall = (tall_before - tall_after).square().mean().sqrt()
        square = (before - after).square().mean().sqrt()
        self.assertLess(abs((tall / square).item() - 1), 0.1)

    def test_decays_only_the_weights_the_update_shrinks_when_cautious(self):
        name = 'transformer.h.0.mlp.c_fc.weight'
        before, undecayed = self.step_with_random_gradient(name)
        _, decayed = self.step_with_random_gradient(name, weight_decay=0.5)
        shrinks = (before - undecayed) * before >= 0
        decay = undecayed - decayed
        lr = 0.04 * 2 # this matrix is four times as tall as it is wide
        self.assertTrue(shrinks.any() and (~shrinks).any())
        self.assertTrue(torch.allclose(decay[shrinks], lr * 0.5 * before[shrinks], atol=1e-5))
        self.assertTrue(torch.allclose(decay[~shrinks], torch.zeros(1), atol=1e-5))

    def test_decays_every_weight_when_not_cautious(self):
        name = 'transformer.h.0.mlp.c_fc.weight'
        before, undecayed = self.step_with_random_gradient(name, cautious_decay=False)
        _, decayed = self.step_with_random_gradient(name, weight_decay=0.5, cautious_decay=False)
        self.assertTrue(torch.allclose(undecayed - decayed, 0.04 * 2 * 0.5 * before, atol=1e-5))

    def test_continues_identically_after_saving_and_loading_its_state(self):
        def run(reload):
            model = small_model()
            optimizer = configure(model)
            for update in range(4):
                gradients(model)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if reload and update == 1:
                    state = optimizer.state_dict()
                    optimizer = configure(model)
                    optimizer.load_state_dict(state)
            return model.transformer.h[0].mlp.c_fc.weight.detach().clone()
        self.assertTrue(torch.equal(run(reload=False), run(reload=True)))


class ScheduleTest(unittest.TestCase):
    def setUp(self):
        self.optimizer = configure(small_model())
        self.adamw, self.muon, _ = self.optimizer.param_groups

    def test_holds_the_learning_rate_for_half_of_training_and_ends_at_zero(self):
        self.assertEqual(module.set_schedule(self.optimizer, 0, 1000, 0.2), 1.0)
        self.assertEqual(module.set_schedule(self.optimizer, 500, 1000, 0.2), 1.0)
        self.assertAlmostEqual(self.muon['lr'], 0.04)
        self.assertAlmostEqual(module.set_schedule(self.optimizer, 750, 1000, 0.2), 0.5)
        self.assertAlmostEqual(self.adamw['lr'], 0.002)
        self.assertEqual(module.set_schedule(self.optimizer, 1000, 1000, 0.2), 0.0)

    def test_stays_at_zero_after_the_last_scheduled_update(self):
        self.assertEqual(module.set_schedule(self.optimizer, 1001, 1000, 0.2), 0.0)
        self.assertEqual(self.muon['weight_decay'], 0.0)

    def test_raises_momentum_over_the_first_updates(self):
        module.set_schedule(self.optimizer, 0, 1000, 0.2)
        self.assertAlmostEqual(self.muon['momentum'], 0.85)
        module.set_schedule(self.optimizer, 150, 1000, 0.2)
        self.assertAlmostEqual(self.muon['momentum'], 0.90)
        module.set_schedule(self.optimizer, 600, 1000, 0.2)
        self.assertAlmostEqual(self.muon['momentum'], 0.95)

    def test_lowers_weight_decay_to_zero_over_training(self):
        module.set_schedule(self.optimizer, 0, 1000, 0.2)
        self.assertAlmostEqual(self.muon['weight_decay'], 0.2)
        module.set_schedule(self.optimizer, 500, 1000, 0.2)
        self.assertAlmostEqual(self.muon['weight_decay'], 0.1)
        module.set_schedule(self.optimizer, 1000, 1000, 0.2)
        self.assertEqual(self.muon['weight_decay'], 0.0)


if __name__ == "__main__":
    unittest.main()
