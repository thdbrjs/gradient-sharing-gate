import unittest

import torch

from gradient_sharing_gate.gating import (
    apply_gate_to_update,
    moments_from_gradients,
    q_to_gate,
    sharing_q,
    update_moments,
)


class GatingTest(unittest.TestCase):
    def test_q_is_one_for_equal_magnitudes_despite_sign_conflict(self):
        gradients = torch.tensor([[2.0, -3.0], [-2.0, 3.0]])
        first, second = moments_from_gradients(gradients, mode="abs")
        self.assertTrue(torch.allclose(sharing_q(first, second), torch.ones(2)))

    def test_signed_q_cancels_opposite_directions(self):
        gradients = torch.tensor([[2.0, -3.0], [-2.0, 3.0]])
        first, second = moments_from_gradients(gradients, mode="signed")
        self.assertTrue(torch.equal(sharing_q(first, second), torch.zeros(2)))

    def test_q_is_inverse_batch_size_for_one_contributing_example(self):
        gradients = torch.tensor([[4.0], [0.0], [0.0], [0.0]])
        first, second = moments_from_gradients(gradients)
        self.assertAlmostEqual(sharing_q(first, second).item(), 0.25)

    def test_q_to_gate_preserves_original_range(self):
        q = torch.tensor([0.0, 0.25, 0.5, 0.75])
        self.assertTrue(torch.equal(q_to_gate(q), q))

    def test_update_moments_uses_ema(self):
        first = torch.tensor([1.0])
        second = torch.tensor([4.0])
        new_first, new_second = update_moments(
            first, second, torch.tensor([[3.0], [-3.0]]), beta=0.5
        )
        self.assertAlmostEqual(new_first.item(), 2.0)
        self.assertAlmostEqual(new_second.item(), 6.5)

    def test_apply_gate_scales_optimizer_displacement(self):
        parameter = torch.nn.Parameter(torch.tensor([12.0, 12.0]))
        before = [torch.tensor([10.0, 10.0])]
        apply_gate_to_update([("p", parameter)], before, torch.tensor([0.0, 0.5]))
        self.assertTrue(torch.equal(parameter, torch.tensor([10.0, 11.0])))


if __name__ == "__main__":
    unittest.main()
