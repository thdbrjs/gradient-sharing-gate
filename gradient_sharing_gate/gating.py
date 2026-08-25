"""Core statistics and update gating for shared gradients."""

from collections.abc import Sequence

import torch


def _stack(gradients: torch.Tensor | Sequence[torch.Tensor]) -> torch.Tensor:
    if isinstance(gradients, torch.Tensor):
        if gradients.ndim != 2:
            raise ValueError("gradients must have shape [examples, parameters]")
        return gradients
    if not gradients:
        raise ValueError("at least one per-example gradient is required")
    return torch.stack(tuple(gradients))


def moments_from_gradients(
    gradients: torch.Tensor | Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return E[|g|] and E[g^2] across examples for every parameter."""
    stacked = _stack(gradients)
    return stacked.abs().mean(dim=0), stacked.square().mean(dim=0)


def update_moments(
    first_abs: torch.Tensor,
    second_square: torch.Tensor,
    gradients: torch.Tensor | Sequence[torch.Tensor],
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update the two sharing moments with an exponential moving average."""
    if not 0.0 <= beta < 1.0:
        raise ValueError("beta must satisfy 0 <= beta < 1")
    observed_abs, observed_square = moments_from_gradients(gradients)
    return (
        beta * first_abs + (1.0 - beta) * observed_abs,
        beta * second_square + (1.0 - beta) * observed_square,
    )


def sharing_q(
    first_abs: torch.Tensor,
    second_square: torch.Tensor,
    epsilon: float = 1e-30,
) -> torch.Tensor:
    """Estimate how broadly each parameter's gradient is shared across examples.

    q = E[|g|]^2 / (E[g^2] + epsilon). Signs intentionally do not cancel.
    q approaches 1 when magnitudes are similar across examples and approaches
    1/B when only one of B examples contributes.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    return first_abs.square().div(second_square + epsilon).clamp(0.0, 1.0)


def q_to_gate(q: torch.Tensor, q_gate_max: float = 0.5) -> torch.Tensor:
    """Map q in [0, q_gate_max] linearly to an update multiplier in [0, 1]."""
    if q_gate_max <= 0:
        raise ValueError("q_gate_max must be positive")
    return q.div(q_gate_max).clamp(0.0, 1.0)


@torch.no_grad()
def apply_gate_to_update(named_parameters, previous, gate: torch.Tensor) -> None:
    """Scale an optimizer's proposed parameter displacement element-wise."""
    expected = sum(parameter.numel() for _, parameter in named_parameters)
    if gate.numel() != expected:
        raise ValueError(f"gate has {gate.numel()} values; expected {expected}")
    if len(previous) != len(named_parameters):
        raise ValueError("previous parameter list does not match named_parameters")

    offset = 0
    for (_, parameter), before in zip(named_parameters, previous):
        end = offset + parameter.numel()
        local_gate = gate[offset:end].view_as(parameter).to(parameter.dtype)
        parameter.copy_(before + local_gate * (parameter - before))
        offset = end
