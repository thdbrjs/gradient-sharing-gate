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
    mode: str = "abs",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the selected first moment and E[g^2] for every parameter."""
    stacked = _stack(gradients)
    if mode == "abs":
        first = stacked.abs().mean(dim=0)
    elif mode == "signed":
        first = stacked.mean(dim=0)
    else:
        raise ValueError("mode must be 'abs' or 'signed'")
    return first, stacked.square().mean(dim=0)


def update_moments(
    first_moment: torch.Tensor,
    second_square: torch.Tensor,
    gradients: torch.Tensor | Sequence[torch.Tensor],
    beta: float,
    mode: str = "abs",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update the two sharing moments with an exponential moving average."""
    if not 0.0 <= beta < 1.0:
        raise ValueError("beta must satisfy 0 <= beta < 1")
    observed_first, observed_square = moments_from_gradients(gradients, mode=mode)
    return (
        beta * first_moment + (1.0 - beta) * observed_first,
        beta * second_square + (1.0 - beta) * observed_square,
    )


def sharing_q(
    first_moment: torch.Tensor,
    second_square: torch.Tensor,
    epsilon: float = 1e-30,
) -> torch.Tensor:
    """Compute a normalized first-to-second-moment ratio in [0, 1].

    The caller chooses whether the first moment is E[|g|] or E[g].
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    return first_moment.square().div(second_square + epsilon).clamp(0.0, 1.0)


def q_to_gate(q: torch.Tensor) -> torch.Tensor:
    """Use q's original [0, 1] range directly as the update multiplier."""
    return q.clamp(0.0, 1.0)


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
