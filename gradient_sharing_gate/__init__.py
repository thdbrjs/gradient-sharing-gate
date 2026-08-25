"""Utilities for modifying gradient updates using cross-example sharing."""

from .gating import (
    apply_gate_to_update,
    moments_from_gradients,
    q_to_gate,
    sharing_q,
    update_moments,
)

__all__ = [
    "apply_gate_to_update",
    "moments_from_gradients",
    "q_to_gate",
    "sharing_q",
    "update_moments",
]
