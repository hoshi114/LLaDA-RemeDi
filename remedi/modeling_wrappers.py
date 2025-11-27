import torch
import torch.nn as nn


class UPSHead(nn.Module):
    """
    Minimal per-token UPS head that maps hidden states to a scalar confidence logit.

    Design: Linear -> GELU -> Linear. Keep it lightweight to allow training only
    this head in early stages (freeze backbone) to save memory/compute.
    """

    def __init__(self, hidden_size: int, width: int = 0):
        super().__init__()
        # If width == 0, use a single linear layer for minimal overhead.
        if width and width > 0:
            self.net = nn.Sequential(
                nn.Linear(hidden_size, width),
                nn.GELU(),
                nn.Linear(width, 1),
            )
        else:
            self.net = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (b, l, h) last-layer hidden states.
        Returns:
            scores: (b, l) raw logits before sigmoid.
        """
        scores = self.net(hidden_states)  # (b, l, 1)
        return scores.squeeze(-1)


def get_last_hidden_states(outputs) -> torch.Tensor:
    """Best-effort helper to extract last hidden states from a HF output.

    It prefers outputs.hidden_states[-1] when available. If not present but
    outputs.last_hidden_state exists (e.g., encoder models), returns it.
    """
    if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
        return outputs.hidden_states[-1]
    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        return outputs.last_hidden_state
    # Not available; caller must handle None (fallback confidence source)
    return None


def load_ups_head(path: str, hidden_size: int = None):
    """Load a saved UPSHead checkpoint and construct the head.

    Args:
        path: torch.load()-able checkpoint path containing
              {'state_dict', 'hidden_size', 'width', ...}
        hidden_size: optional override if checkpoint omits it
    Returns:
        head: UPSHead with loaded weights
        meta: dict metadata from checkpoint
    """
    ckpt = torch.load(path, map_location='cpu')
    width = int(ckpt.get('width', 0))
    hs = int(ckpt.get('hidden_size', hidden_size))
    if hs is None:
        raise ValueError('hidden_size must be provided either in checkpoint or as argument')
    head = UPSHead(hidden_size=hs, width=width)
    head.load_state_dict(ckpt['state_dict'])
    return head, ckpt
