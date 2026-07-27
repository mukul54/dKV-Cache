"""Memory-efficient token confidence for low-confidence remasking.

The reference implementation computes

    p = F.softmax(logits.to(torch.float64), dim=-1)
    x0_p = p.gather(-1, x0.unsqueeze(-1)).squeeze(-1)

which materialises a full ``(B, L, V)`` float64 tensor. For LLaDA that is
``16 x 1312 x 126464 x 8B = 21 GB`` per denoising step at batch size 16, so
batched evaluation OOMs before the caching mechanism is ever exercised.

``token_confidence`` returns the same quantity -- ``P(x0)`` under the model, in
float64 -- but evaluates it chunk-by-chunk via ``logsumexp``, so peak memory is
bounded by ``chunk x V``. The value is used only to rank positions for
remasking, and logsumexp is at least as numerically accurate as the explicit
softmax, so decoding is unaffected.

Set ``LEGACY_SOFTMAX = True`` (``--legacy_confidence`` in the eval driver) to
restore the original code path verbatim.
"""

import torch
import torch.nn.functional as F

LEGACY_SOFTMAX = False
CHUNK_ROWS = 1024


def token_confidence(logits, x0):
    """P(x0 | logits) in float64, with shape ``logits.shape[:-1]``."""
    if LEGACY_SOFTMAX:
        p = F.softmax(logits.to(torch.float64), dim=-1)
        return torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)

    lead_shape = logits.shape[:-1]
    vocab = logits.shape[-1]
    flat_logits = logits.reshape(-1, vocab)
    flat_x0 = x0.reshape(-1)

    out = torch.empty(flat_logits.shape[0], dtype=torch.float64, device=logits.device)
    for start in range(0, flat_logits.shape[0], CHUNK_ROWS):
        end = min(start + CHUNK_ROWS, flat_logits.shape[0])
        chunk = flat_logits[start:end].to(torch.float64)
        selected = torch.gather(chunk, -1, flat_x0[start:end].unsqueeze(-1)).squeeze(-1)
        out[start:end] = torch.exp(selected - torch.logsumexp(chunk, dim=-1))
    return out.view(lead_shape)
