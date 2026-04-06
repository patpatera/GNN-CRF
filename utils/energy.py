import torch
from torch import einsum

def hopfield_energy(state_patterns, stored_patterns, scale, mask=None):
    kinetic = 0.5 * einsum("b i d, b i d -> b i", state_patterns, state_patterns)
    scaled_dot_product = scale * einsum(
        "b i d, b j d -> b i j", state_patterns, stored_patterns
    )

    if mask is not None:
        max_neg_value = -torch.finfo(scaled_dot_product.dtype).max
        scaled_dot_product.masked_fill_(~mask, max_neg_value)

    potential = -(1.0 / scale) * torch.logsumexp(scaled_dot_product, dim=-1)
    return kinetic + potential, scaled_dot_product