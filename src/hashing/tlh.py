from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch

from .encoding import encode_signed_decimal_17bit


@dataclass
class TlhResult:
    method: str
    num_blocks: int
    block_means: list[float]
    values: list[float]
    encoded: list[str]

    def to_dict(self):
        return asdict(self)


def split_model_into_blocks(model, num_blocks: int = 450):
    flat_params = torch.cat([parameter.detach().cpu().view(-1) for parameter in model.parameters()])
    total_params = flat_params.numel()
    block_size = math.ceil(total_params / num_blocks)
    blocks = [
        (index * block_size, min((index + 1) * block_size, total_params))
        for index in range(num_blocks)
    ]
    return flat_params, blocks


def compute_block_means(flat_params, blocks) -> list[float]:
    means = []
    for start, end in blocks:
        block = flat_params[start:end]
        means.append(float(block.mean().item()) if block.numel() else 0.0)
    return means


def generate_chaos_values(block_averages, mu: float = 0.2, k: float = 2.0, iterations: int = 100):
    chaos_values = []
    for average in block_averages:
        x_n = float(average)
        q_n = x_n * 0.5
        for _ in range(iterations):
            x_next = mu * x_n * (1 - x_n) + k * np.cos(q_n) * x_n
            q_next = q_n + x_n
            x_n, q_n = x_next, q_next
        chaos_values.append(float(x_n))
    return chaos_values


def generate_tlh(
    model,
    method: str = "chaos",
    num_blocks: int = 450,
    mu: float = 0.2,
    k: float = 2.0,
    iterations: int = 100,
) -> TlhResult:
    method = method.lower()
    flat_params, blocks = split_model_into_blocks(model, num_blocks=num_blocks)
    block_means = compute_block_means(flat_params, blocks)

    if method == "chaos":
        values = generate_chaos_values(block_means, mu=mu, k=k, iterations=iterations)
    elif method in {"baseline", "nochaos"}:
        values = block_means
    else:
        raise ValueError("method must be 'chaos', 'baseline', or 'nochaos'")

    encoded = [encode_signed_decimal_17bit(value) for value in values]
    return TlhResult(method=method, num_blocks=num_blocks, block_means=block_means, values=values, encoded=encoded)
