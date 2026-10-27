from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class TamperResult:
    tampered_blocks: list[dict]
    total_params: int
    num_blocks: int
    block_size: int
    tamper_rate: float
    noise_std: float
    seed: int
    attack_type: str = "gaussian_noise"
    attack_params: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "metadata": {
                "attack_type": self.attack_type,
                "total_params": self.total_params,
                "num_blocks": self.num_blocks,
                "block_size": self.block_size,
                "tamper_rate": self.tamper_rate,
                "noise_std": self.noise_std,
                "seed": self.seed,
                "attack_params": self.attack_params,
            },
            "tampered_blocks": self.tampered_blocks,
        }


def resolve_tlh_blocks(model_name: str) -> int:
    normalized = model_name.lower().replace("_", "").replace("-", "")
    return 100 if normalized in {"mnist", "mnistnet", "lenet"} else 450


def split_model_parameters(model, num_blocks: int):
    flat_params = torch.cat([parameter.data.view(-1) for parameter in model.parameters()])
    total_params = flat_params.numel()
    block_size = math.ceil(total_params / num_blocks)
    blocks = [
        (index * block_size, min((index + 1) * block_size, total_params))
        for index in range(num_blocks)
    ]
    return flat_params, blocks, total_params, block_size


def restore_flat_parameters(model, flat_params) -> None:
    offset = 0
    for parameter in model.parameters():
        numel = parameter.numel()
        parameter.data.copy_(flat_params[offset : offset + numel].view_as(parameter))
        offset += numel


def tamper_model_blockwise_fixed_n(
    model,
    tamper_rate: float,
    num_blocks: int,
    noise_std: float = 0.1,
    seed: int = 42,
) -> TamperResult:
    if not 0 < tamper_rate <= 1:
        raise ValueError("tamper_rate must be in the interval (0, 1].")
    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive.")

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    flat_params, blocks, total_params, block_size = split_model_parameters(model, num_blocks)
    num_tamper_blocks = max(1, int(num_blocks * tamper_rate))
    selected_blocks = sorted(random.sample(range(num_blocks), num_tamper_blocks))

    tampered_blocks = []
    for block_index in selected_blocks:
        start, end = blocks[block_index]
        block_length = end - start
        if block_length <= 0:
            continue

        noise = torch.randn(block_length, device=flat_params.device, dtype=flat_params.dtype) * noise_std
        flat_params[start:end] += noise
        tampered_blocks.append(
            {
                "block_index": block_index,
                "start_param_index": start,
                "end_param_index": end - 1,
                "block_size": block_length,
            }
        )

    restore_flat_parameters(model, flat_params)
    return TamperResult(
        tampered_blocks=tampered_blocks,
        total_params=total_params,
        num_blocks=num_blocks,
        block_size=block_size,
        tamper_rate=tamper_rate,
        noise_std=noise_std,
        seed=seed,
        attack_type="gaussian_noise",
    )


tamper_model_blockwise_fixedN = tamper_model_blockwise_fixed_n


def _set_random_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _select_blocks(num_blocks: int, tamper_rate: float, seed: int) -> list[int]:
    if not 0 < tamper_rate <= 1:
        raise ValueError("tamper_rate must be in the interval (0, 1].")
    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive.")
    random.seed(seed)
    num_tamper_blocks = max(1, int(num_blocks * tamper_rate))
    return sorted(random.sample(range(num_blocks), num_tamper_blocks))


def _block_log_entry(block_index: int, start: int, end: int) -> dict:
    return {
        "block_index": block_index,
        "start_param_index": start,
        "end_param_index": end - 1,
        "block_size": end - start,
    }


def tamper_model_sign_flip(
    model,
    tamper_rate: float,
    num_blocks: int,
    seed: int = 42,
) -> TamperResult:
    _set_random_seeds(seed)
    flat_params, blocks, total_params, block_size = split_model_parameters(model, num_blocks)
    selected_blocks = _select_blocks(num_blocks, tamper_rate, seed)

    tampered_blocks = []
    for block_index in selected_blocks:
        start, end = blocks[block_index]
        if end <= start:
            continue
        flat_params[start:end] *= -1
        tampered_blocks.append(_block_log_entry(block_index, start, end))

    restore_flat_parameters(model, flat_params)
    return TamperResult(
        tampered_blocks=tampered_blocks,
        total_params=total_params,
        num_blocks=num_blocks,
        block_size=block_size,
        tamper_rate=tamper_rate,
        noise_std=0.0,
        seed=seed,
        attack_type="sign_flip",
    )


def tamper_model_weight_scaling(
    model,
    tamper_rate: float,
    num_blocks: int,
    seed: int = 42,
    alpha: float = 1.5,
) -> TamperResult:
    _set_random_seeds(seed)
    flat_params, blocks, total_params, block_size = split_model_parameters(model, num_blocks)
    selected_blocks = _select_blocks(num_blocks, tamper_rate, seed)

    tampered_blocks = []
    for block_index in selected_blocks:
        start, end = blocks[block_index]
        if end <= start:
            continue
        flat_params[start:end] *= alpha
        tampered_blocks.append(_block_log_entry(block_index, start, end))

    restore_flat_parameters(model, flat_params)
    return TamperResult(
        tampered_blocks=tampered_blocks,
        total_params=total_params,
        num_blocks=num_blocks,
        block_size=block_size,
        tamper_rate=tamper_rate,
        noise_std=0.0,
        seed=seed,
        attack_type="weight_scaling",
        attack_params={"alpha": alpha},
    )


def tamper_model_weight_zeroing(
    model,
    tamper_rate: float,
    num_blocks: int,
    seed: int = 42,
) -> TamperResult:
    _set_random_seeds(seed)
    flat_params, blocks, total_params, block_size = split_model_parameters(model, num_blocks)
    selected_blocks = _select_blocks(num_blocks, tamper_rate, seed)

    tampered_blocks = []
    for block_index in selected_blocks:
        start, end = blocks[block_index]
        if end <= start:
            continue
        flat_params[start:end] = 0
        tampered_blocks.append(_block_log_entry(block_index, start, end))

    restore_flat_parameters(model, flat_params)
    return TamperResult(
        tampered_blocks=tampered_blocks,
        total_params=total_params,
        num_blocks=num_blocks,
        block_size=block_size,
        tamper_rate=tamper_rate,
        noise_std=0.0,
        seed=seed,
        attack_type="weight_zeroing",
    )


def _reinit_block(
    flat_params: torch.Tensor,
    start: int,
    end: int,
    mode: str,
    source_mean: float,
    source_std: float,
) -> None:
    block_length = end - start
    if mode == "normal":
        flat_params[start:end] = torch.normal(
            mean=source_mean,
            std=max(source_std, 1e-8),
            size=(block_length,),
            device=flat_params.device,
            dtype=flat_params.dtype,
        )
        return
    if mode == "xavier":
        bound = math.sqrt(6.0 / max(2 * block_length, 1))
        flat_params[start:end] = torch.empty(
            block_length,
            device=flat_params.device,
            dtype=flat_params.dtype,
        ).uniform_(-bound, bound)
        return
    if mode == "kaiming":
        std = math.sqrt(2.0 / max(block_length, 1))
        flat_params[start:end] = torch.randn(
            block_length,
            device=flat_params.device,
            dtype=flat_params.dtype,
        ) * std
        return
    raise ValueError("reinit_mode must be one of: normal, xavier, kaiming.")


def tamper_model_random_reinit(
    model,
    tamper_rate: float,
    num_blocks: int,
    seed: int = 42,
    mode: str = "normal",
) -> TamperResult:
    if mode not in {"normal", "xavier", "kaiming"}:
        raise ValueError("mode must be one of: normal, xavier, kaiming.")

    _set_random_seeds(seed)
    flat_params, blocks, total_params, block_size = split_model_parameters(model, num_blocks)
    selected_blocks = _select_blocks(num_blocks, tamper_rate, seed)
    source = flat_params.detach()
    source_std = float(source.std().item()) if source.numel() else 0.0
    source_mean = float(source.mean().item()) if source.numel() else 0.0

    tampered_blocks = []
    for block_index in selected_blocks:
        start, end = blocks[block_index]
        if end <= start:
            continue
        _reinit_block(flat_params, start, end, mode, source_mean, source_std)
        tampered_blocks.append(_block_log_entry(block_index, start, end))

    restore_flat_parameters(model, flat_params)
    return TamperResult(
        tampered_blocks=tampered_blocks,
        total_params=total_params,
        num_blocks=num_blocks,
        block_size=block_size,
        tamper_rate=tamper_rate,
        noise_std=source_std,
        seed=seed,
        attack_type="random_reinit",
        attack_params={"mode": mode},
    )


def tamper_model_weight_permutation(
    model,
    tamper_rate: float,
    num_blocks: int,
    seed: int = 42,
) -> TamperResult:
    _set_random_seeds(seed)
    flat_params, blocks, total_params, block_size = split_model_parameters(model, num_blocks)
    selected_blocks = _select_blocks(num_blocks, tamper_rate, seed)

    tampered_blocks = []
    for block_index in selected_blocks:
        start, end = blocks[block_index]
        block_length = end - start
        if block_length <= 1:
            continue
        permutation = torch.randperm(block_length, device=flat_params.device)
        flat_params[start:end] = flat_params[start:end][permutation]
        tampered_blocks.append(_block_log_entry(block_index, start, end))

    restore_flat_parameters(model, flat_params)
    return TamperResult(
        tampered_blocks=tampered_blocks,
        total_params=total_params,
        num_blocks=num_blocks,
        block_size=block_size,
        tamper_rate=tamper_rate,
        noise_std=0.0,
        seed=seed,
        attack_type="weight_permutation",
    )


def tamper_model_quantization(
    model,
    tamper_rate: float,
    num_blocks: int,
    seed: int = 42,
    bits: int = 8,
    decimals: int | None = None,
) -> TamperResult:
    if bits < 2:
        raise ValueError("bits must be at least 2.")
    if decimals is not None and decimals < 0:
        raise ValueError("decimals must be non-negative.")

    _set_random_seeds(seed)
    flat_params, blocks, total_params, block_size = split_model_parameters(model, num_blocks)
    selected_blocks = _select_blocks(num_blocks, tamper_rate, seed)

    tampered_blocks = []
    for block_index in selected_blocks:
        start, end = blocks[block_index]
        if end <= start:
            continue
        block = flat_params[start:end]
        if decimals is not None:
            flat_params[start:end] = torch.round(block * (10**decimals)) / (10**decimals)
        else:
            max_abs = block.abs().max()
            if float(max_abs.item()) > 0:
                qmax = (2 ** (bits - 1)) - 1
                scale = max_abs / qmax
                flat_params[start:end] = torch.round(block / scale).clamp(-qmax, qmax) * scale
        tampered_blocks.append(_block_log_entry(block_index, start, end))

    restore_flat_parameters(model, flat_params)
    return TamperResult(
        tampered_blocks=tampered_blocks,
        total_params=total_params,
        num_blocks=num_blocks,
        block_size=block_size,
        tamper_rate=tamper_rate,
        noise_std=0.0,
        seed=seed,
        attack_type="quantization",
        attack_params={"bits": bits, "decimals": decimals},
    )


def tamper_model_by_attack(
    model,
    attack: str,
    tamper_rate: float,
    num_blocks: int,
    seed: int = 42,
    noise_std: float = 0.1,
    alpha: float = 1.5,
    reinit_mode: str = "normal",
    quant_bits: int = 8,
    quant_decimals: int | None = None,
) -> TamperResult:
    if attack == "gaussian_noise":
        return tamper_model_blockwise_fixed_n(
            model,
            tamper_rate=tamper_rate,
            num_blocks=num_blocks,
            noise_std=noise_std,
            seed=seed,
        )
    if attack == "sign_flip":
        return tamper_model_sign_flip(model, tamper_rate=tamper_rate, num_blocks=num_blocks, seed=seed)
    if attack == "weight_scaling":
        return tamper_model_weight_scaling(
            model,
            tamper_rate=tamper_rate,
            num_blocks=num_blocks,
            seed=seed,
            alpha=alpha,
        )
    if attack == "weight_zeroing":
        return tamper_model_weight_zeroing(model, tamper_rate=tamper_rate, num_blocks=num_blocks, seed=seed)
    if attack == "random_reinit":
        return tamper_model_random_reinit(
            model,
            tamper_rate=tamper_rate,
            num_blocks=num_blocks,
            seed=seed,
            mode=reinit_mode,
        )
    if attack == "weight_permutation":
        return tamper_model_weight_permutation(model, tamper_rate=tamper_rate, num_blocks=num_blocks, seed=seed)
    if attack == "quantization":
        return tamper_model_quantization(
            model,
            tamper_rate=tamper_rate,
            num_blocks=num_blocks,
            seed=seed,
            bits=quant_bits,
            decimals=quant_decimals,
        )
    raise ValueError(f"Unsupported attack: {attack}")


def tamper_model_parameter_replacement(
    model,
    tamper_rate: float,
    num_blocks: int,
    seed: int = 42,
    replacement: str = "shuffle",
) -> TamperResult:
    if replacement not in {"shuffle", "normal"}:
        raise ValueError("replacement must be either 'shuffle' or 'normal'.")

    _set_random_seeds(seed)
    flat_params, blocks, total_params, block_size = split_model_parameters(model, num_blocks)
    selected_blocks = _select_blocks(num_blocks, tamper_rate, seed)
    source = flat_params.detach().clone()
    source_std = float(source.std().item()) if source.numel() else 0.0
    source_mean = float(source.mean().item()) if source.numel() else 0.0

    tampered_blocks = []
    for block_index in selected_blocks:
        start, end = blocks[block_index]
        block_length = end - start
        if block_length <= 0:
            continue

        if replacement == "shuffle":
            random_start = random.randint(0, max(0, total_params - block_length))
            flat_params[start:end] = source[random_start : random_start + block_length]
        else:
            flat_params[start:end] = torch.normal(
                mean=source_mean,
                std=max(source_std, 1e-8),
                size=(block_length,),
                device=flat_params.device,
                dtype=flat_params.dtype,
            )
        tampered_blocks.append(_block_log_entry(block_index, start, end))

    restore_flat_parameters(model, flat_params)
    return TamperResult(
        tampered_blocks=tampered_blocks,
        total_params=total_params,
        num_blocks=num_blocks,
        block_size=block_size,
        tamper_rate=tamper_rate,
        noise_std=source_std,
        seed=seed,
        attack_type=f"parameter_replacement_{replacement}",
    )


def tamper_model_layer_edit(
    model,
    tamper_rate: float,
    num_blocks: int,
    seed: int = 42,
    scale: float = 1.25,
    bias: float = 0.0,
) -> TamperResult:
    _set_random_seeds(seed)
    flat_params, blocks, total_params, block_size = split_model_parameters(model, num_blocks)

    parameter_ranges = []
    offset = 0
    for name, parameter in model.named_parameters():
        length = parameter.numel()
        if "weight" in name and parameter.ndim >= 2:
            parameter_ranges.append((name, offset, offset + length))
        offset += length

    if not parameter_ranges:
        raise ValueError("No editable weight layers found.")

    num_layers = max(1, int(len(parameter_ranges) * tamper_rate))
    selected_layers = sorted(random.sample(range(len(parameter_ranges)), num_layers))
    edited_ranges = []
    for layer_index in selected_layers:
        _, start, end = parameter_ranges[layer_index]
        flat_params[start:end] = flat_params[start:end] * scale + bias
        edited_ranges.append((start, end))

    tampered_blocks = []
    for block_index, (block_start, block_end) in enumerate(blocks):
        if any(max(block_start, edit_start) < min(block_end, edit_end) for edit_start, edit_end in edited_ranges):
            tampered_blocks.append(_block_log_entry(block_index, block_start, block_end))

    restore_flat_parameters(model, flat_params)
    return TamperResult(
        tampered_blocks=tampered_blocks,
        total_params=total_params,
        num_blocks=num_blocks,
        block_size=block_size,
        tamper_rate=tamper_rate,
        noise_std=0.0,
        seed=seed,
        attack_type="layer_edit",
    )
