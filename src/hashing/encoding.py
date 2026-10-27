from __future__ import annotations

import numpy as np


def generate_pseudo_random_bits(key: int, length: int) -> list[int]:
    rng = np.random.RandomState(key)
    return rng.randint(0, 2, length).astype(int).tolist()


def xor_bits(bits: list[int], key_bits: list[int]) -> list[int]:
    return [bit ^ key_bit for bit, key_bit in zip(bits, key_bits)]


def bits_to_string(bits: list[int]) -> str:
    return "".join(str(int(bit)) for bit in bits)


def normalize_to_bits(values, num_bits_per_value: int) -> list[int]:
    values_array = np.asarray(values, dtype=np.float32)
    if values_array.size == 0:
        return []
    values_array = np.nan_to_num(values_array, nan=0.0, posinf=0.0, neginf=0.0)
    min_value = float(np.min(values_array))
    max_value = float(np.max(values_array))
    if max_value - min_value == 0:
        normalized = np.zeros_like(values_array)
    else:
        normalized = (values_array - min_value) / (max_value - min_value)

    max_code = 2**num_bits_per_value - 1
    bits: list[int] = []
    for value in normalized:
        binary = bin(int(value * max_code))[2:].zfill(num_bits_per_value)
        bits.extend(int(bit) for bit in binary)
    return bits


def encode_signed_decimal_17bit(value: float) -> str:
    sign = 0 if value >= 0 else 1
    abs_value = abs(float(value))
    value_str = f"{abs_value:.4f}".replace(".", "")[:4]
    value_str = value_str.ljust(4, "0")
    digit_binaries = [f"{int(char):04b}" for char in value_str]
    return f"{sign} {' '.join(digit_binaries)}"
