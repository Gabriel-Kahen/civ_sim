from __future__ import annotations

import math


def _hash_u32(*values: int) -> int:
    state = 0x9E3779B9
    for value in values:
        state ^= value + 0x9E3779B9 + ((state << 6) & 0xFFFFFFFF) + (state >> 2)
        state &= 0xFFFFFFFF
    state ^= state >> 15
    state = (state * 0x2C1B3C6D) & 0xFFFFFFFF
    state ^= state >> 12
    state = (state * 0x297A2D39) & 0xFFFFFFFF
    state ^= state >> 15
    return state


def hash_float(seed: int, x: int, y: int, z: int = 0) -> float:
    return _hash_u32(seed, x, y, z) / 0xFFFFFFFF


def smoothstep(value: float) -> float:
    return value * value * (3.0 - 2.0 * value)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def value_noise(seed: int, x: float, y: float, scale: float) -> float:
    x /= scale
    y /= scale
    x0 = math.floor(x)
    y0 = math.floor(y)
    tx = smoothstep(x - x0)
    ty = smoothstep(y - y0)
    v00 = hash_float(seed, x0, y0)
    v10 = hash_float(seed, x0 + 1, y0)
    v01 = hash_float(seed, x0, y0 + 1)
    v11 = hash_float(seed, x0 + 1, y0 + 1)
    a = lerp(v00, v10, tx)
    b = lerp(v01, v11, tx)
    return lerp(a, b, ty)


def fractal_noise(seed: int, x: float, y: float, scales: tuple[float, ...]) -> float:
    amplitude = 1.0
    total = 0.0
    amplitude_sum = 0.0
    for octave, scale in enumerate(scales):
        total += value_noise(seed + octave * 997, x, y, scale) * amplitude
        amplitude_sum += amplitude
        amplitude *= 0.5
    return total / max(amplitude_sum, 1e-6)


def warped_noise(seed: int, x: float, y: float, base_scale: float, warp_scale: float) -> float:
    warp_x = fractal_noise(seed + 17, x, y, (warp_scale, warp_scale * 0.5))
    warp_y = fractal_noise(seed + 43, x, y, (warp_scale, warp_scale * 0.5))
    return fractal_noise(
        seed,
        x + (warp_x - 0.5) * base_scale,
        y + (warp_y - 0.5) * base_scale,
        (base_scale, base_scale * 0.5, base_scale * 0.25),
    )
