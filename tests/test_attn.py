import math

import pytest
import torch
import torch.testing
from torch.nn.functional import scaled_dot_product_attention

from miniqwen.attn import _flash_gqa


def test_flash_gqa():
    B = 4
    Hq = 16
    Hkv = 16
    Sq = 1000
    Skv = 900
    D = 128

    Q = torch.randn(B, Hq, Sq, D, dtype=torch.float16, device="cuda")
    K = torch.randn(B, Hkv, Skv, D, dtype=torch.float16, device="cuda")
    V = torch.randn(B, Hkv, Skv, D, dtype=torch.float16, device="cuda")
    scale = 1 / math.sqrt(D)

    O_flash = _flash_gqa(Q, K, V, is_causal=False, scale=scale)
    O_torch = scaled_dot_product_attention(
        Q, K, V, is_causal=False, scale=scale, enable_gqa=True
    )
    assert O_flash.shape == O_torch.shape
    torch.testing.assert_close(O_flash, O_torch, rtol=1e-2, atol=1e-2)


def test_flash_gqa_causal():
    B = 4
    Hq = 16
    Hkv = 16
    Sq = 1000
    Skv = 800
    D = 128

    Q = torch.randn(B, Hq, Sq, D, dtype=torch.float16, device="cuda")
    K = torch.randn(B, Hkv, Skv, D, dtype=torch.float16, device="cuda")
    V = torch.randn(B, Hkv, Skv, D, dtype=torch.float16, device="cuda")
    scale = 1 / math.sqrt(D)

    O_flash = _flash_gqa(Q, K, V, is_causal=True, scale=scale)
    O_torch = scaled_dot_product_attention(
        Q, K, V, is_causal=True, scale=scale, enable_gqa=True
    )
    assert O_flash.shape == O_torch.shape
    torch.testing.assert_close(O_flash, O_torch, rtol=1e-2, atol=1e-2)


@pytest.mark.benchmark(group="gqa")
def test_flash_gqa_bench(benchmark):
    args = _prepare_benchmark_args(is_causal=False)
    benchmark.pedantic(_flash_gqa, **args)


@pytest.mark.benchmark(group="gqa")
def test_torch_gqa_bench(benchmark):
    args = _prepare_benchmark_args(is_causal=False)
    args["kwargs"]["enable_gqa"] = True
    benchmark.pedantic(scaled_dot_product_attention, **args)


@pytest.mark.benchmark(group="gqa_causal")
def test_flash_gqa_causal_bench(benchmark):
    args = _prepare_benchmark_args(is_causal=True)
    benchmark.pedantic(_flash_gqa, **args)


@pytest.mark.benchmark(group="gqa_causal")
def test_torch_gqa_causal_bench(benchmark):
    args = _prepare_benchmark_args(is_causal=True)
    args["kwargs"]["enable_gqa"] = True
    benchmark.pedantic(scaled_dot_product_attention, **args)


def _prepare_benchmark_args(*, is_causal: bool):
    B = 4
    Hq = 16
    Hkv = 16
    Sq = 1000
    Skv = 900
    D = 128

    Q = torch.randn(B, Hq, Sq, D, dtype=torch.float16, device="cuda")
    K = torch.randn(B, Hkv, Skv, D, dtype=torch.float16, device="cuda")
    V = torch.randn(B, Hkv, Skv, D, dtype=torch.float16, device="cuda")
    scale = 1 / math.sqrt(D)

    return {
        "args": (Q, K, V),
        "kwargs": {"is_causal": is_causal, "scale": scale},
        "iterations": 50,
        "rounds": 10,
        "warmup_rounds": 10,
    }
