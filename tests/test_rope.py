import pytest
import torch

from miniqwen.rope import RoPE


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def test_position_zero_does_not_change_activations():
    rope = RoPE(theta=10_000.0, head_dim=8, max_seq_len=16)
    q = torch.randn(2, 4, 1, 8)
    k = torch.randn(2, 2, 1, 8)

    rotated_q, rotated_k = rope(q, k, torch.zeros(1, 1, dtype=torch.long))

    torch.testing.assert_close(rotated_q, q)
    torch.testing.assert_close(rotated_k, k)

def test_matches_reference_with_different_head_counts_and_broadcasting():
    theta = 10_000.0
    head_dim = 8
    rope = RoPE(theta=theta, head_dim=head_dim, max_seq_len=8, dtype=torch.float32)
    q = torch.randn(2, 4, 3, head_dim, dtype=torch.float64)
    k = torch.randn(2, 2, 3, head_dim, dtype=torch.float64)
    position_ids = torch.tensor([[0, 2, 7]])

    rotated_q, rotated_k = rope(q, k, position_ids)

    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2) / head_dim)
    )
    freqs = position_ids.float().unsqueeze(-1) * inv_freq
    emb = torch.cat((freqs, freqs), dim=-1).unsqueeze(1)
    expected_q = (q * emb.cos()) + (rotate_half(q) * emb.sin())
    expected_k = (k * emb.cos()) + (rotate_half(k) * emb.sin())

    assert rotated_q.shape == q.shape
    assert rotated_k.shape == k.shape
    assert rotated_q.dtype == q.dtype
    assert rotated_k.dtype == k.dtype
    torch.testing.assert_close(rotated_q, expected_q)
    torch.testing.assert_close(rotated_k, expected_k)

def test_precomputes_cos_and_sin_for_the_maximum_sequence_length():
    rope = RoPE(theta=10_000.0, head_dim=8, max_seq_len=16, dtype=torch.float32)

    assert rope.cos.shape == (16, 8)
    assert rope.sin.shape == (16, 8)
    assert rope.cos.dtype == torch.float32
    assert rope.sin.dtype == torch.float32
    assert not rope.cos.requires_grad
    assert not rope.sin.requires_grad

def test_rejects_positions_beyond_the_precomputed_length():
    rope = RoPE(theta=10_000.0, head_dim=8, max_seq_len=2, dtype=torch.float32)
    q = torch.randn(1, 4, 1, 8)
    k = torch.randn(1, 2, 1, 8)

    with pytest.raises(IndexError):
        rope(q, k, torch.tensor([[2]]))
