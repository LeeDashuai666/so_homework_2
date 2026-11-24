"""Bit-preserving kernels for BF16 static KV on CUDA."""
from types import MethodType

import torch
import triton
import triton.language as tl


@triton.jit
def _append(K, V, KS, VS, C, HEADS: tl.constexpr, DIM: tl.constexpr,
            CAP: tl.constexpr, K_H: tl.constexpr, K_D: tl.constexpr,
            V_H: tl.constexpr, V_D: tl.constexpr, BLOCK: tl.constexpr):
    # A single program owns the two vectors and this layer's cursor.
    slot = tl.load(C)
    lane = tl.arange(0, BLOCK)
    head = lane // DIM
    dim = lane % DIM
    good = (head < HEADS) & (slot >= 0) & (slot < CAP)
    k = tl.load(K + head * K_H + dim * K_D, mask=good, other=0)
    v = tl.load(V + head * V_H + dim * V_D, mask=good, other=0)
    out = (head * CAP + slot) * DIM + dim
    tl.store(KS + out, k, mask=good)
    tl.store(VS + out, v, mask=good)
    tl.store(C, slot + 1, mask=(slot >= 0) & (slot < CAP))


@triton.jit
def _restore(P, W, LENGTH, CAP: tl.constexpr, DIM: tl.constexpr,
             SIZE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    group = i // (CAP * DIM)
    within = i % (CAP * DIM)
    content = tl.load(P + group * LENGTH * DIM + within,
                      mask=(i < SIZE) & (within < LENGTH * DIM), other=0)
    tl.store(W + i, content, mask=i < SIZE)


def bind_fused_updates(cache):
    """Keep each batch-prefill call on the original StaticLayer implementation."""
    for layer in cache.layers:
        layer.original_update = layer.update
        layer.update = MethodType(update, layer)
        layer.fuse_append = False


def update(layer, key, value, *args, **kwargs):
    if not layer.fuse_append or key.shape[-2] != 1:
        return layer.original_update(key, value, *args, **kwargs)
    assert key.shape == value.shape and key.shape[0] == 1
    assert layer.keys.is_contiguous() and layer.values.is_contiguous()
    heads, capacity, width = layer.keys.shape[1:]
    _append[(1,)](key, value, layer.keys, layer.values, layer.cumulative_length,
                  heads, width, capacity, key.stride(1), key.stride(3),
                  value.stride(1), value.stride(3), triton.next_power_of_2(heads * width))
    return layer.keys, layer.values


class Arena:
    """Bind the same StaticCache layer API to views into one backing allocation."""
    def __init__(self, work):
        first = work.layers[0]
        _, heads, capacity, width = first.keys.shape
        self.data = torch.zeros((len(work.layers), 2, heads, capacity, width),
                                device=first.keys.device, dtype=first.keys.dtype)
        self.cursors = torch.zeros(len(work.layers), device=first.keys.device,
                                   dtype=first.cumulative_length.dtype)
        for i, layer in enumerate(work.layers):
            layer.keys = self.data[i, 0].unsqueeze(0)
            layer.values = self.data[i, 1].unsqueeze(0)
            layer.cumulative_length = self.cursors[i:i+1]
        self.capacity = capacity
        self.width = width

    def reset(self):
        self.data.zero_()
        self.cursors.zero_()

    def save(self, length):
        return self.data[..., :length, :].clone(memory_format=torch.contiguous_format)

    def restore(self, prefix, length):
        assert prefix.shape[-2] == length and 0 < length <= self.capacity
        assert prefix.is_contiguous()
        _restore[(triton.cdiv(self.data.numel(), 2048),)](
            prefix, self.data, length, self.capacity, self.width, self.data.numel(), 2048,
            num_warps=4,
        )
        self.cursors.fill_(length)


@torch.inference_mode()
def check_arena(arena):
    """Exercise the exact GPU kernels with stale buffers and strided BF16 input."""
    addresses = (arena.data.data_ptr(), arena.cursors.data_ptr())
    arena.data.fill_(3)
    arena.cursors.fill_(35)
    length = 7
    pattern = torch.arange(arena.data[..., :length, :].numel(), device="cuda") % 251
    prefix = pattern.reshape(arena.data[..., :length, :].shape).to(torch.bfloat16)
    before = prefix.clone()
    arena.restore(prefix, length)
    assert torch.equal(arena.data[..., :length, :], before)
    assert not arena.data[..., length:, :].count_nonzero().item()
    assert arena.cursors.tolist() == [length] * arena.data.shape[0]
    head, width = arena.data.shape[2], arena.width
    key = torch.arange(head * width * 2, device="cuda").reshape(head, -1).to(torch.bfloat16)[:, ::2]
    value = -key
    target = arena.data[0]
    cursor = arena.cursors[:1]
    for slot in (0, length, arena.capacity - 1, arena.capacity):
        cursor.fill_(slot)
        _append[(1,)](key, value, target[0], target[1], cursor, head, width, arena.capacity,
                      key.stride(0), key.stride(1), value.stride(0), value.stride(1),
                      triton.next_power_of_2(head * width))
        assert cursor.item() == min(slot + 1, arena.capacity)
        if slot < arena.capacity:
            assert torch.equal(target[0, :, slot, :], key)
            assert torch.equal(target[1, :, slot, :], value)
    assert torch.equal(prefix, before)
    assert (arena.data.data_ptr(), arena.cursors.data_ptr()) == addresses
    arena.reset()
    return {"stale_tail_cleared": True, "persistent_snapshot_immutable": True,
            "fixed_addresses": True, "strided_append_slots": [0, length, arena.capacity - 1],
            "full_capacity_guard": arena.capacity}
