"""LFM3 (Liquid AI) for mlx-lm.

Adapted from mlx-optiq 0.8.3 (optiq/models/lfm3.py), MIT License, Copyright (c) 2026 Thin Signal. Changes from the
original: the model's `full_mask` argument (bidirectional full attention for Liquid's decision models) and the
`install()` helper are removed, and docstrings are reworded. Registered as mlx_lm.models.lfm3 by model.py because
released mlx-lm has no lfm3 model.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions: The above copyright notice and this
permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

LFM3 is LFM2 with four changes; LFM2's short convolution and SwiGLU are reused:

* a third layer type, sliding_attention (window sliding_window), beside conv and
  full_attention, each attention type with its own RoPE base;
* a sigmoid output gate per attention head (gate_proj: hidden -> heads);
* sandwich norms (post_operator_norm, post_ffn_norm) when use_sandwich_norm is set;
* partial RoPE in Liquid's zero-tail layout: the first partial_rotary_factor * head_dim / 2
  frequencies keep the progression of the full head dimension and the rest are zero, so
  those dimensions are not rotated. That differs from mlx's partial RoPE, so it is passed to
  mx.fast.rope as explicit periods, infinite for the unrotated tail.
"""

from dataclasses import dataclass
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache
from mlx_lm.models.lfm2 import MLP, ShortConv

ATTENTION_TYPES = ("full_attention", "sliding_attention")


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    norm_eps: float
    conv_bias: bool
    conv_L_cache: int
    layer_types: List[str]
    rope_parameters: dict
    head_dim: Optional[int] = None
    sliding_window: Optional[int] = None
    use_sandwich_norm: bool = False
    tie_word_embeddings: bool = True
    max_position_embeddings: int = 262144

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads


def rope_periods(args: ModelArgs, layer_type: str) -> mx.array:
    """mx.fast.rope periods (1 / inverse frequency) for one attention type, in the
    zero-tail layout of Lfm3RotaryEmbedding."""
    params = args.rope_parameters[layer_type]
    if params.get("rope_type", "default") != "default":
        raise ValueError(f"LFM3 rope_type {params['rope_type']!r} is not supported")
    dim = args.head_dim
    angles = int(params.get("partial_rotary_factor", 1.0) * dim) // 2
    rotated = params["rope_theta"] ** (mx.arange(0, 2 * angles, 2, dtype=mx.float32) / dim)
    periods = mx.concatenate([rotated, mx.full((dim // 2 - angles,), mx.inf, dtype=mx.float32)])
    # Evaluated now, on the loading thread: it is not a parameter, so nothing else
    # materializes it, and a lazy array evaluated on another thread (a server's
    # request thread) aborts with "There is no Stream(gpu, N) in current thread".
    mx.eval(periods)
    return periods


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_type: str):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.norm_eps)
        self.gate_proj = nn.Linear(dim, self.n_heads, bias=False)
        self._periods = rope_periods(args, layer_type)

    def _rope(self, x: mx.array, offset) -> mx.array:
        return mx.fast.rope(x, self.head_dim, traditional=False, base=None, scale=1.0,
                            offset=offset, freqs=self._periods)

    def __call__(self, x: mx.array, mask=None, cache: Optional[Any] = None) -> mx.array:
        B, L, _ = x.shape
        q = self.q_norm(self.q_proj(x).reshape(B, L, self.n_heads, -1)).transpose(0, 2, 1, 3)
        k = self.k_norm(self.k_proj(x).reshape(B, L, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        offset = cache.offset if cache is not None else 0
        q, k = self._rope(q, offset), self._rope(k, offset)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        out = scaled_dot_product_attention(q, k, v, cache=cache, mask=mask, scale=self.scale)
        out = out.transpose(0, 2, 1, 3)                                  # B, L, H, D
        gate = mx.sigmoid(self.gate_proj(x))[..., None]                   # B, L, H, 1
        return self.o_proj((out * gate).reshape(B, L, -1))


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_type = args.layer_types[layer_idx]
        self.is_attention_layer = self.layer_type in ATTENTION_TYPES
        if self.is_attention_layer:
            self.self_attn = Attention(args, self.layer_type)
        else:
            self.conv = ShortConv(args, layer_idx)
        self.feed_forward = MLP(args.hidden_size, args.intermediate_size, multiple_of=1,
                                auto_adjust_ff_dim=False, ffn_dim_multiplier=None)
        self.operator_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)
        self.ffn_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)
        self.use_sandwich_norm = args.use_sandwich_norm
        if self.use_sandwich_norm:
            self.post_operator_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)
            self.post_ffn_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)

    def __call__(self, x: mx.array, mask=None, cache: Optional[Any] = None) -> mx.array:
        if self.is_attention_layer:
            r = self.self_attn(self.operator_norm(x), mask=mask, cache=cache)
        else:
            r = self.conv(self.operator_norm(x), mask=mask, cache=cache)
        if self.use_sandwich_norm:
            r = self.post_operator_norm(r)
        h = x + r
        f = self.feed_forward(self.ffn_norm(h))
        if self.use_sandwich_norm:
            f = self.post_ffn_norm(f)
        return h + f


class Lfm3Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.embedding_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)
        # The first layer of each type, whose cache sizes that type's mask.
        self.first = {t: args.layer_types.index(t) for t in set(args.layer_types)}

    def __call__(self, inputs: mx.array, cache=None, input_embeddings: Optional[mx.array] = None):
        h = input_embeddings if input_embeddings is not None else self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)

        masks = {}
        if "full_attention" in self.first:
            masks["full_attention"] = create_attention_mask(h, cache[self.first["full_attention"]])
        if "sliding_attention" in self.first:
            masks["sliding_attention"] = create_attention_mask(
                h, cache[self.first["sliding_attention"]], window_size=self.args.sliding_window)
        if "conv" in self.first:
            masks["conv"] = create_ssm_mask(h, cache[self.first["conv"]])

        for layer, c in zip(self.layers, cache):
            h = layer(h, masks[layer.layer_type], cache=c)
        return self.embedding_norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Lfm3Model(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache=None, input_embeddings: Optional[mx.array] = None):
        out = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    def sanitize(self, weights):
        out = {}
        for name, param in weights.items():
            if self.args.tie_word_embeddings and name == "lm_head.weight":
                continue
            # PyTorch's depthwise conv is [out, 1, k]; mlx's is [out, k, 1].
            if "conv.weight" in name and param.shape[-1] > param.shape[1]:
                param = param.transpose(0, 2, 1)
            out[name] = param
        return out

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        caches = []
        for layer in self.layers:
            if layer.layer_type == "sliding_attention":
                caches.append(RotatingKVCache(max_size=self.args.sliding_window))
            elif layer.layer_type == "full_attention":
                caches.append(KVCache())
            else:
                caches.append(ArraysCache(size=1))
        return caches
