"""Loads a model on MLX (LFM3 or Qwen3.5) with instruments on its layers: a vector added to a layer's output (activation
steering), an additive bias on attention scores over chosen key positions (attention steering), and captures of layer
outputs and attention weights (readouts). One `Instruments` value configures a forward pass and receives its captures."""

import copy
import sys
from dataclasses import dataclass, field

import mlx.core as mx
from mlx import nn
from mlx_lm import load

import lfm3

# Released mlx-lm has no LFM3, so its loader is given ours under the module name it looks a model up by.
sys.modules["mlx_lm.models.lfm3"] = lfm3

SYSTEM_PROMPT = (
    "You are a novelist. The assistant turn is a story in progress, cut off where the reader is. "
    "Continue it from its last word in plain prose: no title, markdown or commentary."
)


@dataclass
class Instruments:
    """What one forward pass adds and records. Arrays broadcast over the batch."""

    residual_add: dict = field(default_factory=dict)
    """Layer index -> vector added to that layer's output, shape (H,) or (B, 1, H)."""
    attention_bias: mx.array | None = None
    """Additive bias on every full-attention layer's scores, shape (B or 1, 1, 1, P) over key positions 0..P-1; later keys get none."""
    capture_residual: tuple = ()
    """Layer indices whose outputs are stored in `residuals`."""
    capture_attention: bool = False
    """Store each full-attention layer's attention weights, averaged over heads by their output gates, in `attention`."""
    residuals: dict = field(default_factory=dict)
    attention: list = field(default_factory=list)


NO_INSTRUMENTS = Instruments()


class Engine:
    """A loaded model, from the folder its weights are in. Every layer is wrapped to steer and capture its output, and every
    full-attention layer's attention to bias and capture its weights; the layers between those (LFM3's convolutions and
    sliding-window attention, Qwen3.5's linear attention) keep their own."""

    def __init__(self, model_id: str, path: str):
        self.id = model_id
        self.model, self.tokenizer = load(path, tokenizer_config={"trust_remote_code": False})
        text_model = getattr(self.model, "language_model", self.model)
        self.hidden_size = text_model.args.hidden_size
        layers = text_model.model.layers
        # The second quarter of the layers, past where the model is still reading tokens and before it settles on words.
        self.steer_layers = tuple(range(len(layers) // 4, len(layers) // 2))
        self.instruments = NO_INSTRUMENTS
        for i, layer in enumerate(layers):
            if is_full_attention(layer):
                layer.self_attn = InstrumentedAttention(layer.self_attn, self)
            layers[i] = InstrumentedLayer(layer, i, self)
        self.template_ids = self._template_ids(SYSTEM_PROMPT)
        self.stop_ids = set(self.tokenizer.eos_token_ids) | {self.tokenizer.convert_tokens_to_ids("<|im_end|>")}

    def _template_ids(self, system: str):
        messages = [{"role": "system", "content": system}, {"role": "user", "content": "Continue the story."}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return self.tokenizer._tokenizer(text, add_special_tokens=False)["input_ids"]

    def story_tokens(self, story: str, system: str | None = None):
        """The chat prompt with `story` as the start of the assistant's turn: token ids, each story token's character span
        within `story` keyed by its position in the ids, and the position of the first story token. A custom `system` prompt
        replaces the default one."""
        template = self.template_ids if system is None else self._template_ids(system)
        encoded = self.tokenizer._tokenizer(story, add_special_tokens=False, return_offsets_mapping=True)
        ids = template + encoded["input_ids"]
        spans = {len(template) + i: span for i, span in enumerate(encoded["offset_mapping"])}
        return ids, spans, len(template)

    def forward(self, ids, cache=None, instruments: Instruments = NO_INSTRUMENTS):
        """Logits for `ids` (a batch, shape (B, L)) with `instruments` active for this pass only."""
        self.instruments = instruments
        try:
            return self.model(ids, cache=cache)
        finally:
            self.instruments = NO_INSTRUMENTS

    def make_cache(self):
        return self.model.make_cache()


class InstrumentedLayer(nn.Module):
    def __init__(self, inner, index: int, engine: Engine):
        super().__init__()
        self.inner = inner
        self.index = index
        self._engine = engine
        # The model's own loop reads these off each layer, to pick its mask and its kind of cache.
        for name in ("layer_type", "is_linear"):
            if hasattr(inner, name):
                setattr(self, name, getattr(inner, name))

    def __call__(self, x, mask=None, cache=None):
        out = self.inner(x, mask=mask, cache=cache)
        instruments = self._engine.instruments
        add = instruments.residual_add.get(self.index)
        if add is not None:
            out = out + add.astype(out.dtype)
        if self.index in instruments.capture_residual:
            instruments.residuals[self.index] = out
        return out


class InstrumentedAttention(nn.Module):
    """Full attention computed as in lfm3.Attention or mlx-lm's Qwen3NextAttention, with the bias and capture of the engine's
    current instruments. Both gate their output with a sigmoid: LFM3 one number per head, Qwen3.5 one per channel, packed
    into its query projection."""

    def __init__(self, inner, engine: Engine):
        super().__init__()
        self.inner = inner
        self._engine = engine

    def __call__(self, x, mask=None, cache=None):
        a = self.inner
        B, L, _ = x.shape
        if hasattr(a, "gate_proj"):
            q = a.q_norm(a.q_proj(x).reshape(B, L, a.n_heads, -1))
            gate = mx.sigmoid(a.gate_proj(x))[..., None]
            kv_heads, rope = a.n_kv_heads, a._rope
        else:
            q, gate = mx.split(a.q_proj(x).reshape(B, L, a.num_attention_heads, -1), 2, axis=-1)
            q, gate = a.q_norm(q), mx.sigmoid(gate)
            kv_heads, rope = a.num_key_value_heads, lambda t, offset: a.rope(t, offset=offset)
        q = q.transpose(0, 2, 1, 3)
        k = a.k_norm(a.k_proj(x).reshape(B, L, kv_heads, -1)).transpose(0, 2, 1, 3)
        v = a.v_proj(x).reshape(B, L, kv_heads, -1).transpose(0, 2, 1, 3)
        offset = cache.offset if cache is not None else 0
        q, k = rope(q, offset), rope(k, offset)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        instruments = self._engine.instruments
        bias = instruments.attention_bias
        if bias is not None or instruments.capture_attention:
            S = k.shape[2]
            mask = additive_mask(mask, L, S, q.dtype)
            if bias is not None:
                mask = mask + padded(bias, S).astype(q.dtype)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=a.scale, mask=mask)
        if instruments.capture_attention:
            instruments.attention.append(gated_weights(q, k, mask, a.scale, gate.mean(axis=-1)))
        return a.o_proj((out.transpose(0, 2, 1, 3) * gate).reshape(B, L, -1))


def is_full_attention(layer):
    """LFM3 names each layer's type; in Qwen3.5 every layer that has `self_attn` attends over the whole context."""
    return hasattr(layer, "self_attn") and getattr(layer, "layer_type", "full_attention") == "full_attention"


def additive_mask(mask, L: int, S: int, dtype):
    """Any mask the model passes (None, "causal", or a boolean or additive array) as an additive array of shape (.., L, S)."""
    if mask is None:
        return mx.zeros((1, 1, L, S), dtype=dtype)
    if isinstance(mask, str):
        rows = mx.arange(L)[:, None] + (S - L)
        allowed = mx.arange(S)[None, :] <= rows
        return mx.where(allowed, 0.0, -mx.inf).astype(dtype)[None, None]
    if mask.dtype == mx.bool_:
        return mx.where(mask, 0.0, -mx.inf).astype(dtype)
    return mask.astype(dtype)


def padded(bias, S: int):
    P = bias.shape[-1]
    if P >= S:
        return bias[..., :S]
    return mx.concatenate([bias, mx.zeros((*bias.shape[:-1], S - P), dtype=bias.dtype)], axis=-1)


def gated_weights(q, k, mask, scale, gate):
    """Attention weights averaged over heads, each head weighted by its output gate (`gate`, shape (B, L, H)): how much each
    key feeds the output. Returns shape (B, L, S)."""
    repeats = q.shape[1] // k.shape[1]
    keys = mx.repeat(k, repeats, axis=1)
    scores = (q.astype(mx.float32) * scale) @ keys.astype(mx.float32).transpose(0, 1, 3, 2) + mask.astype(mx.float32)
    weights = mx.softmax(scores, axis=-1)
    head_gate = gate.astype(mx.float32).transpose(0, 2, 1)[..., None]
    return (weights * head_gate).sum(axis=1) / head_gate.sum(axis=1)


def positions_in(spans: dict, span: tuple):
    """The token positions, from `story_tokens`' spans, whose characters overlap the character span."""
    start, end = span
    return {position for position, (s, e) in spans.items() if s < end and e > start}


def fork_cache(cache, batch: int):
    """Independent copies of a batch-1 cache, stacked `batch` times, so several continuations can branch from one prefill."""
    forked = []
    for c in cache:
        f = copy.copy(c)
        if hasattr(c, "keys") and c.keys is not None:
            f.keys = mx.repeat(c.keys, batch, axis=0)
            f.values = mx.repeat(c.values, batch, axis=0)
        if hasattr(c, "cache"):
            f.cache = [None if a is None else mx.repeat(a, batch, axis=0) for a in c.cache]
        forked.append(f)
    return forked
