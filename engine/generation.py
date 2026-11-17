"""Story continuation, steered inside the model. The story so far is read without steering, so what is already written is
encoded exactly as the reader sees it; from the first generated token on, the future knobs' vectors are added to the
residual stream and the attention boxes' tokens get a bias in every full-attention layer's scores. Tokens are sampled
after a repetition penalty, at a temperature, within the top-p nucleus."""

import re

import mlx.core as mx
from mlx_lm.sample_utils import make_logits_processors

from model import Engine, Instruments, fork_cache, positions_in
from steering import Vectors

PREFILL_CHUNK = 256
MAX_TOKENS = 360
# Past this many tokens, generation stops at the next sentence end.
SOFT_MAX_TOKENS = 220
# Added to the attention scores of an attention box's tokens, and on top of that to its keyword's, in log space:
# e^1.5 is about 4.5 times the weight before normalization.
PHRASE_BIAS = 1.5
KEYWORD_BIAS = 1.0
TEMPERATURE = 0.9
TOP_P = 0.95
REPETITION_PENALTY = 1.1
REPETITION_CONTEXT = 64
# Candidates a compared step reports from each of the steered and unsteered distributions.
TOP_WORDS = 5
SENTENCE_END = re.compile(r"[.!?][\"”’]?\s*$")


def complete(engine: Engine, vectors: Vectors, prefix: str, leads: list, controls: list, compare: bool = False):
    """Yields the continuation of `prefix` as {"text"} items, one per token, and None between prefill steps, so a scheduler
    can interleave.

    With `compare`, an unsteered copy of the model runs beside the steered one as a second row of the batch: row 0 is steered
    and writes the story; row 1 is unsteered and reads exactly what row 0 writes, so each token's probability can be compared
    with and without steering on the same text. Both rows sample from the same random draw, so they pick the same token
    wherever steering does not change the odds; row 1's pick is what the unsteered model would have written in row 0's place,
    so a token steering flipped is one where the two differ. Row 1's pick is then discarded and it reads row 0's. Each item
    then has a "step": `candidates` under "top" and, where steering flipped the token, the unsteered pick under "instead"."""
    story = prefix.rstrip()
    ids, spans, _ = engine.story_tokens(story)
    cache = engine.make_cache()
    for start in range(0, len(ids) - 1, PREFILL_CHUNK):
        engine.forward(mx.array([ids[start : min(start + PREFILL_CHUNK, len(ids) - 1)]]), cache)
        mx.eval([c.state for c in cache])
        yield None

    adds = vectors.residual_adds(controls, story)
    bias = lock_bias(lead_positions(leads, story, spans), len(ids))
    if compare:
        cache = fork_cache(cache, 2)
        adds = {layer: mx.stack([add, mx.zeros_like(add)])[:, None, :] for layer, add in adds.items()}
        bias = None if bias is None else mx.concatenate([bias, mx.zeros_like(bias)])
    instruments = Instruments(residual_add=adds, attention_bias=bias)
    [penalty] = make_logits_processors(repetition_penalty=REPETITION_PENALTY, repetition_context_size=REPETITION_CONTEXT)
    writer = Writer(engine, needs_space=bool(story) and prefix == story)
    history = list(ids)
    rows = 2 if compare else 1
    for step in range(MAX_TOKENS):
        logits = engine.forward(mx.array([[history[-1]]] * rows), cache, instruments)[:, -1, :].astype(mx.float32)
        picks, odds = coupled_sample(penalty(mx.array(history), logits))
        token, *unsteered = picks.tolist()
        if token in engine.stop_ids:
            break
        history.append(token)
        item = {"text": writer.add(token)}
        if compare:
            instead = engine.tokenizer.decode(unsteered) if unsteered != [token] else None
            item["step"] = {"top": candidates(engine, odds, token), "instead": instead}
        yield item
        if step >= SOFT_MAX_TOKENS and SENTENCE_END.search(writer.text):
            break


def coupled_sample(logits):
    """One token per row, sampled at `TEMPERATURE` within the top-p nucleus, by Gumbel-max with one noise vector shared by the
    rows: rows with the same distribution pick the same token, and rows whose distributions differ pick differently only where
    the difference is large enough to move the winner. Returns the picks and each row's odds they were drawn from: the
    nucleus, renormalized, and zero outside it."""
    scaled = logits / TEMPERATURE
    probabilities = mx.softmax(scaled, axis=-1)
    ordered = -mx.sort(-probabilities, axis=-1)
    before = mx.cumsum(ordered, axis=-1) - ordered
    floor = mx.where(before < TOP_P, ordered, mx.inf).min(axis=-1, keepdims=True)
    nucleus = probabilities >= floor
    noise = -mx.log(-mx.log(mx.random.uniform(low=1e-9, high=1.0, shape=(logits.shape[-1],))))
    odds = mx.where(nucleus, probabilities, 0.0)
    return mx.argmax(mx.where(nucleus, scaled + noise, -mx.inf), axis=-1), odds / odds.sum(axis=-1, keepdims=True)


def candidates(engine: Engine, odds, chosen: int):
    """The likeliest tokens at this step under the steered row (0) and the unsteered row that read the same text (1), and the
    chosen one, each with the odds it had of being drawn under both, likeliest when steered first. A nucleus can hold fewer
    than `TOP_WORDS` tokens; a token neither row could have drawn is left out."""
    tokens = {chosen}
    for row in odds:
        tokens |= set(mx.argpartition(-row, TOP_WORDS)[:TOP_WORDS].tolist())
    tokens = list(tokens)
    decode = engine.tokenizer.decode
    listed = [
        {"token": decode([token]), "on": round(on, 6), "off": round(off, 6), "chosen": token == chosen}
        for token, on, off in zip(tokens, *odds[:, mx.array(tokens)].tolist())
        if on or off
    ]
    return sorted(listed, key=lambda candidate: (-candidate["on"], -candidate["off"]))


class Writer:
    """The text of a growing list of tokens, handed out in pieces as it grows. A token that ends partway through a character
    adds nothing until the rest arrives. Leading whitespace is kept only if the story it continues needs a space after it."""

    def __init__(self, engine: Engine, needs_space: bool):
        self.tokenizer = engine.tokenizer._tokenizer
        self.needs_space = needs_space
        self.tokens = []
        self.text = ""

    def add(self, token: int):
        self.tokens.append(token)
        text = self.tokenizer.decode(self.tokens, clean_up_tokenization_spaces=False)
        if text.endswith("�"):
            return ""
        if not self.needs_space:
            text = text.lstrip()
        piece = text[len(self.text) :]
        self.text = text
        return piece


def lead_positions(leads: list, story: str, spans: dict):
    """Token positions of the attention boxes' phrases and of their keywords."""
    phrase_positions, keyword_positions = set(), set()
    for phrase_span, keyword_span in filter(None, (lead_spans(lead, story) for lead in leads)):
        phrase_positions |= positions_in(spans, phrase_span)
        keyword_positions |= positions_in(spans, keyword_span)
    return phrase_positions, keyword_positions


def lead_spans(lead: dict, text: str):
    """Character spans of an attention box's phrase and of its keyword, at the phrase's last occurrence in `text`; the keyword's
    for both if the phrase is no longer there, and None if the keyword is not either."""
    phrase = lead["before"] + lead["keyword"] + lead["after"]
    at = text.rfind(phrase)
    if at != -1:
        start = at + len(lead["before"])
        return (at, at + len(phrase)), (start, start + len(lead["keyword"]))
    at = text.rfind(lead["keyword"])
    if at == -1:
        return None
    keyword_span = (at, at + len(lead["keyword"]))
    return keyword_span, keyword_span


def lock_bias(positions: tuple, length: int):
    phrase_positions, keyword_positions = positions
    if not phrase_positions:
        return None
    bias = [0.0] * length
    for p in phrase_positions:
        bias[p] += PHRASE_BIAS
    for p in keyword_positions:
        bias[p] += KEYWORD_BIAS
    return mx.array(bias)[None, None, None, :]
