"""Readouts of what the model is doing, from its own attention weights and likelihoods rather than from asking it."""

import math
import re

import mlx.core as mx

from generation import lead_spans
from model import Engine, Instruments, positions_in
from steering import Vectors

# Greedy tokens the model is run ahead for: the attention boxes show where those next words' attention goes.
FOCUS_STEPS = 6
MAX_BOXES = 3
# A box after the first is kept only if its keyword draws at least this share of the first's attention.
SECOND_BOX_SHARE = 0.5
PHRASE_REACH = 2
# Attention per keyword token, as a multiple of the average story token's, that counts as fully relevant.
FULL_RELEVANCE_RATIO = 12.0
# Per-token log-likelihood gap between the toward- and away-steered model, beyond the story so far's own gap, that reads
# as a clear yes (about 73%).
REALIZATION_GAP_SCALE = 0.3

STOPWORDS = set(
    """a an the and or but so if then than that this these those there here of in on at to for from with by as into onto upon
    about over under after before again is was were be been being am are has have had do does did not no nor it its he him his
    she her hers they them their we us our i me my you your yours who whom whose which what when where why how all any each
    few more most other some such only own same too very can will just would could should may might must shall up down out off
    one""".split()
)
WORD = re.compile(r"[\w’']+")
CLAUSE_BREAK = re.compile(r"[.!?;:,—\"“”()]")


def focus_boxes(engine: Engine, text: str):
    """The 1 to 3 phrases of `text` that the model's next few words attend to most, as character spans of the phrase and its
    keyword. Attention is read from the full-attention layers, averaged over heads by their output gates."""
    story = text.rstrip()
    if not story:
        return []
    ids, spans, _ = engine.story_tokens(story)
    cache = engine.make_cache()
    engine.forward(mx.array([ids[:-1]]), cache)
    token = ids[-1]
    scores = mx.zeros((len(ids),))
    for _ in range(FOCUS_STEPS):
        instruments = Instruments(capture_attention=True)
        logits = engine.forward(mx.array([[token]]), cache, instruments)
        scores = scores + mx.stack([w[0, -1, : len(ids)] for w in instruments.attention]).mean(axis=0)
        token = mx.argmax(logits[0, -1]).item()
        if token in engine.stop_ids:
            break
    token_scores = scores.tolist()

    words = [(m.start(), m.end(), m.group(0)) for m in WORD.finditer(story)]
    word_scores = [sum(token_scores[p] for p, (s, e) in spans.items() if s < end and e > start) for start, end, _ in words]
    candidates = sorted(
        (i for i, (_, _, word) in enumerate(words) if word.lower() not in STOPWORDS), key=lambda i: word_scores[i], reverse=True
    )
    boxes, covered = [], set()
    for i in candidates:
        if len(boxes) == MAX_BOXES or (boxes and word_scores[i] < SECOND_BOX_SHARE * boxes[0]["score"]):
            break
        if i in covered:
            continue
        first, last = phrase_around(story, words, i)
        if covered & set(range(first, last + 1)):
            continue
        covered |= set(range(first - 1, last + 2))
        box = {"start": words[first][0], "end": words[last][1], "keyStart": words[i][0]}
        boxes.append({**{key: utf16_offset(story, at) for key, at in box.items()}, "score": word_scores[i]})
    return boxes


def phrase_around(story: str, words: list, index: int):
    """Word indices of the phrase around a keyword: up to `PHRASE_REACH` words each side, within the keyword's clause."""
    first = last = index
    while first > 0 and index - first < PHRASE_REACH and not CLAUSE_BREAK.search(story[words[first - 1][1] : words[first][0]]):
        first -= 1
    while last < len(words) - 1 and last - index < PHRASE_REACH and not CLAUSE_BREAK.search(story[words[last][1] : words[last + 1][0]]):
        last += 1
    return first, last


def influence(engine: Engine, sentences: list, focus: dict):
    """How strongly each sentence feeds the focus, or is fed by it, measured as the average share of attention one side's
    tokens put on the other's: sentences before the focus by the focus's attention to them, sentences after it by their
    attention to the focus. Scaled so the strongest is 1."""
    text, sentence_spans = joined(sentences)
    focus_span = find_last(text, focus["text"].strip())
    if focus_span is None:
        text, (_, focus_span) = joined([text, focus["text"].strip()])
    ids, spans, _ = engine.story_tokens(text)
    weights = attention_matrix(engine, ids)
    focus_positions = positions_in(spans, focus_span)
    shares = []
    for span in sentence_spans:
        positions = positions_in(spans, span)
        if not positions or overlaps(span, focus_span):
            shares.append(None)
        elif span[1] <= focus_span[0]:
            shares.append(mean_share(weights, focus_positions, positions))
        else:
            shares.append(mean_share(weights, positions, focus_positions))
    strongest = max((s for s in shares if s is not None), default=0.0) or 1.0
    return [{"index": i, "strength": 1.0 if s is None else s / strongest} for i, s in enumerate(shares)]


def relevance(engine: Engine, story: str, tip: str, leads: list):
    """How much attention the tip sentence's tokens put on each attention box's keyword, per keyword token, against the
    average story token."""
    text, (_, tip_span) = joined([story, tip])
    ids, spans, start = engine.story_tokens(text)
    weights = attention_matrix(engine, ids)
    tip_positions = positions_in(spans, tip_span)
    story_positions = [p for p in range(start, len(ids)) if p not in tip_positions]
    baseline = mean_share(weights, tip_positions, story_positions) / max(1, len(story_positions))
    results = []
    for i, lead in enumerate(leads):
        found = lead_spans(lead, story)
        if found is None:
            continue
        keyword_positions = positions_in(spans, found[1])
        share = mean_share(weights, tip_positions, keyword_positions) / max(1, len(keyword_positions))
        ratio = share / baseline if baseline > 0 else 0.0
        results.append({"index": i, "relevance": min(1.0, math.log1p(ratio) / math.log1p(FULL_RELEVANCE_RATIO))})
    return results


def realization(engine: Engine, vectors: Vectors, context: str, sentences: list, asked: list, control: str):
    """Whether each asked sentence delivers the control, read as a likelihood contrast: the text is scored twice, once with the
    control's steering vector pushing toward the asked value and once pushing away from it, and a sentence the toward-steered
    model finds more likely, per token, delivers it. A midway scale asks for a sentence neither push explains better."""
    target = vectors.target_of(control)
    if target is None:
        return []
    direction, sign = target
    toward = sign or 1
    body, sentence_spans = joined(sentences)
    text, (_, (offset, _)) = joined([context, body])
    ids, spans, _ = engine.story_tokens(text)
    adds = vectors.batched([vectors.push(direction, toward), vectors.push(direction, -toward)])
    logits = engine.forward(mx.array([ids, ids]), instruments=Instruments(residual_add=adds))[:, :-1, :].astype(mx.float32)
    targets = mx.broadcast_to(mx.array(ids[1:])[None, :, None], (2, len(ids) - 1, 1))
    scored = mx.take_along_axis(logits, targets, axis=-1)[..., 0] - mx.logsumexp(logits, axis=-1)
    contrast = (scored[0] - scored[1]).tolist()

    def mean_gap(span):
        positions = [p - 1 for p in positions_in(spans, span) if p > 0]
        return sum(contrast[p] for p in positions) / len(positions) if positions else None

    gaps = [mean_gap((s + offset, e + offset)) for s, e in sentence_spans]
    baseline = mean_gap((0, len(context))) if context.strip() else None
    if baseline is None:
        known = [g for g in gaps if g is not None]
        baseline = sum(known) / len(known) if known else 0.0
    results = []
    for index in asked:
        if gaps[index] is None:
            continue
        gap = (gaps[index] - baseline) / REALIZATION_GAP_SCALE
        confidence = math.exp(-gap * gap) if sign == 0 else sigmoid(gap)
        results.append({"index": index, "confidence": confidence})
    return results


def attention_matrix(engine: Engine, ids: list):
    """Gate-weighted attention of every token to every token, averaged over the full-attention layers. Shape (L, L)."""
    instruments = Instruments(capture_attention=True)
    engine.forward(mx.array([ids]), instruments=instruments)
    return mx.stack([w[0] for w in instruments.attention]).mean(axis=0)


def mean_share(weights, queries, keys):
    """Average over `queries` of the attention they put on `keys`, in total."""
    if not queries or not keys:
        return 0.0
    rows = weights[mx.array(sorted(queries))][:, mx.array(sorted(keys))]
    return rows.sum(axis=1).mean().item()


def joined(sentences: list):
    text, spans = "", []
    for sentence in sentences:
        if text and not text[-1].isspace() and not sentence[:1].isspace():
            text += " "
        spans.append((len(text), len(text) + len(sentence.rstrip())))
        text += sentence
    return text, spans


def find_last(text: str, part: str):
    at = text.rfind(part) if part else -1
    return None if at == -1 else (at, at + len(part))


def utf16_offset(text: str, at: int):
    """A character offset in `text` as the page counts it, in UTF-16 code units, where a character beyond the basic plane is two."""
    return at + sum(1 for c in text[:at] if ord(c) > 0xFFFF)


def overlaps(a: tuple, b: tuple):
    return a[0] < b[1] and b[0] < a[1]


def sigmoid(x: float):
    return 1 / (1 + math.exp(-x))
