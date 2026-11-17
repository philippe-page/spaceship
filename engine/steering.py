"""Steering vectors for the future knobs' dimensions, found by contrastive activation addition: the same story openings are
read under instructions for each pole or option, and the mean difference of the model's layer outputs at the end of the
story is that dimension's direction at each layer. Adding a direction to the residual stream while generating moves the
text along it."""

import json
import math
import re
from collections import Counter

import mlx.core as mx

from model import SYSTEM_PROMPT, Engine, Instruments, fork_cache

# Size of the push at full strength, as a share of the residual stream's typical norm at each steered layer. Tuned by eye
# on LFM3 until the strongest push still wrote grammatical prose (0.2 does not); Qwen3.5 holds up at the same share.
GAIN = 0.12
# Story tokens, counted back from the end, whose activations are averaged into a direction.
TAIL_TOKENS = 6
# How much of the story's end a character focus is read against.
FOCUS_CONTEXT_CHARS = 600

OPENINGS = [
    "Mara had locked the kitchen an hour ago. Now",
    "The train pulled into the station and Theo",
    "On the last morning of summer, the lake",
    "The letter arrived on a Tuesday. Ines read it twice and",
    "Nobody had opened the attic in years, so when",
    "The market was crowded with the smell of bread and",
    "Grandpa's old radio crackled to life just as",
    "The ship's lights flickered over the dark water while",
]

CATALOG = [
    {"name": "tension", "kind": "scale", "low": "relaxed", "high": "strained"},
    {"name": "resolution", "kind": "scale", "low": "open question", "high": "answered"},
    {"name": "pace", "kind": "scale", "low": "compressed", "high": "expanded"},
    {"name": "mode", "kind": "choice", "options": ["dialogue", "action", "description"]},
    {"name": "tone", "kind": "choice", "options": ["wry", "tender", "somber", "wondrous"]},
    {"name": "distance", "kind": "scale", "low": "outside view", "high": "inside mind"},
    {"name": "fortune", "kind": "scale", "low": "worsens", "high": "improves"},
    {"name": "surprise", "kind": "scale", "low": "expected", "high": "unexpected"},
]

CAPITALIZED = re.compile(r"\b[A-Z][a-z]{2,}\b")
# What a word that starts a sentence comes after, past any whitespace: nothing, or how the sentence before it ended.
SENTENCE_STARTS_AFTER = ("", ".", "!", "?", '"', "“")
NOT_NAMES = set(
    """The And But Then Now When While Its She They You His Her Their Our This That These Those There Here What Who Why How
    Where Once Long Nobody Somebody Everyone Yes Not For With From After Before Even Still Just Only All One Two Monday
    Tuesday Wednesday Thursday Friday Saturday Sunday God Mrs""".split()
)


class Vectors:
    """Directions per dimension, computed on first use and kept for the session. Dimensions are remembered by name so a
    realization request, which names a control only in words, finds the direction it was steered by."""

    def __init__(self, engine: Engine):
        self.engine = engine
        self.directions = {}
        self.dimensions = {}
        ends = [self._residuals(opening) for opening in OPENINGS]
        # Each steered layer's typical norm at the end of a story, which sizes every push.
        self.norms = {layer: sum(mx.linalg.norm(end[layer][-1]).item() for end in ends) / len(ends) for layer in engine.steer_layers}

    def directions_of(self, dimension: dict, story: str = ""):
        """Value -> per-layer unit direction: one value, "high", for a scale; one per option for a choice. A character focus
        is read against the story as it was when first offered, since its options name the story's people."""
        self.dimensions[dimension["name"]] = dimension
        key = json.dumps(dimension, sort_keys=True)
        if key not in self.directions:
            contexts = [story[-FOCUS_CONTEXT_CHARS:]] if dimension["name"] == "focus" and story else OPENINGS
            self.directions[key] = self._extract(dimension, contexts)
        return self.directions[key]

    def _extract(self, dimension: dict, contexts: list):
        layers = self.engine.steer_layers
        if dimension["kind"] == "scale":
            high = self._mean_states(instruction(dimension, dimension["high"]), contexts)
            low = self._mean_states(instruction(dimension, dimension["low"]), contexts)
            return {"high": {layer: unit(high[layer] - low[layer]) for layer in layers}}
        states = {option: self._mean_states(instruction(dimension, option), contexts) for option in dimension["options"]}
        center = {layer: sum(state[layer] for state in states.values()) / len(states) for layer in layers}
        return {option: {layer: unit(state[layer] - center[layer]) for layer in layers} for option, state in states.items()}

    def _mean_states(self, guidance: str, contexts: list):
        """Each steered layer's output at the end of the contexts, read under `guidance`, averaged over them."""
        ends = [self._residuals(context, f"{SYSTEM_PROMPT} {guidance}") for context in contexts]
        states = {layer: sum(end[layer][-TAIL_TOKENS:].mean(axis=0) for end in ends) / len(ends) for layer in self.engine.steer_layers}
        mx.eval(states)
        return states

    def _residuals(self, story: str, system: str | None = None):
        """Each steered layer's output over the prompt for `story`, shape (L, H)."""
        ids, _, _ = self.engine.story_tokens(story, system)
        instruments = Instruments(capture_residual=self.engine.steer_layers)
        self.engine.forward(mx.array([ids]), instruments=instruments)
        return {layer: output[0].astype(mx.float32) for layer, output in instruments.residuals.items()}

    def residual_adds(self, controls: list, story: str = ""):
        """The vector each steered layer adds for these future-knob controls: a scale pushes toward its high pole past the
        midpoint and toward its low pole before it; a choice pushes toward its selected option. Strength scales the push."""
        adds = {}
        for control in controls:
            dimension, position, strength = control["dimension"], control["position"], control["strength"]
            directions = self.directions_of(dimension, story)
            if dimension["kind"] == "scale":
                value, amount = "high", (2 * position - 1) * strength
            else:
                value, amount = dimension["options"][option_index(len(dimension["options"]), position)], strength
            for layer, push in self.push(directions[value], amount).items():
                adds[layer] = adds[layer] + push if layer in adds else push
        return adds

    def push(self, direction: dict, amount: float):
        """What each steered layer adds to move `amount` along `direction`: at 1, `GAIN` times the layer's typical norm."""
        return {layer: direction[layer] * (amount * GAIN * self.norms[layer]) for layer in self.engine.steer_layers}

    def batched(self, pushes: list):
        """Several pushes as one per-layer stack, shape (B, 1, H), for a batch that tries each push on its own row."""
        return {layer: mx.stack([push[layer] for push in pushes])[:, None, :] for layer in self.engine.steer_layers}

    def target_of(self, control: str):
        """A realization control in words ("tension: somewhat strained", "tone: wry") as the direction it asks to move along and
        which way: 1 toward a scale's high pole or a choice's option, -1 toward the low pole, 0 for a scale's midway."""
        name, _, value = control.partition(":")
        dimension = self.dimensions.get(name.strip())
        if dimension is None:
            return None
        value = value.strip()
        if dimension["kind"] == "choice":
            return (self.directions_of(dimension)[value], 1) if value in dimension["options"] else None
        direction = self.directions_of(dimension)["high"]
        if value.startswith("midway"):
            return direction, 0
        return direction, 1 if value.endswith(dimension["high"]) else -1

    def menu(self, text: str, taken: list, size: int):
        """The free catalog dimensions, plus a character focus when the story has named people, ranked by how much pushing each
        one either way changes the model's next-token distribution at the end of `text`, averaged over a choice's options so
        a dimension with more options is not favored for having more of them."""
        taken_names = {d["name"].lower() for d in taken}
        free = [d for d in CATALOG if d["name"] not in taken_names]
        characters = characters_of(text)
        if "focus" not in taken_names and characters:
            free.append({"name": "focus", "kind": "choice", "options": characters[:3] + ["narrator"]})
        if not free:
            return []
        ids, _, _ = self.engine.story_tokens(text)
        cache = self.engine.make_cache()
        self.engine.forward(mx.array([ids[:-1]]), cache)
        pushes = []
        for dimension in free:
            for direction in self.directions_of(dimension, text).values():
                pushes.append((dimension["name"], direction, 1.0))
                pushes.append((dimension["name"], direction, -1.0))
        batch = len(pushes)
        adds = self.batched([self.push(direction, sign) for _, direction, sign in pushes])
        logits = self.engine.forward(mx.array([[ids[-1]]] * batch), fork_cache(cache, batch), Instruments(residual_add=adds))
        logprobs = logits[:, -1, :].astype(mx.float32)
        logprobs = logprobs - mx.logsumexp(logprobs, axis=-1, keepdims=True)
        shifts = {}
        for i in range(0, batch, 2):
            divergence = (mx.exp(logprobs[i]) * (logprobs[i] - logprobs[i + 1])).sum().item()
            shifts.setdefault(pushes[i][0], []).append(divergence)
        ranked = sorted(free, key=lambda d: sum(shifts[d["name"]]) / len(shifts[d["name"]]), reverse=True)
        return ranked[:size]


def instruction(dimension: dict, value: str):
    if dimension["name"] == "focus":
        return f"Write the next passage following {value}." if value != "narrator" else "Write the next passage from the narrator's view."
    if dimension["kind"] == "scale":
        return f"Write the next passage so its {dimension['name']} is extremely {value}."
    return f"Write the next passage as {value} ({dimension['name']})."


def option_index(count: int, position: float):
    return min(count - 1, math.floor(position * count))


def unit(vector):
    return vector / (mx.linalg.norm(vector) + 1e-6)


def characters_of(text: str):
    """Likely names, most frequent first: capitalized words, other than common ones, that appear inside a sentence or more
    than once."""
    found = [match for match in CAPITALIZED.finditer(text) if match.group() not in NOT_NAMES]
    counts = Counter(match.group() for match in found)
    inside = {match.group() for match in found if text[: match.start()].rstrip()[-1:] not in SENTENCE_STARTS_AFTER}
    return [word for word, count in counts.most_common() if word in inside or count > 1]
