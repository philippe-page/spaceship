import { streamEngine } from "../llm/local-engine.ts"
import { expandToStoryStart } from "../story/openings.ts"
import type { AttentionSource } from "./attention.ts"
import type { Dimension, DimensionSource } from "./dimension.ts"
import type { Influence, InfluenceSource } from "./influence.ts"
import { nonSpaceLength, type ComparedStep, type InsideView } from "./inside.ts"
import type { Relevance, RelevanceSource } from "./keyword-relevance.ts"
import type { Realization, RealizationSource } from "./realization.ts"
import { wordSpans } from "./segments.ts"
import type { TextSource } from "./text-stream.ts"

/**
 * Sources backed by the local MLX engine (see engine/), which works inside the model instead of prompting it:
 * future knobs add steering vectors to the residual stream, attention boxes bias the attention scores toward their tokens,
 * and every score is read from the model's own attention weights, activations or likelihoods.
 */

const MENU_SIZE = 5

/** Character spans of a phrase and its keyword in the text. */
type FocusSpan = { start: number; end: number; keyStart: number }

/** A line of a continuation: its text, and while the inside view is on, the token's comparison with the unsteered model. */
type CompletionChunk = { text?: string; step?: ComparedStep }

/** Continuations from the engine; while the inside view is on, the engine compares each token with an unsteered copy of the model and the view shows it. */
export function createLocalSource(inside: InsideView): TextSource {
  return {
    /** A story that starts over keeps none of the last one's tokens. */
    expand(text) {
      const opening = expandToStoryStart(text)
      if (opening !== text) inside.start(0)
      return opening
    },
    async *complete(prefix, { leads, controls }, signal) {
      const compare = inside.isOn()
      let at = nonSpaceLength(prefix)
      inside.start(at)
      for await (const { text = "", step } of streamEngine<CompletionChunk>("/complete", { prefix, leads, controls, compare }, signal)) {
        if (step) inside.record(at, nonSpaceLength(text), step)
        if (!text) continue
        at += nonSpaceLength(text)
        yield text
      }
    },
  }
}

/** Where the model's next few words put the most attention, in its full-attention layers. */
export const localAttention: AttentionSource = async function* (text, signal) {
  const words = wordSpans(text)
  for await (const { start, end, keyStart } of streamEngine<FocusSpan>("/attention", { text }, signal)) {
    const keyword = words.findIndex((word) => word.end > keyStart)
    const first = words.findIndex((word) => word.end > start)
    const last = words.findLastIndex((word) => word.start < end)
    if (keyword === -1 || first === -1 || first > keyword || last < keyword) continue
    yield { keyword, before: keyword - first, after: last - keyword }
  }
}

/** The attention between each sentence and the focus. */
export const localInfluence: InfluenceSource = (sentences, focus, signal) => streamEngine<Influence>("/influence", { sentences, focus }, signal)

/** The attention the tip's tokens put on each keyword's. */
export const localRelevance: RelevanceSource = (request, signal) => streamEngine<Relevance>("/relevance", request, signal)

/** How much likelier each sentence is under the control's steering vector than under its opposite. */
export const localRealization: RealizationSource = (request, signal) => streamEngine<Realization>("/realization", request, signal)

/** Dimensions, the story's characters among them, ranked by how far pushing each one moves the model's next-token distribution. */
export const localDimension: DimensionSource = ({ text, taken }, signal) => streamEngine<Dimension>("/dimensions", { text, taken, size: MENU_SIZE }, signal)
