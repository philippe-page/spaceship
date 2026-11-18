import { streamEngine } from "../llm/local-engine.ts"
import { expandToStoryStart } from "../story/openings.ts"
import type { AttentionSource } from "./attention.ts"
import type { Dimension, DimensionSource } from "./dimension.ts"
import type { Influence, InfluenceSource } from "./influence.ts"
import type { FocusBox } from "../engine/analysis.ts"
import type { CompletionChunk, CompletionRequest } from "../engine/generation.ts"
import type { ModelLayout } from "../engine/host.ts"
import { nonSpaceLength, type InsideView } from "./inside.ts"
import type { Relevance, RelevanceSource } from "./keyword-relevance.ts"
import type { Realization, RealizationSource } from "./realization.ts"
import { wordSpans } from "./segments.ts"
import type { TextSource } from "./text-stream.ts"

const MENU_SIZE = 5
/** Written where the model ended the story itself; nothing more is asked for past it. */
const END_MARK = " ∎"

export function createLocalSource(inside: InsideView): TextSource {
  let isLaidOut = false

  return {
    /** A story that starts over keeps none of the last one's tokens. */
    expand(text) {
      const opening = expandToStoryStart(text)
      if (opening !== text) inside.start(0)
      return opening
    },
    async *complete(prefix, { leads, controls }, signal) {
      if (prefix.endsWith(END_MARK)) return
      const compare = inside.isOn()
      if (compare && !isLaidOut) {
        for await (const layout of streamEngine<ModelLayout>("/layers", {}, signal)) inside.lay(layout)
        isLaidOut = true
      }
      let at = nonSpaceLength(prefix)
      inside.start(at)
      const request: CompletionRequest = { prefix, leads, controls, compare }
      for await (const { text, step, ended } of streamEngine<CompletionChunk>("/complete", request, signal)) {
        if (ended) return yield END_MARK
        if (step) inside.record(at, nonSpaceLength(text), step)
        if (!text) continue
        at += nonSpaceLength(text)
        yield text
      }
    },
  }
}

export const localAttention: AttentionSource = async function* (text, signal) {
  const words = wordSpans(text)
  for await (const { start, end, keyStart } of streamEngine<FocusBox>("/attention", { text }, signal)) {
    const keyword = words.findIndex((word) => word.end > keyStart)
    const first = words.findIndex((word) => word.end > start)
    const last = words.findLastIndex((word) => word.start < end)
    if (keyword === -1 || first === -1 || first > keyword || last < keyword) continue
    yield { keyword, before: keyword - first, after: last - keyword }
  }
}

export const localInfluence: InfluenceSource = (sentences, focus, signal) => streamEngine<Influence>("/influence", { sentences, focus }, signal)

export const localRelevance: RelevanceSource = (request, signal) => streamEngine<Relevance>("/relevance", request, signal)

export const localRealization: RealizationSource = (request, signal) => streamEngine<Realization>("/realization", request, signal)

export const localDimension: DimensionSource = ({ text, taken }, signal) => streamEngine<Dimension>("/dimensions", { text, taken, size: MENU_SIZE }, signal)
