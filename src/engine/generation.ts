import type { Lead } from "../text/attention.ts"
import type { Control } from "../text/dimension.ts"
import type { Engine, Span } from "./model.ts"
import type { Vectors } from "./steering.ts"

/** The story so far is read without steering, so it is encoded as the reader sees it; steering applies from the first generated token. */

export type CompletionRequest = { prefix: string; leads: Lead[]; controls: Control[]; compare: boolean }

/** `ended` is the model ending the story itself. */
export type CompletionChunk = { text: string; step?: ComparedStep; ended?: true }

export type Candidate = { token: string; on: number; off: number; chosen: boolean }

/**
 * Per layer, `drift` (the steered state's distance from the unsteered one) and `push` (the size of the added vector) are shares of the unsteered
 * state's size.
 */
export type ComparedStep = { top: Candidate[]; instead: string | null; drift: number[]; push: number[]; isLocked: boolean }

const PREFILL_CHUNK = 256
const MAX_TOKENS = 360
const SOFT_MAX_TOKENS = 220
/** Added to the attention scores of an attention box's tokens, and on top of that to its keyword's, in log space: e^1.5 is about 4.5 times the weight before normalization. */
const PHRASE_BIAS = 1.5
const KEYWORD_BIAS = 1.0
const TEMPERATURE = 0.9
const TOP_P = 0.95
const REPETITION_PENALTY = 1.1
const REPETITION_CONTEXT = 64
const TOP_WORDS = 5
const SENTENCE_END = /[.!?]["”’]?\s*$/
/** No token likelier than this can be left out of a nucleus by looking only at tokens above it: all the others together stay under 1 - TOP_P. */
const NUCLEUS_FLOOR = 3e-7

/**
 * With `compare`, row 1 is an unsteered copy that reads exactly what the steered row 0 writes. Both sample from the same random draw, so they pick
 * the same token wherever steering does not change the odds; row 1's pick is what the unsteered model would have written there, and is then
 * discarded.
 */
export async function* complete(
  engine: Engine,
  vectors: Vectors,
  { prefix, leads, controls, compare }: CompletionRequest,
  signal?: AbortSignal,
): AsyncGenerator<CompletionChunk> {
  const story = prefix.trimEnd()
  const { ids, spans } = engine.storyTokens(story)
  const read = engine.emptyCache(1)
  let cache = read
  try {
    for (let start = 0; start < ids.length - 1; start += PREFILL_CHUNK) {
      await engine.forward([ids.slice(start, Math.min(start + PREFILL_CHUNK, ids.length - 1))], { cache: read, signal })
    }

    const rows = compare ? 2 : 1
    if (compare) {
      cache = await engine.forkCache(read, 2)
      engine.release(read)
    }
    const adds = await vectors.residualAdds(controls, story)
    const bias = lockBias(leadPositions(leads, story, spans), ids.length)
    const steer = Object.fromEntries(Object.entries(adds).map(([layer, add]) => [layer, compare ? [add, new Float32Array(add.length)] : [add]]))
    const lock = bias && (compare ? [bias, new Float32Array(bias.length)] : [bias])
    const writer = createWriter(engine, story !== "" && prefix === story)
    const history = [...ids]
    const addSizes = engine.layerKinds.map((_, layer) => (adds[layer] ? Math.hypot(...adds[layer]) : 0))

    for (let step = 0; step < MAX_TOKENS; step++) {
      const { logits, layerStates } = await engine.forward(
        Array.from({ length: rows }, () => [history.at(-1)!]),
        { cache, steer, lock, layerStates: compare, signal },
      )
      const { picks, odds } = coupledSample(logits, history.slice(-REPETITION_CONTEXT))
      const [token, unsteered] = picks
      if (engine.stopIds.has(token)) return yield { text: "", ended: true }
      history.push(token)
      const text = writer.add(token)
      if (layerStates) {
        const instead = unsteered === token ? null : engine.decode([unsteered])
        yield { text, step: { top: candidates(engine, odds, token), instead, ...layerMoves(layerStates, addSizes), isLocked: !!bias } }
      } else yield { text }
      if (step >= SOFT_MAX_TOKENS && SENTENCE_END.test(writer.text())) return
    }
  } finally {
    engine.release(cache)
  }
}

export function leadPositions(leads: Lead[], story: string, spans: (Span | undefined)[]) {
  const found = leads.flatMap((lead) => leadSpans(lead, story) ?? [])
  return { phrase: found.flatMap(({ phrase }) => positionsIn(spans, phrase)), keyword: found.flatMap(({ keyword }) => positionsIn(spans, keyword)) }
}

export function leadSpans(lead: Lead, text: string): { phrase: Span; keyword: Span } | undefined {
  const phrase = lead.before + lead.keyword + lead.after
  const at = text.lastIndexOf(phrase)
  if (at !== -1) {
    const start = at + lead.before.length
    return { phrase: [at, at + phrase.length], keyword: [start, start + lead.keyword.length] }
  }
  const keywordAt = text.lastIndexOf(lead.keyword)
  if (keywordAt === -1) return undefined
  const keyword: Span = [keywordAt, keywordAt + lead.keyword.length]
  return { phrase: keyword, keyword }
}

export function positionsIn(spans: (Span | undefined)[], [start, end]: Span) {
  return spans.flatMap((span, position) => (span && span[0] < end && span[1] > start ? [position] : []))
}

function lockBias({ phrase, keyword }: { phrase: number[]; keyword: number[] }, length: number) {
  if (!phrase.length) return undefined
  const bias = new Float32Array(length)
  for (const position of new Set(phrase)) bias[position] += PHRASE_BIAS
  for (const position of new Set(keyword)) bias[position] += KEYWORD_BIAS
  return bias
}

/** A token that ends partway through a character adds nothing until the rest arrives. */
function createWriter(engine: Engine, needsSpace: boolean) {
  const tokens: number[] = []
  let written = ""

  return {
    text: () => written,
    add(token: number) {
      tokens.push(token)
      const decoded = engine.decode(tokens)
      if (decoded.endsWith("�")) return ""
      const text = needsSpace ? decoded : decoded.trimStart()
      const piece = text.slice(written.length)
      written = text
      return piece
    },
  }
}

const SCALED_BY_BITS = Float32Array.from(new Float16Array(Uint16Array.from({ length: 1 << 16 }, (_, bits) => bits).buffer), (logit) => logit / TEMPERATURE)
const counts = new Uint32Array(1 << 16)

/**
 * Gumbel-max sampling within the top-p nucleus, with noise shared by the rows, so rows with the same distribution pick the same token. A row's
 * logits are 16-bit floats, so the softmax takes one exponential per distinct value, not per token.
 */
function coupledSample(logits: Float16Array[], recent: number[]) {
  const noise = new Map<number, number>()
  const noiseOf = (token: number) => noise.get(token) ?? noise.set(token, -Math.log(-Math.log(Math.max(Math.random(), 1e-9)))).get(token)!
  const picks: number[] = []
  const odds: Map<number, number>[] = []
  for (const row of logits) {
    const bits = new Uint16Array(row.buffer, row.byteOffset, row.length)
    counts.fill(0)
    for (let i = 0; i < bits.length; i++) counts[bits[i]]++
    const penalized = new Map<number, number>()
    for (const token of new Set(recent)) {
      counts[bits[token]]--
      const logit = row[token]
      penalized.set(token, Math.fround(Math.fround(logit < 0 ? logit * REPETITION_PENALTY : logit / REPETITION_PENALTY) / TEMPERATURE))
    }
    const scaled = (token: number) => penalized.get(token) ?? SCALED_BY_BITS[bits[token]]
    let max = Math.max(...penalized.values())
    for (let value = 0; value < counts.length; value++) if (counts[value] && SCALED_BY_BITS[value] > max) max = SCALED_BY_BITS[value]
    let total = 0
    for (let value = 0; value < counts.length; value++) if (counts[value]) total += counts[value] * Math.exp(SCALED_BY_BITS[value] - max)
    for (const logit of penalized.values()) total += Math.exp(logit - max)
    const probabilityOf = (token: number) => Math.exp(scaled(token) - max) / total
    // The floor's logit, less a margin for rounding; the exact test decides.
    const floor = max + Math.log(NUCLEUS_FLOOR * total) - 1e-3
    const likely = [...penalized.keys()].filter((token) => probabilityOf(token) > NUCLEUS_FLOOR)
    for (let i = 0; i < bits.length; i++) if (SCALED_BY_BITS[bits[i]] > floor && !penalized.has(i) && probabilityOf(i) > NUCLEUS_FLOOR) likely.push(i)
    likely.sort((a, b) => scaled(b) - scaled(a) || a - b)

    const nucleus = new Map<number, number>()
    let mass = 0
    let pick = likely[0]
    for (const token of likely) {
      if (mass >= TOP_P) break
      nucleus.set(token, probabilityOf(token))
      mass += probabilityOf(token)
      if (scaled(token) + noiseOf(token) > scaled(pick) + noiseOf(pick)) pick = token
    }
    for (const [token, probability] of nucleus) nucleus.set(token, probability / mass)
    picks.push(pick)
    odds.push(nucleus)
  }
  return { picks, odds }
}

function candidates(engine: Engine, odds: Map<number, number>[], chosen: number): Candidate[] {
  // A nucleus lists its tokens likeliest first.
  const tokens = new Set([chosen, ...odds.flatMap((row) => [...row.keys()].slice(0, TOP_WORDS))])
  const [steered, unsteered] = odds
  return [...tokens]
    .map((token) => ({ token: engine.decode([token]), on: round(steered.get(token) ?? 0), off: round(unsteered.get(token) ?? 0), chosen: token === chosen }))
    .sort((a, b) => b.on - a.on || b.off - a.off)
}

function round(odds: number) {
  return Math.round(odds * 1e6) / 1e6
}

function layerMoves(states: Float32Array, addSizes: number[]) {
  const layers = addSizes.length
  // A row holds each layer's squared distance from the rows' mean, which is half the distance between two rows, then its squared size.
  const unsteeredSize = (layer: number) => Math.sqrt(states[3 * layers + layer])
  const share = (size: number, layer: number) => Math.round((1e4 * size) / unsteeredSize(layer)) / 1e4
  return { drift: addSizes.map((_, layer) => share(2 * Math.sqrt(states[layer]), layer)), push: addSizes.map(share) }
}
