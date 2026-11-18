import type { Lead } from "../text/attention.ts"
import type { Focus, Influence } from "../text/influence.ts"
import type { Relevance, RelevanceRequest } from "../text/keyword-relevance.ts"
import type { Realization, RealizationRequest } from "../text/realization.ts"
import { leadSpans, positionsIn } from "./generation.ts"
import type { Engine, Span } from "./model.ts"
import type { Vectors } from "./steering.ts"

export type FocusBox = { start: number; end: number; keyStart: number; score: number }

/** Greedy tokens the model is run ahead for: the attention boxes show where those next words' attention goes. */
const FOCUS_STEPS = 6
const MAX_BOXES = 3
const SECOND_BOX_SHARE = 0.5
const PHRASE_REACH = 2
/** Attention per keyword token, as a multiple of the average story token's, that counts as fully relevant. */
const FULL_RELEVANCE_RATIO = 12
/** Per-token log-likelihood gap between the toward- and away-steered model, beyond the story so far's own gap, that reads as a clear yes (about 73%). */
const REALIZATION_GAP_SCALE = 0.3

const WORD = /[\p{L}\p{N}_’']+/gu
const CLAUSE_BREAK = /[.!?;:,—"“”()]/

export async function focusBoxes(engine: Engine, text: string, signal?: AbortSignal): Promise<FocusBox[]> {
  const story = text.trimEnd()
  if (!story) return []
  const { ids, spans } = engine.storyTokens(story)
  const cache = engine.emptyCache(1)
  const tokenScores = new Float32Array(ids.length)
  try {
    await engine.forward([ids.slice(0, -1)], { cache, signal })
    let token = ids.at(-1)!
    for (let step = 0; step < FOCUS_STEPS; step++) {
      const { logits, attention } = await engine.forward([[token]], { cache, attention: true, signal })
      tokenScores.forEach((_, position) => (tokenScores[position] += attention![position]))
      token = argmax(logits[0])
      if (engine.stopIds.has(token)) break
    }
  } finally {
    engine.release(cache)
  }

  const words = [...story.matchAll(WORD)].map((match) => ({ start: match.index, end: match.index + match[0].length }))
  const wordScores = words.map(({ start, end }) => sum(positionsIn(spans, [start, end]).map((position) => tokenScores[position])))
  const ranked = words.map((_, index) => index).sort((a, b) => wordScores[b] - wordScores[a])
  const boxes: FocusBox[] = []
  const covered = new Set<number>()
  for (const index of ranked) {
    if (boxes.length === MAX_BOXES || (boxes.length && wordScores[index] < SECOND_BOX_SHARE * boxes[0].score)) break
    if (covered.has(index)) continue
    const [first, last] = phraseAround(story, words, index)
    if (range(first, last).some((word) => covered.has(word))) continue
    for (const word of range(first - 1, last + 1)) covered.add(word)
    boxes.push({ start: words[first].start, end: words[last].end, keyStart: words[index].start, score: wordScores[index] })
  }
  return boxes
}

/** Sentences before the focus are scored by the focus's attention to them, sentences after it by their attention to the focus. */
export async function influence(engine: Engine, sentences: string[], focus: Focus, signal?: AbortSignal): Promise<Influence[]> {
  const whole = joined(sentences)
  const focusText = focus.text.trim()
  const found = focusText ? whole.text.lastIndexOf(focusText) : -1
  const withFocus = found === -1 ? joined([whole.text, focusText]) : undefined
  const text = withFocus?.text ?? whole.text
  const focusSpan: Span = withFocus?.spans[1] ?? [found, found + focusText.length]
  const { ids, spans } = engine.storyTokens(text)
  const share = shareOf(await attentionMatrix(engine, ids, signal), ids.length)
  const focusPositions = positionsIn(spans, focusSpan)
  const shares = whole.spans.map((span) => {
    const positions = positionsIn(spans, span)
    if (!positions.length || (span[0] < focusSpan[1] && focusSpan[0] < span[1])) return undefined
    return span[1] <= focusSpan[0] ? share(focusPositions, positions) : share(positions, focusPositions)
  })
  const strongest = Math.max(0, ...shares.map((value) => value ?? 0)) || 1
  return shares.map((value, index) => ({ index, strength: value === undefined ? 1 : value / strongest }))
}

export async function relevance(engine: Engine, { story, tip, leads }: RelevanceRequest, signal?: AbortSignal): Promise<Relevance[]> {
  const whole = joined([story, tip])
  const tipSpan = whole.spans[1]
  const { ids, spans, first } = engine.storyTokens(whole.text)
  const share = shareOf(await attentionMatrix(engine, ids, signal), ids.length)
  const tipPositions = positionsIn(spans, tipSpan)
  const storyPositions = range(first, ids.length - 1).filter((position) => !tipPositions.includes(position))
  const baseline = share(tipPositions, storyPositions) / Math.max(1, storyPositions.length)
  return leads.flatMap((lead: Lead, index) => {
    const found = leadSpans(lead, story)
    if (!found) return []
    const keywordPositions = positionsIn(spans, found.keyword)
    const ratio = baseline > 0 ? share(tipPositions, keywordPositions) / Math.max(1, keywordPositions.length) / baseline : 0
    return { index, relevance: Math.min(1, Math.log1p(ratio) / Math.log1p(FULL_RELEVANCE_RATIO)) }
  })
}

/**
 * The text is scored twice, with the control's vector pushing toward the asked value and away from it; a sentence likelier per token under the first
 * delivers it.
 */
export async function realization(
  engine: Engine,
  vectors: Vectors,
  { context, sentences, asked, control }: RealizationRequest,
  signal?: AbortSignal,
): Promise<Realization[]> {
  const target = await vectors.targetOf(control)
  if (!target) return []
  const toward = target.sign || 1
  const body = joined(sentences)
  const whole = joined([context, body.text])
  const offset = whole.spans[1][0]
  const { ids, spans } = engine.storyTokens(whole.text)
  const [pushed, pulled] = [vectors.push(target.direction, toward), vectors.push(target.direction, -toward)]
  const steer = Object.fromEntries(engine.steerLayers.map((layer) => [layer, [pushed[layer], pulled[layer]]]))
  // Each position is scored on the token that follows it; the last has none.
  const next = [...ids.slice(1), 0]
  const { scored } = await engine.forward([ids, ids], { steer, targets: [next, next], signal })
  const contrast = Array.from({ length: ids.length - 1 }, (_, position) => scored![position] - scored![ids.length + position])

  const meanGap = (span: Span) => {
    const positions = positionsIn(spans, span).filter((position) => position > 0)
    return positions.length ? sum(positions.map((position) => contrast[position - 1])) / positions.length : undefined
  }
  const gaps = body.spans.map(([start, end]) => meanGap([start + offset, end + offset]))
  const known = gaps.filter((gap) => gap !== undefined)
  const baseline = (context.trim() ? meanGap([0, context.length]) : undefined) ?? (known.length ? sum(known) / known.length : 0)
  return asked.flatMap((index) => {
    if (gaps[index] === undefined) return []
    const gap = (gaps[index] - baseline) / REALIZATION_GAP_SCALE
    return { index, confidence: target.sign === 0 ? Math.exp(-gap * gap) : 1 / (1 + Math.exp(-gap)) }
  })
}

async function attentionMatrix(engine: Engine, ids: number[], signal?: AbortSignal) {
  return (await engine.forward([ids], { attention: true, signal })).attention!
}

function shareOf(weights: Float16Array, length: number) {
  return (queries: number[], keys: number[]) => {
    if (!queries.length || !keys.length) return 0
    return sum(queries.map((query) => sum(keys.map((key) => weights[query * length + key])))) / queries.length
  }
}

function phraseAround(story: string, words: { start: number; end: number }[], index: number) {
  const breaksBetween = (left: number) => CLAUSE_BREAK.test(story.slice(words[left].end, words[left + 1].start))
  let [first, last] = [index, index]
  while (first > 0 && index - first < PHRASE_REACH && !breaksBetween(first - 1)) first--
  while (last < words.length - 1 && last - index < PHRASE_REACH && !breaksBetween(last)) last++
  return [first, last]
}

function joined(sentences: string[]) {
  let text = ""
  const spans: Span[] = []
  for (const sentence of sentences) {
    if (text && !/\s$/.test(text) && !/^\s/.test(sentence)) text += " "
    spans.push([text.length, text.length + sentence.trimEnd().length])
    text += sentence
  }
  return { text, spans }
}

function argmax(values: ArrayLike<number>) {
  let best = 0
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i
  return best
}

function range(first: number, last: number) {
  return Array.from({ length: Math.max(0, last - first + 1) }, (_, index) => first + index)
}

function sum(values: number[]) {
  return values.reduce((total, value) => total + value, 0)
}
