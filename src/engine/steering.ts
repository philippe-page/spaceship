import { optionIndexAt, type Control, type Dimension } from "../text/dimension.ts"
import { SYSTEM_PROMPT, type Engine } from "./model.ts"

/**
 * Steering vectors by contrastive activation addition: the same story openings are read under instructions for each pole or option, and the mean
 * difference of the layer outputs at the end of the story is the direction.
 */

export type Direction = Record<number, Float32Array>

export type Vectors = Awaited<ReturnType<typeof findVectors>>

/**
 * Size of the push at full strength, as a share of the residual stream's typical norm at each steered layer. Tuned by eye
 * until the strongest push still wrote grammatical prose (0.2 does not).
 */
const GAIN = 0.12
const TAIL_TOKENS = 6
const FOCUS_CONTEXT_CHARS = 600
const MENU_BATCH = 8

const OPENINGS = [
  "Mara had locked the kitchen an hour ago. Now",
  "The train pulled into the station and Theo",
  "On the last morning of summer, the lake",
  "The letter arrived on a Tuesday. Ines read it twice and",
  "Nobody had opened the attic in years, so when",
  "The market was crowded with the smell of bread and",
  "Grandpa's old radio crackled to life just as",
  "The ship's lights flickered over the dark water while",
]

export const CATALOG: Dimension[] = [
  { name: "tension", kind: "scale", low: "relaxed", high: "strained" },
  { name: "resolution", kind: "scale", low: "open question", high: "answered" },
  { name: "pace", kind: "scale", low: "compressed", high: "expanded" },
  { name: "mode", kind: "choice", options: ["dialogue", "action", "description"] },
  { name: "tone", kind: "choice", options: ["wry", "tender", "somber", "wondrous"] },
  { name: "distance", kind: "scale", low: "outside view", high: "inside mind" },
  { name: "fortune", kind: "scale", low: "worsens", high: "improves" },
  { name: "surprise", kind: "scale", low: "expected", high: "unexpected" },
]

const CAPITALIZED = /\b[A-Z][a-z]{2,}\b/g
/** What a word that starts a sentence comes after, past any whitespace: nothing, or how the sentence before it ended. */
const SENTENCE_STARTS_AFTER = ["", ".", "!", "?", '"', "“"]
const NOT_NAMES = new Set(
  `The And But Then Now When While Its She They You His Her Their Our This That These Those There Here What Who Why How
  Where Once Long Nobody Somebody Everyone Yes Not For With From After Before Even Still Just Only All One Two Monday
  Tuesday Wednesday Thursday Friday Saturday Sunday God Mrs`.split(/\s+/),
)

/** Dimensions are remembered by name, so a realization request, which names a control only in words, finds its direction. */
export async function findVectors(engine: Engine) {
  const { steerLayers, hidden } = engine
  const directions = new Map<string, Promise<Record<string, Direction>>>()
  const dimensions = new Map<string, Dimension>()
  const ends = await tails(OPENINGS, SYSTEM_PROMPT, 1)
  const norms = Object.fromEntries(steerLayers.map((layer) => [layer, mean(ends.map((end) => Math.hypot(...end[layer])))]))

  return { directionsOf, residualAdds, push, targetOf, menu }

  /** A character focus is read against the story as it was when first offered, since its options name the story's people. */
  function directionsOf(dimension: Dimension, story = "") {
    dimensions.set(dimension.name, dimension)
    const key = JSON.stringify(dimension)
    if (!directions.has(key)) {
      const contexts = dimension.name === "focus" && story ? [story.slice(-FOCUS_CONTEXT_CHARS)] : OPENINGS
      directions.set(key, extract(dimension, contexts))
    }
    return directions.get(key)!
  }

  async function extract(dimension: Dimension, contexts: string[]): Promise<Record<string, Direction>> {
    if (dimension.kind === "scale") {
      const high = await meanStates(instruction(dimension, dimension.high), contexts)
      const low = await meanStates(instruction(dimension, dimension.low), contexts)
      return { high: perLayer((layer) => unit(high[layer].map((value, i) => value - low[layer][i]))) }
    }
    const states: Direction[] = []
    for (const option of dimension.options) states.push(await meanStates(instruction(dimension, option), contexts))
    const center = perLayer((layer) => meanVector(states.map((state) => state[layer])))
    return Object.fromEntries(
      dimension.options.map((option, index) => [option, perLayer((layer) => unit(states[index][layer].map((value, i) => value - center[layer][i])))]),
    )
  }

  async function meanStates(guidance: string, contexts: string[]) {
    const ends = await tails(contexts, `${SYSTEM_PROMPT} ${guidance}`, TAIL_TOKENS)
    return perLayer((layer) => meanVector(ends.map((end) => end[layer])))
  }

  /** Shorter stories are padded at the end, which the tokens before never see. */
  async function tails(stories: string[], system: string, count: number) {
    const rows = stories.map((story) => engine.storyTokens(story, system).ids)
    const length = Math.max(...rows.map((row) => row.length))
    const { residuals } = await engine.forward(
      rows.map((row) => [...row, ...Array<number>(length - row.length).fill(row.at(-1)!)]),
      { residuals: true },
    )
    return rows.map((row, index) =>
      perLayer((layer) => {
        const end = (index * length + row.length) * hidden
        return meanVector(Array.from({ length: count }, (_, back) => residuals[layer].subarray(end - (back + 1) * hidden, end - back * hidden)))
      }),
    )
  }

  async function residualAdds(controls: Control[], story = "") {
    const adds = perLayer(() => new Float32Array(hidden))
    for (const { dimension, position, strength } of controls) {
      const found = await directionsOf(dimension, story)
      const [value, amount] =
        dimension.kind === "scale" ? ["high", (2 * position - 1) * strength] : [dimension.options[optionIndexAt(dimension.options.length, position)], strength]
      const pushed = push(found[value], amount)
      for (const layer of steerLayers) adds[layer].forEach((_, i) => (adds[layer][i] += pushed[layer][i]))
    }
    return adds
  }

  function push(direction: Direction, amount: number) {
    return perLayer((layer) => direction[layer].map((value) => value * amount * GAIN * norms[layer]))
  }

  /**
   * A control in words ("tension: somewhat strained", "tone: wry") as its direction and sign: 1 toward the high pole or the option, -1 toward the
   * low pole, 0 for midway.
   */
  async function targetOf(control: string): Promise<{ direction: Direction; sign: number } | undefined> {
    const [name, ...rest] = control.split(":")
    const value = rest.join(":").trim()
    const dimension = dimensions.get(name.trim())
    if (!dimension) return undefined
    const found = await directionsOf(dimension)
    if (dimension.kind === "choice") return dimension.options.includes(value) ? { direction: found[value], sign: 1 } : undefined
    if (value.startsWith("midway")) return { direction: found.high, sign: 0 }
    return { direction: found.high, sign: value.endsWith(dimension.high) ? 1 : -1 }
  }

  /**
   * Ranked by how far a push either way moves the next-token distribution, averaged over a choice's options so that having more options is not
   * favored.
   */
  async function menu(text: string, taken: Dimension[], size: number) {
    const takenNames = new Set(taken.map((dimension) => dimension.name.toLowerCase()))
    const free = CATALOG.filter((dimension) => !takenNames.has(dimension.name))
    const characters = charactersOf(text)
    if (!takenNames.has("focus") && characters.length) free.push({ name: "focus", kind: "choice", options: [...characters.slice(0, 3), "narrator"] })
    if (!free.length) return []

    const { ids } = engine.storyTokens(text)
    const tries: { name: string; direction: Direction }[] = []
    for (const dimension of free) {
      for (const direction of Object.values(await directionsOf(dimension, text))) tries.push({ name: dimension.name, direction })
    }
    const read = engine.emptyCache(1)
    const shifts = new Map<string, number[]>()
    try {
      await engine.forward([ids.slice(0, -1)], { cache: read })
      for (let start = 0; start < tries.length; start += MENU_BATCH / 2) {
        const batch = tries.slice(start, start + MENU_BATCH / 2)
        const pushes = batch.flatMap(({ direction }) => [push(direction, 1), push(direction, -1)])
        const forked = await engine.forkCache(read, pushes.length)
        const { logits } = await engine
          .forward(
            pushes.map(() => [ids.at(-1)!]),
            { cache: forked, steer: perLayer((layer) => pushes.map((pushed) => pushed[layer])) },
          )
          .finally(() => engine.release(forked))
        batch.forEach(({ name }, index) => shifts.set(name, [...(shifts.get(name) ?? []), divergence(logits[2 * index], logits[2 * index + 1])]))
      }
    } finally {
      engine.release(read)
    }
    return free.sort((a, b) => mean(shifts.get(b.name)!) - mean(shifts.get(a.name)!)).slice(0, size)
  }

  function perLayer<T>(of: (layer: number) => T): Record<number, T> {
    return Object.fromEntries(steerLayers.map((layer) => [layer, of(layer)]))
  }
}

function instruction(dimension: Dimension, value: string) {
  if (dimension.name === "focus")
    return value === "narrator" ? "Write the next passage from the narrator's view." : `Write the next passage following ${value}.`
  if (dimension.kind === "scale") return `Write the next passage so its ${dimension.name} is extremely ${value}.`
  return `Write the next passage as ${value} (${dimension.name}).`
}

function charactersOf(text: string) {
  const found = [...text.matchAll(CAPITALIZED)].filter((match) => !NOT_NAMES.has(match[0]))
  const counts = new Map<string, number>()
  for (const [word] of found) counts.set(word, (counts.get(word) ?? 0) + 1)
  const inside = new Set(found.filter((match) => !SENTENCE_STARTS_AFTER.includes(text.slice(0, match.index).trimEnd().slice(-1))).map(([word]) => word))
  return [...counts]
    .sort((a, b) => b[1] - a[1])
    .filter(([word, count]) => inside.has(word) || count > 1)
    .map(([word]) => word)
}

function divergence(from: ArrayLike<number>, to: ArrayLike<number>) {
  const [p, q] = [logSoftmax(from), logSoftmax(to)]
  let sum = 0
  for (let i = 0; i < p.length; i++) sum += Math.exp(p[i]) * (p[i] - q[i])
  return sum
}

function logSoftmax(logits: ArrayLike<number>) {
  let max = -Infinity
  for (let i = 0; i < logits.length; i++) max = Math.max(max, logits[i])
  let total = 0
  for (let i = 0; i < logits.length; i++) total += Math.exp(logits[i] - max)
  const offset = max + Math.log(total)
  return Float64Array.from(logits, (value) => value - offset)
}

function unit(vector: Float32Array) {
  const norm = Math.hypot(...vector) + 1e-6
  return vector.map((value) => value / norm)
}

function meanVector(vectors: ArrayLike<number>[]) {
  const sum = new Float32Array(vectors[0].length)
  for (const vector of vectors) for (let i = 0; i < sum.length; i++) sum[i] += vector[i] / vectors.length
  return sum
}

function mean(values: number[]) {
  return values.reduce((sum, value) => sum + value, 0) / values.length
}
