import { domPointAt, textOffsetOf } from "./text-offsets.ts"

/** A token the model weighed for one position, with its probability under the steered model and under an unsteered copy that read the same text. */
export type Candidate = { token: string; on: number; off: number; chosen: boolean }

/** One generated token: the candidates for it, and the token the unsteered copy would have written instead, if steering flipped it. */
export type ComparedStep = { top: Candidate[]; instead: string | null }

export type InsideView = ReturnType<typeof createInsideView>

/**
 * Where a generated token sits in the story, in non-space characters from the start: the shown text breaks lines differently from the generated text,
 * but keeps every other character in order.
 */
type Token = { at: number; length: number; step: ComparedStep }

type InsideViewOptions = { textElement: HTMLElement; sidebar: HTMLElement; oddsElement: HTMLElement; isAvailable: boolean }

const STORAGE_KEY = "inside"
/** Flipped words are underlined in steps of ink, 1 to 4, by how many times likelier steering made them: one step up at each bound. */
const LIFT_BOUNDS = [1.5, 4, 12]
const CANDIDATE_ROWS = 6
const FLIP_ROWS = 16
const ODDS_OFFSET_PX = 6

/**
 * Shows what steering does inside the local model, while the engine runs an unsteered copy of the model beside the steered one on the same text:
 * words steering flipped get a dotted underline in the story; a sidebar shows, at the main knob, the candidates the model weighed for the next word with and without
 * steering, and lists the words steering flipped so far; hovering a generated word shows its candidates the same way.
 * Only available in local mode, where it starts on; turning it off is remembered.
 */
export function createInsideView({ textElement, sidebar, oddsElement, isAvailable }: InsideViewOptions) {
  const candidateList = sidebar.querySelector<HTMLOListElement>(".at-knob ol")!
  const flipSection = sidebar.querySelector<HTMLElement>(".flips")!
  const flipCount = sidebar.querySelector<HTMLElement>("[data-field='flips']")!
  const flipList = sidebar.querySelector<HTMLOListElement>(".flips ol")!
  let tokens: Token[] = []
  let isOn = isAvailable && localStorage.getItem(STORAGE_KEY) !== "off"
  let frame: number | undefined
  sidebar.hidden = !isOn

  if (isAvailable) addEventListener("pointermove", showOddsAt, { passive: true })

  return { isOn: () => isOn, toggle, start, record, repaint }

  function toggle() {
    isOn = !isOn
    localStorage.setItem(STORAGE_KEY, isOn ? "on" : "off")
    sidebar.hidden = !isOn
    oddsElement.hidden = true
    repaint()
  }

  /** A generation starts after `from` non-space characters; what was recorded past there belonged to text it replaces. */
  function start(from: number) {
    tokens = tokens.filter((token) => token.at < from)
  }

  function record(at: number, length: number, step: ComparedStep) {
    if (length) tokens.push({ at, length, step })
    repaint()
  }

  /** Call whenever the shown text changes. */
  function repaint() {
    if (isAvailable) frame ??= requestAnimationFrame(paint)
  }

  function paint() {
    frame = undefined
    const positions = isOn ? nonSpacePositions(textElement.textContent ?? "") : []
    const shown = tokens.filter(({ at, length }) => at + length <= positions.length)
    const flipped = shown.filter(({ step }) => step.instead !== null)
    for (let liftStep = 1; liftStep <= LIFT_BOUNDS.length + 1; liftStep++) {
      const inStep = flipped.filter(({ step }) => liftStepOf(step) === liftStep)
      CSS.highlights.set(`steering-flipped-${liftStep}`, new Highlight(...inStep.flatMap((token) => rangeOfToken(positions, token) ?? [])))
    }
    if (isOn) paintSidebar(positions.length, shown.length, flipped)
  }

  /** The step for the next word at the knob (or the last one written, at the end of the text), and the flipped words so far, newest first. */
  function paintSidebar(knob: number, shownCount: number, flipped: Token[]) {
    const atKnob = tokens.findLast(({ at }) => at <= knob)
    candidateList.replaceChildren(...(atKnob ? candidateRows(atKnob.step) : []))
    flipSection.hidden = shownCount === 0
    flipCount.textContent = `${flipped.length} of ${shownCount}`
    flipList.replaceChildren(...flipped.slice(-FLIP_ROWS).reverse().map(flipRow))
  }

  function showOddsAt(event: PointerEvent) {
    const token = isOn && !event.buttons ? tokenUnder(event.clientX, event.clientY) : undefined
    if (!token) return void (oddsElement.hidden = true)
    const { step, rect } = token
    const list = Object.assign(document.createElement("ol"), { className: "candidates" })
    list.append(...candidateRows(step))
    oddsElement.replaceChildren(list)
    oddsElement.style.translate = `${rect.left + scrollX}px ${rect.bottom + scrollY + ODDS_OFFSET_PX}px`
    oddsElement.hidden = false
  }

  /** The recorded token whose characters are under the viewport point, with its box there. */
  function tokenUnder(x: number, y: number) {
    const position = document.caretPositionFromPoint(x, y)
    if (!position || !textElement.contains(position.offsetNode)) return
    const content = textElement.textContent ?? ""
    const count = nonSpaceLength(content.slice(0, textOffsetOf(textElement, position.offsetNode, position.offset)))
    const near = tokens.filter(({ at, length }) => count >= at && count <= at + length)
    if (!near.length) return
    const positions = nonSpacePositions(content)
    for (const token of near) {
      const rect = [...(rangeOfToken(positions, token)?.getClientRects() ?? [])].find((box) => box.width > 0 && contains(box, x, y))
      if (rect) return { step: token.step, rect }
    }
  }

  /** `positions` are the shown text's non-space characters' offsets (see `nonSpacePositions`). */
  function rangeOfToken(positions: number[], { at, length }: Token) {
    if (at + length > positions.length) return
    const range = document.createRange()
    range.setStart(...domPointAt(textElement, positions[at]))
    range.setEnd(...domPointAt(textElement, positions[at + length - 1] + 1))
    return range
  }
}

/** How far into the story `text` reaches, in the count the inside view places tokens by. */
export function nonSpaceLength(text: string) {
  return text.replace(/\s/g, "").length
}

/** The offset in `text` of each non-space character, in order. */
function nonSpacePositions(text: string) {
  const positions: number[] = []
  for (let i = 0; i < text.length; i++) if (!/\s/.test(text[i])) positions.push(i)
  return positions
}

function contains({ left, right, top, bottom }: DOMRect, x: number, y: number) {
  return x >= left && x <= right && y >= top && y <= bottom
}

/** How many times likelier steering made the word it wrote: infinitely, for a word the unsteered model could not have drawn at all. */
function liftOf({ top }: ComparedStep) {
  const chosen = top.find((candidate) => candidate.chosen)
  return chosen ? chosen.on / chosen.off : 1
}

function liftStepOf(step: ComparedStep) {
  return 1 + LIFT_BOUNDS.filter((bound) => liftOf(step) >= bound).length
}

function liftLabel(step: ComparedStep) {
  const lift = liftOf(step)
  if (!Number.isFinite(lift)) return "×∞"
  return `×${lift < 10 ? lift.toFixed(1) : Math.round(lift)}`
}

/**
 * A legend row, then the likeliest candidates with their probability steered (solid bar) and unsteered (hairline bar);
 * the word written, when sampled from further down, takes the last row.
 */
function candidateRows({ top }: ComparedStep) {
  const legend = Object.assign(document.createElement("li"), { className: "legend", innerHTML: `<span></span><span></span><span class="pct">  on  off</span>` })
  const shown = top.slice(0, CANDIDATE_ROWS)
  const chosen = top.find((candidate) => candidate.chosen)
  if (chosen && !shown.includes(chosen)) shown[shown.length - 1] = chosen
  return [legend, ...shown.map(candidateRow)]
}

function candidateRow({ token, on, off, chosen }: Candidate) {
  const row = document.createElement("li")
  row.toggleAttribute("data-chosen", chosen)
  row.style.setProperty("--on", `${on}`)
  row.style.setProperty("--off", `${off}`)
  const label = Object.assign(document.createElement("span"), { className: "token", textContent: `${chosen ? "›" : " "} ${visible(token)}` })
  const bars = Object.assign(document.createElement("span"), { className: "bars", innerHTML: `<i class="on"></i><i class="off"></i>` })
  const numbers = Object.assign(document.createElement("span"), { className: "pct", textContent: `${percent(on)} ${percent(off)}` })
  row.append(label, bars, numbers)
  return row
}

/** What the unsteered model would have written, then what steering wrote instead. */
function flipRow({ step }: Token) {
  const row = document.createElement("li")
  const written = step.top.find((candidate) => candidate.chosen)?.token ?? ""
  row.append(
    Object.assign(document.createElement("s"), { textContent: visible(step.instead ?? "") }),
    document.createTextNode(" → "),
    Object.assign(document.createElement("b"), { textContent: visible(written) }),
    Object.assign(document.createElement("small"), { textContent: liftLabel(step) }),
  )
  return row
}

/** A token as a word: without the space most tokens start with, line breaks drawn, and a token of spaces alone shown as one. */
function visible(token: string) {
  const shown = token.replace(/^ /, "").replace(/\n/g, "↵")
  return shown.trim() || shown.includes("↵") ? shown : "␣"
}

function percent(p: number) {
  return `${Math.round(p * 100)}%`.padStart(4)
}
