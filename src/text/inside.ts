import type { Candidate, ComparedStep } from "../engine/generation.ts"
import type { ModelLayout } from "../engine/host.ts"
import type { LayerKind } from "../engine/instruments.ts"
import { domPointAt, textOffsetOf } from "./text-offsets.ts"

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
/** Drift, and push on the same scale, that fills a layer's bar. */
const DRIFT_FULL = 0.5
const HISTORY_STEPS = 48
/** Layer bars show drift averaged over this many tokens up to the knob's, so they move with the text rather than with each token. */
const SMOOTHING_STEPS = 8
const KIND_LABELS: Record<LayerKind, string> = { attention: "attn", "windowed attention": "slide", "linear attention": "linear", convolution: "conv" }

export function createInsideView({ textElement, sidebar, oddsElement, isAvailable }: InsideViewOptions) {
  const layerSection = sidebar.querySelector<HTMLElement>(".layers")!
  const modelField = sidebar.querySelector<HTMLElement>("[data-field='model']")!
  const layerList = sidebar.querySelector<HTMLOListElement>(".layers ol")!
  const history = sidebar.querySelector<HTMLCanvasElement>(".layers canvas")!
  const candidateList = sidebar.querySelector<HTMLOListElement>(".at-knob ol")!
  const flipSection = sidebar.querySelector<HTMLElement>(".flips")!
  const flipCount = sidebar.querySelector<HTMLElement>("[data-field='flips']")!
  const flipList = sidebar.querySelector<HTMLOListElement>(".flips ol")!
  let tokens: Token[] = []
  let layerRows: HTMLElement[] = []
  let prefersOn = localStorage.getItem(STORAGE_KEY) !== "off"
  let frame: number | undefined
  sidebar.hidden = !isOn()

  addEventListener("pointermove", showOddsAt, { passive: true })

  return { isOn, enable, toggle, lay, start, record, repaint }

  function isOn() {
    return isAvailable && prefersOn
  }

  function enable() {
    isAvailable = true
    sidebar.hidden = !isOn()
    repaint()
  }

  function toggle() {
    prefersOn = !prefersOn
    localStorage.setItem(STORAGE_KEY, prefersOn ? "on" : "off")
    sidebar.hidden = !isOn()
    oddsElement.hidden = true
    repaint()
  }

  function lay({ model, layers }: ModelLayout) {
    modelField.textContent = model
    layerRows = layers.map(layerRow)
    layerList.replaceChildren(...layerRows.toReversed())
    history.style.gridRow = `1 / span ${layers.length}`
  }

  /** A generation starts after `from` non-space characters; what was recorded past there belonged to text it replaces. */
  function start(from: number) {
    tokens = tokens.filter((token) => token.at < from)
  }

  function record(at: number, length: number, step: ComparedStep) {
    if (length) tokens.push({ at, length, step })
    repaint()
  }

  function repaint() {
    if (isAvailable) frame ??= requestAnimationFrame(paint)
  }

  function paint() {
    frame = undefined
    const positions = isOn() ? nonSpacePositions(textElement.textContent ?? "") : []
    const shown = tokens.filter(({ at, length }) => at + length <= positions.length)
    const flipped = shown.filter(({ step }) => step.instead !== null)
    for (let liftStep = 1; liftStep <= LIFT_BOUNDS.length + 1; liftStep++) {
      const inStep = flipped.filter(({ step }) => liftStepOf(step) === liftStep)
      CSS.highlights.set(`steering-flipped-${liftStep}`, new Highlight(...inStep.flatMap((token) => rangeOfToken(positions, token) ?? [])))
    }
    if (isOn()) paintSidebar(positions.length, shown.length, flipped)
  }

  function paintSidebar(knob: number, shownCount: number, flipped: Token[]) {
    const index = tokens.findLastIndex(({ at }) => at <= knob)
    const atKnob = index === -1 ? undefined : tokens[index]
    paintLayers(tokens.slice(Math.max(0, index + 1 - HISTORY_STEPS), index + 1).map(({ step }) => step))
    candidateList.replaceChildren(...(atKnob ? candidateRows(atKnob.step) : []))
    flipSection.hidden = shownCount === 0
    flipCount.textContent = `${flipped.length} of ${shownCount}`
    flipList.replaceChildren(...flipped.slice(-FLIP_ROWS).reverse().map(flipRow))
  }

  function paintLayers(steps: ComparedStep[]) {
    layerSection.hidden = !steps.length || !layerRows.length
    if (layerSection.hidden) return
    const atKnob = steps.at(-1)!
    const recent = steps.slice(-SMOOTHING_STEPS)
    layerRows.forEach((row, layer) => {
      const drift = recent.reduce((sum, step) => sum + step.drift[layer], 0) / recent.length
      row.style.setProperty("--drift", `${Math.min(1, drift / DRIFT_FULL)}`)
      row.style.setProperty("--push", `${Math.min(1, atKnob.push[layer] / DRIFT_FULL)}`)
      row.toggleAttribute("data-locked", atKnob.isLocked && row.dataset.kind === "attention")
    })
    paintHistory(steps)
  }

  function paintHistory(steps: ComparedStep[]) {
    const scale = devicePixelRatio
    const { clientWidth, clientHeight } = history
    if (history.width !== clientWidth * scale || history.height !== clientHeight * scale) {
      history.width = clientWidth * scale
      history.height = clientHeight * scale
    }
    const context = history.getContext("2d")!
    context.setTransform(scale, 0, 0, scale, 0, 0)
    context.clearRect(0, 0, clientWidth, clientHeight)
    context.fillStyle = getComputedStyle(history).color
    const column = clientWidth / HISTORY_STEPS
    const row = clientHeight / layerRows.length
    const offset = HISTORY_STEPS - steps.length
    steps.forEach(({ drift }, step) => {
      drift.forEach((value, layer) => {
        context.globalAlpha = Math.min(1, value / DRIFT_FULL)
        context.fillRect((offset + step) * column, (layerRows.length - 1 - layer) * row, Math.max(1, column - 1), Math.max(1, row - 1))
      })
    })
    context.globalAlpha = 1
  }

  function showOddsAt(event: PointerEvent) {
    const token = isOn() && !event.buttons ? tokenUnder(event.clientX, event.clientY) : undefined
    if (!token) return void (oddsElement.hidden = true)
    const { step, rect } = token
    const list = Object.assign(document.createElement("ol"), { className: "candidates" })
    list.append(...candidateRows(step))
    oddsElement.replaceChildren(list)
    oddsElement.style.translate = `${rect.left + scrollX}px ${rect.bottom + scrollY + ODDS_OFFSET_PX}px`
    oddsElement.hidden = false
  }

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

  function rangeOfToken(positions: number[], { at, length }: Token) {
    if (at + length > positions.length) return
    const range = document.createRange()
    range.setStart(...domPointAt(textElement, positions[at]))
    range.setEnd(...domPointAt(textElement, positions[at + length - 1] + 1))
    return range
  }
}

export function nonSpaceLength(text: string) {
  return text.replace(/\s/g, "").length
}

function nonSpacePositions(text: string) {
  const positions: number[] = []
  for (let i = 0; i < text.length; i++) if (!/\s/.test(text[i])) positions.push(i)
  return positions
}

function contains({ left, right, top, bottom }: DOMRect, x: number, y: number) {
  return x >= left && x <= right && y >= top && y <= bottom
}

function layerRow(kind: LayerKind, index: number) {
  const row = document.createElement("li")
  row.dataset.kind = kind
  row.innerHTML = `<span class="num">${index}</span><span class="kind">${KIND_LABELS[kind]}</span><span class="bar"><i class="drift"></i><i class="push"></i></span><span class="lock">◆</span>`
  return row
}

/** Infinite for a word the unsteered model could not have drawn. */
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

/** The word written, when sampled from further down, takes the last row. */
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

function visible(token: string) {
  const shown = token.replace(/^ /, "").replace(/\n/g, "↵")
  return shown.trim() || shown.includes("↵") ? shown : "␣"
}

function percent(p: number) {
  return `${Math.round(p * 100)}%`.padStart(4)
}
