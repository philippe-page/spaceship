import { KEY_STORAGE as CEREBRAS_KEY_STORAGE } from "../llm/cerebras.ts"
import { KEY_STORAGE as GEMINI_KEY_STORAGE } from "../llm/decisions.ts"

/** Each key's field in the dialog's form, by name, and where it is kept for reuse. */
const KEY_FIELDS = { cerebras: CEREBRAS_KEY_STORAGE, gemini: GEMINI_KEY_STORAGE }

/** Sim plays simulated sources; flight calls the hosted models; local runs the MLX engine on this machine. */
export type Mode = "sim" | "flight" | "local"

export function modeOf(url: URL): Mode {
  if (url.searchParams.get("local") === "true") return "local"
  if (url.searchParams.get("live") === "true") return "flight"
  return "sim"
}

/**
 * The sources are chosen when the page loads, so switching reloads it in the other mode; leaving flight or local returns to sim.
 * Launching asks for both API keys, prefilled with the ones kept from last time, and the dialog's submit button returns "live".
 * Its local section, which needs no keys, is told when the dialog opens (`onOpen`) and reloads the page itself once a model is loaded.
 */
export function bindModeToggle(toggle: HTMLButtonElement, dialog: HTMLDialogElement, mode: Mode, onOpen: () => void) {
  const form = dialog.querySelector("form")!
  const inputOf = (name: string) => form.elements.namedItem(name) as HTMLInputElement
  toggle.setAttribute("aria-pressed", `${mode !== "sim"}`)
  toggle.querySelector("b")!.textContent = mode

  toggle.addEventListener("click", () => {
    if (mode !== "sim") return reloadAs("sim")
    for (const [name, storageKey] of Object.entries(KEY_FIELDS)) inputOf(name).value = localStorage.getItem(storageKey) ?? ""
    dialog.showModal()
    onOpen()
  })
  dialog.addEventListener("close", () => {
    const choice = dialog.returnValue
    dialog.returnValue = ""
    if (choice !== "live") return
    for (const [name, storageKey] of Object.entries(KEY_FIELDS)) localStorage.setItem(storageKey, inputOf(name).value.trim())
    reloadAs("flight")
  })
}

export function reloadAs(mode: Mode) {
  const url = new URL(location.href)
  url.searchParams.delete("live")
  url.searchParams.delete("local")
  if (mode === "flight") url.searchParams.set("live", "true")
  if (mode === "local") url.searchParams.set("local", "true")
  location.assign(url)
}
