import { showError } from "../dom.ts"
import { engineHealth, streamEngine, type EngineModel, type LoadProgress } from "../llm/local-engine.ts"

const HEALTH_TIMEOUT_MS = 2000

type LocalLaunchOptions = { dialog: HTMLDialogElement; list: HTMLElement; status: HTMLOutputElement; onReady: () => void }

/**
 * The launch dialog's local section: one button per model the engine on this machine can steer. Pressing one loads it, downloading it first
 * if it is not on disk yet, with the progress on the button, and `onReady` runs once it is loaded. Nothing is downloaded until then.
 */
export function createLocalLaunch({ dialog, list, status, onReady }: LocalLaunchOptions) {
  let launching: AbortController | undefined

  // Closing the dialog stops following a launch; the engine finishes a download it started, so the model is on disk next time.
  dialog.addEventListener("close", () => launching?.abort())

  return { refresh, openUnlessLoaded }

  /** Lists the engine's models; resolves to the one it has loaded, if any. */
  async function refresh() {
    list.replaceChildren()
    status.value = "looking for the engine…"
    try {
      const { model, models } = await engineHealth(AbortSignal.timeout(HEALTH_TIMEOUT_MS))
      list.replaceChildren(...models.map((option) => modelItem(option, option.id === model, launch)))
      status.value = ""
      return model
    } catch {
      status.value = "no engine: run npm run local from a clone of the repo"
      return null
    }
  }

  /** For a page already in local mode: the engine starts with no model, and forgets it when restarted, so the page asks for one again. */
  async function openUnlessLoaded() {
    if (!(await refresh())) dialog.showModal()
  }

  async function launch(id: string, note: HTMLElement) {
    launching = new AbortController()
    const { signal } = launching
    // The keys' launch button goes too: pressed while a model loads, it would ask for the keys local mode does not need.
    const buttons = [...list.querySelectorAll("button"), ...dialog.querySelectorAll<HTMLButtonElement>("button[value=live]")]
    const before = note.textContent
    for (const button of buttons) button.disabled = true
    status.value = ""
    try {
      for await (const progress of streamEngine<LoadProgress>("/load", { model: id }, signal)) note.textContent = describe(progress)
      onReady()
    } catch (error) {
      note.textContent = before
      for (const button of buttons) button.disabled = false
      if (!signal.aborted) showError(status, error)
    }
  }
}

function modelItem(model: EngineModel, isLoaded: boolean, launch: (id: string, note: HTMLElement) => void) {
  const item = document.createElement("li")
  const button = Object.assign(document.createElement("button"), { type: "button" })
  const note = document.createElement("small")
  note.textContent = isLoaded ? "loaded" : model.downloaded ? "on disk" : `download ${gigabytes(model.size)}`
  button.append(model.name, note)
  button.addEventListener("click", () => launch(model.id, note))
  item.append(button)
  return item
}

function describe(progress: LoadProgress) {
  if (progress.status === "downloading") return `${gigabytes(progress.done)} of ${gigabytes(progress.total)}`
  if (progress.status === "loading") return "loading…"
  return "ready"
}

function gigabytes(bytes: number) {
  return `${(bytes / 1e9).toFixed(1)} GB`
}
