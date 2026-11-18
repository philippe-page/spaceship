import { showError } from "../dom.ts"
import { engineHealth, streamEngine, type EngineModel, type LoadProgress } from "../llm/local-engine.ts"

const MODEL_STORAGE = "local-model"
const HEALTH_TIMEOUT_MS = 10_000
const FLIGHT = "flight"

type LocalLaunchOptions = { dialog: HTMLDialogElement; list: HTMLElement; status: HTMLOutputElement; isLocal: () => boolean; enterLocal: () => void }

/** A model lives only as long as its page, so one launched from another mode loads here, in the dialog, and local mode is entered without a reload. */
export function createLocalLaunch({ dialog, list, status, isLocal, enterLocal }: LocalLaunchOptions) {
  const form = dialog.querySelector("form")!
  const engines = form.querySelector<HTMLFieldSetElement>(".engines")!
  const keys = form.querySelector<HTMLFieldSetElement>(".keys")!
  const launchButton = dialog.querySelector<HTMLButtonElement>("button[value=live]")!
  let following: AbortController | undefined
  let picked: string | undefined

  // The engine finishes a download or load the dialog stops following, so the model is there the next time.
  dialog.addEventListener("close", () => following?.abort())
  form.addEventListener("change", () => {
    picked = engine()
    showKeys()
  })
  form.addEventListener("submit", (event) => {
    if (engine() === FLIGHT) return
    event.preventDefault()
    launch(engine())
  })

  return { refresh, start }

  async function refresh() {
    list.replaceChildren()
    status.value = ""
    try {
      await listModels()
    } catch (error) {
      showError(status, error)
    }
    check(picked ?? localStorage.getItem(MODEL_STORAGE) ?? FLIGHT)
  }

  async function listModels() {
    const { models, unsupported } = await engineHealth(AbortSignal.timeout(HEALTH_TIMEOUT_MS))
    if (unsupported) status.value = `no local mode in this browser: ${unsupported}`
    else list.replaceChildren(...models.map((model) => modelItem(model, { download, remove })))
  }

  function engine() {
    return (form.elements.namedItem("engine") as RadioNodeList).value
  }

  function check(value: string) {
    const radio = radioOf(form, value) ?? radioOf(form, FLIGHT)!
    radio.checked = true
    showKeys()
  }

  // A disabled fieldset's inputs are neither required nor sent, so a model launches without keys.
  function showKeys() {
    keys.disabled = engine() !== FLIGHT
  }

  /** A page that starts in local mode loads the model last launched, with the dialog up until it is ready. */
  function start() {
    dialog.showModal()
    const chosen = localStorage.getItem(MODEL_STORAGE)
    if (!chosen) return refresh()
    listModels().then(
      () => check(chosen),
      () => {},
    )
    return launch(chosen)
  }

  async function launch(id: string) {
    if (!(await follow("/load", id, describe({ status: "loading" })))) return
    localStorage.setItem(MODEL_STORAGE, id)
    dialog.close()
    if (!isLocal()) enterLocal()
  }

  async function download(model: EngineModel) {
    if (!(await follow("/download", model.id))) return
    await refresh()
    status.value = `downloaded ${model.name}`
  }

  // Deleting the chosen model forgets the choice too, or the next visit would download it again.
  async function remove(model: EngineModel) {
    try {
      for await (const _ of streamEngine("/delete", { model: model.id }, AbortSignal.timeout(HEALTH_TIMEOUT_MS)));
      if (localStorage.getItem(MODEL_STORAGE) === model.id) localStorage.removeItem(MODEL_STORAGE)
      await refresh()
      status.value = `deleted ${model.name}, ${gigabytes(model.size)} freed`
    } catch (error) {
      showError(status, error)
    }
  }

  /** Shows the engine's progress on a model while the choices wait; resolves to whether the work finished. */
  async function follow(path: string, id: string, firstMessage = "") {
    const { signal } = (following = new AbortController())
    engines.disabled = launchButton.disabled = true
    status.value = firstMessage
    try {
      for await (const progress of streamEngine<LoadProgress>(path, { model: id }, signal)) status.value = describe(progress)
      return true
    } catch (error) {
      if (signal.aborted) return false
      await refresh()
      showError(status, error)
      return false
    } finally {
      engines.disabled = launchButton.disabled = false
    }
  }
}

type StorageActions = { download: (model: EngineModel) => void; remove: (model: EngineModel) => void }

function modelItem(model: EngineModel, { download, remove }: StorageActions) {
  const item = document.createElement("li")
  const radio = Object.assign(document.createElement("input"), { type: "radio", name: "engine", value: model.id })
  const name = Object.assign(document.createElement("span"), { textContent: model.name })
  const size = Object.assign(document.createElement("small"), { textContent: gigabytes(model.size) })
  const label = Object.assign(document.createElement("label"), { className: "engine" })
  label.append(radio, name, size)
  const [text, ariaLabel, act] = model.downloaded
    ? ["delete", `delete ${model.name} from this browser`, remove]
    : ["download", `download ${model.name} to this browser`, download]
  const storage = Object.assign(document.createElement("button"), { type: "button", textContent: text })
  storage.setAttribute("aria-label", ariaLabel)
  storage.addEventListener("click", () => act(model))
  item.append(label, storage)
  return item
}

function radioOf(form: HTMLFormElement, value: string) {
  return form.querySelector<HTMLInputElement>(`input[name=engine][value="${CSS.escape(value)}"]`)
}

function describe(progress: LoadProgress) {
  if (progress.status === "downloading") return `downloading the model: ${gigabytes(progress.done)} of ${gigabytes(progress.total)}`
  if (progress.status === "loading") return "loading the model…"
  return "ready"
}

function gigabytes(bytes: number) {
  return `${(bytes / 1e9).toFixed(1)} GB`
}
