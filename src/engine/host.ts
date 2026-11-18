import { focusBoxes, influence, realization, relevance } from "./analysis.ts"
import { complete } from "./generation.ts"
import type { LayerKind } from "./instruments.ts"
import { loadEngine, type Engine } from "./model.ts"
import { deleteModel, fetchModel, isDownloaded, MODELS, type Model, type ModelFiles } from "./models.ts"
import { CATALOG, findVectors, type Vectors } from "./steering.ts"

export type EngineModel = {
  id: string
  name: string
  size: number
  downloaded: boolean
}

export type EngineHealth = { model: string | null; models: EngineModel[]; unsupported?: string }

export type ModelLayout = { model: string; layers: LayerKind[] }

export type LoadProgress = { status: "downloading"; done: number; total: number } | { status: "loading" } | { status: "ready"; model: string }

const PROGRESS_INTERVAL_MS = 500

/** An undefined item is only a place to stop. */
export function createHost() {
  let loaded: { engine: Engine; vectors: Vectors } | undefined
  let preparing: ReturnType<typeof prepare> | undefined
  const fetching = new Map<string, { done: number; files: Promise<ModelFiles> }>()

  return { answer }

  // The requests are the page's own, typed where each source makes them (see local-sources.ts).
  async function* answer(path: string, request: any): AsyncGenerator<unknown> {
    if (path === "/health") return yield await health()
    if (path === "/load") return yield* load(request.model)
    if (path === "/download") return yield* download(request.model)
    if (path === "/delete") return await deleteModel(modelOf(request.model))
    // A request that arrives while a model loads waits for it.
    await preparing?.ready.catch(() => {})
    if (!loaded) throw new Error("no model loaded")
    const { engine, vectors } = loaded
    if (path === "/complete") yield* complete(engine, vectors, request)
    else if (path === "/attention") yield* await focusBoxes(engine, request.text)
    else if (path === "/influence") yield* await influence(engine, request.sentences, request.focus)
    else if (path === "/relevance") yield* await relevance(engine, request)
    else if (path === "/realization") yield* await realization(engine, vectors, request)
    else if (path === "/dimensions") yield* await vectors.menu(request.text, request.taken, request.size)
    else if (path === "/layers") yield { model: engine.name, layers: engine.layerKinds } satisfies ModelLayout
    else throw new Error(`unknown request ${path}`)
  }

  async function health(): Promise<EngineHealth> {
    const models = await Promise.all(MODELS.map(async (model) => ({ id: model.id, name: model.name, size: model.size, downloaded: await isDownloaded(model) })))
    return { model: loaded?.engine.id ?? null, models, unsupported: await unsupported() }
  }

  /** A page that stops listening leaves the load running. */
  async function* load(id: string): AsyncGenerator<LoadProgress> {
    const model = modelOf(id)
    if (loaded?.engine.id !== id) {
      const current = preparing?.id === id ? preparing : (preparing = prepare(model))
      yield* follow(current.ready, () => current.progress())
    }
    yield { status: "ready", model: id }
  }

  async function* download(id: string): AsyncGenerator<LoadProgress> {
    const model = modelOf(id)
    if (await isDownloaded(model)) return
    const fetch = fetchOnce(model)
    yield* follow(fetch.files, (): LoadProgress => ({ status: "downloading", done: fetch.done, total: model.size }))
  }

  function prepare(model: Model) {
    const state = { id: model.id, progress: (): LoadProgress => ({ status: "loading" }), ready: Promise.resolve() }
    state.ready = (async () => {
      const isNew = !(await isDownloaded(model))
      const fetch = fetchOnce(model)
      if (isNew) state.progress = () => ({ status: "downloading", done: fetch.done, total: model.size })
      const files = await fetch.files
      state.progress = () => ({ status: "loading" })
      const engine = await loadEngine(model, files)
      const vectors = await findVectors(engine)
      for (const dimension of CATALOG) await vectors.directionsOf(dimension)
      loaded = { engine, vectors }
    })().finally(() => {
      if (preparing === state) preparing = undefined
    })
    return state
  }

  function fetchOnce(model: Model) {
    const under = fetching.get(model.id)
    if (under) return under
    const fetch = { done: 0, files: fetchModel(model, (done) => (fetch.done = done)).finally(() => fetching.delete(model.id)) }
    fetching.set(model.id, fetch)
    return fetch
  }
}

async function* follow<T>(ready: Promise<unknown>, progress: () => T) {
  while (!(await Promise.race([ready.then(() => true), wait(PROGRESS_INTERVAL_MS)]))) yield progress()
}

async function unsupported() {
  if (!("gpu" in navigator) || !("Float16Array" in globalThis)) return "it needs WebGPU, which Chrome and Edge have"
  const adapter = await navigator.gpu.requestAdapter()
  // The models are stored in 16-bit floats, which a GPU supports only as an optional feature.
  if (!adapter?.features.has("shader-f16")) return "its GPU cannot compute in 16-bit floats, which the models need"
}

function modelOf(id: string) {
  const model = MODELS.find((candidate) => candidate.id === id)
  if (!model) throw new Error(`unknown model ${id}`)
  return model
}

function wait(ms: number) {
  return new Promise<false>((resolve) => setTimeout(() => resolve(false), ms))
}
