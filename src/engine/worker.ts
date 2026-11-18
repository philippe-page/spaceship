import * as ort from "onnxruntime-web/webgpu"
import { createHost } from "./host.ts"

/** The page posts a request with an id and gets back its items, then `done` or an `error`; posting `cancel` for the id stops it at its next item. */

export type Request = { id: number } & ({ path: string; body: unknown } | { cancel: true })
export type Reply = { id: number } & ({ item: unknown } | { done: true } | { error: string })

// The runtime's WebAssembly is 29 MB, so the site does not carry it: it comes from the package's own CDN copy when a model first loads.
ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ort.env.versions.web}/dist/`
// More threads would need a cross-origin isolated page, which a static host cannot give; the work is on the GPU anyway.
ort.env.wasm.numThreads = 1

const host = createHost()
const running = new Set<number>()

addEventListener("message", async ({ data }: MessageEvent<Request>) => {
  if ("cancel" in data) return void running.delete(data.id)
  const { id, path, body } = data
  running.add(id)
  try {
    for await (const item of host.answer(path, body)) {
      if (!running.has(id)) return
      if (item !== undefined) reply({ id, item })
    }
    reply({ id, done: true })
  } catch (error) {
    reply({ id, error: error instanceof Error ? error.message : String(error) })
  } finally {
    running.delete(id)
  }
})

function reply(message: Reply) {
  postMessage(message)
}
