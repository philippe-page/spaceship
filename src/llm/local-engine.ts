import type { EngineHealth } from "../engine/host.ts"
import type { Reply, Request } from "../engine/worker.ts"

export type { EngineHealth, EngineModel, LoadProgress } from "../engine/host.ts"

let worker: Worker | undefined
let lastId = 0
const listeners = new Map<number, (reply: Reply) => void>()

export async function engineHealth(signal: AbortSignal): Promise<EngineHealth> {
  for await (const health of streamEngine<EngineHealth>("/health", {}, signal)) return health
  throw new Error("engine: no answer")
}

export async function* streamEngine<T>(path: string, body: unknown, signal: AbortSignal): AsyncIterable<T> {
  const id = ++lastId
  const replies: Reply[] = []
  let wake = () => {}
  listeners.set(id, (reply) => {
    replies.push(reply)
    wake()
  })
  const onAbort = () => wake()
  signal.addEventListener("abort", onAbort)
  post({ id, path, body })
  try {
    while (true) {
      signal.throwIfAborted()
      const reply = replies.shift()
      if (!reply) await new Promise<void>((resolve) => (wake = resolve))
      else if ("error" in reply) throw new Error(`engine: ${reply.error}`)
      else if ("done" in reply) return
      else yield reply.item as T
    }
  } finally {
    signal.removeEventListener("abort", onAbort)
    listeners.delete(id)
    post({ id, cancel: true })
  }
}

function post(request: Request) {
  if (!worker) {
    worker = new Worker(new URL("../engine/worker.ts", import.meta.url), { type: "module" })
    worker.addEventListener("message", ({ data }: MessageEvent<Reply>) => listeners.get(data.id)?.(data))
    worker.addEventListener("error", (event) => {
      for (const [id, listener] of listeners) listener({ id, error: event.message || "the engine stopped" })
    })
  }
  worker.postMessage(request)
}
