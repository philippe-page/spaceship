const URL_STORAGE = "local-engine-url"

const DEFAULT_URL = "http://localhost:8765"

export type EngineModel = {
  id: string
  name: string
  /** Download size in bytes. */
  size: number
  downloaded: boolean
}

/** The model the engine has loaded, if any, and the ones it can load. */
export type EngineHealth = { model: string | null; models: EngineModel[] }

export type LoadProgress = { status: "downloading"; done: number; total: number } | { status: "loading" } | { status: "ready"; model: string }

export async function engineHealth(signal: AbortSignal): Promise<EngineHealth> {
  const response = await fetchEngine("/health", { signal })
  if (!response.ok) throw new Error(`${response.status} ${await response.text()}`)
  return response.json()
}

/** Posts `body` to the local MLX engine and streams the newline-delimited JSON it answers with; an `error` line throws. */
export async function* streamEngine<T>(path: string, body: unknown, signal: AbortSignal): AsyncIterable<T> {
  const response = await fetchEngine(path, {
    method: "POST",
    signal,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
  if (!response.ok) throw new Error(`${response.status} ${await response.text()}`)
  if (!response.body) throw new Error("no response body")

  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader()
  let buffer = ""

  while (true) {
    const { done, value } = await reader.read()
    if (done) return
    buffer += value

    const lines = buffer.split("\n")
    buffer = lines.pop() ?? ""

    for (const line of lines) {
      if (!line.trim()) continue
      const item = JSON.parse(line)
      if (item.error) throw new Error(`engine: ${item.error}`)
      yield item as T
    }
  }
}

/**
 * The engine only answers pages served from this machine, so a page served from anywhere else, like the published demo,
 * does not go looking for it: a browser asks its visitor for permission before a public page may reach their machine.
 */
async function fetchEngine(path: string, init: RequestInit) {
  if (!["localhost", "127.0.0.1"].includes(location.hostname)) throw new Error("the engine only answers pages served from this machine")
  return fetch(`${localStorage.getItem(URL_STORAGE)?.trim() || DEFAULT_URL}${path}`, init)
}
