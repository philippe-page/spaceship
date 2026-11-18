export type Model = {
  id: string
  name: string
  repo: string
  revision: string
  network: NetworkFiles
  embedding?: NetworkFiles
  size: number
  hidden: number
}

type NetworkFiles = { graph: string; weights: string }

export type Network = { graph: Uint8Array; weights: Uint8Array; weightsName: string }

export type ModelFiles = { tokenizer: object; tokenizerConfig: Record<string, unknown>; chatTemplate: string; network: Network; embedding?: Network }

export const MODELS: Model[] = [
  {
    id: "lfm3-1.7b",
    name: "LFM3 1.7B",
    repo: "onnx-community/LFM3-1.7B-ONNX",
    revision: "ab9fec4c681a25bec3b48b0b97a62fc2660a2985",
    network: { graph: "onnx/model_q4f16.onnx", weights: "onnx/model_q4f16.onnx_data" },
    size: 1_087_789_056,
    hidden: 2048,
  },
  {
    id: "qwen3.5-2b",
    name: "Qwen3.5 2B",
    repo: "onnx-community/Qwen3.5-2B-ONNX-OPT",
    revision: "ed446636739bf426e73c5a00714fd8e010d36c21",
    network: { graph: "onnx/decoder_model_merged_q4f16.onnx", weights: "onnx/decoder_model_merged_q4f16.onnx_data" },
    embedding: { graph: "onnx/embed_tokens_q4f16.onnx", weights: "onnx/embed_tokens_q4f16.onnx_data" },
    size: 1_382_903_808,
    hidden: 2048,
  },
]

const CACHE_NAME = "local-models"
const TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]

/** Whether the file fetched last, the main network's weights, is in the cache. */
export async function isDownloaded(model: Model) {
  const cache = await caches.open(CACHE_NAME)
  return (await cache.match(urlOf(model, model.network.weights))) !== undefined
}

export async function fetchModel(model: Model, onProgress: (done: number) => void): Promise<ModelFiles> {
  const [tokenizer, tokenizerConfig, chatTemplate] = await Promise.all(TOKENIZER_FILES.map((file) => fetchCached(urlOf(model, file))))
  const embedding = model.embedding && (await fetchNetwork(model, model.embedding, onProgress))
  const before = embedding?.weights.length ?? 0
  const network = await fetchNetwork(model, model.network, (done) => onProgress(before + done))
  const json = (bytes: Uint8Array) => JSON.parse(new TextDecoder().decode(bytes))
  return { tokenizer: json(tokenizer), tokenizerConfig: json(tokenizerConfig), chatTemplate: new TextDecoder().decode(chatTemplate), network, embedding }
}

export async function deleteModel(model: Model) {
  const cache = await caches.open(CACHE_NAME)
  const networks = model.embedding ? [model.network, model.embedding] : [model.network]
  const files = [...TOKENIZER_FILES, ...networks.flatMap(({ graph, weights }) => [graph, weights])]
  const kept = (await cache.keys()).filter(({ url }) => url.includes(keptPathOf(model)))
  await Promise.all([...files.map((file) => urlOf(model, file)), ...kept].map((entry) => cache.delete(entry)))
}

/** What `make` works out from the model, kept beside its files under a digest of `recipe`, so a later visit need not work it out again. */
export async function kept(model: Model, recipe: string, make: () => Promise<Float32Array<ArrayBuffer>>) {
  const cache = await caches.open(CACHE_NAME)
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(recipe)))
  const key = keptPathOf(model) + Array.from(digest, (byte) => byte.toString(16).padStart(2, "0")).join("")
  const saved = await cache.match(key)
  if (saved) return new Float32Array(await saved.arrayBuffer())
  const made = await make()
  await cache.put(key, new Response(made)).catch(() => {})
  return made
}

function keptPathOf(model: Model) {
  return `kept/${model.id}/${model.revision}/`
}

async function fetchNetwork(model: Model, files: NetworkFiles, onProgress: (done: number) => void): Promise<Network> {
  const graph = await fetchCached(urlOf(model, files.graph))
  const weights = await fetchCached(urlOf(model, files.weights), onProgress)
  return { graph, weights, weightsName: files.weights.split("/").pop()! }
}

function urlOf(model: Model, file: string) {
  return `https://huggingface.co/${model.repo}/resolve/${model.revision}/${file}`
}

async function fetchCached(url: string, onProgress?: (done: number) => void) {
  const cache = await caches.open(CACHE_NAME)
  const cached = await cache.match(url)
  if (cached) return new Uint8Array(await cached.arrayBuffer())
  const response = await fetch(url)
  if (!response.ok || !response.body) throw new Error(`${response.status} fetching ${url}`)
  const bytes = await readAll(response, onProgress)
  // A cache that is full or unavailable costs the next visit a download, not this one its model.
  await cache.put(url, new Response(bytes)).catch(() => {})
  return bytes
}

async function readAll(response: Response, onProgress?: (done: number) => void) {
  const length = Number(response.headers.get("Content-Length"))
  const reader = response.body!.getReader()
  let bytes = new Uint8Array(length || 1 << 20)
  let done = 0
  while (true) {
    const { done: ended, value } = await reader.read()
    if (ended) return bytes.subarray(0, done)
    if (done + value.length > bytes.length) {
      const grown = new Uint8Array(Math.max(bytes.length * 2, done + value.length))
      grown.set(bytes.subarray(0, done))
      bytes = grown
    }
    bytes.set(value, done)
    done += value.length
    onProgress?.(done)
  }
}
