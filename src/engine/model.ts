import { Template } from "@huggingface/jinja"
import { Tokenizer } from "@huggingface/tokenizers"
import * as ort from "onnxruntime-web/webgpu"
import { instrument, steerInputOf, type Instruments } from "./instruments.ts"
import type { Model, ModelFiles, Network } from "./models.ts"
import { editGraph } from "./onnx-graph.ts"

export const SYSTEM_PROMPT =
  "You are a novelist. The assistant turn is a story in progress, cut off where the reader is. " +
  "Continue it from its last word in plain prose: no title, markdown or commentary."

export type Span = [start: number, end: number]

/** Stays on the GPU between passes, so whoever makes one must `release` it. */
export type Cache = { length: number; tensors: Record<string, ort.Tensor> }

/** Per steered layer, one vector per row of the batch, or one for all rows. */
export type Adds = Record<number, Float32Array[]>

type Pass = {
  cache?: Cache
  steer?: Adds
  lock?: Float32Array[]
  residuals?: boolean
  attention?: boolean
  layerStates?: boolean
  targets?: number[][]
  /** Once aborted, the pass fails when its turn comes instead of running. */
  signal?: AbortSignal
}

type Output = {
  logits: Float16Array[]
  residuals: Record<number, Float16Array>
  /** Shape (B, L, T): from each of this pass's tokens to every token so far. */
  attention?: Float16Array
  scored?: Float32Array
  /** Shape (B, 2N) for N layers: each layer's output's squared distance from the mean of the rows, then its squared size. */
  layerStates?: Float32Array
}

export type Engine = Awaited<ReturnType<typeof loadEngine>>

/** Float16's most negative number, which the model's own masks use for a position that cannot be seen. */
const MASKED = -65504

export async function loadEngine(model: Model, files: ModelFiles) {
  const tokenizer = new Tokenizer(files.tokenizer, files.tokenizerConfig)
  const template = new Template(files.chatTemplate)
  let instruments!: Instruments
  let cacheOutputs: string[] = []
  const graph = editGraph(files.network.graph, (edit) => {
    instruments = instrument(edit, model.hidden)
    cacheOutputs = edit.outputs.filter((name) => name.startsWith("present"))
  })
  const session = await createSession({ ...files.network, graph }, Object.fromEntries(cacheOutputs.map((name) => [name, "gpu-buffer" as const])))
  const embedding = files.embedding && (await createSession(files.embedding, "gpu-buffer"))

  const { hidden } = model
  const { steerLayers, residualOutputs, layerKinds } = instruments
  const takesPositions = session.inputNames.includes("position_ids")
  const cacheInputs = session.inputNames.filter((name) => name.startsWith("past_"))
  const presentOf = (name: string) => name.replace("past_key_values", "present").replace("past_", "present_")
  const stopIds = new Set(encode(String(files.tokenizerConfig.eos_token)).concat(encode("<|im_end|>")))
  const templates = new Map<string, number[]>()
  let lastPass: Promise<unknown> = Promise.resolve()

  return { id: model.id, name: model.name, hidden, steerLayers, layerKinds, stopIds, decode, storyTokens, emptyCache, forkCache, release, forward }

  function encode(text: string): number[] {
    return tokenizer.encode(text, { add_special_tokens: false }).ids
  }

  function decode(ids: number[]) {
    return tokenizer.decode(ids, { clean_up_tokenization_spaces: false })
  }

  function templateIds(system: string) {
    if (!templates.has(system)) {
      const messages = [
        { role: "system", content: system },
        { role: "user", content: "Continue the story." },
      ]
      const { bos_token, eos_token } = files.tokenizerConfig
      templates.set(system, encode(template.render({ messages, add_generation_prompt: true, enable_thinking: false, bos_token, eos_token })))
    }
    return templates.get(system)!
  }

  function storyTokens(story: string, system = SYSTEM_PROMPT) {
    const prompt = templateIds(system)
    const storyIds = encode(story)
    const spans: (Span | undefined)[] = [...prompt.map(() => undefined), ...spansOf(storyIds, story)]
    return { ids: [...prompt, ...storyIds], spans, first: prompt.length }
  }

  /**
   * The tokenizer reports no offsets, so each token's span is found by decoding; a token that ends partway through a character shares the span of
   * the one that completes it.
   */
  function spansOf(ids: number[], text: string) {
    const spans: Span[] = []
    let pending: number[] = []
    let at = 0
    for (const id of ids) {
      pending.push(id)
      const piece = decode(pending)
      if (piece.endsWith("�")) continue
      const found = text.indexOf(piece, at)
      const start = found === -1 || found > at + piece.length ? at : found
      const span: Span = [start, Math.min(text.length, start + piece.length)]
      for (let i = 0; i < pending.length; i++) spans.push(span)
      at = span[1]
      pending = []
    }
    for (let i = 0; i < pending.length; i++) spans.push([at, text.length])
    return spans
  }

  function emptyCache(batch: number): Cache {
    const tensors = Object.fromEntries(
      cacheInputs.map((name) => {
        const input = session.inputMetadata.find((input) => input.name === name)!
        const shape = input.isTensor ? input.shape : []
        return [name, half(shape.map((dim) => (dim === "batch_size" ? batch : typeof dim === "string" ? 0 : dim)))]
      }),
    )
    return { length: 0, tensors }
  }

  async function forkCache(cache: Cache, batch: number): Promise<Cache> {
    const tensors = Object.fromEntries(
      await Promise.all(
        Object.entries(cache.tensors).map(async ([name, tensor]) => {
          const data = halves(await tensor.getData())
          const repeated = new Float32Array(data.length * batch)
          for (let row = 0; row < batch; row++) repeated.set(data, row * data.length)
          return [name, half([batch, ...tensor.dims.slice(1)], repeated)]
        }),
      ),
    )
    return { length: cache.length, tensors }
  }

  /** Frees the GPU memory a cache holds, which dropping it does not. */
  function release(cache: Cache) {
    for (const tensor of Object.values(cache.tensors)) tensor.dispose()
  }

  async function forward(
    rows: number[][],
    { cache, steer = {}, lock, residuals = false, attention = false, layerStates = false, targets, signal }: Pass = {},
  ): Promise<Output> {
    const batch = rows.length
    const length = rows[0].length
    const total = (cache?.length ?? 0) + length
    const feeds: Record<string, ort.Tensor> = {
      attention_mask: int64([batch, total], Array<number>(batch * total).fill(1)),
      num_logits_to_keep: int64([], [targets ? length : 1]),
      lock_bias: lock ? half([lock.length, 1, 1, total], padded(lock, total)) : half([1, 1, 1, total]),
      // The last two only matter to a pass that reads them back; any other gets the smallest tensor that fits.
      attention_causal: attention ? half([length, total], causalMask(length, total)) : half([1, 1]),
      score_targets: targets ? int64([batch, length, 1], targets.flat()) : int64([1, 1, 1], [0]),
      ...(cache ?? emptyCache(batch)).tensors,
    }
    for (const layer of steerLayers) {
      const vectors = steer[layer]
      feeds[steerInputOf(layer)] = vectors ? half([vectors.length, 1, hidden], joined(vectors)) : half([1, 1, hidden])
    }
    // Text positions, the same along each of the three axes a model that also reads images counts them on.
    if (takesPositions)
      feeds.position_ids = int64(
        [3, batch, length],
        Array.from({ length: 3 * batch * length }, (_, i) => total - length + (i % length)),
      )
    const fetches = [
      ...(targets ? ["scored"] : ["logits"]),
      ...(cache ? cacheInputs.map(presentOf) : []),
      ...(residuals ? steerLayers.map((layer) => residualOutputs[layer]) : []),
      ...(attention ? ["attention"] : []),
      ...(layerStates ? ["layer_states"] : []),
    ]
    const ids = { input_ids: int64([batch, length], rows.flat()) }

    // The model runs one pass at a time; a pass that fails does not hold up the ones behind it.
    const run = lastPass.then(async () => {
      signal?.throwIfAborted()
      const embedded = embedding ? await embedding.run(ids) : ids
      try {
        return await session.run({ ...feeds, ...embedded }, fetches)
      } finally {
        if (embedding) for (const tensor of Object.values(embedded)) tensor.dispose()
      }
    })
    lastPass = run.catch(() => {})
    const out = await run

    if (cache) {
      for (const name of cacheInputs) {
        cache.tensors[name].dispose()
        cache.tensors[name] = out[presentOf(name)]
      }
      cache.length = total
    }
    return {
      logits: targets ? [] : rowsOf(floatsOf(out.logits), batch),
      residuals: residuals ? Object.fromEntries(steerLayers.map((layer) => [layer, floatsOf(out[residualOutputs[layer]])])) : {},
      attention: attention ? floatsOf(out.attention) : undefined,
      scored: out.scored?.data as Float32Array | undefined,
      layerStates: out.layer_states?.data as Float32Array | undefined,
    }
  }
}

function createSession({ graph, weights, weightsName }: Network, preferredOutputLocation?: ort.InferenceSession.SessionOptions["preferredOutputLocation"]) {
  return ort.InferenceSession.create(graph, { executionProviders: ["webgpu"], externalData: [{ path: weightsName, data: weights }], preferredOutputLocation })
}

/** ONNX Runtime's types say float16 data is raw 16-bit words; a browser that has Float16Array gives and takes that instead. */
function half(dims: number[], values?: ArrayLike<number>) {
  const data = new Float16Array(dims.reduce((size, dim) => size * dim, 1))
  if (values) data.set(values)
  return new ort.Tensor("float16", data as unknown as Uint16Array, dims)
}

function floatsOf({ data }: ort.Tensor) {
  return halves(data)
}

function halves(data: ort.Tensor["data"]) {
  return data instanceof Uint16Array ? new Float16Array(data.buffer, data.byteOffset, data.length) : (data as unknown as Float16Array)
}

function int64(dims: number[], values: number[]) {
  return new ort.Tensor("int64", BigInt64Array.from(values, BigInt), dims)
}

function rowsOf(data: Float16Array, rows: number) {
  const width = data.length / rows
  return Array.from({ length: rows }, (_, row) => data.subarray(row * width, (row + 1) * width))
}

function joined(vectors: Float32Array[]) {
  const all = new Float32Array(vectors.length * vectors[0].length)
  vectors.forEach((vector, row) => all.set(vector, row * vector.length))
  return all
}

function padded(rows: Float32Array[], total: number) {
  const all = new Float32Array(rows.length * total)
  rows.forEach((row, index) => all.set(row.subarray(0, total), index * total))
  return all
}

function causalMask(length: number, total: number) {
  const mask = new Float32Array(length * total)
  for (let row = 0; row < length; row++) mask.fill(MASKED, row * total + (total - length) + row + 1, (row + 1) * total)
  return mask
}
