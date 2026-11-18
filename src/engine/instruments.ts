import { FLOAT, FLOAT16, INT64, type GraphEdit, type GraphNode } from "./onnx-graph.ts"

/**
 * Written for the graphs ONNX Runtime's model builder exports, as LFM3's and Qwen3.5's are: layers are `/model/layers.N/...`, each layer's output is
 * the sum a SkipSimplifiedLayerNormalization node forms at the start of the next layer, and full-attention layers are GroupQueryAttention nodes with
 * no window. Some keep their norms in 32-bit floats and cast on the way in and out.
 */

export type LayerKind = "attention" | "windowed attention" | "linear attention" | "convolution"

export type Instruments = {
  steerLayers: number[]
  residualOutputs: Record<number, string>
  layerKinds: LayerKind[]
}

/** Dimensions per attention head that the rotary position embedding turns inside the attention operator, in pairs. */
const ROTARY_PAIRS = 32
const PAST_LENGTH = ["/model/shared_dims/attention_mask_total_seq_len_1d/output_0", "/model/shared_dims/root_input_seq_len_1d/output_0"]

const layerOf = (node: GraphNode) => Number(/\/layers\.(\d+)\//.exec(node.name)?.[1])
export const steerInputOf = (layer: number) => `steer.${layer}`

export function instrument(graph: GraphEdit, hidden: number): Instruments {
  const sums = graph.nodes.filter((node) => node.op === "SkipSimplifiedLayerNormalization")
  // Two sums per layer: one after its attention or convolution, one after its feed-forward block.
  const layerCount = sums.length / 2
  // The second quarter of the layers: late enough to carry meaning, early enough for the rest of the model to write with it.
  const steerLayers = Array.from({ length: layerCount / 2 - layerCount / 4 }, (_, index) => layerCount / 4 + index)
  const fullAttention = graph.nodes.filter((node) => node.op === "GroupQueryAttention" && node.ints.local_window_size === -1)
  const residualOutputs = Object.fromEntries(
    steerLayers.map((layer) => [
      layer,
      addSteering(
        graph,
        sums.find((node) => layerOf(node) === layer + 1)!,
        layer,
        hidden,
      ),
    ]),
  )
  addLock(graph, fullAttention)
  addAttentionReadout(graph, fullAttention, hidden)
  addScoring(graph)
  const layers = Array.from({ length: layerCount }, (_, layer) => layer)
  // A layer's output is summed where the next layer starts; the last layer's, at the final norm, which is named as one more layer.
  addLayerReadout(
    graph,
    layers.map((layer) => sums.find((node) => layerOf(node) === layer + 1)!),
  )
  return { steerLayers, residualOutputs, layerKinds: layers.map((layer) => kindOf(graph.nodes.filter((node) => layerOf(node) === layer))) }
}

function kindOf(nodes: GraphNode[]): LayerKind {
  const attention = nodes.find((node) => node.op === "GroupQueryAttention")
  if (attention) return attention.ints.local_window_size === -1 ? "attention" : "windowed attention"
  return nodes.some((node) => node.op === "LinearAttention") ? "linear attention" : "convolution"
}

/** `steer.N`, shape (B or 1, 1, H), is added to layer N's output where `sum`, the first sum of the next layer, takes it in. */
function addSteering(graph: GraphEdit, sum: GraphNode, layer: number, hidden: number) {
  const castIn = graph.nodes.find((node) => node.op === "Cast" && node.outputs[0] === sum.inputs[1])
  const [taker, slot] = castIn ? [castIn, 0] : [sum, 1]
  const steered = `${steerInputOf(layer)}/sum`
  graph.insertBefore(taker, "Add", `${steerInputOf(layer)}/Add`, [taker.inputs[slot], steerInputOf(layer)], [steered])
  taker.inputs[slot] = steered
  graph.addInput(steerInputOf(layer), FLOAT16, ["steer_batch", 1, hidden])
  const castOut = graph.nodes.find((node) => node.op === "Cast" && node.inputs[0] === sum.outputs[3])
  const residual = castOut?.outputs[0] ?? sum.outputs[3]
  graph.addOutput(residual, FLOAT16, ["batch_size", "sequence_length", hidden])
  return residual
}

/** `lock_bias`, shape (B or 1, 1, 1, T), is added to the attention scores of the full-attention layers, on the padding mask they already take. */
function addLock(graph: GraphEdit, fullAttention: GraphNode[]) {
  graph.insertBefore(fullAttention[0], "Add", "lock/Add", [fullAttention[0].inputs[10], "lock_bias"], ["lock/bias"])
  graph.addInput("lock_bias", FLOAT16, ["lock_batch", 1, 1, "total_sequence_length"])
  for (const node of fullAttention) node.inputs[10] = "lock/bias"
}

/**
 * `attention`, shape (B, L, T), averaged over heads by each head's output gate, then over the full-attention layers. WebGPU refuses the operator's
 * own weights output, so they are rebuilt from its queries and the keys it hands back, which the rotary embedding has already turned.
 * `attention_causal`, shape (L, T), masks the tokens each one cannot see.
 */
function addAttentionReadout(graph: GraphEdit, fullAttention: GraphNode[], hidden: number) {
  const { append, ints } = graph
  const [cosCache, sinCache] = [fullAttention[0].inputs[7], fullAttention[0].inputs[8]]
  const [total, length] = PAST_LENGTH
  const past = cosCache && append("Sub", "attention/past", [total, length])
  const [cos, sin] = [cosCache, sinCache].map((cache, index) => cache && append("Slice", `attention/rotary.${index}`, [cache, past, total, ints([0])]))
  graph.addInput("attention_causal", FLOAT16, ["attention_length", "attention_total"])
  const lock = append("Unsqueeze", "attention/lock", ["lock_bias", ints([1])])

  const perLayer = fullAttention.map((node) => {
    const at = `attention/${layerOf(node)}`
    const { num_heads: heads, kv_num_heads: keyHeads } = node.ints
    const headSize = hidden / heads
    const group = heads / keyHeads

    const byHead = append("Transpose", `${at}/by_head`, [append("Reshape", `${at}/heads`, [node.inputs[0], ints([0, 0, heads, headSize])])], {
      perm: [0, 2, 1, 3],
    })
    const queries = append("Reshape", `${at}/grouped`, [cos ? turned(byHead) : byHead, ints([0, keyHeads, group, -1, headSize])])
    const keys = append("Unsqueeze", `${at}/keys`, [append("Transpose", `${at}/keys_last`, [node.outputs[1]], { perm: [0, 1, 3, 2] }), ints([2])])
    const scaled = append("Mul", `${at}/scaled`, [append("MatMul", `${at}/scores`, [queries, keys]), graph.half(1 / Math.sqrt(headSize))])
    const weights = append("Softmax", `${at}/weights`, [append("Add", `${at}/locked`, [append("Add", `${at}/masked`, [scaled, "attention_causal"]), lock])], {
      axis: -1,
    })

    // One gate per head, or one per channel, whose mean over a head stands for it.
    const gates = graph.nodes.find((other) => other.op === "Sigmoid" && other.name.startsWith(`/model/layers.${layerOf(node)}/attn/`))!
    const gate = append("ReduceMean", `${at}/gate`, [append("Reshape", `${at}/gate_heads`, [gates.outputs[0], ints([0, 0, heads, -1])]), ints([-1])], {
      keepdims: 0,
    })
    const gateByHead = append("Reshape", `${at}/gate_grouped`, [
      append("Transpose", `${at}/gate_by_head`, [gate], { perm: [0, 2, 1] }),
      ints([0, keyHeads, group, -1]),
    ])
    const gated = append("Mul", `${at}/gated`, [weights, append("Unsqueeze", `${at}/gate_per_key`, [gateByHead, ints([-1])])])
    const summed = append("ReduceSum", `${at}/summed`, [gated, ints([1, 2])], { keepdims: 0 })
    return append("Div", `${at}/mean`, [summed, append("ReduceSum", `${at}/gate_sum`, [gate, ints([2])], { keepdims: 1 })])

    function turned(vectors: string) {
      const [a, b, rest] = [`${at}/a`, `${at}/b`, `${at}/rest`]
      append("Split", `${at}/split`, [vectors, ints([ROTARY_PAIRS, ROTARY_PAIRS, headSize - 2 * ROTARY_PAIRS])], { axis: -1 }, [a, b, rest])
      const product = (x: string, y: string, name: string) => append("Mul", `${at}/${name}`, [x, y])
      const turnedA = append("Sub", `${at}/turned_a`, [product(a, cos, "a_cos"), product(b, sin, "b_sin")])
      const turnedB = append("Add", `${at}/turned_b`, [product(b, cos, "b_cos"), product(a, sin, "a_sin")])
      return append("Concat", `${at}/turned`, [turnedA, turnedB, rest], { axis: -1 })
    }
  })
  // Stacked and reduced rather than by Mean, which ONNX Runtime's WebGPU backend lacks: run on the CPU, it would read the GPU back
  // mid-pass on every pass, whether the attention is read or not.
  const stacked = perLayer.map((layer, index) => append("Unsqueeze", `attention/stack.${index}`, [layer, ints([0])]))
  append("ReduceMean", "attention/mean", [append("Concat", "attention/stacked", stacked, { axis: 0 }), ints([0])], { keepdims: 0 }, ["attention"])
  graph.addOutput("attention", FLOAT16, ["batch_size", "sequence_length", "total_sequence_length"])
}

/** `scored`, shape (B, L): the log-probability the model gives, at each position, to the token `score_targets` (B, L, 1) names for it. */
function addScoring(graph: GraphEdit) {
  const { append, ints } = graph
  // The logits less their log-sum-exp, rather than LogSoftmax, which the WebGPU backend lacks too.
  const floats = append("Cast", "score/cast", ["logits"], { to: FLOAT })
  const logProbabilities = append("Sub", "score/log_softmax", [floats, append("ReduceLogSumExp", "score/normalizer", [floats, ints([-1])], { keepdims: 1 })])
  const picked = append("GatherElements", "score/gather", [logProbabilities, "score_targets"], { axis: -1 })
  append("Squeeze", "score/squeeze", [picked, ints([-1])], {}, ["scored"])
  graph.addInput("score_targets", INT64, ["score_batch", "score_length", 1])
  graph.addOutput("scored", FLOAT, ["batch_size", "sequence_length"])
}

/**
 * `layer_states`, shape (B, 2N) for N layers, at the pass's last token: each layer's squared distance from the mean of the batch's rows, then its
 * squared size. Two rows are each half their distance from that mean, so a steered row is compared with an unsteered one without reading either
 * state back.
 */
function addLayerReadout(graph: GraphEdit, sums: GraphNode[]) {
  const { append, ints } = graph
  const last = sums.map((sum, layer) => {
    if (!sum.outputs[3]) sum.outputs = [sum.outputs[0], "", "", `layers/output.${layer}`]
    return append("Gather", `layers/last.${layer}`, [sum.outputs[3], ints([-1])], { axis: 1 })
  })
  // In 32-bit floats: a state's squared size is past what 16 bits hold.
  const states = append("Cast", "layers/cast", [append("Concat", "layers/stacked", last, { axis: 1 })], { to: FLOAT })
  const apart = append("Sub", "layers/apart", [states, append("ReduceMean", "layers/center", [states, ints([0])], { keepdims: 1 })])
  const squaredSum = (values: string, name: string) =>
    append("ReduceSum", `layers/${name}`, [append("Mul", `layers/${name}_squares`, [values, values]), ints([-1])], { keepdims: 0 })
  append("Concat", "layers/joined", [squaredSum(apart, "spread"), squaredSum(states, "size")], { axis: 1 }, ["layer_states"])
  graph.addOutput("layer_states", FLOAT, ["batch_size", "layer_states"])
}
