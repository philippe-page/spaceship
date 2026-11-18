/**
 * Edits an ONNX graph file at the protobuf wire level, so every field not named here is written back byte for byte.
 *
 * Field numbers, from onnx.proto: ModelProto.graph 7; GraphProto.node 1, initializer 5, input 11, output 12;
 * NodeProto.input 1, output 2, name 3, op_type 4, attribute 5; AttributeProto.name 1, i 3, ints 8, type 20;
 * TensorProto.dims 1, data_type 2, name 8, raw_data 9; ValueInfoProto.name 1, type 2.
 */

export const FLOAT = 1
export const INT64 = 7
export const FLOAT16 = 10

export type GraphNode = { name: string; op: string; inputs: string[]; outputs: string[]; ints: Record<string, number> }

type Attributes = Record<string, number | number[]>

type Dims = (number | string)[]

export type GraphEdit = {
  nodes: GraphNode[]
  outputs: string[]
  insertBefore(node: GraphNode, op: string, name: string, inputs: string[], outputs: string[]): void
  append(op: string, name: string, inputs: string[], attributes?: Attributes, outputs?: string[]): string
  ints(values: number[]): string
  half(value: number): string
  addInput(name: string, elementType: number, dims: Dims): void
  addOutput(name: string, elementType: number, dims: Dims): void
}

type Field = { number: number; wire: number; value: bigint | Uint8Array }

const VARINT = 0
const FIXED64 = 1
const BYTES = 2
const INT_ATTRIBUTE = 2
const INTS_ATTRIBUTE = 7

const utf8 = new TextEncoder()
const text = new TextDecoder()

export function editGraph(model: Uint8Array, edit: (graph: GraphEdit) => void) {
  const modelFields = decode(model)
  const graphField = modelFields.find((field) => field.number === 7)
  if (!graphField) throw new Error("not an ONNX model: no graph")
  const items = decode(graphField.value as Uint8Array).map((field) => ({ field, node: field.number === 1 ? readNode(field) : undefined }))
  const appended: Field[] = []
  const constants = new Map<string, Field>()

  edit({
    nodes: items.flatMap((item) => item.node ?? []),
    outputs: items.flatMap(({ field }) => (field.number === 12 ? [nameOf(field)] : [])),
    insertBefore(node, op, name, inputs, outputs) {
      const at = items.findIndex((item) => item.node === node)
      items.splice(at, 0, { field: nodeField(op, name, inputs, outputs), node: undefined })
    },
    append(op, name, inputs, attributes = {}, outputs = [`${name}/output`]) {
      appended.push(nodeField(op, name, inputs, outputs, attributes))
      return outputs[0]
    },
    ints(values) {
      const name = `constant/ints/${values.join(",")}`
      constants.set(name, constantField(name, INT64, [values.length], new Uint8Array(BigInt64Array.from(values, BigInt).buffer)))
      return name
    },
    half(value) {
      const name = `constant/half/${value}`
      constants.set(name, constantField(name, FLOAT16, [], new Uint8Array(Float16Array.of(value).buffer)))
      return name
    },
    addInput: (name, elementType, dims) => appended.push(valueField(11, name, elementType, dims)),
    addOutput: (name, elementType, dims) => appended.push(valueField(12, name, elementType, dims)),
  })

  const written = items.map(({ field, node }) => (node ? writeNode(field, node) : field))
  graphField.value = encode([...written, ...constants.values(), ...appended])
  return encode(modelFields)
}

function nameOf(field: Field) {
  return text.decode(decode(field.value as Uint8Array).find((f) => f.number === 1)!.value as Uint8Array)
}

function readNode(field: Field): GraphNode {
  const fields = decode(field.value as Uint8Array)
  const strings = (number: number) => fields.filter((f) => f.number === number).map((f) => text.decode(f.value as Uint8Array))
  const ints: Record<string, number> = {}
  for (const attribute of fields.filter((f) => f.number === 5).map((f) => decode(f.value as Uint8Array))) {
    const name = attribute.find((f) => f.number === 1)
    const value = attribute.find((f) => f.number === 3)
    if (name && value) ints[text.decode(name.value as Uint8Array)] = Number(BigInt.asIntN(64, value.value as bigint))
  }
  return { name: strings(3)[0] ?? "", op: strings(4)[0] ?? "", inputs: strings(1), outputs: strings(2), ints }
}

function writeNode(field: Field, node: GraphNode) {
  const others = decode(field.value as Uint8Array).filter((f) => f.number !== 1 && f.number !== 2)
  return message(1, [...node.inputs.map((name) => string(1, name)), ...node.outputs.map((name) => string(2, name)), ...others])
}

function nodeField(op: string, name: string, inputs: string[], outputs: string[], attributes: Attributes = {}) {
  const attributeFields = Object.entries(attributes).map(([key, value]) =>
    message(
      5,
      Array.isArray(value)
        ? [string(1, key), ...value.map((item) => int(8, item)), int(20, INTS_ATTRIBUTE)]
        : [string(1, key), int(3, value), int(20, INT_ATTRIBUTE)],
    ),
  )
  return message(1, [...inputs.map((i) => string(1, i)), ...outputs.map((o) => string(2, o)), string(3, name), string(4, op), ...attributeFields])
}

function constantField(name: string, elementType: number, dims: number[], raw: Uint8Array) {
  return message(5, [...dims.map((dim) => int(1, dim)), int(2, elementType), string(8, name), { number: 9, wire: BYTES, value: raw }])
}

/** A graph input or output: ValueInfoProto{name, type{tensor_type{elem_type, shape{dim{dim_value | dim_param}...}}}}. */
function valueField(number: number, name: string, elementType: number, dims: Dims) {
  const shape = message(
    2,
    dims.map((dim) => message(1, [typeof dim === "string" ? string(2, dim) : int(1, dim)])),
  )
  return message(number, [string(1, name), message(2, [message(1, [int(1, elementType), shape])])])
}

function string(number: number, value: string): Field {
  return { number, wire: BYTES, value: utf8.encode(value) }
}

function message(number: number, fields: Field[]): Field {
  return { number, wire: BYTES, value: encode(fields) }
}

function int(number: number, value: number): Field {
  return { number, wire: VARINT, value: BigInt.asUintN(64, BigInt(value)) }
}

function decode(bytes: Uint8Array) {
  const fields: Field[] = []
  let at = 0
  while (at < bytes.length) {
    const [tag, afterTag] = readVarint(bytes, at)
    const number = Number(tag >> 3n)
    const wire = Number(tag & 7n)
    if (wire === VARINT) {
      const [value, next] = readVarint(bytes, afterTag)
      fields.push({ number, wire, value })
      at = next
    } else if (wire === BYTES) {
      const [length, start] = readVarint(bytes, afterTag)
      at = start + Number(length)
      fields.push({ number, wire, value: bytes.subarray(start, at) })
    } else {
      at = afterTag + (wire === FIXED64 ? 8 : 4)
      fields.push({ number, wire, value: bytes.subarray(afterTag, at) })
    }
  }
  return fields
}

function encode(fields: Field[]) {
  const parts = fields.flatMap(({ number, wire, value }) => {
    const tag = varint((BigInt(number) << 3n) | BigInt(wire))
    if (typeof value === "bigint") return [tag, varint(value)]
    return wire === BYTES ? [tag, varint(BigInt(value.length)), value] : [tag, value]
  })
  const bytes = new Uint8Array(parts.reduce((sum, part) => sum + part.length, 0))
  let at = 0
  for (const part of parts) {
    bytes.set(part, at)
    at += part.length
  }
  return bytes
}

function readVarint(bytes: Uint8Array, start: number): [bigint, number] {
  let value = 0n
  let shift = 0n
  let at = start
  while (true) {
    const byte = bytes[at++]
    value |= BigInt(byte & 0x7f) << shift
    if (byte < 0x80) return [value, at]
    shift += 7n
  }
}

function varint(value: bigint) {
  const bytes: number[] = []
  let rest = value
  do {
    const byte = Number(rest & 0x7fn)
    rest >>= 7n
    bytes.push(rest ? byte | 0x80 : byte)
  } while (rest)
  return Uint8Array.from(bytes)
}
