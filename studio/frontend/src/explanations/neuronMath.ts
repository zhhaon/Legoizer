/** Executable scalar teaching networks. Tooltips and node values use the same operations. */
export type Activation = 'identity' | 'relu' | 'tanh' | 'sigmoid' | 'silu' | 'gelu';
export type Operation = 'input' | 'affine' | 'sum' | 'product' | 'mean' | 'max' | 'softmax' | 'layernorm' | 'logloss' | 'bce';
export type NodeKind = 'input' | 'parameter' | 'neuron' | 'operator' | 'gradient' | 'output';
export interface Term { id: string; weight: number }
export interface Neuron {
  id: string; label: string; stage: number; depth: number; kind: NodeKind; op: Operation;
  terms: Term[]; bias: number; activation: Activation; value: number; pre: number;
  index?: number; epsilon?: number; target?: number; note?: string;
}
export interface NeuralGraph { nodes: Neuron[]; outputs: string[]; note: string }
export const round = (x: number) => String(Number(x.toFixed(5)));
export function activate(x: number, activation: Activation): number {
  switch (activation) {
    case 'relu': return Math.max(0, x);
    case 'tanh': return Math.tanh(x);
    case 'sigmoid': return x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x));
    case 'silu': return x * activate(x, 'sigmoid');
    case 'gelu': return .5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + .044715 * x ** 3)));
    default: return x;
  }
}
export function evaluate(node: Neuron, values: Map<string, number>): { pre: number; value: number } {
  const xs = node.terms.map(t => values.get(t.id)! * t.weight);
  let pre: number;
  switch (node.op) {
    case 'input': return { pre: node.pre, value: node.pre };
    case 'product': pre = xs.reduce((a, b) => a * b, 1); break;
    case 'mean': pre = xs.reduce((a, b) => a + b, 0) / xs.length; break;
    case 'max': pre = Math.max(...xs); break;
    case 'softmax': {
      const max = Math.max(...xs), ex = xs.map(x => Math.exp(x - max));
      pre = ex[node.index!] / ex.reduce((a, b) => a + b, 0); break;
    }
    case 'layernorm': {
      const mean = xs.reduce((a, b) => a + b, 0) / xs.length;
      const variance = xs.reduce((a, b) => a + (b - mean) ** 2, 0) / xs.length;
      pre = (xs[node.index!] - mean) / Math.sqrt(variance + (node.epsilon ?? 1e-5)); break;
    }
    case 'logloss': pre = -Math.log(Math.max(xs[0], Number.MIN_VALUE)); break;
    case 'bce': pre = Math.max(xs[0], 0) - xs[0] * (xs[1] ?? node.target!) + Math.log1p(Math.exp(-Math.abs(xs[0]))); break;
    default: pre = xs.reduce((a, b) => a + b, 0) + node.bias;
  }
  return { pre, value: activate(pre, node.activation) };
}
export class Network {
  nodes: Neuron[] = [];
  private lookup = new Map<string, Neuron>();
  input(label: string, value: number, kind: NodeKind = 'input', note?: string): string {
    return this.add(label, 'input', [], 0, { value, kind, note });
  }
  add(label: string, op: Operation, ids: string[], stage: number,
    options: { weights?: number[]; bias?: number; activation?: Activation; kind?: NodeKind; value?: number; index?: number; target?: number; note?: string } = {}): string {
    const node: Neuron = { id: `n${this.nodes.length}`, label, op, stage,
      depth: ids.length ? Math.max(...ids.map(id => this.lookup.get(id)!.depth)) + 1 : 0,
      terms: ids.map((id, i) => ({ id, weight: options.weights?.[i] ?? 1 })),
      bias: options.bias ?? 0, activation: options.activation ?? 'identity',
      kind: options.kind ?? (op === 'affine' ? 'neuron' : 'operator'),
      pre: options.value ?? 0, value: 0, index: options.index, epsilon: 1e-5, target: options.target, note: options.note };
    Object.assign(node, evaluate(node, new Map(this.nodes.map(n => [n.id, n.value]))));
    if (!Number.isFinite(node.value)) throw new Error(`Non-finite teaching neuron: ${label}`);
    this.nodes.push(node); this.lookup.set(node.id, node); return node.id;
  }
  affine(label: string, ids: string[], weights: number[], stage: number, activation: Activation = 'identity', bias = 0, note?: string) {
    return this.add(label, 'affine', ids, stage, { weights, activation, bias, note });
  }
  product(label: string, ids: string[], stage: number) { return this.add(label, 'product', ids, stage); }
  softmax(label: string, ids: string[], stage: number) { return ids.map((_, index) => this.add(`${label}${index + 1}`, 'softmax', ids, stage, { index })); }
  norm(label: string, ids: string[], stage: number) { return ids.map((_, index) => this.add(`${label}${index + 1}`, 'layernorm', ids, stage, { index })); }
  output(label: string, id: string) { return this.add(label, 'affine', [id], 3, { kind: 'output' }); }
  finish(outputs: string[], note = 'tiny') { return { nodes: this.nodes, outputs, note }; }
}

/** Formula strings are generated from actual dependencies, never independent canned values. */
export function calculation(node: Neuron, nodes: Neuron[]): { symbolic: string; substituted: string; details: string[] } {
  const lookup = new Map(nodes.map(n => [n.id, n]));
  const labels = node.terms.map(t => lookup.get(t.id)!.label);
  const values = node.terms.map(t => lookup.get(t.id)!.value * t.weight);
  const f = round;
  let symbolic: string, substituted: string;
  const details: string[] = [];
  if (node.op === 'input') return { symbolic: node.label, substituted: `${node.label} = ${f(node.value)}`, details };
  if (node.op === 'sum') {
    symbolic = labels.join(' + '); substituted = values.map(x => `(${f(x)})`).join(' + ') + ` = ${f(node.pre)}`;
  } else if (node.op === 'affine') {
    symbolic = node.terms.map((t, i) => `${f(t.weight)}·${labels[i]}`).join(' + ') + ` + b`;
    substituted = node.terms.map(t => `${f(t.weight)}×(${f(lookup.get(t.id)!.value)})`).join(' + ') + ` + (${f(node.bias)}) = ${f(node.pre)}`;
  } else if (node.op === 'product') {
    symbolic = labels.join(' × '); substituted = values.map(x => `(${f(x)})`).join(' × ') + ` = ${f(node.pre)}`;
  } else if (node.op === 'softmax') {
    symbolic = `exp(${labels[node.index!]}) / Σⱼ exp(sⱼ)`;
    const shift = Math.max(...values);
    substituted = `exp(${f(values[node.index!])}−${f(shift)}) / (${values.map(x => `exp(${f(x)}−${f(shift)})`).join(' + ')}) = ${f(node.pre)}`;
    details.push(`m = max(s) = ${f(shift)}`);
  } else if (node.op === 'layernorm') {
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const variance = values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length;
    symbolic = `(${labels[node.index!]} − μ) / √(σ² + ε)`;
    substituted = `(${f(values[node.index!])}−${f(mean)}) / √(${f(variance)}+0.00001) = ${f(node.pre)}`;
    details.push(`μ = (${values.map(f).join(' + ')})/${values.length} = ${f(mean)}`, `σ² = ${f(variance)}, γ = 1, β = 0`);
  } else if (node.op === 'bce') {
    symbolic = 'max(z,0) − z·y + ln(1+exp(−|z|))';
    substituted = `max(${f(values[0])},0) − (${f(values[0])})×${f(values[1] ?? node.target!)} + ln(1+exp(−|${f(values[0])}|)) = ${f(node.pre)}`;
  } else if (node.op === 'logloss') {
    symbolic = `−ln(${labels[0]})`; substituted = `−ln(${f(values[0])}) = ${f(node.pre)}`;
  } else {
    symbolic = `${node.op}(${labels.join(', ')})`; substituted = `${node.op}(${values.map(f).join(', ')}) = ${f(node.pre)}`;
  }
  if (node.activation !== 'identity') details.push(`${node.activation}(${f(node.pre)}) = ${f(node.value)}`);
  return { symbolic, substituted, details };
}
