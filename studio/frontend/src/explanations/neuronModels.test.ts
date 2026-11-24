import { describe, expect, it } from 'vitest';
import { activate, calculation, evaluate, Network } from './neuronMath';
import { buildNeuralGraph, defaultExample, type ExampleOptions } from './neuronModels';
import en from '@/i18n/locales/en.json';
import zh from '@/i18n/locales/zh-CN.json';

const kinds = ['cnn', 'cnncat', 'ada', 'attention', 'polyattention', 'gru', 'pooling', 'transformer', 'flattentransformer', 'fastformer', 'flattenfastformer', 'lm', 'iisan', 'externalhf', 'nullsimple', 'nullconcat', 'scsimple', 'scflatten', 'dot', 'dnn', 'pnn', 'deepfm', 'dcn', 'dcnv2', 'gdcn', 'autoint', 'din', 'miner', 'masknet', 'finalmlp', 'matching_task', 'ranking_task'];
const graph = (kind: string, overrides: Partial<ExampleOptions> = {}, id = kind) => buildNeuralGraph(kind, id, { ...defaultExample, ...overrides });
const outputs = (kind: string, overrides: Partial<ExampleOptions> = {}) => { const g = graph(kind, overrides); return g.outputs.map(id => g.nodes.find(n => n.id === id)!.value); };

describe('executable neuron examples', () => {
  for (const kind of kinds) it(`${kind}: all four stages, valid dependencies, executable formulas, bilingual notes`, () => {
    for (const options of [{}, { candidate: 'cooking', pooling: 'max', reverse: true, hf: 'tokens', label: 0, reducer: 'max', context: 'recipe' }, { reducer: 'mean' }] as Partial<ExampleOptions>[]) {
      const g = graph(kind, options), values = new Map<string, number>();
      expect([...new Set(g.nodes.map(n => n.stage))].sort()).toEqual([0, 1, 2, 3]);
      expect(new Set(g.nodes.map(n => n.id)).size).toBe(g.nodes.length);
      for (const node of g.nodes) {
        for (const term of node.terms) expect(values.has(term.id), `${node.label} upstream exists`).toBe(true);
        expect(node.value).toBeCloseTo(evaluate(node, values).value, 12);
        expect(Number.isFinite(node.value)).toBe(true);
        const formula = calculation(node, g.nodes);
        expect(formula.substituted).not.toMatch(/undefined|NaN|Infinity/);
        expect(formula.symbolic.length).toBeGreaterThan(0);
        for (const locale of [en, zh]) {
          expect(locale).toHaveProperty(`neuron.op.${node.op}`);
          expect(locale).toHaveProperty(`neuron.kind.${node.kind}`);
          if (node.note) expect(locale).toHaveProperty(`neuron.note.${node.note}`);
        }
        values.set(node.id, node.value);
      }
      expect(g.outputs.every(id => values.has(id))).toBe(true);
      for (const locale of [en, zh]) {
        expect(locale).toHaveProperty(`neuron.example.${g.note}`);
        for (let i = 0; i < 4; i++) expect(locale).toHaveProperty(`explain.${kind}.step.${i}.body`);
      }
    }
  });

  it('dot and pooling agree with hand calculations', () => {
    expect(outputs('dot')[0]).toBeCloseTo(.8 * .9 + .2 * .1, 12);
    expect(outputs('dot', { candidate: 'cooking' })[0]).toBeCloseTo(.26, 12);
    expect(outputs('pooling')[0]).toBeCloseTo(.6, 12);
    expect(outputs('pooling')[1]).toBeCloseTo(.4, 12);
    expect(outputs('pooling', { pooling: 'max' })).toEqual([1, 1]);
  });
  it('convolution uses a shared width-3 kernel and actual normalized weights', () => {
    const g = graph('cnn'), conv = g.nodes.filter(n => /^c[1-3]$/.test(n.label));
    [.7, .875, .675].forEach((value, i) => expect(conv[i].value).toBeCloseTo(value, 12));
    expect(conv.map(n => n.terms.map(t => t.weight))).toEqual(Array(3).fill([.25, .5, .25]));
    const alphas = g.nodes.filter(n => n.op === 'softmax').map(n => n.value);
    expect(alphas.reduce((a, b) => a + b)).toBeCloseTo(1, 12);
    expect(outputs('cnn')[0]).toBeCloseTo(conv.reduce((a, n, i) => a + alphas[i] * n.value, 0), 12);
  });
  it('GRU is order-dependent; embedding-only path skips contextual attention', () => {
    expect(outputs('gru')).not.toEqual(outputs('gru', { reverse: true }));
    expect(graph('externalhf', { hf: 'tokens' }).nodes.some(n => n.op === 'softmax')).toBe(false);
    expect(graph('externalhf').nodes.some(n => n.op === 'softmax')).toBe(true);
    expect(outputs('externalhf')).not.toEqual(outputs('externalhf', { context: 'recipe' }));
  });
  it('BERT is bidirectional and autoregressive examples have causal edges', () => {
    expect(graph('lm', {}, 'item.bertbase').nodes.some(n => n.label === 'LMe1.1.2')).toBe(true);
    expect(graph('lm', {}, 'item.llama3').nodes.some(n => n.label === 'LMe1.1.2')).toBe(false);
  });
  it('pass-through components preserve features and introduce no trainable neurons', () => {
    for (const kind of ['nullsimple', 'nullconcat', 'scsimple', 'scflatten']) {
      const g = graph(kind);
      expect(g.nodes.some(n => n.kind === 'neuron' || n.kind === 'parameter')).toBe(false);
      expect(outputs(kind).slice(0, 6)).toEqual([1, 0, .8, .2, 0, 1]);
    }
  });
  it('training gradients match finite differences and update actual parameters', () => {
    const xs = [[.9, .1], [.1, .9], [.3, .7]], w = [.8, .2], eta = .5;
    const matchingLoss = (weights: number[]) => { const ex = xs.map(row => Math.exp(row.reduce((s, x, d) => s + weights[d] * x, 0))); return -Math.log(ex[0] / ex.reduce((a, b) => a + b)); };
    const g = graph('matching_task');
    const get = (label: string) => g.nodes.find(n => n.label === label)!.value;
    const dw = w.map((_, d) => { const hi = [...w], lo = [...w]; hi[d] += 1e-5; lo[d] -= 1e-5; return (matchingLoss(hi) - matchingLoss(lo)) / 2e-5; });
    dw.forEach((v, d) => { expect(get(`∂L/∂w${d + 1}`)).toBeCloseTo(v, 9); expect(get(`w′${d + 1}`)).toBeCloseTo(w[d] - eta * v, 9); });
    expect(outputs('matching_task')[1]).toBeCloseTo(matchingLoss(w.map((v, d) => v - eta * dw[d])), 9);
    for (const label of [0, 1] as const) {
      const r = graph('ranking_task', { label }), find = (key: string) => r.nodes.find(n => n.label === key)!.value;
      const loss = (z: number) => Math.max(z, 0) - z * label + Math.log1p(Math.exp(-Math.abs(z)));
      const numericalDz = (loss(find('z') + 1e-5) - loss(find('z') - 1e-5)) / 2e-5;
      expect(find('∂L/∂z')).toBeCloseTo(numericalDz, 9);
      expect(find('w′1')).toBeCloseTo(.8 - eta * numericalDz * .9, 9);
      expect(find('b′')).toBeCloseTo(-.2 - eta * numericalDz, 9);
      expect(find('BCE′')).toBeLessThan(find('BCE'));
    }
    expect(get('CE′')).toBeLessThan(get('CE'));
  });
});

describe('scalar calculation engine', () => {
  it('evaluates weighted neurons with bias and activation', () => {
    const g = new Network(), a = g.input('a', -2), b = g.input('b', 3);
    g.affine('h', [a, b], [.5, 2], 1, 'relu', -.5);
    expect(g.nodes.at(-1)!.pre).toBe(4.5);
    expect(g.nodes.at(-1)!.value).toBe(4.5);
    expect(calculation(g.nodes.at(-1)!, g.nodes).substituted).toContain('4.5');
    expect(activate(-2, 'relu')).toBe(0);
    expect(activate(0, 'sigmoid')).toBe(.5);
    expect(activate(-1000, 'sigmoid')).toBe(0);
    expect(activate(0, 'gelu')).toBe(0);
  });
  it('normalizes softmax stably and applies layer norm', () => {
    const g = new Network(), xs = [g.input('a', 1000), g.input('b', 1001)];
    const sm = g.softmax('p', xs, 1), norm = g.norm('LN', xs, 1);
    const find = (id: string) => g.nodes.find(n => n.id === id)!.value;
    expect(sm.map(find).reduce((a, b) => a + b)).toBeCloseTo(1, 12);
    expect(find(sm[1])).toBeCloseTo(Math.exp(1) / (1 + Math.exp(1)), 12);
    expect(find(norm[0])).toBeCloseTo(-.5 / Math.sqrt(.25 + 1e-5), 12);
    expect(find(norm[1])).toBeCloseTo(.5 / Math.sqrt(.25 + 1e-5), 12);
  });
});
