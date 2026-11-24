import { describe, expect, it } from 'vitest';
import { buildNeuralGraph, defaultExample, type ExampleOptions } from './neuronModels';
import { organizationGroup } from './neuronOrganization';

const kinds = ['cnn', 'cnncat', 'ada', 'attention', 'polyattention', 'gru', 'pooling', 'transformer', 'flattentransformer', 'fastformer', 'flattenfastformer', 'lm', 'iisan', 'externalhf', 'nullsimple', 'nullconcat', 'scsimple', 'scflatten', 'dot', 'dnn', 'pnn', 'deepfm', 'dcn', 'dcnv2', 'gdcn', 'autoint', 'din', 'miner', 'masknet', 'finalmlp', 'matching_task', 'ranking_task'];
const model = (kind: string, options: Partial<ExampleOptions> = {}) => buildNeuralGraph(kind, `item.${kind}`, { ...defaultExample, ...options });
function group(kind: string, label: string) {
  const node = model(kind).nodes.find(n => n.label === label);
  expect(node, label).toBeDefined();
  return organizationGroup(kind, node!);
}

describe('structural explanations linked to scalar networks', () => {
  for (const kind of kinds) it(`${kind}: every node belongs to a nonempty structural section in every example mode`, () => {
    for (const options of [{}, { hf: 'tokens', reducer: 'max', pooling: 'max', reverse: true, context: 'recipe', label: 0 }, { reducer: 'mean' }] as Partial<ExampleOptions>[]) {
      const g = model(kind, options);
      const groups = g.nodes.map(node => organizationGroup(kind, node));
      expect([...new Set(groups)].sort()).toEqual([0, 1, 2]);
      for (const node of g.nodes.filter(n => n.kind === 'output')) expect(organizationGroup(kind, node)).toBe(2);
    }
  });
  it('links a shared convolution kernel independently of animation stage', () => {
    expect(group('cnn', 'x1')).toBe(0);
    for (const name of ['c1', 'c2', 'c3']) expect(group('cnn', name)).toBe(0);
    expect(group('cnn', 'convα1')).toBe(1);
    expect(group('cnn', 'title[1]')).toBe(2);
    expect(model('cnn').nodes.find(n => n.label === 'c1')!.stage).toBe(1);
    expect(group('cnncat', 'category[1]')).toBe(1);
  });
  it('separates projections, attention interactions, and head merging', () => {
    expect(group('attention', 'attnQ1.1')).toBe(0);
    expect(group('attention', 'attnhead1.1')).toBe(1);
    expect(group('attention', 'attnO1[1]')).toBe(2);
    expect(group('transformer', 'TLN1.1.1')).toBe(1);
  });
  it('groups recurrent gates across time, rather than one diagram column', () => {
    for (const name of ['r1.1', 'z2.1', 'h3.1']) expect(group('gru', name)).toBe(1);
    expect(group('gru', 'h0[1]')).toBe(0);
    expect(group('gru', 'Wout[1]')).toBe(2);
  });
  it('keeps gradients with the loss explanation and updated parameters with recomputation', () => {
    expect(group('matching_task', 'p1')).toBe(0);
    expect(group('matching_task', '∂L/∂w1')).toBe(1);
    expect(group('matching_task', 'w′1')).toBe(2);
    expect(group('ranking_task', 'BCE')).toBe(1);
    expect(group('ranking_task', 'b′')).toBe(2);
  });
});
