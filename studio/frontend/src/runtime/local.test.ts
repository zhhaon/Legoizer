import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { parse, stringify } from 'yaml';
import { modelConfig, expConfig, resources, validateLocal, type Snapshot } from './local';
import { safeFolder } from './bundle';
const s: Snapshot = JSON.parse(readFileSync('public/catalog.json', 'utf8'));
const parity = JSON.parse(readFileSync('src/runtime/parity.json', 'utf8'));
describe('static catalog and compiler', () => {
  for (const t of s.templates.filter(t => t.available && t.graph)) {
    it(`compiles ${t.id} with Python-equivalent model, experiment and resource configs`, () => {
      const g = structuredClone(t.graph!);
      expect(validateLocal(g, s).errors).toEqual([]);
      expect(modelConfig(g, s)).toEqual(parity[t.id].model);
      expect(expConfig(g)).toEqual(parity[t.id].exp);
      const r = resources(g, s);
      expect(r.data).toEqual(parity[t.id].data);
      expect(r.embed).toEqual(parity[t.id].embed);
      expect(parse(stringify(modelConfig(g, s)))).toEqual(parity[t.id].model);
    });
  }
  it('rejects broken connections, cycles, unknown components and invalid training values', () => {
    const g = structuredClone(s.templates.find(t => t.graph)!.graph!);
    g.edges.push({ id: 'bad', source: g.nodes[0].id, target: g.nodes[0].id, sourceHandle: 'bad', targetHandle: 'bad' });
    g.nodes[0].component = 'unknown'; g.train.batch_size = 0;
    expect(validateLocal(g, s).valid).toBe(false);
  });
  it('rejects shell syntax and traversal in export directory', () => {
    for (const p of ['../bad', '/tmp/x', 'a;echo', 'a b', 'x/../y', 'a\nb']) expect(() => safeFolder(p)).toThrow();
    expect(safeFolder('my_model/v2')).toBe('my_model/v2');
  });
});
