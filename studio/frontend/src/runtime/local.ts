import { staticBenchmark, staticBenchmarkDetail } from './benchmark';
import { parse, stringify } from 'yaml';
import type { ComponentCatalog, TemplateSummary } from '@/types/catalog';
import { DEFAULT_TRAIN, emptyGraph, type GraphJson, type GraphNode } from '@/types/graph';
import type { CompileResult, ValidationIssue, ValidationReport } from '@/types/api';
import { rt } from './messages';

type Dict = Record<string, any>;
export interface Snapshot {
  schema_version: number; revision: string; catalog: ComponentCatalog; templates: TemplateSummary[];
  resources: Record<string, { data: Dict; embed: Dict }>; references: Record<string, string>;
  archetypes: { key: string; name: string; type: string; components: string[]; reference: string }[];
  lm_models: Record<string, [string, number]>;
}
let pending: Promise<Snapshot> | undefined;
export function snapshot(): Promise<Snapshot> {
  return pending ??= fetch(`${import.meta.env.BASE_URL}catalog.json`).then(async r => {
    if (!r.ok) throw new Error(rt('unavailable'));
    const data = await r.json();
    if (data.schema_version !== 1) throw new Error(rt('unavailable'));
    return data;
  }).catch(error => { pending = undefined; throw error; });
}
const clone = <T>(v: T): T => JSON.parse(JSON.stringify(v));
export function roles(g: GraphJson, s: Snapshot, role: string) {
  return g.nodes.filter(n => s.catalog.components.find(c => c.id === n.component)?.role === role);
}
function archetype(g: GraphJson, s: Snapshot) {
  const signature = ['item_encoder', 'user_encoder', 'predictor'].flatMap(r => roles(g, s, r).map(n => n.component));
  return s.archetypes.find(a => a.type === g.task && JSON.stringify(a.components) === JSON.stringify(signature));
}
export function validateLocal(g: GraphJson, s: Snapshot): ValidationReport {
  const errors: ValidationIssue[] = [];
  const add = (code: string, node: string | null = null) => errors.push({ code, severity: 'error', message: `${rt('structure')} (${code})`, reason: '', fix: '', node_id: node, edge_id: null });
  if (!Array.isArray(g.nodes) || !Array.isArray(g.edges) || !g.train) throw new Error(rt('invalid'));
  const index = new Map(s.catalog.components.map(c => [c.id, c]));
  const ids = new Set<string>();
  const byId = new Map(g.nodes.map(n => [n.id, n]));
  for (const n of g.nodes) {
    if (ids.has(n.id)) add('duplicate_node', n.id);
    ids.add(n.id);
    const c = index.get(n.component);
    if (!c) { add('unknown_component', n.id); continue; }
    if (c.tasks.length && !c.tasks.includes(g.task) && !c.conceptual) add('task', n.id);
    for (const p of c.parameters) {
      const v = n.params[p.name] ?? p.default;
      if (v == null) { if (p.required && !p.nullable) add(`required:${p.name}`, n.id); continue; }
      const number = (x: unknown) => typeof x === 'number' && Number.isFinite(x);
      const types: Record<string, boolean> = {
        integer: number(v) && Number.isInteger(v), number: number(v), boolean: typeof v === 'boolean',
        string: typeof v === 'string', enum: p.choices.includes(String(v)), list: Array.isArray(v),
        object: typeof v === 'object' && !Array.isArray(v),
        number_or_list: number(v) || (Array.isArray(v) && v.length > 0 && v.every(number)),
        string_or_list: typeof v === 'string' || (Array.isArray(v) && v.every(x => typeof x === 'string')),
      };
      if (!types[p.type] || (typeof v === 'number' && ((p.minimum != null && v < p.minimum) || (p.maximum != null && v > p.maximum)))) add(`parameter:${p.name}`, n.id);
      if (p.name === 'hidden_size' && v !== g.train.hidden_size) add('hidden_size', n.id);
    }
  }
  for (const r of ['dataset', 'user_encoder', 'predictor']) if (roles(g, s, r).length !== 1) add(`count:${r}`);
  if (roles(g, s, 'item_encoder').length > 1 || (g.task === 'matching' && roles(g, s, 'item_encoder').length !== 1)) add('count:item_encoder');
  for (const r of ['training', 'evaluation']) if (roles(g, s, r).length > 1) add(`count:${r}`);
  const edgeKeys = new Set<string>();
  for (const e of g.edges) {
    const source = index.get(byId.get(e.source)?.component || '');
    const target = index.get(byId.get(e.target)?.component || '');
    const out = source?.outputs.find(p => p.name === e.sourceHandle);
    const input = target?.inputs.find(p => p.name === e.targetHandle);
    const key = JSON.stringify([e.source, e.sourceHandle, e.target, e.targetHandle]);
    if (edgeKeys.has(key)) add('duplicate_edge');
    edgeKeys.add(key);
    if (!out || !input || e.source === e.target) add('edge');
    else if (out.type !== input.type && !s.catalog.port_types.find(p => p.id === input.type)?.accepts.includes(out.type)) add('port_type');
  }
  const visiting = new Set<string>(), done = new Set<string>();
  const visit = (id: string) => {
    if (visiting.has(id)) { add('cycle', id); return; }
    if (done.has(id)) return;
    visiting.add(id);
    g.edges.filter(e => e.source === id).forEach(e => visit(e.target));
    visiting.delete(id); done.add(id);
  };
  g.nodes.forEach(n => {
    visit(n.id);
    const spec = index.get(n.component);
    for (const p of spec?.inputs || []) if (p.required && !g.edges.some(e => e.target === n.id && e.targetHandle === p.name)) add(`input:${p.name}`, n.id);
    if (!spec?.conceptual && !g.edges.some(e => e.source === n.id || e.target === n.id)) add('orphan', n.id);
  });
  for (const k of ['batch_size', 'accumulate_batch', 'hidden_size', 'epochs', 'patience'] as const) if (!Number.isInteger(g.train[k]) || g.train[k] < 1) add(`train:${k}`);
  if (!Number.isFinite(g.train.lr) || g.train.lr <= 0 || !Number.isInteger(g.train.seed)) add('train');
  if (!g.train.metrics.length || !g.train.dev_metric) add('metrics');
  const first = (r: string) => roles(g, s, r)[0]?.component || null;
  return { valid: !errors.length, errors, warnings: [], runtime_checks: [{ code: 'server_runtime', severity: 'info', message: rt('staticCheck'), reason: '', fix: '', node_id: null, edge_id: null }], summary: { nodes: g.nodes.length, edges: g.edges.length, task: g.task, model_label: archetype(g, s)?.name || 'Custom Model', exact_template: null, dataset: first('dataset'), item_encoder: first('item_encoder'), user_encoder: first('user_encoder'), predictor: first('predictor') } };
}
function setPath(obj: Dict, path: string, value: unknown) {
  const parts = path.split('.'); let cursor = obj;
  for (const p of parts.slice(0, -1)) { if (['__proto__', 'constructor', 'prototype'].includes(p)) throw new Error(rt('invalid')); cursor = cursor[p] ??= {}; }
  const last = parts.at(-1)!;
  if (['__proto__', 'constructor', 'prototype'].includes(last)) throw new Error(rt('invalid'));
  cursor[last] = value;
}
export function modelConfig(g: GraphJson, s: Snapshot): Dict {
  const config: Dict = { use_item_content: !!roles(g, s, 'item_encoder').length, hidden_size: g.train.hidden_size, item_hidden_size: g.train.item_hidden_size || g.train.hidden_size, use_fast_eval: true, use_neg_sampling: g.task === 'matching' };
  const meta: Dict = {};
  for (const [role, section, key] of [['item_encoder', 'item_config', 'item'], ['user_encoder', 'user_config', 'user'], ['predictor', 'predictor_config', 'predictor']]) {
    const n = roles(g, s, role)[0]; if (!n) continue;
    const c = s.catalog.components.find(c => c.id === n.component)!;
    meta[key] = c.meta_value;
    for (const p of c.parameters) if (p.config_path.startsWith(`config.${section}.`) && p.name in n.params) setPath(config, p.config_path.slice(7), n.params[p.name]);
    if (key === 'item') {
      const checkpoint = (c.meta_value || '').toLowerCase().replace(/iisan$/, '');
      config.item_hidden_size = s.lm_models[checkpoint]?.[1] || n.params.item_hidden_size || config.item_hidden_size;
      if (n.params.item_page_size || s.lm_models[checkpoint]) config.item_page_size = n.params.item_page_size || 64;
    }
  }
  const train = roles(g, s, 'training')[0], predictor = roles(g, s, 'predictor')[0];
  config.use_fast_eval = train?.params.use_fast_eval ?? predictor?.params.use_fast_eval ?? true;
  if (g.task === 'matching') config.neg_count = train?.params.neg_count ?? 4;
  return { name: archetype(g, s)?.name || 'CustomModel', meta, config };
}
export function expConfig(g: GraphJson): Dict {
  const t = g.train;
  return { load: { sign: null, model_only: true, strict: false }, store: { metric: t.dev_metric, patience: t.patience }, policy: { epoch: t.epochs, lr: t.lr, item_lr: null, freeze_emb: false, pin_memory: false, epoch_batch: 0, batch_size: t.batch_size, accumulate_batch: t.accumulate_batch, device: t.device === 'cpu' ? 'cpu' : 'gpu', n_warmup: 0, check_interval: -2, simple_dev: t.simple_dev, disable_tqdm: false }, metrics: [...new Set([...(t.metrics.includes(t.dev_metric) ? [] : [t.dev_metric]), ...t.metrics])] };
}
export function resources(g: GraphJson, s: Snapshot) {
  const n = roles(g, s, 'dataset')[0];
  if (!n) throw new Error(rt('invalid'));
  const spec = s.catalog.components.find(c => c.id === n.component)!;
  const val = (k: string) => n.params[k] ?? spec.parameters.find(p => p.name === k)?.default;
  const dataPath = String(val('data_config'));
  const item = roles(g, s, 'item_encoder')[0];
  const meta = s.catalog.components.find(c => c.id === item?.component)?.meta_value || '';
  const checkpoint = meta.toLowerCase().replace(/iisan$/, '');
  let lm = s.lm_models[checkpoint]?.[0] || String(val('lm') || 'glove');
  if (lm === 'opt') lm = 'optbase';
  const lang = String(val('lang') || 'cmn');
  const preset = s.resources[`${dataPath}|${lm}|${dataPath.endsWith('/xmind.yaml') ? lang : 'cmn'}`];
  if (!preset) throw new Error(rt('unavailable'));
  const { data, embed } = clone(preset);
  const externalItem = item?.component === 'item.externalhf';
  const externalDataset = !externalItem && !s.lm_models[checkpoint] && val('token_source') === 'external_hf';
  const vocab = externalItem || externalDataset ? 'external_hf' : s.lm_models[checkpoint]?.[0] || embed.embeddings?.[0]?.vocab_name;
  const entries = (n.params.item_inputs as { column: string; truncate: number | null }[] | undefined) || data.item.inputs.map((e: any) => typeof e === 'string' ? { column: e, truncate: null } : { column: Object.keys(e)[0], truncate: Object.values(e)[0] });
  data.item.inputs = entries.filter((e: any) => !(s.lm_models[checkpoint] || externalItem) || e.column.includes('@')).map((e: any) => {
    const column = vocab ? e.column.replace(/@[^\s]+/g, `@${vocab}`) : e.column;
    return e.truncate == null ? column : { [column]: e.truncate };
  });
  if ((externalItem || externalDataset) && !data.item.inputs.some((e: any) => JSON.stringify(e).includes('@external_hf'))) data.item.inputs.push({ 'title@external_hf': 30 });
  if (externalItem || externalDataset || s.lm_models[checkpoint]) Object.assign(embed, { name: null, embeddings: [], transformation: 'auto', transformation_dropout: 0 });
  const externalConfig = externalDataset ? Object.fromEntries(['model_id', 'revision', 'tokenizer_id', 'tokenizer_revision', 'freeze_token_embeddings'].map(k => [k, val(`external_${k}`)])) : null;
  return { data, embed, dataPath, plan: { data: clone(data), embed: clone(embed), lang, data_config: dataPath, checkpoint: s.lm_models[checkpoint] ? checkpoint : null, external_hf: externalItem, dataset_external_hf: externalConfig } };
}
export async function compileLocal(g: GraphJson): Promise<CompileResult> {
  const s = await snapshot(), report = validateLocal(g, s);
  if (!report.valid) throw new Error(rt('invalid'));
  const model = modelConfig(g, s), r = resources(g, s), a = archetype(g, s);
  return { yaml_text: stringify(model), model_dict: model, model_label: report.summary.model_label, exact_template: a?.reference || null, reference_yaml: a?.reference || null, notes: [rt('staticCheck')], generated_file: null, data_config: r.dataPath, exp_config: null, embed_config: null, validation: report, compatible_with_reference: false };
}
function importModel(text: string, s: Snapshot): GraphJson {
  const raw = parse(text);
  if (!raw?.meta || !raw?.config || text.includes('$$import') || text.includes('${')) throw new Error(rt('imports'));
  const g = emptyGraph(String(raw.name || 'Imported Model'));
  g.task = raw.config.use_neg_sampling === false ? 'ranking' : 'matching';
  g.train = { ...DEFAULT_TRAIN, hidden_size: raw.config.hidden_size || 256, item_hidden_size: raw.config.item_hidden_size || null };
  for (const [role, metaKey] of [['dataset', ''], ['item_encoder', 'item'], ['user_encoder', 'user'], ['predictor', 'predictor']]) {
    const c = role === 'dataset' ? s.catalog.components.find(c => c.id === 'dataset.mind') : s.catalog.components.find(c => c.role === role && c.meta_value === raw.meta[metaKey]);
    if (!c) { if (role === 'item_encoder' && !raw.meta.item) continue; throw new Error(rt('invalid')); }
    const params: Dict = Object.fromEntries(c.parameters.filter(p => p.default != null).map(p => [p.name, p.default]));
    for (const p of c.parameters) if (p.config_path.startsWith('config.')) {
      const v = p.config_path.split('.').reduce((v: any, k) => v?.[k], raw); if (v != null) params[p.name] = v;
    }
    if ('hidden_size' in params) params.hidden_size = g.train.hidden_size;
    g.nodes.push({ id: role, component: c.id, params, position: { x: g.nodes.length * 260, y: 120 } });
  }
  const link = (source: string, sourceHandle: string, target: string, targetHandle: string) => g.edges.push({ id: `e${g.edges.length}`, source, sourceHandle, target, targetHandle });
  link('user_encoder', 'user', 'predictor', 'user');
  link('dataset', 'features', 'user_encoder', 'features');
  if (g.nodes.some(n => n.id === 'item_encoder')) { link('dataset', 'content', 'item_encoder', 'content'); link('item_encoder', 'history', 'user_encoder', 'history'); link('item_encoder', 'candidate', 'predictor', 'candidate'); }
  else { link('dataset', 'history', 'user_encoder', 'history'); link('dataset', 'features', 'predictor', 'features'); }
  return g;
}
export const localApi = {
  benchmark: staticBenchmark,
  benchmarkDetail: staticBenchmarkDetail,
  components: async () => (await snapshot()).catalog,
  templates: async () => ({ templates: (await snapshot()).templates }),
  template: async (id: string) => { const t = (await snapshot()).templates.find(t => t.id === id); if (!t) throw new Error(rt('invalid')); return clone(t); },
  validate: async (g: GraphJson) => validateLocal(g, await snapshot()), compile: compileLocal,
  referenceYaml: async (path: string) => (await snapshot()).references[path] || '',
  importGraph: async (payload: { graph?: unknown; yaml?: string }) => { const s = await snapshot(), graph = payload.yaml ? importModel(payload.yaml, s) : clone(payload.graph as GraphJson); return { graph, validation: validateLocal(graph, s) }; },
  compatible: async (id: string, direction: string) => {
    const s = await snapshot(), c = s.catalog.components.find(c => c.id === id);
    return { results: s.catalog.components.filter(other => {
      const from = direction === 'upstream' ? other : c, to = direction === 'upstream' ? c : other;
      return from?.outputs.some(o => to?.inputs.some(i => o.type === i.type || s.catalog.port_types.find(p => p.id === i.type)?.accepts.includes(o.type)));
    }).map(c => ({ ...c, reasons: [] })) };
  },
  datasetInputs: async (g: { nodes: GraphNode[] }) => {
    const r = resources({ ...emptyGraph(), nodes: g.nodes }, await snapshot());
    const defaults = r.data.item.inputs.map((e: any) => typeof e === 'string' ? { column: e, truncate: null } : { column: Object.keys(e)[0], truncate: Object.values(e)[0] });
    return { defaults, columns: defaults.map((e: any) => e.column) };
  },
  saveGraph: async (g: GraphJson) => { localStorage.setItem(`studio-local:${g.graph_id}`, JSON.stringify(g)); return { saved: true, graph_id: g.graph_id }; },
  savedGraphs: async () => ({ graphs: Object.keys(localStorage).filter(k => k.startsWith('studio-local:')).flatMap(k => { try { const g = JSON.parse(localStorage.getItem(k)!); return [{ graph_id: g.graph_id, name: g.name, task: g.task, nodes: g.nodes.length, saved_at: 0 }]; } catch { return []; } }) }),
  loadGraph: async (id: string) => { const graph = JSON.parse(localStorage.getItem(`studio-local:${id}`) || 'null'); return { graph, validation: validateLocal(graph, await snapshot()) }; },
};
