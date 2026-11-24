import { Network, type NeuralGraph, type Activation } from './neuronMath';

export interface ExampleOptions { candidate: 'sports' | 'cooking'; pooling: 'mean' | 'max'; reverse: boolean; hf: 'contextual' | 'tokens'; label: 0 | 1; reducer: 'weighted' | 'max' | 'mean'; context: 'phone' | 'recipe' }
export const defaultExample: ExampleOptions = { candidate: 'sports', pooling: 'mean', reverse: false, hf: 'contextual', label: 1, reducer: 'weighted', context: 'phone' };
function vec(g: Network, prefix: string, xs: number[], kind: 'input' | 'parameter' = 'input') { return xs.map((x, d) => g.input(`${prefix}[${d + 1}]`, x, kind)); }
function projection(g: Network, prefix: string, xs: string[], matrix: number[][], stage: number, activation: Activation = 'identity') { return matrix.map((row, d) => g.affine(`${prefix}[${d + 1}]`, xs, row, stage, activation)); }
function scalarDot(g: Network, prefix: string, a: string[], b: string[], stage: number, sumStage = stage) {
  const products = a.map((x, d) => g.product(`${prefix}×${d + 1}`, [x, b[d]], stage));
  return g.add(prefix, 'sum', products, sumStage);
}
function weighted(g: Network, prefix: string, xs: string[][], weights: string[], stage: number) {
  return xs[0].map((_, d) => g.add(`${prefix}[${d + 1}]`, 'sum', xs.map((row, i) => g.product(`${prefix}α${i + 1}v${d + 1}`, [weights[i], row[d]], stage)), stage));
}
function additive(g: Network, prefix: string, xs: string[][], stage: number) {
  const hidden = xs.map((row, i) => g.affine(`${prefix}t${i + 1}`, row, row.map((_, d) => d ? -.35 : .8), stage, 'tanh', .1));
  const scores = hidden.map((id, i) => g.affine(`${prefix}s${i + 1}`, [id], [1.2], stage));
  return g.softmax(`${prefix}α`, scores, stage);
}
function pool(g: Network, prefix: string, xs: string[][], stage: number, op: 'mean' | 'max') { return xs[0].map((_, d) => g.add(`${prefix}[${d + 1}]`, op, xs.map(row => row[d]), stage)); }
function mlp(g: Network, prefix: string, xs: string[], stage: number, readoutStage = stage) {
  const a = g.affine(`${prefix}h1`, xs, xs.map((_, i) => i % 2 ? -.2 : .7), stage, 'relu', -.1);
  const b = g.affine(`${prefix}h2`, xs, xs.map((_, i) => i % 2 ? .6 : -.4), stage, 'relu', .1);
  return g.affine(`${prefix}z`, [a, b], [1, -.5], readoutStage, 'identity', .05);
}
function attention(g: Network, prefix: string, xs: string[][], stage: number, heads = 2, causal = false) {
  const headResults = Array.from({ length: heads }, (_, h) => {
    const q = xs.map((x, i) => g.affine(`${prefix}Q${h + 1}.${i + 1}`, x, h ? [.2, 1] : [1, .2], stage));
    const k = xs.map((x, i) => g.affine(`${prefix}K${h + 1}.${i + 1}`, x, h ? [.1, .9] : [.8, .1], stage));
    const v = xs.map((x, i) => g.affine(`${prefix}V${h + 1}.${i + 1}`, x, h ? [0, 1] : [1, 0], stage));
    return xs.map((_, i) => {
      const allowed = k.map((id, j) => ({ id, j })).filter(({ j }) => !causal || j <= i);
      const scores = allowed.map(({ id, j }) => g.product(`${prefix}e${h + 1}.${i + 1}.${j + 1}`, [q[i], id], stage));
      const ws = g.softmax(`${prefix}a${h + 1}.${i + 1}.`, scores, stage);
      return g.add(`${prefix}head${h + 1}.${i + 1}`, 'sum', allowed.map(({ j }, t) => g.product(`${prefix}av${h + 1}.${i + 1}.${j + 1}`, [ws[t], v[j]], stage)), stage);
    });
  });
  return xs.map((_, i) => projection(g, `${prefix}O${i + 1}`, headResults.map(result => result[i]), [[.8, .2], [.1, .9]].map(row => row.slice(0, heads)), stage));
}
function transformer(g: Network, prefix: string, xs: string[][], stage: number, causal = false) {
  const attended = attention(g, prefix, xs, stage, 2, causal);
  return xs.map((row, i) => {
    const residual = row.map((x, d) => g.add(`${prefix}r1.${i + 1}.${d + 1}`, 'sum', [x, attended[i][d]], stage));
    const normalized = g.norm(`${prefix}LN1.${i + 1}.`, residual, stage);
    const ff = projection(g, `${prefix}FF${i + 1}`, normalized, [[.5, .2], [-.1, 1.2]], stage, 'gelu');
    const down = projection(g, `${prefix}W2.${i + 1}`, ff, [[1, .1], [.2, .8]], stage);
    return g.norm(`${prefix}LN2.${i + 1}.`, row.map((_, d) => g.add(`${prefix}r2.${i + 1}.${d + 1}`, 'sum', [normalized[d], down[d]], stage)), stage);
  });
}
function fastformer(g: Network, xs: string[][]) {
  const q = xs.map((row, i) => projection(g, `Q${i + 1}`, row, [[.8, .2], [.1, .9]], 1));
  const k = xs.map((row, i) => projection(g, `K${i + 1}`, row, [[.9, .1], [-.1, .8]], 1));
  const alpha = g.softmax('α', q.map((row, i) => g.affine(`qa${i + 1}`, row, [.6, .4], 1)), 1);
  const globalQ = weighted(g, 'q̄', q, alpha, 1);
  const qk = k.map((row, i) => row.map((id, d) => g.product(`qK${i + 1}.${d + 1}`, [id, globalQ[d]], 2)));
  const beta = g.softmax('β', qk.map((row, i) => g.affine(`ka${i + 1}`, row, [.3, .7], 2)), 2);
  const globalK = weighted(g, 'k̄', qk, beta, 2);
  const values = xs.map((row, i) => row.map((id, d) => g.product(`Vḱ${i + 1}.${d + 1}`, [id, globalK[d]], 2)));
  const contextual = values.map((row, i) => row.map((id, d) => g.add(`res${i + 1}.${d + 1}`, 'sum', [id, q[i][d]], 2)));
  return weighted(g, 'pool', contextual, additive(g, 'pool', contextual, 2), 2);
}
function outputs(g: Network, xs: string[], prefix = 'out') { return xs.map((id, i) => g.output(`${prefix}[${i + 1}]`, id)); }

export function buildNeuralGraph(kind: string, componentId: string, options: ExampleOptions): NeuralGraph {
  const g = new Network();
  const history = (count = 3) => [[1, 0], [.8, .2], [0, 1]].slice(0, count).map((x, i) => vec(g, `x${i + 1}`, x));
  if (kind === 'cnn' || kind === 'cnncat') {
    const x = [.2, .8, 1, .7, .3].map((value, i) => g.input(`x${i + 1}`, value));
    const c = [0, 1, 2].map(i => g.affine(`c${i + 1}`, x.slice(i, i + 3), [.25, .5, .25], 1, 'relu', 0, 'sharedKernel'));
    const ws = additive(g, 'conv', c.map(id => [id]), 2);
    const pooled = weighted(g, 'title', c.map(id => [id]), ws, 2);
    if (kind === 'cnncat') return g.finish(outputs(g, [...pooled, ...vec(g, 'category', [.1, .9])]), 'cnncat');
    return g.finish(outputs(g, pooled), 'cnn');
  }
  if (['ada', 'attention', 'polyattention'].includes(kind)) {
    const x = history(kind === 'attention' ? 2 : 3);
    if (kind === 'ada') return g.finish(outputs(g, weighted(g, 'pool', x, additive(g, 'add', x, 1), 2)), 'ada');
    if (kind === 'polyattention') {
      const projected = x.map((row, i) => projectionFn(`t${i + 1}`, row));
      function projectionFn(prefix: string, row: string[]) { return projection(g, prefix, row, [[1, .2], [-.2, 1]], 1, 'tanh'); }
      const interests = [[1, .1], [.1, 1]].map((query, i) => {
        const logits = projected.map((row, j) => g.affine(`interest${i + 1}.s${j + 1}`, row, query, 1));
        return weighted(g, `interest${i + 1}`, x, g.softmax(`interest${i + 1}.α`, logits, 1), 2);
      });
      return g.finish(outputs(g, interests.flat()), 'polyattention');
    }
    const contextual = attention(g, 'attn', x.slice(0, 2), 1);
    return g.finish(outputs(g, weighted(g, 'pool', contextual, additive(g, 'pool', contextual, 2), 2)), 'attention');
  }
  if (kind === 'gru') {
    const x = history(); if (options.reverse) x.reverse();
    let state = vec(g, 'h0', [0, 0]);
    x.forEach((row, i) => {
      const both = [...row, ...state];
      const r = [0, 1].map(d => g.affine(`r${i + 1}.${d + 1}`, both, d ? [.1, .7, .2, .4] : [.8, -.1, .3, .2], 1, 'sigmoid', -.1));
      const z = [0, 1].map(d => g.affine(`z${i + 1}.${d + 1}`, both, d ? [.2, .6, .1, .5] : [.6, .2, .4, -.1], 1, 'sigmoid', .1));
      // PyTorch GRU: n=tanh(W_in*x + r ⊙ W_hn*h); h=(1-z)⊙n + z⊙h_prev.
      const recurrent = projection(g, `Wh${i + 1}`, state, [[.5, .2], [-.1, .6]], 1);
      state = row.map((_, d) => {
        const reset = g.product(`reset${i + 1}.${d + 1}`, [r[d], recurrent[d]], 1);
        const input = g.affine(`Wx${i + 1}.${d + 1}`, row, d ? [.1, .9] : [.8, -.2], 1, 'identity', .05);
        const candidate = g.affine(`n${i + 1}.${d + 1}`, [input, reset], [1, 1], 1, 'tanh');
        const complement = g.affine(`1-z${i + 1}.${d + 1}`, [z[d]], [-1], 1, 'identity', 1);
        return g.add(`h${i + 1}.${d + 1}`, 'sum', [g.product(`new${i + 1}.${d + 1}`, [complement, candidate], 1), g.product(`keep${i + 1}.${d + 1}`, [z[d], state[d]], 1)], 1);
      });
    });
    const projected = projection(g, 'Wout', state, [[1, .1], [.2, .9]], 2);
    return g.finish(outputs(g, projected), 'gru');
  }
  if (kind === 'pooling') {
    const x = history();
    const valid = x.map((row, i) => { const mask = g.input(`mask${i + 1}`, 1); return row.map((id, d) => g.product(`valid${i + 1}.${d + 1}`, [id, mask], 1)); });
    return g.finish(outputs(g, pool(g, options.pooling, valid, 2, options.pooling)), 'pooling');
  }
  if (['transformer', 'flattentransformer', 'fastformer', 'flattenfastformer'].includes(kind)) {
    const x = history(2);
    if (kind.includes('fastformer')) return g.finish(outputs(g, projection(g, 'Wout', fastformer(g, x), [[1, .1], [.1, .9]], 2)), 'fastformer');
    const states = transformer(g, 'T', x, 1);
    const projected = states.map((row, i) => projection(g, `P${i + 1}`, row, [[.8, .2], [-.1, 1]], 2));
    return g.finish(outputs(g, weighted(g, 'pool', projected, additive(g, 'pool', projected, 2), 2)), 'transformer');
  }
  if (['lm', 'iisan', 'externalhf'].includes(kind)) {
    if (kind === 'iisan') {
      const layerTokens = options.context === 'phone' ? [[[1, 0], [.8, .2]], [[.6, .4], [.4, .6]], [[.2, .8], [.1, .9]]] : [[[0, 1], [.2, .8]], [[.4, .6], [.6, .4]], [[.8, .2], [.9, .1]]];
      const cached = layerTokens.map((rows, i) => pool(g, `cacheL${i + 1}`, rows.map((row, j) => vec(g, `L${i + 1}.t${j + 1}`, row)), 0, 'mean'));
      let state = cached[0];
      cached.slice(1).forEach((row, i) => {
        const fusionStage = i === 0 ? 1 : 2;
        const gate = g.input(`gateLogit${i + 1}`, .5, 'parameter');
        const mu = g.affine(`μ${i + 1}`, [gate], [1], fusionStage, 'sigmoid');
        const complement = g.affine(`1-μ${i + 1}`, [mu], [-1], fusionStage, 'identity', 1);
        const fusion = state.map((id, d) => g.add(`fuse${i + 1}.${d + 1}`, 'sum', [g.product(`old${i + 1}.${d + 1}`, [mu, id], fusionStage), g.product(`next${i + 1}.${d + 1}`, [complement, row[d]], fusionStage)], fusionStage));
        const up = projection(g, `SANup${i + 1}`, fusion, [[1, .2], [.1, 1], [-.3, .8], [.7, -.2]], 2, 'relu');
        const down = projection(g, `SANdown${i + 1}`, up, [[.5, .1, .2, .3], [.1, .6, .3, -.1]], 2);
        state = g.norm(`SANnorm${i + 1}.`, fusion.map((id, d) => g.add(`SANres${i + 1}.${d + 1}`, 'sum', [id, down[d]], 2)), 2);
      });
      return g.finish(outputs(g, projection(g, 'Wout', state, [[.7, .1], [.2, .8]], 2)), 'iisan');
    }
    const embeddings = options.context === 'phone' ? [[.5, .5], [.9, .1]] : [[.5, .5], [.1, .9]];
    const x = embeddings.map((row, i) => vec(g, `token${i + 1}`, row));
    const tokensOnly = kind === 'externalhf' && options.hf === 'tokens';
    const contextual = tokensOnly ? x.map((row, i) => row.map((id, d) => g.affine(`lookup${i + 1}.${d + 1}`, [id], [1], 1))) : attention(g, 'LM', x, 1, 2, kind === 'lm' && !componentId.includes('bert'));
    if (kind === 'externalhf') return g.finish(outputs(g, projection(g, 'Wout', pool(g, 'mean', contextual, 2, 'mean'), [[.8, .2], [.1, .9]], 2)), tokensOnly ? 'tokenOnly' : 'lm');
    const projected = contextual.map((row, i) => projection(g, `W${i + 1}`, row, [[.8, .2], [.1, .9]], 2));
    return g.finish(outputs(g, weighted(g, 'pool', projected, additive(g, 'pool', projected, 2), 2)), 'lm');
  }
  if (['nullsimple', 'nullconcat', 'scsimple', 'scflatten'].includes(kind)) {
    const x = history();
    const passthrough = x.flatMap((row, i) => row.map((id, d) => g.add(`copy${i + 1}.${d + 1}`, 'affine', [id], 1, { kind: 'operator' })));
    if (kind.startsWith('null')) for (let i = 0; i < 3; i++) passthrough.push(g.add(`maskCopy${i + 1}`, 'affine', [g.input(`mask${i + 1}`, 1)], 1, { kind: 'operator' }));
    return g.finish(outputs(g, passthrough.map((id, i) => g.add(`pass${i + 1}`, 'affine', [id], 2, { kind: 'operator' }))), 'passthrough');
  }
  if (kind === 'matching_task' || kind === 'ranking_task') {
    const x = (kind === 'matching_task' ? [[.9, .1], [.1, .9], [.3, .7]] : [[.9, .1]]).map((row, i) => vec(g, `candidate${i + 1}`, row));
    const w = vec(g, 'w', [.8, .2], 'parameter');
    const eta = .5;
    if (kind === 'matching_task') {
      const logits = x.map((row, i) => scalarDot(g, `s${i + 1}`, row, w, 1));
      const probabilities = g.softmax('p', logits, 1);
      const loss = g.add('CE', 'logloss', [probabilities[0]], 2);
      const dz = probabilities.map((id, i) => g.add(`∂L/∂s${i + 1}`, 'affine', [id], 3, { bias: i === 0 ? -1 : 0, kind: 'gradient', note: 'ceGradient' }));
      const dw = w.map((_, d) => g.add(`∂L/∂w${d + 1}`, 'sum', x.map((row, i) => g.product(`grad${i + 1}.${d + 1}`, [dz[i], row[d]], 3)), 3, { kind: 'gradient' }));
      const updated = w.map((id, d) => g.add(`w′${d + 1}`, 'affine', [id, dw[d]], 3, { weights: [1, -eta], kind: 'gradient', note: 'weightUpdate' }));
      const newScores = x.map((row, i) => scalarDot(g, `s′${i + 1}`, row, updated, 3));
      const newP = g.softmax('p′', newScores, 3);
      return g.finish([loss, g.add('CE′', 'logloss', [newP[0]], 3, { kind: 'output' })], 'matching');
    }
    const bias = g.input('b', -.2, 'parameter'), y = g.input('y', options.label);
    const z = g.add('z', 'sum', [scalarDot(g, 'w·x', x[0], w, 1), bias], 1);
    const p = g.affine('p', [z], [1], 1, 'sigmoid');
    const loss = g.add('BCE', 'bce', [z, y], 2, { target: options.label });
    const dz = g.add('∂L/∂z', 'affine', [p, y], 3, { weights: [1, -1], kind: 'gradient', note: 'bceGradient' });
    const updated = w.map((id, d) => g.add(`w′${d + 1}`, 'affine', [id, g.product(`∂L/∂w${d + 1}`, [dz, x[0][d]], 3)], 3, { weights: [1, -eta], kind: 'gradient', note: 'weightUpdate' }));
    const b = g.add('b′', 'affine', [bias, dz], 3, { weights: [1, -eta], kind: 'gradient' });
    const newZ = g.add('z′', 'sum', [scalarDot(g, 'w′·x', x[0], updated, 3), b], 3);
    return g.finish([loss, g.add('BCE′', 'bce', [newZ, y], 3, { target: options.label, kind: 'output' })], 'ranking');
  }
  const u = ['din', 'miner'].includes(kind) ? [] : vec(g, 'u', [.8, .2]), v = vec(g, 'v', options.candidate === 'sports' ? [.9, .1] : [.1, .9]);
  const x = [...u, ...v];
  if (kind === 'dot') return g.finish([g.output('score', scalarDot(g, 'u·v', u, v, 1, 2))], 'dot');
  if (kind === 'dnn') return g.finish([g.output('score', mlp(g, 'MLP', x, 1, 2))], 'dnn');
  if (kind === 'pnn') return g.finish([g.output('score', mlp(g, 'PNN', [...x, scalarDot(g, 'u·v', u, v, 1)], 2))], 'pnn');
  if (kind === 'deepfm') {
    const fm = scalarDot(g, 'FM', u, v, 1), deep = mlp(g, 'DNN', x, 1);
    return g.finish([g.output('score', g.add('average', 'mean', [fm, deep], 2))], 'deepfm');
  }
  if (['dcn', 'dcnv2', 'gdcn'].includes(kind)) {
    const transformed = kind === 'dcn' ? x.map((_, d) => g.affine(`crossScale${d + 1}`, x, [.25, .25, .25, .25], 1)) : projection(g, 'Wx', x, [[1, .1, 0, 0], [.1, 1.2, 0, .1], [0, .1, .8, .1], [.1, 0, .2, 1]], 1);
    const crosses = x.map((id, d) => {
      let term = g.product(`cross${d + 1}`, [id, transformed[d]], 1);
      if (kind === 'gdcn') term = g.product(`gated${d + 1}`, [term, g.affine(`gate${d + 1}`, x, x.map((_, i) => i === d ? .7 : .1), 1, 'sigmoid')], 1);
      return g.add(`x1.${d + 1}`, 'sum', [id, term], 1);
    });
    const deep = mlp(g, 'DNN', x, 1);
    return g.finish([g.output('score', g.affine('fusion', [...crosses, deep], [.2, .2, .2, .2, .5], 2, 'identity', .05))], kind);
  }
  if (kind === 'autoint') {
    const contextual = attention(g, 'field', [u, v], 1);
    return g.finish([g.output('score', g.affine('readout', contextual.flat(), [1, -.2, .5, -.1], 2))], 'autoint');
  }
  if (kind === 'din' || kind === 'miner') {
    if (kind === 'din') {
      const h = history();
      const logits = h.map((row, i) => {
        const difference = row.map((id, d) => g.affine(`v-h${i + 1}.${d + 1}`, [v[d], id], [1, -1], 1));
        const products = row.map((id, d) => g.product(`v×h${i + 1}.${d + 1}`, [v[d], id], 1));
        return mlp(g, `DIN${i + 1}`, [...v, ...row, ...difference, ...products], 1);
      });
      const pooled = weighted(g, 'interest', h, g.softmax('α', logits, 1), 2);
      return g.finish([g.output('score', mlp(g, 'DNN', pooled, 2))], 'din');
    }
    const interests = [vec(g, 'interest1', [.84, .16]), vec(g, 'interest2', [.18, .82])];
    const scores = interests.map((row, i) => scalarDot(g, `interestScore${i + 1}`, row, v, 1, 2));
    let result: string;
    if (options.reducer === 'weighted') {
      const projected = interests.map((row, i) => projection(g, `target${i + 1}`, row, [[.8, .2], [.1, .9]], 2, 'gelu'));
      const ws = g.softmax('targetα', projected.map((row, i) => scalarDot(g, `targetScore${i + 1}`, row, v, 2)), 2);
      result = g.add('weightedScore', 'sum', scores.map((id, i) => g.product(`αscore${i + 1}`, [id, ws[i]], 2)), 2);
    } else result = g.add('reduce', options.reducer, scores, 2);
    return g.finish([g.output('score', result)], 'miner');
  }
  if (kind === 'masknet') {
    const mask = projection(g, 'mask', x, [[.8, .1, .2, 0], [.1, .7, 0, .1], [.2, 0, .9, .1], [0, .1, .2, .8]], 1, 'relu');
    const normalized = g.norm('norm', x, 1);
    const modulated = normalized.map((id, d) => g.product(`mask×h${d + 1}`, [id, mask[d]], 1));
    const hidden = projection(g, 'block', modulated, [[.6, .2, .3, -.1], [.1, .5, -.2, .7]], 2, 'relu');
    return g.finish([g.output('score', g.affine('readout', hidden, [1, -.5], 2))], 'masknet');
  }
  if (kind === 'finalmlp') {
    const a = projection(g, 'MLP1', x, [[.8, .1, .6, -.1], [-.2, .7, .1, .8]], 1, 'relu');
    const b = projection(g, 'MLP2', x, [[.2, .8, .1, .7], [.7, -.1, .8, .2]], 1, 'relu');
    const bilinear = a.flatMap((id, i) => b.map((other, j) => g.affine(`B${i + 1}.${j + 1}`, [g.product(`h1h2.${i + 1}.${j + 1}`, [id, other], 2)], [i === j ? .7 : .2], 2)));
    const linear = g.affine('linear', [...a, ...b], [.2, .1, .3, -.1], 2, 'identity', .05);
    return g.finish([g.output('score', g.add('fusion', 'sum', [linear, ...bilinear], 2))], 'finalmlp');
  }
  throw new Error(`Missing neuron model: ${kind}`);
}
