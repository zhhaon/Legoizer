import type { Neuron } from './neuronMath';

export type OrganizationGroup = 0 | 1 | 2;
export interface OrganizationFocus { group: OrganizationGroup; revision: number }

/** Semantic groups match the structural notes, independently of animation stages or layout columns. */
export function organizationGroup(kind: string, node: Neuron): OrganizationGroup {
  const name = node.label;
  if (node.kind === 'output') return 2;
  switch (kind) {
    case 'cnn': return node.op === 'input' || /^c[1-3]$/.test(name) ? 0 : name === 'title[1]' ? 2 : 1;
    case 'cnncat': return name.startsWith('category') ? 1 : 0;
    case 'ada': return name.startsWith('pool') ? 2 : node.op === 'softmax' ? 1 : 0;
    case 'attention': return /^(attnO|pool)/.test(name) ? 2 : node.op === 'input' || /^attn[QKV]/.test(name) ? 0 : 1;
    case 'polyattention': return /^interest[12]\[/.test(name) ? 2 : node.op === 'input' || /^t[123]\[/.test(name) ? 0 : 1;
    case 'gru': return name.startsWith('Wout') ? 2 : node.op === 'input' ? 0 : 1;
    case 'pooling': return ['mean', 'max'].includes(node.op) ? 1 : 0;
    case 'transformer': case 'flattentransformer':
      return /^(P|pool)/.test(name) ? 2 : /^(Tr[12]|TLN|TFF|TW2)/.test(name) ? 1 : 0;
    case 'fastformer': case 'flattenfastformer':
      return /^(res|pool|Wout)/.test(name) ? 2 : /^(qK|ka|β|k̄|Vḱ)/.test(name) ? 1 : 0;
    case 'lm': return /^(W|pool)/.test(name) ? 2 : node.op === 'input' || /^LM[QKV]/.test(name) ? 0 : 1;
    case 'externalhf': return /^(mean|Wout)/.test(name) ? 2 : node.op === 'input' || /^LM[QKV]/.test(name) ? 0 : 1;
    case 'iisan': return name.startsWith('Wout') ? 2 : /^(L[123]|cacheL)/.test(name) ? 0 : 1;
    case 'nullsimple': case 'nullconcat': case 'scsimple': case 'scflatten':
      return name.startsWith('pass') ? 2 : node.op === 'input' ? 0 : 1;
    case 'dot': return node.op === 'input' ? 0 : node.op === 'product' ? 1 : 2;
    case 'dnn': return node.op === 'input' ? 0 : name === 'MLPz' ? 2 : 1;
    case 'pnn': return node.op === 'input' ? 0 : name.startsWith('PNN') ? 2 : 1;
    case 'deepfm': return node.op === 'input' ? 0 : name === 'average' ? 2 : 1;
    case 'dcn': case 'dcnv2': case 'gdcn':
      return node.op === 'input' ? 0 : /^(DNN|fusion)/.test(name) ? 2 : 1;
    case 'autoint': return name === 'readout' ? 2 : node.op === 'input' || /^field[QKV]/.test(name) ? 0 : 1;
    case 'din': return node.op === 'input' ? 0 : /^(interest|DNN)/.test(name) ? 2 : 1;
    case 'miner': return node.op === 'input' ? 0 : name.startsWith('interestScore') ? 1 : 2;
    case 'masknet': return /^(block|readout)/.test(name) ? 2 : name.startsWith('mask×h') ? 1 : 0;
    case 'finalmlp': return /^(linear|fusion)/.test(name) ? 2 : /^(B|h1h2)/.test(name) ? 1 : 0;
    case 'matching_task': return name.includes('′') ? 2 : name === 'CE' || /^(∂|grad)/.test(name) ? 1 : 0;
    case 'ranking_task': return name.includes('′') ? 2 : name === 'BCE' || name.startsWith('∂') ? 1 : 0;
    default: throw new Error(`Missing neuron organization: ${kind}`);
  }
}
