import { stringify } from 'yaml';
import type { GraphJson } from '@/types/graph';
import { snapshot, modelConfig, expConfig, resources, validateLocal } from './local';
import { apiBase, localMode } from './mode';
import { rt } from './messages';

export function safeFolder(folder: string) {
  if (!/^[A-Za-z0-9_-]+(?:\/[A-Za-z0-9_-]+)*$/.test(folder)) throw new Error(rt('path'));
  return folder;
}
export async function buildBundle(g: GraphJson, folder: string) {
  safeFolder(folder);
  let model, exp, data, embed, plan, revision;
  if (localMode) {
    const s = await snapshot();
    if (!validateLocal(g, s).valid) throw new Error(rt('invalid'));
    model = modelConfig(g, s); exp = expConfig(g);
    ({ data, embed, plan } = resources(g, s)); revision = s.revision;
  } else {
    const response = await fetch(`${apiBase}/graph/export`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(g) });
    if (!response.ok) throw new Error(`${rt('error')}: ${await response.text()}`);
    ({ model, exp, data, embed, plan, revision } = await response.json());
  }
  const path = `config/exported/${folder}`;
  const cuda = g.train.device === 'cpu' ? '-1' : String(g.train.cuda_index ?? 0);
  const modelName = plan.external_hf ? 'prepared-model.yaml' : 'model.yaml';
  const commands = `python -m studio.backend.executor.provision ${path}/prepare.json\n\npython trainer.py \\\n  --model ${path}/${modelName} \\\n  --exp ${path}/exp.yaml \\\n  --data ${path}/data.yaml \\\n  --embed ${path}/embed.yaml \\\n  --cuda ${cuda} \\\n  --seed ${g.train.seed}\n`;
  const files: Record<string, string> = {
    'model.yaml': stringify(model), 'exp.yaml': stringify(exp), 'data.yaml': stringify(data), 'embed.yaml': stringify(embed),
    'prepare.json': JSON.stringify(plan, null, 2), 'graph.json': JSON.stringify(g, null, 2),
    'README.txt': `${rt('version')}: ${revision}\n\n${rt('requirements')}\n\n${rt('folder')}: ${path}\n${rt('prepare')}\n\n${commands}`,
  };
  if (Object.values(files).slice(0, 4).some(text => text.includes('${') || text.includes('$$import'))) throw new Error(rt('imports'));
  return { files, commands, folder: path };
}
