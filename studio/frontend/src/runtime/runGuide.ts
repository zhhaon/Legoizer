import type { GraphJson } from '@/types/graph';
import type { buildBundle } from './bundle';
import { tr } from '@/i18n';
export interface RunOptions { project: string; archive: string; firstInstall: boolean }
export const shellQuote = (value: string) => "'" + value.replace(/'/g, "'\\''") + "'";
export function serverPath(value: string): boolean {
  return value.startsWith('/') && value.trim() === value && !/[\u0000-\u001f\u007f]/.test(value)
    && value.split('/').filter(Boolean).length > 0 && !value.split('/').some(p => p === '.' || p === '..');
}
export function makeRunGuide(bundle: Awaited<ReturnType<typeof buildBundle>>, graph: GraphJson, options: RunOptions) {
  if (!serverPath(options.project) || !serverPath(options.archive)) throw new Error(tr('guide.pathError'));
  const project = options.project.replace(/\/+$/, '');
  if (options.firstInstall && options.archive.startsWith(project + '/')) throw new Error(tr('guide.archiveOutside'));
  const cd = `cd ${shellQuote(project)}`, py = '.venv/bin/python';
  const cache = ['export HF_HOME="$PWD/studio/backend/workspace/resources/huggingface"',
    'export HF_HUB_CACHE="$HF_HOME/hub"', 'export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"',
    'export TRANSFORMERS_CACHE="$HF_HUB_CACHE"'];
  const chain = (commands: string[]) => commands.join(' && \\\n');
  const install = options.firstInstall ? chain([
    `git clone https://github.com/zhhaon/Legoizer.git ${shellQuote(project)}`, cd,
    `bash studio/setup.sh --torch ${graph.train.device === 'cpu' ? 'cpu' : 'cu128'}`,
  ]) : chain([cd, `test -x ${py}`]);
  const prepare = chain([cd, `${py} -m zipfile -e ${shellQuote(options.archive)} .`, ...cache,
    `${py} -m studio.backend.executor.provision ${shellQuote(bundle.folder + '/prepare.json')}`]);
  const plan = JSON.parse(bundle.files['prepare.json']);
  const model = plan.external_hf ? 'prepared-model.yaml' : 'model.yaml';
  const cuda = graph.train.device === 'cpu' ? -1 : graph.train.cuda_index ?? 0;
  const train = chain([cd, ...cache,
    `${py} trainer.py \\\n  --model ${shellQuote(bundle.folder + '/' + model)} \\\n  --exp ${shellQuote(bundle.folder + '/exp.yaml')} \\\n  --data ${shellQuote(bundle.folder + '/data.yaml')} \\\n  --embed ${shellQuote(bundle.folder + '/embed.yaml')} \\\n  --cuda ${cuda} \\\n  --seed ${graph.train.seed}`]);
  const steps = [null, install, prepare, train];
  const commands = chain([install, prepare, train]) + '\n';
  const guide = `# ${tr('guide.title')}\n\n${tr('guide.offlineIntro')}\n\n${tr('guide.prerequisites')}\n\n` +
    `1. ${tr('guide.step1')}\n\n${tr('guide.downloadHelp')}\n\n` +
    [1, 2, 3].map(i => `## ${i + 1}. ${tr('guide.step' + (i + 1))}\n\n${tr('guide.server')}\n\n\`\`\`bash\n${steps[i]}\n\`\`\`\n`).join('\n') +
    `\n${tr('guide.failure')}\n\n${tr('guide.scriptHelp')}\n\n${tr('guide.resources')}\n`;
  const script = '#!/usr/bin/env bash\nset -euo pipefail\n\n' +
    [1, 2, 3].map(i => `# ${tr('guide.step' + (i + 1))}\n${steps[i]}`).join('\n\n') + '\n';
  return { steps, commands, files: { ...bundle.files, 'RUN_GUIDE.md': guide, 'commands.sh': script,
    'commands.txt': commands, 'README.txt': bundle.files['README.txt'].split('\n\n').slice(0, 3).join('\n\n') + '\n\n' + guide } as Record<string, string> };
}
