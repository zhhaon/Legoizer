import { afterEach, expect, it } from 'vitest';
import { execFileSync, spawnSync } from 'node:child_process';
import { makeRunGuide, serverPath, shellQuote } from './runGuide';
import { emptyGraph } from '@/types/graph';
import { setLocale } from '@/i18n';
const base = { folder: 'config/exported/my_model', commands: '', files: {
  'prepare.json': '{"external_hf":false}', 'model.yaml': 'name: test\n',
  'exp.yaml': '', 'data.yaml': '', 'embed.yaml': '', 'graph.json': '{}', 'README.txt': 'revision\n\nrequirements\n\nfolder\n\nold commands',
} };
const options = { project: "/srv/my project's Legoizer", archive: '/tmp/my export.zip', firstInstall: false };
afterEach(async () => { await setLocale('en'); });
it('generates parseable shell commands with safely quoted paths and correct device', () => {
  const graph = emptyGraph(); graph.train.device = 'cuda'; graph.train.cuda_index = 3; graph.train.seed = 42;
  const result = makeRunGuide(base, graph, options);
  for (const text of [result.commands, result.files['commands.sh'], ...result.steps.filter(Boolean)]) {
    expect(spawnSync('bash', ['-n'], { input: text! }).status).toBe(0);
  }
  expect(result.commands).toContain('--cuda 3'); expect(result.commands).toContain('--seed 42');
  expect(result.steps[1]).not.toContain('git clone');
  expect(result.commands).toContain('&&'); expect(result.steps[2]).toContain('HF_HOME');
  expect(result.steps[2]).toContain(shellQuote(options.archive));
  expect(Object.keys(result.files)).toContain('RUN_GUIDE.md');
  expect(Object.keys(result.files)).toContain('commands.sh');
  expect(result.files['RUN_GUIDE.md']).not.toContain('old commands');
  expect(result.files['README.txt']).not.toContain('old commands');
});
it('supports first install, CPU and External HF prepared filenames', () => {
  const result = makeRunGuide({ ...base, files: { ...base.files, 'prepare.json': '{"external_hf":true}' } }, emptyGraph(), { ...options, firstInstall: true });
  expect(result.steps[1]).toContain('git clone'); expect(result.steps[1]).toContain('--torch cpu');
  expect(result.steps[3]).toContain('prepared-model.yaml'); expect(result.steps[3]).toContain('--cuda -1');
});
it('rejects missing paths, traversal and invalid first-install ZIP placement', () => {
  for (const path of ['', '/', '~/project', 'relative', '/srv/../x', '/tmp/x\ncmd', '/tmp/x\0']) expect(serverPath(path)).toBe(false);
  expect(() => makeRunGuide(base, emptyGraph(), { ...options, project: '' })).toThrow();
  expect(() => makeRunGuide(base, emptyGraph(), { project: '/srv/project', archive: '/srv/project/export.zip', firstInstall: true })).toThrow();
  const evil = "/srv/project'; printf unsafe; '$(echo bad)";
  expect(execFileSync('bash', ['-c', `printf '%s' ${shellQuote(evil)}`], { encoding: 'utf8' })).toBe(evil);
});
it('stops a failed full flow before preparation or training', () => {
  const result = makeRunGuide(base, emptyGraph(), { ...options, project: '/path/that/does/not/exist' });
  const output = spawnSync('bash', ['-c', result.commands + "\nprintf 'shell alive'"], { encoding: 'utf8' });
  expect(output.stderr).toContain('No such file or directory');
  expect(output.stderr).not.toContain('No module named');
  expect(result.files['commands.sh']).toContain('set -euo pipefail');
});
it('localizes offline guides into Chinese and English', async () => {
  await setLocale('zh-CN'); expect(makeRunGuide(base, emptyGraph(), options).files['RUN_GUIDE.md']).toContain('在服务器运行');
  await setLocale('en'); expect(makeRunGuide(base, emptyGraph(), options).files['RUN_GUIDE.md']).toContain('Run on your server');
});
