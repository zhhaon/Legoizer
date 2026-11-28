import { renderToStaticMarkup } from 'react-dom/server';
import { afterEach, expect, it, vi } from 'vitest';
import { setLocale } from '@/i18n';
import type { IssueSeverity, ValidationReport } from '@/types/api';
import { ConsolePanel } from './ConsolePanel';

const state = vi.hoisted(() => ({
  validation: null as ValidationReport | null,
  consoleLines: [] as string[], clearConsole: () => {}, focusNode: () => {},
}));
vi.mock('@/store/graphStore', () => ({ useStudio: (selector: (value: typeof state) => unknown) => selector(state) }));
afterEach(async () => { state.validation = null; await setLocale('en'); });

function report(severity: IssueSeverity, fix = 'Check the model') : ValidationReport {
  const issue = { severity, code: 'example_check', message: 'Check details', reason: 'Context', fix, node_id: 'node-1', edge_id: null };
  return { valid: severity !== 'error', errors: severity === 'error' ? [issue] : [],
    warnings: severity === 'warning' ? [issue] : [], runtime_checks: severity === 'info' ? [issue] : [],
    summary: { nodes: 1, edges: 0, task: 'matching', model_label: 'Example', exact_template: null,
      dataset: null, item_encoder: null, user_encoder: null, predictor: null } };
}

it.each([
  ['en', ['Error', 'Warning', 'Information'], ['Fix:', 'Suggested action:', 'How to verify:']],
  ['zh-CN', ['错误', '警告', '提示'], ['修复建议：', '建议操作：', '验证方式：']],
] as const)('distinguishes corrections, suggested actions and verification in %s', async (locale, levels, actions) => {
  await setLocale(locale);
  for (const [index, severity] of (['error', 'warning', 'info'] as const).entries()) {
    state.validation = report(severity);
    const html = renderToStaticMarkup(<ConsolePanel />);
    expect(html).toContain(`<b>${levels[index]}</b>`);
    expect(html).toContain(actions[index]);
    for (const other of actions.filter((_, i) => i !== index)) expect(html).not.toContain(other);
    expect(html).toContain('Check the model');
    expect(html).toContain('node-1');
  }
});

it('omits the action heading when an informational check has no instructions', async () => {
  for (const [locale, heading] of [['en', 'How to verify:'], ['zh-CN', '验证方式：']]) {
    await setLocale(locale);
    state.validation = report('info', '');
    expect(renderToStaticMarkup(<ConsolePanel />)).not.toContain(heading);
  }
});
