import { describe, it, expect, vi, afterEach } from 'vitest';
import { readFileSync } from 'node:fs';
const catalog = JSON.parse(readFileSync('public/catalog.json', 'utf8'));
afterEach(() => { vi.unstubAllEnvs(); vi.unstubAllGlobals(); vi.resetModules(); });
describe('explicit local mode', () => {
  it('initializes and exports without any backend request', async () => {
    vi.stubEnv('VITE_STUDIO_MODE', 'local');
    const fetcher = vi.fn(async (url: string) => { expect(url).not.toContain('/api'); return { ok: true, json: async () => catalog }; });
    vi.stubGlobal('fetch', fetcher);
    const { api } = await import('@/api/client');
    expect((await api.components()).components.length).toBeGreaterThan(50);
    expect(await api.environment()).toBeNull();
    expect((await api.runs()).runs).toEqual([]);
    const { buildBundle } = await import('./bundle');
    const g = catalog.templates.find((t: any) => t.id === 'naml').graph;
    const bundle = await buildBundle(g, 'test_run');
    expect(Object.keys(bundle.files)).toHaveLength(7);
    expect(bundle.commands).toContain('studio.backend.executor.provision');
    expect(bundle.commands).toContain('--seed 2023');
    await expect(api.train(g)).rejects.toThrow();
    expect(fetcher).toHaveBeenCalledTimes(1);
  });
  it('switches new messages between Chinese and English', async () => {
    const { rt } = await import('./messages');
    const i18n = (await import('@/i18n')).default;
    await i18n.changeLanguage('zh-CN'); expect(rt('zip')).toBe('下载 ZIP');
    await i18n.changeLanguage('en'); expect(rt('zip')).toBe('Download ZIP');
  });
});
