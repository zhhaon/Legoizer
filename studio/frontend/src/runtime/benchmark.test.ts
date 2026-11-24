import { readFileSync } from 'node:fs';
import { afterEach, describe, expect, it, vi } from 'vitest';
const data = JSON.parse(readFileSync('public/benchmark.json', 'utf8'));
afterEach(() => { vi.unstubAllEnvs(); vi.unstubAllGlobals(); vi.resetModules(); });
describe('static Benchmark', () => {
  it('ships every real configuration detail and no capture timestamp', () => {
    expect(data.index.total).toBe(828);
    expect(data.index.evaluations).toHaveLength(data.index.total);
    expect(data.index.snapshot_date).toBe('2026-11-16');
    expect(data.index.fetched_at).toBeUndefined();
    expect(data.index.captured_at).toBeUndefined();
    for (const row of data.index.evaluations) {
      const detail = data.details[row.signature].evaluation;
      expect(detail.signature).toBe(row.signature);
      expect(detail.configuration).toBeTruthy();
      expect(detail.experiments).toEqual(row.experiments);
    }
  });
  it('loads index and details through one static request in local mode', async () => {
    vi.stubEnv('VITE_STUDIO_MODE', 'local');
    vi.stubEnv('BASE_URL', '/Legoizer/');
    const fetcher = vi.fn(async (url: string) => {
      expect(url).toBe('/Legoizer/benchmark.json');
      return { ok: true, json: async () => data };
    });
    vi.stubGlobal('fetch', fetcher);
    const { api } = await import('@/api/client');
    const index = await api.benchmark();
    const detail = await api.benchmarkDetail(index.evaluations[0].signature);
    expect(detail.evaluation.configuration).toBeTruthy();
    await expect(api.benchmarkDetail('missing')).rejects.toThrow();
    expect(fetcher).toHaveBeenCalledTimes(1);
  });
  it('allows retry after a failed static request', async () => {
    const fetcher = vi.fn().mockResolvedValueOnce({ ok: false, status: 503 })
      .mockResolvedValueOnce({ ok: true, json: async () => data });
    vi.stubGlobal('fetch', fetcher);
    const { staticBenchmark } = await import('./benchmark');
    await expect(staticBenchmark()).rejects.toThrow('503');
    expect((await staticBenchmark()).total).toBe(data.index.total);
  });
});
