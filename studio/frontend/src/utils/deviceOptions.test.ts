import { expect, it } from 'vitest';
import { devicePatch, deviceValue } from './deviceOptions';
it('round trips explicit GPU selection, including GPU zero', () => {
  for (const index of [0, 1, 3]) {
    expect(devicePatch(`cuda:${index}`)).toEqual({ device: 'cuda', cuda_index: index });
    expect(deviceValue({ device: 'cuda', cuda_index: index })).toBe(`cuda:${index}`);
  }
});
it('clears explicit GPU selection when switching modes', () => {
  for (const device of ['cpu', 'cuda', 'auto']) {
    expect(deviceValue(devicePatch(device))).toBe(device);
    expect(devicePatch(device).cuda_index).toBeNull();
  }
});

it('accepts non-negative integer GPU indices and rejects invalid values', async () => {
  const { parseGpuIndex } = await import('./deviceOptions');
  for (const index of ['0', '1', '3', '12']) expect(parseGpuIndex(index)).toBe(Number(index));
  for (const invalid of ['', ' ', '-1', '1.5', 'NaN', 'Infinity', '1e3', '9007199254740992']) expect(parseGpuIndex(invalid)).toBeNull();
});
