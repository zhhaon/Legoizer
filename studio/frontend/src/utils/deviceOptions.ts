import type { TrainConfig } from '@/types/graph';
export function deviceValue(settings: Pick<TrainConfig, 'device' | 'cuda_index'>): string {
  return settings.device === 'cuda' && settings.cuda_index != null
    ? `cuda:${settings.cuda_index}` : settings.device;
}
export function devicePatch(value: string): Pick<TrainConfig, 'device' | 'cuda_index'> {
  if (value.startsWith('cuda:')) return { device: 'cuda', cuda_index: Number(value.slice(5)) };
  return { device: value as TrainConfig['device'], cuda_index: null };
}

/** A remote GPU index is configuration, not a claim about browser hardware. */
export function parseGpuIndex(value: string): number | null {
  if (!/^\d+$/.test(value)) return null;
  const index = Number(value);
  return Number.isSafeInteger(index) && index >= 0 ? index : null;
}
