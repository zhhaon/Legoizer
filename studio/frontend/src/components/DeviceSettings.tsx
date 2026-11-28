import { useState } from 'react';
import { tr, useLocale } from '@/i18n';
import { devicePatch, deviceValue, parseGpuIndex } from '@/utils/deviceOptions';
import type { TrainConfig } from '@/types/graph';
import type { EnvironmentReport } from '@/types/api';

type DeviceConfig = Pick<TrainConfig, 'device' | 'cuda_index'>;
export function DeviceSettings({ settings, environment, local, manual = false, onChange }: {
  settings: DeviceConfig; environment: EnvironmentReport | null; local: boolean; manual?: boolean;
  onChange: (patch: DeviceConfig) => void;
}) {
  useLocale();
  const [draft, setDraft] = useState<string | null>(null);
  const devices = environment?.cuda_devices || Array.from(
    { length: environment?.cuda_device_count || 0 }, (_, index) => ({ index, name: 'GPU' }));
  const missingDevice = settings.device === 'cuda' && settings.cuda_index != null
    && !devices.some(gpu => gpu.index === settings.cuda_index);
  const shown = draft ?? String(settings.cuda_index ?? 0);
  const valid = parseGpuIndex(shown) !== null;
  return <>
    <label className="field-label">{tr('ui.device_a5a74a')}
      <select className="field-input" value={local ? settings.device : deviceValue(settings)}
        onChange={event => {
          setDraft(null);
          onChange(local && event.target.value === 'cuda'
            ? { device: 'cuda', cuda_index: settings.cuda_index ?? (manual ? 0 : null) } : devicePatch(event.target.value));
        }}>
        <option value="cpu">CPU</option>
        <option value="cuda">{local ? manual ? tr('device.manualCuda') : 'CUDA' : tr('ui.cuda_auto__d8bb29')}</option>
        <option value="auto">{tr('ui.auto_c614ba')}</option>
        {!local && devices.map(gpu => <option key={gpu.index} value={`cuda:${gpu.index}`}>CUDA:{gpu.index}</option>)}
        {!local && missingDevice && <option value={`cuda:${settings.cuda_index}`} disabled>
          CUDA:{settings.cuda_index} {tr('ui._unavailable__b2cac2')}</option>}
      </select>
    </label>
    {local && manual && settings.device === 'cuda' && <label className="field-label">
      {tr('device.gpuIndex')}
      <input className="field-input" type="number" min={0} step={1} inputMode="numeric"
        value={shown} aria-invalid={!valid}
        onChange={event => {
          const raw = event.target.value;
          setDraft(raw);
          const index = parseGpuIndex(raw);
          if (index !== null) onChange({ device: 'cuda', cuda_index: index });
        }} onBlur={() => setDraft(null)} />
      <span className="field-help">{tr('device.gpuHint')}</span>
      {!valid && <span className="field-help text-red-500" role="alert">{tr('device.invalidIndex')}</span>}
    </label>}
    {local && settings.device === 'auto' && <p className="field-help">{tr('device.localAutoHint')}</p>}
  </>;
}
