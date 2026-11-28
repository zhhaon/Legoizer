import { renderToStaticMarkup } from 'react-dom/server';
import { afterEach, expect, it } from 'vitest';
import { DeviceSettings } from './DeviceSettings';
import { setLocale } from '@/i18n';
import type { EnvironmentReport } from '@/types/api';
afterEach(async () => { await setLocale('en'); });
it('offers manual GPU indices in the run guide in both languages', async () => {
  for (const [locale, label] of [['en', 'GPU index'], ['zh-CN', '显卡编号']]) {
    await setLocale(locale);
    const html = renderToStaticMarkup(<DeviceSettings local manual settings={{ device: 'cuda', cuda_index: 3 }} environment={null} onChange={() => {}} />);
    expect(html).toContain(label);
    expect(html).toContain('min="0"');
    expect(html).toContain('value="3"');
    expect(html).not.toContain('value="cuda:3"');
  }
});
it('preserves backend GPU options, selection and unavailable-device behavior', () => {
  const environment = { cuda_devices: [{ index: 0, name: 'A' }, { index: 2, name: 'B' }], cuda_device_count: 2 } as EnvironmentReport;
  const html = renderToStaticMarkup(<DeviceSettings local={false} settings={{ device: 'cuda', cuda_index: 2 }} environment={environment} onChange={() => {}} />);
  expect(html).toContain('value="cuda:0"');
  expect(html).toContain('value="cuda:2" selected');
  expect(html).not.toContain('type="number"');
  const missing = renderToStaticMarkup(<DeviceSettings local={false} settings={{ device: 'cuda', cuda_index: 4 }} environment={environment} onChange={() => {}} />);
  expect(missing).toContain('value="cuda:4" disabled');
});
it('hides the manual GPU field for CPU and Auto', () => {
  for (const device of ['cpu', 'auto'] as const) {
    const html = renderToStaticMarkup(<DeviceSettings local settings={{ device, cuda_index: null }} environment={null} onChange={() => {}} />);
    expect(html).not.toContain('type="number"');
  }
});

it('keeps the original local device selector free of manual GPU inputs', async () => {
  for (const locale of ['en', 'zh-CN']) {
    await setLocale(locale);
    const html = renderToStaticMarkup(<DeviceSettings local settings={{ device: 'cuda', cuda_index: 3 }} environment={null} onChange={() => {}} />);
    expect(html).not.toContain('type="number"');
    expect(html).not.toContain('value="cuda:3"');
    expect(html).toContain('value="cuda" selected');
    expect(html).toContain('>CUDA</option>');
  }
});
