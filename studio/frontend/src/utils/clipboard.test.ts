import { afterEach, describe, expect, it, vi } from 'vitest';
import { copyText } from './clipboard';

afterEach(() => vi.unstubAllGlobals());
function fallback(success = true) {
  class Field {
    value = ''; style = {}; readOnly = false;
    focus = vi.fn(); select = vi.fn(); setSelectionRange = vi.fn();
    setAttribute = vi.fn(); remove = vi.fn();
  }
  const field = new Field(), previous = new Field();
  const execCommand = vi.fn(() => success);
  vi.stubGlobal('HTMLElement', Field);
  vi.stubGlobal('document', {
    activeElement: previous, getSelection: () => null,
    createElement: vi.fn(() => field), body: { appendChild: vi.fn() }, execCommand,
  });
  return { field, previous, execCommand };
}
describe('copyText', () => {
  const text = 'name: 测试\nconfig:\n  enabled: true\n';
  it('uses the modern clipboard when available', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    vi.stubGlobal('navigator', { clipboard: { writeText } });
    await copyText(text);
    expect(writeText).toHaveBeenCalledWith(text);
  });
  it('copies the entire multiline text on HTTP without Clipboard API', async () => {
    vi.stubGlobal('navigator', {});
    const { field, previous, execCommand } = fallback();
    await copyText(text);
    expect(field.value).toBe(text);
    expect(field.setSelectionRange).toHaveBeenCalledWith(0, text.length);
    expect(execCommand).toHaveBeenCalledWith('copy');
    expect(field.remove).toHaveBeenCalled();
    expect(previous.focus).toHaveBeenCalledWith({ preventScroll: true });
  });
  it('falls back if the modern API rejects', async () => {
    vi.stubGlobal('navigator', { clipboard: { writeText: vi.fn().mockRejectedValue(new Error('denied')) } });
    const { execCommand } = fallback();
    await copyText(text);
    expect(execCommand).toHaveBeenCalledWith('copy');
  });
  it('rejects and cleans up when copying fails', async () => {
    vi.stubGlobal('navigator', {});
    const { field } = fallback(false);
    await expect(copyText(text)).rejects.toThrow();
    expect(field.remove).toHaveBeenCalled();
  });
});
