import { rt } from '@/runtime/messages';

/** Copy in HTTPS and ordinary HTTP deployments. Never report a failed copy as success. */
export async function copyText(text: string): Promise<void> {
  if (typeof navigator !== 'undefined' && navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(text);
      return;
    } catch { /* Denied clipboard permission: try the user-gesture fallback. */ }
  }
  const active = document.activeElement instanceof HTMLElement ? document.activeElement : null;
  const selection = document.getSelection();
  const ranges = selection ? Array.from({ length: selection.rangeCount }, (_, i) => selection.getRangeAt(i).cloneRange()) : [];
  const field = document.createElement('textarea');
  field.value = text;
  field.readOnly = true;
  field.setAttribute('aria-hidden', 'true');
  Object.assign(field.style, { position: 'fixed', left: '0', top: '0', width: '1px', height: '1px', opacity: '0', fontSize: '16px' });
  document.body.appendChild(field);
  try {
    field.focus({ preventScroll: true });
    field.select();
    field.setSelectionRange(0, text.length);
    if (!document.execCommand('copy')) throw new Error(rt('copyFailed'));
  } catch {
    throw new Error(rt('copyFailed'));
  } finally {
    field.remove();
    active?.focus({ preventScroll: true });
    if (selection) {
      selection.removeAllRanges();
      for (const range of ranges) selection.addRange(range);
    }
  }
}
