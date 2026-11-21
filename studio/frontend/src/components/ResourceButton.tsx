import { tr, useLocale } from '@/i18n';
import { createPortal } from 'react-dom';
import { useEffect, useRef, useState } from 'react';
import { Check, Download, Loader2, Settings, RotateCcw, Trash2 } from 'lucide-react';
import { useTrainingCredentials } from './TrainingCredentials';

type Status = { state: string; message?: string };
export function ResourceButton({ component, params = {} }: { component: string; params?: Record<string, unknown> }) {
  useLocale();
  const [status, setStatus] = useState<Status>({ state: 'checking' });
  const [confirmDelete, setConfirmDelete] = useState(false);
  const { token } = useTrainingCredentials();
  const identity = JSON.stringify([component, params]);
  useEffect(() => {
    let active = true;
    const check = async () => {
      try {
        const response = await fetch('/api/resources/status', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify([{ component, params }]) });
        if (!response.ok) throw new Error(tr("ui.resource_check_failed_094244"));
        const body = await response.json();
        if (active) setStatus(body.items[0]);
      } catch { if (active) setStatus({ state: 'error', message: tr("ui.resource_check_unavailable_fb7c54") }); }
    };
    setStatus({ state: 'checking' });
    void check();
    window.addEventListener('studio-resources-changed', check);
    const timer = window.setInterval(check, 5000);
    return () => { active = false; window.clearInterval(timer); window.removeEventListener('studio-resources-changed', check); };
  }, [identity]);
  if (status.state === 'unnecessary') return null;
  const busy = ['checking', 'downloading'].includes(status.state);
  const ready = status.state === 'ready';
  const configure = status.state === 'configure';
  return <span className="flex shrink-0 items-center"> <button draggable={false} className={`shrink-0 rounded p-1 ${ready ? 'text-green-500' : 'text-muted hover:text-ink'}`}
    title={status.message || (ready ? tr("ui.downloaded_available_locally_caedd1") : busy ? tr("ui.preparing_resources__899f24") : tr("ui.download_resources_4c0710"))}
    aria-label={ready ? tr('actions.downloaded') : busy ? tr("ui.preparing_resources_320982") : configure ? tr("ui.configure_model_first_d1d276") : tr("ui.download_resources_4c0710")}
    disabled={busy || ready || configure || status.state === 'error'}
    onDragStart={e => e.stopPropagation()} onClick={async e => {
      e.stopPropagation();
      setStatus({ state: 'downloading' });
      try {
        const response = await fetch('/api/resources/download', { method: 'POST', headers: { 'Content-Type': 'application/json', ...(token ? { 'X-Hugging-Face-Token': token } : {}) }, body: JSON.stringify({ component, params }) });
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail || 'Download failed');
        setStatus(result);
      } catch (error) { setStatus({ state: 'failed', message: String(error) }); }
    }}>
    {ready ? <Check size={15} /> : busy ? <Loader2 size={14} className="animate-spin" /> : configure ? <Settings size={14} /> : status.state === 'failed' ? <RotateCcw size={14} /> : <Download size={14} />}
  </button>
    {ready && <button draggable={false} className="shrink-0 rounded p-1 text-muted hover:text-red-500"
      title={tr("ui.delete_local_resources_5d56bb")} aria-label={tr("ui.delete_local_resources_5d56bb")}
      onDragStart={e => e.stopPropagation()} onClick={e => {
        e.stopPropagation();
        setConfirmDelete(true);
      }}><Trash2 size={14} /></button>}
    {confirmDelete && <DeleteResourceDialog onCancel={() => setConfirmDelete(false)} onDelete={async () => {
        setConfirmDelete(false);
        setStatus({ state: 'checking' });
        try {
          const response = await fetch('/api/resources/delete', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ component, params }) });
          const result = await response.json();
          if (!response.ok) throw new Error(result.detail || 'Delete failed');
          setStatus(result);
          window.dispatchEvent(new Event('studio-resources-changed'));
        } catch (error) { setStatus({ state: 'ready', message: String(error) }); }
      }} />}
  </span>;
}


export function DeleteResourceDialog({ onCancel, onDelete, title = tr("ui.delete_local_resources__ca91fd"), description = tr("ui.these_local_resources_will_be_deleted_components_tha_40551d") }: { onCancel: () => void; onDelete: () => void; title?: string; description?: string }) {
  useLocale();
  const cancel = useRef<HTMLButtonElement>(null);
  const remove = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    cancel.current?.focus();
    return () => previous?.focus();
  }, []);
  return createPortal(
    <div className="modal-backdrop fixed inset-0 z-[100] grid place-items-center p-6"
      onClick={event => { event.stopPropagation(); if (event.target === event.currentTarget) onCancel(); }}
      onKeyDown={event => {
        event.stopPropagation();
        if (event.key === 'Escape') { event.preventDefault(); onCancel(); }
        if (event.key === 'Tab') {
          event.preventDefault();
          (document.activeElement === cancel.current ? remove.current : cancel.current)?.focus();
        }
      }}>
      <div role="alertdialog" aria-modal="true" aria-labelledby="resource-delete-title" aria-describedby="resource-delete-description"
        className="w-full max-w-[420px] rounded-xl border border-line bg-surface shadow-2xl">
        <div className="flex items-center gap-3 border-b border-line p-4">
          <div className="rounded-lg bg-red-500/10 p-2 text-red-500"><Trash2 size={18} /></div>
          <h3 id="resource-delete-title" className="text-sm font-bold">{title}</h3>
        </div>
        <p id="resource-delete-description" className="p-4 text-[11px] leading-relaxed text-muted">
          {description}
        </p>
        <div className="flex justify-end gap-2 border-t border-line p-3">
          <button ref={cancel} className="tool-btn outline" onClick={onCancel}>{tr("ui.cancel_77dfd2")}</button>
          <button ref={remove} className="tool-btn !border-red-600 !bg-red-600 !text-white hover:!bg-red-700" onClick={onDelete}>{tr("ui.delete_f6fdbe")}</button>
        </div>
      </div>
    </div>, document.body,
  );
}
