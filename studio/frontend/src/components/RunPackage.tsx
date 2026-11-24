import { copyText } from '@/utils/clipboard';
import { useState } from 'react';
import { strToU8, zipSync } from 'fflate';
import { useStudio } from '@/store/graphStore';
import { downloadText } from '@/api/client';
import { buildBundle } from '@/runtime/bundle';
import { localMode } from '@/runtime/mode';
import { rt } from '@/runtime/messages';
import { useLocale } from '@/i18n';

export function RunPackage() {
  const locale = useLocale();
  const graph = useStudio(s => s.graph);
  const [folder, setFolder] = useState('my_model');
  const [selected, select] = useState('Commands');
  const [bundle, setBundle] = useState<Awaited<ReturnType<typeof buildBundle>> | null>(null);
  const [source, setSource] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const identity = JSON.stringify([graph, folder, locale]);
  const current = source === identity ? bundle : null;
  const prepare = async () => {
    setBusy(true); setError('');
    try { const result = await buildBundle(graph, folder); setBundle(result); setSource(identity); }
    catch (e) { setError(String(e)); }
    finally { setBusy(false); }
  };
  const content = selected === 'Commands' ? current?.commands : current?.files[selected];
  return <section className="min-w-0 shrink-0 border-b border-line p-2 text-xs">
    <div className="flex flex-wrap items-center gap-2">
      <span>{localMode ? rt('local') : rt('server')}</span>
      <label>{rt('folder')} <input className="field-input !w-40" value={folder} onChange={e => setFolder(e.target.value)} /></label>
      <button className="tool-btn outline" disabled={busy} onClick={() => void prepare()}>{rt('export')}</button>
      {current && <>
        <select className="field-input !w-40" value={selected} onChange={e => select(e.target.value)}>
          <option value="Commands">{rt('command')}</option>{Object.keys(current.files).map(name => <option key={name}>{name}</option>)}
        </select>
        <button className="tool-btn outline" onClick={() => { setError(''); void copyText(content || '').then(() => useStudio.getState().toast('success', rt('copied'))).catch(e => setError(String(e))); }}>{rt('copy')}</button>
        <button className="tool-btn outline" onClick={() => downloadText(selected === 'Commands' ? 'commands.txt' : selected, content || '')}>{rt('download')}</button>
        <button className="tool-btn outline" onClick={() => {
          const zipped = zipSync(Object.fromEntries(Object.entries(current.files).map(([name, text]) => [`${current.folder}/${name}`, strToU8(text)])));
          const url = URL.createObjectURL(new Blob([zipped as BlobPart], { type: 'application/zip' }));
          const a = document.createElement('a'); a.href = url; a.download = 'legoizer-export.zip'; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
        }}>{rt('zip')}</button>
      </>}
    </div>
    <p className="mt-1 text-muted">{rt('requirements')}</p>
    {error && <p role="alert" className="text-red-500">{error}</p>}
    {current && <pre data-testid="run-package-code" tabIndex={0} className="mt-2 max-h-40 overflow-auto whitespace-pre-wrap break-words pb-3">{content}</pre>}
  </section>;
}
