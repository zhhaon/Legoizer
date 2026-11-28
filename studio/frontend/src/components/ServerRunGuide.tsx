import { useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { zipSync, strToU8 } from 'fflate';
import { useStudio } from '@/store/graphStore';
import { useLocale, tr } from '@/i18n';
import { rt } from '@/runtime/messages';
import { copyText } from '@/utils/clipboard';
import { downloadText } from '@/api/client';
import { buildBundle } from '@/runtime/bundle';
import { makeRunGuide, serverPath } from '@/runtime/runGuide';
import { DeviceSettings } from './DeviceSettings';

export function ServerRunGuide() {
  const open = useStudio(s => s.runGuideOpen);
  return open ? <GuideDialog /> : null;
}
function GuideDialog() {
  const locale = useLocale();
  const graph = useStudio(s => s.graph);
  const [step, setStep] = useState(0), [confirmed, confirm] = useState(false);
  const [project, setProject] = useState('/home/you/projects/Legoizer'), [archive, setArchive] = useState('/tmp/legoizer-export.zip');
  const [folder, setFolder] = useState('my_model'), [firstInstall, setFirstInstall] = useState(true);
  const [selected, select] = useState('model.yaml');
  const [base, setBase] = useState<Awaited<ReturnType<typeof buildBundle>> | null>(null);
  const [error, setError] = useState(''), [copyError, setCopyError] = useState('');
  const closeButton = useRef<HTMLButtonElement>(null);
  const identity = JSON.stringify([graph, folder, locale]);
  const [source, setSource] = useState('');
  const close = () => useStudio.setState({ runGuideOpen: false });
  useEffect(() => {
    const active = document.activeElement as HTMLElement | null;
    closeButton.current?.focus();
    return () => active?.focus();
  }, []);
  useEffect(() => {
    let active = true;
    setError(''); setBase(null);
    void buildBundle(graph, folder).then(result => { if (active) { setBase(result); setSource(identity); } })
      .catch(e => { if (active) setError(String(e)); });
    return () => { active = false; };
  }, [identity]);
  useEffect(() => { confirm(false); setCopyError(''); }, [identity, project, archive, firstInstall, step]);
  const result = useMemo(() => {
    if (!base || source !== identity || !serverPath(project) || !serverPath(archive)) return null;
    try { return { guide: makeRunGuide(base, graph, { project, archive, firstInstall }), error: '' }; }
    catch (e) { return { guide: null, error: String(e) }; }
  }, [base, source, identity, project, archive, firstInstall]);
  const guide = result?.guide;
  const pathIssue = (project !== '' && !serverPath(project)) || (archive !== '' && !serverPath(archive));
  const copy = async (value: string) => {
    setCopyError('');
    try { await copyText(value); useStudio.getState().toast('success', rt('copied')); }
    catch (e) { setCopyError(String(e)); }
  };
  const download = () => {
    if (!guide || !base) return;
    const bytes = zipSync(Object.fromEntries(Object.entries(guide.files).map(([name, text]) => [`${base.folder}/${name}`, strToU8(text)])));
    const url = URL.createObjectURL(new Blob([bytes as BlobPart], { type: 'application/zip' }));
    const link = document.createElement('a'); link.href = url; link.download = 'legoizer-export.zip'; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  return createPortal(<div className="modal-backdrop fixed inset-0 z-[100] grid place-items-center p-3" onClick={e => { if (e.target === e.currentTarget) close(); }}>
    <section role="dialog" aria-modal="true" aria-labelledby="run-guide-title" className="flex max-h-[92vh] w-full max-w-4xl flex-col rounded-xl border border-line bg-surface shadow-xl"
      onKeyDown={e => {
        e.stopPropagation();
        if (e.key === 'Escape') close();
        if (e.key === 'Tab') {
          const nodes = [...e.currentTarget.querySelectorAll<HTMLElement>('button:not(:disabled), input:not(:disabled), select, textarea, [tabindex="0"]')];
          const first = nodes[0], last = nodes.at(-1);
          if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last?.focus(); }
          if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first?.focus(); }
        }
      }}>
      <header className="flex shrink-0 items-center justify-between border-b border-line p-3">
        <h2 id="run-guide-title" className="text-lg font-bold">{tr('guide.title')}</h2>
        <button ref={closeButton} className="tool-btn outline" onClick={close}>{tr('guide.close')}</button>
      </header>
      <div className="min-h-0 flex-1 overflow-y-auto p-4 text-sm space-y-4">
        <ol className="grid grid-cols-2 gap-2 md:grid-cols-4">{[0, 1, 2, 3].map(i => <li key={i} aria-current={step === i ? 'step' : undefined} className={`rounded border p-2 ${step === i ? 'border-blue-500 text-blue-500' : 'border-line text-muted'}`}>{i + 1}. {tr('guide.step' + (i + 1))}</li>)}</ol>
        <details open className="rounded border border-line p-3">
          <summary className="cursor-pointer font-semibold">{tr('guide.settings')}</summary>
          <div className="mt-3 grid gap-3 md:grid-cols-2">
            <label className="field-label">{tr('guide.project')}<input className="field-input" placeholder="/home/you/projects/Legoizer" value={project} onChange={e => setProject(e.target.value)} /></label>
            <label className="field-label">{tr('guide.archive')}<input className="field-input" placeholder="/tmp/legoizer-export.zip" value={archive} onChange={e => setArchive(e.target.value)} /></label>
            <label className="field-label">{tr('guide.folder')}<input className="field-input" value={folder} onChange={e => setFolder(e.target.value)} /></label>
            <label className="field-label">{tr('guide.step2')}<select className="field-input" value={firstInstall ? 'first' : 'existing'} onChange={e => setFirstInstall(e.target.value === 'first')}><option value="first">{tr('guide.first')}</option><option value="existing">{tr('guide.existing')}</option></select></label>
            <DeviceSettings settings={graph.train} environment={null} local manual onChange={patch => useStudio.getState().updateTrain(patch)} />
          </div>
          <p className="mt-2 text-muted">{tr('guide.firstHelp')}</p>
        </details>
        {!guide && !error && <p role="status">{base ? tr('guide.pathsNeeded') : tr('guide.loading')}</p>}
        {error && <p role="alert" className="text-red-500">{error}</p>}
        {pathIssue && <p role="alert" className="text-red-500">{tr('guide.pathError')}</p>}
        {result?.error && <p role="alert" className="text-red-500">{result.error}</p>}
        {copyError && <p role="alert" className="text-red-500">{copyError}</p>}
        <article className="space-y-3">
          <h3 className="font-bold">{step + 1}. {tr('guide.step' + (step + 1))}</h3>
          <p className="font-semibold text-blue-500">{tr(step === 0 ? 'guide.browser' : 'guide.server')}</p>
          {step === 0 ? <>
            <p>{tr('guide.downloadHelp')}</p><p className="text-muted">{tr('guide.downloadLocation')}</p>
            <button className="tool-btn primary" disabled={!guide} onClick={download}>{tr('guide.zip')}</button>
            {base && <code className="block break-all">{base.folder}/</code>}
          </> : <>
            <p>{tr(step === 1 ? 'guide.prerequisites' : step === 2 ? 'guide.prepareHelp' : 'guide.trainHelp')}</p>
            <button className="tool-btn outline" disabled={!guide} onClick={() => guide && void copy(guide.steps[step] || '')}>{tr('guide.copyStep')}</button>
            {guide && <pre tabIndex={0} data-testid="guide-step-code" className="max-h-64 overflow-auto whitespace-pre-wrap break-words rounded border border-line p-3">{guide.steps[step]}</pre>}
            {step >= 2 && <p className="text-muted">{tr('guide.failure')}</p>}
          </>}
        </article>
        {guide && <details className="rounded border border-line p-3">
          <summary className="cursor-pointer">{tr('guide.files')}</summary>
          <div className="my-2 flex flex-wrap gap-2">
            <select aria-label={tr('guide.files')} className="field-input !w-auto" value={selected} onChange={e => select(e.target.value)}>{Object.keys(guide.files).map(name => <option key={name}>{name}</option>)}</select>
            <button className="tool-btn outline" onClick={() => void copy(guide.files[selected])}>{tr('guide.copyFile')}</button>
            <button className="tool-btn outline" onClick={() => downloadText(selected, guide.files[selected])}>{tr('guide.downloadFile')}</button>
          </div>
          <pre tabIndex={0} className="max-h-64 overflow-auto whitespace-pre-wrap break-words">{guide.files[selected]}</pre>
        </details>}
        <p className="text-muted">{tr('guide.resources')}</p>
        <p className="text-muted">{tr('guide.manual')}</p>
        {step < 3 && <label className="flex items-center gap-2"><input type="checkbox" checked={confirmed} disabled={!guide} onChange={e => confirm(e.target.checked)} />{tr('guide.confirm')}</label>}
      </div>
      <footer className="flex shrink-0 flex-wrap justify-between gap-2 border-t border-line p-3">
        <button className="tool-btn outline" disabled={step === 0} onClick={() => setStep(step - 1)}>{tr('guide.back')}</button>
        <button className="tool-btn outline" disabled={!guide} onClick={() => guide && void copy(guide.commands)}>{tr('guide.copyAll')}</button>
        <button className="tool-btn outline" disabled={!guide} onClick={download}>{tr('guide.zip')}</button>
        {step < 3 ? <button className="tool-btn primary" disabled={!guide || !confirmed} onClick={() => setStep(step + 1)}>{tr('guide.next')}</button>
          : <button className="tool-btn primary" onClick={close}>{tr('guide.close')}</button>}
      </footer>
    </section>
  </div>, document.body);
}
