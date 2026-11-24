import { useEffect, useId, useLayoutEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { ArrowLeft, ArrowRight, Check, X } from 'lucide-react';
import { tr, useLocale } from '@/i18n';
import { LanguageSelect } from './LanguageSelect';

type Area = 'canvas' | 'library' | 'properties' | 'toolbar' | 'benchmark' | 'results';
const steps: Area[] = ['canvas', 'library', 'properties', 'toolbar', 'benchmark', 'results'];
type Rect = { x: number; y: number; width: number; height: number };
type Placement = { target: Rect | null; x: number; y: number; width: number; height: number };
const clamp = (value: number, min: number, max: number) => Math.min(Math.max(value, min), Math.max(min, max));

/** Keep the explanation close to its target while avoiding the highlighted content. */
function place(target: Rect | null, card: { width: number; height: number }, width: number, height: number): Placement {
  const margin = 12, gap = 16;
  const centered = { x: (width - card.width) / 2, y: (height - card.height) / 2 };
  const candidates = target ? [
    { x: target.x + target.width + gap, y: target.y + (target.height - card.height) / 2 },
    { x: target.x - card.width - gap, y: target.y + (target.height - card.height) / 2 },
    { x: target.x + (target.width - card.width) / 2, y: target.y + target.height + gap },
    { x: target.x + (target.width - card.width) / 2, y: target.y - card.height - gap },
  ] : [centered];
  const positions = candidates.map(p => ({ x: clamp(p.x, margin, width - card.width - margin), y: clamp(p.y, margin, height - card.height - margin) }));
  const overlap = (p: { x: number; y: number }) => !target ? 0 :
    Math.max(0, Math.min(p.x + card.width, target.x + target.width) - Math.max(p.x, target.x)) *
    Math.max(0, Math.min(p.y + card.height, target.y + target.height) - Math.max(p.y, target.y));
  const best = positions.reduce((a, b) => overlap(b) < overlap(a) ? b : a);
  return { target, ...best, width, height };
}

export function StudioTour({ onClose }: { onClose: () => void }) {
  const locale = useLocale();
  const [step, setStep] = useState(0);
  const area = steps[step];
  const [placement, setPlacement] = useState<Placement | null>(null);
  const card = useRef<HTMLDivElement>(null);
  const heading = useRef<HTMLHeadingElement>(null);
  const closeAction = useRef(onClose);
  closeAction.current = onClose;
  const titleId = useId(), bodyId = useId(), maskId = useId().replace(/:/g, '');

  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    const shell = document.querySelector<HTMLElement>('.app-shell');
    const wasInert = shell?.inert ?? false;
    if (shell) shell.inert = true;
    const handleKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault(); event.stopPropagation(); closeAction.current(); return;
      }
      if (event.key === 'Tab') {
        const controls = Array.from(card.current?.querySelectorAll<HTMLElement>('button:not(:disabled), select:not(:disabled)') || []);
        const index = controls.indexOf(document.activeElement as HTMLElement);
        if (index < 0 || (!event.shiftKey && index === controls.length - 1) || (event.shiftKey && index === 0)) {
          event.preventDefault();
          (event.shiftKey ? controls.at(-1) : controls[0])?.focus();
        }
      }
      if (!(event.target instanceof HTMLSelectElement) && !event.ctrlKey && !event.metaKey && !event.altKey) {
        if (event.key === 'ArrowRight') { event.preventDefault(); setStep(s => Math.min(steps.length - 1, s + 1)); }
        if (event.key === 'ArrowLeft') { event.preventDefault(); setStep(s => Math.max(0, s - 1)); }
      }
      event.stopPropagation();
    };
    document.addEventListener('keydown', handleKey, true);
    return () => {
      document.removeEventListener('keydown', handleKey, true);
      if (shell) shell.inert = wasInert;
      if (previous?.isConnected && previous !== document.body && previous !== document.documentElement && !previous.closest('[inert]')) previous.focus();
      else document.querySelector<HTMLElement>('[data-tour-target="canvas"]')?.focus();
    };
  }, []);

  const ready = placement !== null;
  useEffect(() => { if (ready) heading.current?.focus(); }, [step, ready]);
  useLayoutEffect(() => {
    let frame = 0;
    const target = document.querySelector<HTMLElement>(`[data-tour-target="${area}"]`);
    const measure = () => {
      if (!card.current) return;
      const width = window.innerWidth, height = window.innerHeight;
      const raw = target?.getBoundingClientRect();
      let visible: Rect | null = null;
      if (raw && raw.width > 0 && raw.height > 0) {
        const x = Math.max(6, raw.left - 4), y = Math.max(6, raw.top - 4);
        const right = Math.min(width - 6, raw.right + 4), bottom = Math.min(height - 6, raw.bottom + 4);
        if (right - x > 20 && bottom - y > 20) visible = { x, y, width: right - x, height: bottom - y };
      }
      const bounds = card.current.getBoundingClientRect();
      setPlacement(place(visible, bounds, width, height));
    };
    const schedule = () => { window.cancelAnimationFrame(frame); frame = window.requestAnimationFrame(measure); };
    const observer = new ResizeObserver(schedule);
    observer.observe(card.current!);
    if (target) observer.observe(target);
    window.addEventListener('resize', schedule);
    document.addEventListener('scroll', schedule, true);
    measure();
    return () => {
      window.cancelAnimationFrame(frame); observer.disconnect();
      window.removeEventListener('resize', schedule);
      document.removeEventListener('scroll', schedule, true);
    };
  }, [area, locale]);

  const rect = placement?.target;
  const width = placement?.width || window.innerWidth, height = placement?.height || window.innerHeight;
  return createPortal(<div className="studio-tour" data-tour-step={area} onClick={event => event.stopPropagation()}>
    <div className="studio-tour-blocker" aria-hidden="true" />
    <svg className="studio-tour-shade" viewBox={`0 0 ${width} ${height}`} aria-hidden="true">
      <defs><mask id={maskId}><rect width={width} height={height} fill="white" />{rect && <rect {...rect} rx={10} fill="black" />}</mask></defs>
      <rect width={width} height={height} fill="rgba(9, 17, 33, .65)" mask={`url(#${maskId})`} />
      {rect && <rect {...rect} rx={10} fill="none" stroke="#60a5fa" strokeWidth={3} data-tour-highlight={area} />}
    </svg>
    <div ref={card} className="studio-tour-card" role="dialog" aria-modal="true" aria-labelledby={titleId} aria-describedby={bodyId}
      style={{ left: placement?.x ?? 12, top: placement?.y ?? 12, visibility: placement ? 'visible' : 'hidden' }}>
      <div className="studio-tour-top"><span>{tr('tour.step', { current: step + 1, total: steps.length })}</span><LanguageSelect />
        <button className="studio-tour-close" aria-label={tr('tour.close')} title={tr('tour.close')} onClick={onClose}><X size={16} /></button>
      </div>
      <h2 ref={heading} tabIndex={-1} id={titleId}>{tr(`tour.${area}.title`)}</h2>
      <p id={bodyId} className="studio-tour-body">{tr(`tour.${area}.body`)}</p>
      <div className="studio-tour-map" aria-label={tr('tour.map')}>
        {(['toolbar', 'library', 'canvas', 'properties', 'results'] as Area[]).map(part => <span key={part} className={`tour-map-${part} ${part === area ? 'is-current' : ''}`} aria-current={part === area ? 'step' : undefined}>{tr(`tour.area.${part}`)}{part === 'toolbar' && <b className={`tour-map-benchmark ${area === 'benchmark' ? 'is-current' : ''}`} aria-current={area === 'benchmark' ? 'step' : undefined}>{tr('tour.area.benchmark')}</b>}</span>)}
      </div>
      <p className="studio-tour-tip">{tr(`tour.${area}.tip`)}</p>
      {placement && !rect && <p className="studio-tour-compact">{tr('tour.compact')}</p>}
      <div className="studio-tour-dots" aria-hidden="true">{steps.map((name, i) => <span key={name} className={i === step ? 'is-current' : ''} />)}</div>
      <footer className="studio-tour-actions">
        <button className="studio-tour-skip" onClick={onClose}>{tr('tour.skip')}</button>
        <button className="studio-tour-secondary" disabled={step === 0} onClick={() => setStep(s => s - 1)}><ArrowLeft size={13} />{tr('tour.previous')}</button>
        <button className="studio-tour-primary" onClick={() => step === steps.length - 1 ? onClose() : setStep(s => s + 1)}>
          {tr(step === steps.length - 1 ? 'tour.finish' : 'tour.next')}{step === steps.length - 1 ? <Check size={13} /> : <ArrowRight size={13} />}
        </button>
      </footer>
    </div>
  </div>, document.body);
}
