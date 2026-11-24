import { MechanismScene } from './MechanismScene';
import { useEffect, useId, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { ChevronLeft, ChevronRight, CircleHelp, Pause, Play, RotateCcw, X } from 'lucide-react';
import { tr, useLocale } from '@/i18n';
import { componentText } from '@/i18n/display';
import type { ComponentSpec } from '@/types/catalog';
import { CATEGORY_COLORS } from '@/utils/ui';
import type { OrganizationFocus, OrganizationGroup } from '@/explanations/neuronOrganization';
import type { Explanation } from '@/explanations/catalog';

function useReducedMotion() {
  const [reduced, setReduced] = useState(() => window.matchMedia('(prefers-reduced-motion: reduce)').matches);
  useEffect(() => {
    const media = window.matchMedia('(prefers-reduced-motion: reduce)');
    const update = () => setReduced(media.matches);
    media.addEventListener('change', update);
    return () => media.removeEventListener('change', update);
  }, []);
  return reduced;
}

export function ComponentExplanation({ component, explanation, onClose, onDetails }: {
  component: ComponentSpec; explanation: Explanation; onClose: () => void; onDetails: () => void;
}) {
  useLocale();
  const reduced = useReducedMotion();
  const [step, setStep] = useState(0);
  const [replayId, setReplayId] = useState(0);
  const [stepDuration, setStepDuration] = useState(3400);
  const [playing, setPlaying] = useState(!reduced);
  const [hoveredGroup, setHoveredGroup] = useState<OrganizationGroup | null>(null);
  const [focusedGroup, setFocusedGroup] = useState<OrganizationGroup | null>(null);
  const [inspectedGroup, setInspectedGroup] = useState<OrganizationGroup | null>(null);
  const [organizationFocus, setOrganizationFocus] = useState<OrganizationFocus | null>(null);
  const activeGroup = hoveredGroup ?? inspectedGroup ?? focusedGroup ?? organizationFocus?.group ?? null;
  const organization = useRef<HTMLElement>(null);
  const root = useRef<HTMLDivElement>(null);
  const close = useRef<HTMLButtonElement>(null);
  const closeAction = useRef(onClose);
  closeAction.current = onClose;
  const titleId = useId();
  const summaryId = useId();
  const key = `explain.${explanation.key}`;
  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    close.current?.focus();
    // Capture Escape even if a newly disabled step button caused focus to leave the dialog.
    const handleKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault(); event.stopPropagation(); closeAction.current(); return;
      }
      if (event.key === 'Tab') {
        const buttons = Array.from(root.current?.querySelectorAll<HTMLButtonElement>('button:not(:disabled)') || []);
        const first = buttons[0], last = buttons[buttons.length - 1];
        const outside = !root.current?.contains(document.activeElement);
        if (event.shiftKey && (outside || document.activeElement === first)) { event.preventDefault(); last?.focus(); }
        if (!event.shiftKey && (outside || document.activeElement === last)) { event.preventDefault(); first?.focus(); }
      }
      if (!root.current?.contains(event.target as Node)) event.stopPropagation();
    };
    document.addEventListener('keydown', handleKey, true);
    return () => {
      document.removeEventListener('keydown', handleKey, true);
      if (previous?.isConnected) previous.focus();
    };
  }, []);
  useEffect(() => { if (reduced) setPlaying(false); }, [reduced]);
  useEffect(() => {
    if (!playing || reduced) return;
    const timer = window.setTimeout(() => {
      if (step === 3) setPlaying(false);
      else setStep(step + 1);
    }, stepDuration);
    return () => window.clearTimeout(timer);
  }, [playing, reduced, step, replayId, stepDuration]);
  useEffect(() => {
    const pane = organization.current;
    const section = pane?.querySelector<HTMLElement>(`[data-explanation-organization="${inspectedGroup}"]`);
    if (!pane || !section || pane.scrollHeight <= pane.clientHeight) return;
    const target = section.getBoundingClientRect(), bounds = pane.getBoundingClientRect();
    if (target.top < bounds.top || target.bottom > bounds.bottom) pane.scrollTop += target.top + target.height / 2 - bounds.top - pane.clientHeight / 2;
  }, [inspectedGroup]);
  const clearOrganization = () => { setHoveredGroup(null); setFocusedGroup(null); setInspectedGroup(null); setOrganizationFocus(null); };
  const go = (index: number) => { clearOrganization(); setPlaying(false); setStep(index); };
  const locateOrganization = (group: OrganizationGroup) => {
    setPlaying(false); setInspectedGroup(null);
    setOrganizationFocus(current => ({ group, revision: (current?.revision ?? 0) + 1 }));
    const body = root.current?.querySelector<HTMLElement>('.explanation-body');
    const visual = root.current?.querySelector<HTMLElement>('.explanation-visual');
    if (!body || !visual) return;
    const target = visual.getBoundingClientRect(), bounds = body.getBoundingClientRect();
    // An explicit locate action also brings the graph into view in the stacked mobile layout.
    if (target.top < bounds.top || target.bottom > bounds.bottom) body.scrollTop += target.top - bounds.top - Math.max(0, (body.clientHeight - target.height) / 2);
  };
  return createPortal(<div className="modal-backdrop explanation-backdrop" onClick={e => { e.stopPropagation(); if (e.target === e.currentTarget) onClose(); }}
    onKeyDown={e => e.stopPropagation()}>
    <div ref={root} role="dialog" aria-modal="true" aria-labelledby={titleId} aria-describedby={summaryId} className="explanation-dialog" data-explanation-id={component.id} style={{ '--explanation-color': CATEGORY_COLORS[component.category] } as React.CSSProperties}>
      <header className="explanation-header">
        <div className="explanation-heading-icon"><CircleHelp size={22} /></div>
        <div className="min-w-0 flex-1"><p className="text-[10px] font-semibold uppercase tracking-widest text-muted">{tr('explain.title')}</p><h2 id={titleId} className="mt-1 text-base font-bold">{componentText(component, 'name')}</h2></div>
        <button ref={close} className="tool-btn !px-2" aria-label={tr('explain.close')} title={tr('explain.close')} onClick={onClose}><X size={16} /></button>
      </header>
      <div className="explanation-body">
        <p id={summaryId} className="text-sm leading-relaxed">{tr(`${key}.summary`)}</p>
        <div className="explanation-main">
          <div className="explanation-stage" data-explanation-step={step}><div className="explanation-visual"><MechanismScene key={`mechanism-${replayId}`} component={component} kind={explanation.key} step={step} playing={playing && !reduced} onInteract={() => setPlaying(false)} onStepDuration={setStepDuration} activeGroup={activeGroup} organizationFocus={organizationFocus} onInspectGroup={setInspectedGroup} organizationPreview={hoveredGroup !== null || focusedGroup !== null} /></div>

          </div>
          <section ref={organization} className="explanation-narration explanation-organization" aria-label={tr('explain.organization.title')}>
            <p className="explanation-organization-hint">{tr('explain.organization.hint')}</p>
            {([0, 1, 2] as const).map(index => <button type="button" key={index}
              className={`explanation-organization-section ${activeGroup === index ? 'is-linked' : ''}`}
              data-explanation-organization={index} aria-pressed={organizationFocus?.group === index}
              aria-label={tr('explain.organization.locate', { name: tr(`${key}.organization.${index}.title`) })}
              onPointerEnter={() => { setPlaying(false); setHoveredGroup(index); }} onPointerLeave={() => setHoveredGroup(null)}
              onFocus={() => { setPlaying(false); setInspectedGroup(null); setFocusedGroup(index); }} onBlur={() => setFocusedGroup(null)}
              onClick={() => locateOrganization(index)}>
              <span className="explanation-organization-heading">{tr(`${key}.organization.${index}.title`)}</span>
              <span data-explanation-organization-body={index}>{tr(`${key}.organization.${index}.body`)}</span>
            </button>)}
          </section>
        </div>
        <div className="explanation-notes">
          <section><h3>{tr('explain.use')}</h3><p>{tr(`${key}.use`)}</p></section>
          <section><h3>{tr('explain.difference')}</h3><p>{tr(`${key}.difference`)}</p></section>
        </div>
        {reduced && <p className="text-[11px] text-muted">{tr('explain.reduced')}</p>}
      </div>
      <footer className="explanation-controls">
        <div className="flex flex-wrap items-center gap-2">
          <button className="tool-btn explanation-secondary" disabled={step === 0} onClick={() => go(step - 1)}><ChevronLeft size={14} />{tr('explain.previous')}</button>
          {!reduced && <button className="tool-btn primary" onClick={() => { clearOrganization(); if (!playing && step === 3) setStep(0); setPlaying(!playing); }}>{playing ? <Pause size={14} /> : <Play size={14} />}{tr(playing ? 'explain.pause' : 'explain.play')}</button>}
          <button className="tool-btn explanation-secondary" disabled={step === 3} onClick={() => go(step + 1)}>{tr('explain.next')}<ChevronRight size={14} /></button>
          <button className="tool-btn explanation-secondary" onClick={() => { clearOrganization(); setStep(0); setReplayId(id => id + 1); setPlaying(!reduced); }}><RotateCcw size={14} />{tr('explain.replay')}</button>
        </div>
        <button className="tool-btn explanation-secondary" onClick={onDetails}>{tr('explain.parameters')}</button>
      </footer>
    </div>
  </div>, document.body);
}
