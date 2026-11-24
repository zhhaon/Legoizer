import { useEffect, useId, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { calculation, round, type NeuralGraph, type Neuron } from '@/explanations/neuronMath';
import { organizationGroup, type OrganizationFocus, type OrganizationGroup } from '@/explanations/neuronOrganization';
import { tr, useLocale } from '@/i18n';

const t = (key: string) => tr(`neuron.${key}`);
const glyph = (n: Neuron) => n.kind === 'output' ? 'out' : n.op === 'input' ? n.kind === 'parameter' ? 'w' : 'x' : n.op === 'affine' ? n.activation === 'identity' ? 'Σ' : n.activation : ({ product: '×', softmax: 'SM', layernorm: 'LN', mean: 'mean', max: 'max', logloss: '−ln', bce: 'BCE', sum: '+' }[n.op] || n.op);
interface Point { x: number; y: number }
function connectionPath(a: Point, b: Point, skip: boolean, lane: number) {
  if (!skip) return `M${a.x + 23},${a.y} C${a.x + 62},${a.y} ${b.x - 62},${b.y} ${b.x - 25},${b.y}`;
  // Skip connections use the free space above the neurons, rather than crossing intermediate circles.
  return `M${a.x + 23},${a.y} C${a.x + 45},${a.y} ${a.x + 52},${a.y - 12} ${a.x + 52},${a.y - 26}
    L${a.x + 52},${lane + 12} Q${a.x + 52},${lane} ${a.x + 64},${lane}
    L${b.x - 64},${lane} Q${b.x - 52},${lane} ${b.x - 52},${lane + 12}
    L${b.x - 52},${b.y - 26} C${b.x - 52},${b.y - 12} ${b.x - 45},${b.y} ${b.x - 25},${b.y}`;
}
function arrange(graph: NeuralGraph) {
  const columns = new Map<number, Neuron[]>();
  for (const node of graph.nodes) columns.set(node.depth, [...(columns.get(node.depth) || []), node]);
  const height = Math.max(...[...columns.values()].map(xs => xs.length)) * 66 + 90;
  const width = Math.max(...columns.keys()) * 120 + 150;
  const points = new Map<string, Point>();
  for (const [depth, nodes] of columns) nodes.forEach((node, i) => points.set(node.id, { x: 65 + depth * 120, y: (height - (nodes.length - 1) * 66) / 2 + i * 66 }));
  return { points, columns, width, height };
}

function NeuronTooltip({ node, graph, anchor }: { node: Neuron; graph: NeuralGraph; anchor: HTMLElement }) {
  const locale = useLocale();
  const pane = useRef<HTMLDivElement>(null);
  const [position, setPosition] = useState({ x: 12, y: 12 });
  const computed = useMemo(() => calculation(node, graph.nodes), [node, graph]);
  const lookup = useMemo(() => new Map(graph.nodes.map(n => [n.id, n])), [graph]);
  useLayoutEffect(() => {
    const update = () => {
      const source = anchor.getBoundingClientRect(), box = pane.current!.getBoundingClientRect();
      const visual = anchor.closest('.explanation-visual')?.getBoundingClientRect();
      // Keep the explanation pane readable while its matching neuron is inspected.
      const withinGraph = visual && visual.width >= box.width + 16;
      const minX = withinGraph ? Math.max(12, visual.left + 8) : 12;
      const maxX = withinGraph ? Math.min(innerWidth - box.width - 12, visual.right - box.width - 8) : innerWidth - box.width - 12;
      const x = source.right + box.width + 12 <= maxX + box.width ? source.right + 12 : source.left - box.width - 12;
      setPosition({ x: Math.max(minX, Math.min(x, maxX)), y: Math.max(12, Math.min(source.top - 12, innerHeight - box.height - 12)) });
    };
    const observer = new ResizeObserver(update); observer.observe(pane.current!); observer.observe(anchor);
    addEventListener('resize', update); document.addEventListener('scroll', update, true); update();
    return () => { observer.disconnect(); removeEventListener('resize', update); document.removeEventListener('scroll', update, true); };
  }, [anchor, locale, node]);
  return createPortal(<div ref={pane} id="neuron-calculation-tooltip" role="tooltip" data-neuron-tooltip={node.id} className="neuron-tooltip" style={{ left: position.x, top: position.y }}>
    <div className="neuron-tooltip-heading"><b>{node.label}</b><span>{t(`kind.${node.kind}`)}</span></div>
    <p className="neuron-operation">{t(`op.${node.op}`)}{node.activation !== 'identity' ? ` · ${node.activation}` : ''}</p>
    {node.terms.length ? <>
      <h4>{t('inputs')}</h4><table><thead><tr><th>{t('source')}</th><th>{t('value')}</th><th>{t('weight')}</th></tr></thead><tbody>
        {node.terms.map((term, i) => <tr key={`${term.id}-${i}`}><td>{lookup.get(term.id)!.label}</td><td>{round(lookup.get(term.id)!.value)}</td><td>{round(term.weight)}</td></tr>)}
      </tbody></table>
      {node.op === 'affine' && <p className="neuron-bias">b = {round(node.bias)}</p>}
      <h4>{t('calculation')}</h4><code>{computed.symbolic}</code><code>{computed.substituted.replace(/ = /g, ' ≈ ')}</code>
      {computed.details.map((line, i) => <code key={i}>{line}</code>)}
      {node.activation !== 'identity' && <><h4>{t('activation')}</h4><code>{{ relu: 'ReLU(z) = max(0,z)', tanh: 'tanh(z) = (exp(z)−exp(−z))/(exp(z)+exp(−z))', sigmoid: 'σ(z) = 1/(1+exp(−z))', silu: 'SiLU(z) = z·σ(z)', gelu: 'GELU(z) ≈ 0.5z[1+tanh(√(2/π)(z+0.044715z³))]' }[node.activation]}</code></>}
    </> : <><p>{t(node.kind === 'parameter' ? 'parameterSource' : 'inputSource')}</p><code>{computed.substituted}</code></>}
    {node.note && <p className="neuron-tooltip-note">{t(`note.${node.note}`)}</p>}
    <div className="neuron-tooltip-output"><span>{t('output')}</span><strong data-neuron-tooltip-result={node.value}>{round(node.value)}</strong></div>
    <small>{t('rounding')}</small>
  </div>, document.body);
}

export function NeuronNetwork({ graph, step, playing, onInteract, onDuration, kind, activeGroup, organizationFocus, onInspectGroup, organizationPreview }: { graph: NeuralGraph; step: number; playing: boolean; onInteract: () => void; onDuration: (duration: number) => void; kind: string; activeGroup: OrganizationGroup | null; organizationFocus: OrganizationFocus | null; onInspectGroup: (group: OrganizationGroup | null) => void; organizationPreview: boolean }) {
  useLocale();
  const root = useRef<HTMLDivElement>(null);
  const [scale, setScale] = useState(.8);
  const [tick, setTick] = useState(0);
  const [selected, setSelected] = useState<{ id: string; anchor: HTMLElement } | null>(null);
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [focusedId, setFocusedId] = useState<string | null>(null);
  const [focusId, setFocusId] = useState(graph.nodes[0].id);
  const uid = useId().replace(/:/g, '');
  const layout = useMemo(() => arrange(graph), [graph]);
  const lookup = useMemo(() => new Map(graph.nodes.map(n => [n.id, n])), [graph]);
  const groups = useMemo(() => new Map(graph.nodes.map(node => [node.id, organizationGroup(kind, node)])), [graph, kind]);
  const inspectedId = hoveredId ?? focusedId ?? selected?.id ?? null;
  const tracingNode = inspectedId !== null && (!organizationPreview || hoveredId !== null);
  const highlightingGroup = !tracingNode && activeGroup !== null;
  useEffect(() => { onInspectGroup(inspectedId ? groups.get(inspectedId) ?? null : null); }, [inspectedId, groups, onInspectGroup]);
  const depths = useMemo(() => [...new Set(graph.nodes.filter(n => n.stage === step).map(n => n.depth))].sort((a, b) => a - b), [graph, step]);
  const currentDepth = depths[Math.min(tick, depths.length - 1)];
  const active = useMemo(() => new Set(graph.nodes.filter(n => n.stage === step && n.depth === currentDepth).map(n => n.id)), [graph, step, currentDepth]);
  const stageRect = useMemo(() => {
    const nodes = graph.nodes.filter(node => active.has(node.id)).map(node => layout.points.get(node.id)!);
    return { x: Math.min(...nodes.map(p => p.x)) - 38, y: Math.min(...nodes.map(p => p.y)) - 34, width: 76, height: Math.max(...nodes.map(p => p.y)) - Math.min(...nodes.map(p => p.y)) + 80 };
  }, [graph, active, layout]);
  const groupRect = useMemo(() => {
    if (!organizationFocus) return null;
    const points = graph.nodes.filter(node => groups.get(node.id) === organizationFocus.group).map(node => layout.points.get(node.id)!);
    if (!points.length) return null;
    return { x: Math.min(...points.map(p => p.x)) - 38, y: Math.min(...points.map(p => p.y)) - 34,
      width: Math.max(...points.map(p => p.x)) - Math.min(...points.map(p => p.x)) + 76,
      height: Math.max(...points.map(p => p.y)) - Math.min(...points.map(p => p.y)) + 80 };
  }, [organizationFocus, graph, groups, layout]);
  const focusRect = groupRect ?? stageRect;
  // Fit only on an explicit organization request; hovering never pans or changes zoom.
  useLayoutEffect(() => {
    const viewport = root.current?.closest<HTMLElement>('.explanation-visual');
    if (!groupRect || !viewport) return;
    const toolbar = root.current?.querySelector<HTMLElement>('.neuron-tools')?.offsetHeight ?? 44;
    const availableHeight = Math.max(30, viewport.clientHeight - Math.max(90, toolbar * 2 + 24));
    setScale(current => Math.max(.04, Math.min(current, (viewport.clientWidth - 30) / groupRect.width, availableHeight / groupRect.height)));
  }, [groupRect]);
  useEffect(() => { setTick(0); }, [step, graph]);
  useEffect(() => { setSelected(null); setHoveredId(null); setFocusedId(null); setFocusId(graph.nodes[0].id); }, [graph]);
  useEffect(() => { if (organizationFocus) { setSelected(null); setHoveredId(null); setFocusedId(null); } }, [organizationFocus]);
  useEffect(() => { onDuration(Math.max(2800, depths.length * 450 + 800)); }, [depths.length, onDuration]);
  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(() => setTick(n => n + 1), 450);
    return () => window.clearInterval(timer);
  }, [playing, step]);
  useEffect(() => {
    if (!selected) return;
    const outside = (event: PointerEvent) => {
      const target = event.target as Node;
      if (!selected.anchor.contains(target) && !document.querySelector('[data-neuron-tooltip]')?.contains(target)) { setSelected(null); }
    };
    document.addEventListener('pointerdown', outside, true);
    return () => document.removeEventListener('pointerdown', outside, true);
  }, [selected]);
  useLayoutEffect(() => {
    const viewport = root.current?.closest<HTMLElement>('.explanation-visual');
    const box = root.current?.querySelector<HTMLElement>('[data-neuron-stage-focus]');
    if (!viewport || !box) return;
    box.setAttribute('data-mechanism-centered', 'true');
    const center = () => {
      root.current?.style.setProperty('--neuron-viewport-width', `${Math.max(200, viewport.clientWidth - 20)}px`);
      const target = box.getBoundingClientRect(), bounds = viewport.getBoundingClientRect();
      viewport.scrollTop += target.top + target.height / 2 - bounds.top - viewport.clientHeight / 2;
      viewport.scrollLeft += target.left + target.width / 2 - bounds.left - viewport.clientWidth / 2;
    };
    const observer = new ResizeObserver(center); observer.observe(viewport); center();
    return () => { observer.disconnect(); box.removeAttribute('data-mechanism-centered'); };
  }, [currentDepth, scale, step, focusRect]);
  const selectedNode = selected && lookup.get(selected.id);
  const fit = () => {
    const viewport = root.current?.closest<HTMLElement>('.explanation-visual');
    if (viewport) setScale(Math.max(.04, Math.min(1, (viewport.clientWidth - 24) / layout.width, (viewport.clientHeight - 85) / layout.height)));
  };
  return <div ref={root} className={`neuron-scene ${tracingNode ? 'is-node-trace' : ''}`} data-neuron-trace={tracingNode ? inspectedId : undefined} data-neuron-network data-mechanism-frame={tick} data-mechanism-step={step}>
    <div className="neuron-tools">
      <span>{t('hint')}</span><div><button type="button" aria-label={t('zoomOut')} onClick={() => { onInteract(); setScale(s => Math.max(.04, s - .1)); }}>−</button><span>{Math.round(scale * 100)}%</span><button type="button" aria-label={t('zoomIn')} onClick={() => { onInteract(); setScale(s => Math.min(1.5, s + .1)); }}>＋</button><button type="button" onClick={() => { onInteract(); fit(); }}>{t('fit')}</button></div>
    </div>
    <div className="neuron-canvas" style={{ width: layout.width * scale, height: layout.height * scale }}>
      <div className="neuron-surface" style={{ width: layout.width, height: layout.height, transform: `scale(${scale})` }}>
        <svg width={layout.width} height={layout.height} className="neuron-links" aria-hidden="true"><defs><marker id={uid} markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="context-stroke" /></marker></defs>
          {graph.nodes.flatMap((node, nodeIndex) => node.terms.map((term, i) => {
            const a = layout.points.get(term.id)!, b = layout.points.get(node.id)!;
            const inspected = tracingNode && (inspectedId === node.id || inspectedId === term.id);
            const organizationEdge = highlightingGroup && (groups.get(node.id) === activeGroup || groups.get(term.id) === activeGroup);
            const muted = tracingNode ? !inspected : highlightingGroup && !organizationEdge;
            const flowing = playing && !tracingNode && active.has(node.id);
            return <path key={`${node.id}-${i}`} data-neuron-edge={`${term.id}:${node.id}`} className={`neuron-edge ${organizationEdge ? 'is-organization' : ''} ${muted ? 'is-muted' : ''} ${inspected ? 'is-inspected' : ''} ${flowing ? 'is-flowing' : ''}`}
              d={connectionPath(a, b, node.depth - lookup.get(term.id)!.depth > 1, 14 + ((nodeIndex + i) % 5) * 6)} fill="none" stroke="currentColor" markerEnd={`url(#${uid})`} />;
          }))}
        </svg>
        <div className={`neuron-stage-focus ${groupRect ? 'is-organization-focus' : ''}`} data-neuron-stage-focus
          data-organization-centered={organizationFocus?.group} style={{ left: focusRect.x, top: focusRect.y, width: focusRect.width, height: focusRect.height,
            visibility: tracingNode || activeGroup !== null && (!organizationFocus || organizationFocus.group !== activeGroup) ? 'hidden' : undefined }} />
        {graph.nodes.map(node => {
          const p = layout.points.get(node.id)!;
          return <div key={node.id} className={`neuron-unit kind-${node.kind} ${highlightingGroup ? groups.get(node.id) === activeGroup ? 'is-organization' : 'is-muted' : ''} ${!tracingNode && active.has(node.id) ? 'is-current' : ''} ${selected?.id === node.id ? 'is-selected' : ''}`} style={{ left: p.x - 23, top: p.y - 23 }}>
            <button type="button" data-neuron={node.id} data-neuron-op={node.op} data-neuron-kind={node.kind} data-neuron-value={node.value} data-neuron-stage={node.stage} data-neuron-organization={groups.get(node.id)}
              tabIndex={focusId === node.id ? 0 : -1} aria-label={`${node.label}: ${round(node.value)}. ${t('inspect')}`} aria-describedby={selected?.id === node.id ? 'neuron-calculation-tooltip' : undefined}
              onPointerEnter={() => { onInteract(); setHoveredId(node.id); }} onPointerLeave={() => setHoveredId(current => current === node.id ? null : current)}
              onPointerMove={() => { if (hoveredId !== node.id) { onInteract(); setHoveredId(node.id); } }}
              onFocus={() => { onInteract(); setFocusId(node.id); setFocusedId(node.id); }} onBlur={() => setFocusedId(current => current === node.id ? null : current)}
              onClick={event => { onInteract(); const anchor = event.currentTarget; setSelected(current => current?.id === node.id ? null : { id: node.id, anchor }); }}
              onKeyDown={event => {
                if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End'].includes(event.key)) return;
                event.preventDefault(); event.stopPropagation();
                let candidates = graph.nodes;
                if (event.key === 'ArrowUp' || event.key === 'ArrowDown') candidates = layout.columns.get(node.depth)!;
                const ordered = [...candidates].sort((a, b) => a.depth - b.depth || layout.points.get(a.id)!.y - layout.points.get(b.id)!.y);
                const index = ordered.findIndex(n => n.id === node.id);
                const next = event.key === 'Home' ? ordered[0] : event.key === 'End' ? ordered.at(-1)! : ordered[Math.max(0, Math.min(ordered.length - 1, index + (['ArrowLeft', 'ArrowUp'].includes(event.key) ? -1 : 1)))];
                setFocusId(next.id); root.current?.querySelector<HTMLButtonElement>(`[data-neuron="${next.id}"]`)?.focus();
              }}><small>{glyph(node)}</small><strong>{Number(node.value.toFixed(2))}</strong></button><span className="neuron-label">{node.label}</span>
          </div>;
        })}
      </div>
    </div>
    {selectedNode && selected && <NeuronTooltip node={selectedNode} graph={graph} anchor={selected.anchor} />}
  </div>;
}
