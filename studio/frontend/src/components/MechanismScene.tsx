import { useMemo, useState } from 'react';
import { tr, useLocale } from '@/i18n';
import type { ComponentSpec } from '@/types/catalog';
import { buildNeuralGraph, defaultExample, type ExampleOptions } from '@/explanations/neuronModels';
import { round } from '@/explanations/neuronMath';
import type { OrganizationFocus, OrganizationGroup } from '@/explanations/neuronOrganization';
import { NeuronNetwork } from './NeuronNetwork';

const t = (key: string) => tr(`neuron.${key}`);
function Options({ label, value, options, onChange }: { label: string; value: string; options: [string, string][]; onChange: (value: string) => void }) {
  return <div className="neuron-options" role="group" aria-label={label}><span>{label}</span>{options.map(([id, text]) => <button type="button" key={id} aria-pressed={id === value} onClick={() => onChange(id)}>{text}</button>)}</div>;
}
export function MechanismScene({ component, kind, step, playing, onInteract, onStepDuration, activeGroup, organizationFocus, onInspectGroup, organizationPreview }: { component: ComponentSpec; kind: string; step: number; playing: boolean; onInteract: () => void; onStepDuration: (duration: number) => void; activeGroup: OrganizationGroup | null; organizationFocus: OrganizationFocus | null; onInspectGroup: (group: OrganizationGroup | null) => void; organizationPreview: boolean }) {
  useLocale();
  const [options, setOptions] = useState<ExampleOptions>(defaultExample);
  const graph = useMemo(() => buildNeuralGraph(kind, component.id, options), [kind, component.id, options]);
  const values = useMemo(() => new Map(graph.nodes.map(node => [node.id, node])), [graph]);
  const change = <K extends keyof ExampleOptions>(key: K, value: ExampleOptions[K]) => { onInteract(); setOptions(current => ({ ...current, [key]: value })); };
  const sample = ['cnn', 'cnncat'].includes(kind) ? 'cnn' : ['lm', 'externalhf'].includes(kind) ? 'lm' : kind === 'iisan' ? 'iisan' : kind.includes('_task') ? 'training' : ['din', 'miner'].includes(kind) ? kind : component.category === 'predictor' ? 'predictor' : ['attention', 'transformer', 'flattentransformer', 'fastformer', 'flattenfastformer'].includes(kind) ? 'sequence' : 'history';
  return <div className="mechanism-scene neuron-mechanism" data-mechanism={kind}>
    <div className="neuron-example-controls">
      {component.category === 'predictor' && <Options label={t('candidate')} value={options.candidate} options={['sports', 'cooking'].map(id => [id, t(`candidate.${id}`)] as [string, string])} onChange={value => change('candidate', value as ExampleOptions['candidate'])} />}
      {kind === 'pooling' && <Options label={t('pooling')} value={options.pooling} options={['mean', 'max'].map(id => [id, t(`pooling.${id}`)] as [string, string])} onChange={value => change('pooling', value as ExampleOptions['pooling'])} />}
      {kind === 'gru' && <Options label={t('order')} value={options.reverse ? 'reverse' : 'forward'} options={['forward', 'reverse'].map(id => [id, t(`order.${id}`)] as [string, string])} onChange={value => change('reverse', value === 'reverse')} />}
      {['lm', 'iisan', 'externalhf'].includes(kind) && <Options label={t('context')} value={options.context} options={['phone', 'recipe'].map(id => [id, t(`context.${id}`)] as [string, string])} onChange={value => change('context', value as ExampleOptions['context'])} />}
      {kind === 'externalhf' && <Options label={t('encoding')} value={options.hf} options={['contextual', 'tokens'].map(id => [id, t(`encoding.${id}`)] as [string, string])} onChange={value => change('hf', value as ExampleOptions['hf'])} />}
      {kind === 'miner' && <Options label={t('reducer')} value={options.reducer} options={['weighted', 'max', 'mean'].map(id => [id, t(`reducer.${id}`)] as [string, string])} onChange={value => change('reducer', value as ExampleOptions['reducer'])} />}
      {kind === 'ranking_task' && <Options label={t('label')} value={String(options.label)} options={['1', '0'].map(id => [id, `y = ${id}`])} onChange={value => change('label', Number(value) as 0 | 1)} />}
    </div>
    <p className="neuron-sample">{t(`sample.${sample}`)}</p>
    <div className="neuron-legend">{['input', 'parameter', 'neuron', 'operator', 'gradient', 'output'].map(kind => <span key={kind}><i className={`kind-${kind}`} />{t(`kind.${kind}`)}</span>)}</div>
    <NeuronNetwork organizationPreview={organizationPreview} kind={kind} activeGroup={activeGroup} organizationFocus={organizationFocus} onInspectGroup={onInspectGroup} graph={graph} step={step} playing={playing} onInteract={onInteract} onDuration={onStepDuration} />
    <div className="neuron-results">{graph.outputs.map((id, i) => <div key={id}><span>{kind.includes('_task') ? t(i === 0 ? 'lossBefore' : 'lossAfter') : component.category === 'predictor' ? t('score') : `${t('output')} ${i + 1}`}</span><strong data-neuron-output={id} data-value={values.get(id)!.value}>{round(values.get(id)!.value)}</strong></div>)}</div>
    <p className="neuron-example-note">{t(`example.${graph.note}`)} {t('teaching')}</p>
  </div>;
}
