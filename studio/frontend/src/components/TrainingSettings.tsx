import { tr, useLocale } from '@/i18n';
import { useEffect, useState } from 'react';
import { accumulationSteps } from '@/utils/batchSize';
import { Play } from 'lucide-react';
import { MetricSettings } from '@/components/MetricSettings';
import { useTrainingCredentials } from '@/components/TrainingCredentials';
import { useStudio } from '@/store/graphStore';
import { DeviceSettings } from '@/components/DeviceSettings';
import { localMode } from '@/runtime/mode';

export function TrainingSettings() {
  useLocale();
  const { name, train: settings } = useStudio((state) => state.graph);
  const update = useStudio((state) => state.updateTrain);
  const rename = useStudio((state) => state.renameGraph);
  const train = useStudio((state) => state.train);
  const training = useStudio((state) => state.training);
  const environment = useStudio((state) => state.environment);
  const httpsPort = useStudio((state) => state.environment?.https_port);
  const { token, setToken, startTraining, acceptsToken } = useTrainingCredentials();
  const secureUrl = new URL(window.location.href);
  secureUrl.search = "";
  secureUrl.hash = "";
  secureUrl.protocol = 'https:';
  if (httpsPort) secureUrl.port = String(httpsPort);

  return (
    <section aria-label={tr("ui.training_settings_a9d521")} className="space-y-2">
      <label className="field-label"> {tr("ui.experiment_name_13ecb5")} <input className="field-input" value={name} onChange={(event) => rename(event.target.value)} />
      </label>
      <BatchSizeFields micro={settings.batch_size} steps={settings.accumulate_batch}
        onChange={(micro, steps) => update({ batch_size: micro, accumulate_batch: steps })} />
      <NumberField label={tr("ui.learning_rate_d06fa7")} value={settings.lr} min={0.000001} max={1} decimal
        onChange={(value) => update({ lr: value })} />
      <NumberField label={tr("ui.hidden_size_6e7174")} value={settings.hidden_size} min={8} max={8192}
        onChange={(value) => update({ hidden_size: value })} />
      <NumberField label={tr("ui.epochs_8f4dcf")} value={settings.epochs} min={1} max={200}
        onChange={(value) => update({ epochs: value })} />
      <DeviceSettings settings={settings} environment={environment} local={localMode} onChange={update} />
      <MetricSettings />
      {acceptsToken ? (
        <label className="field-label"> {tr("ui.hugging_face_access_token_optional__70e03b")} <input type="password" className="field-input" aria-label={tr("ui.hugging_face_access_token_optional__70e03b")} value={token}
            autoComplete="off" spellCheck={false} onChange={(event) => setToken(event.target.value)} />
          <span className="field-help">{tr("ui.use_a_token_with_access_to_the_model_it_is_cleared_w_6346bb")}</span>
        </label>
      ) : (
        <div>
          <p className="field-label">{tr("ui.hugging_face_access_token_optional__70e03b")}</p>
          {httpsPort ? (
            <a className="tool-btn outline" href={secureUrl.href} target="_blank" rel="noopener"> {tr("ui.open_https_to_enter_a_token_d50074")} </a>
          ) : <p className="field-help">{tr("ui.open_studio_over_https_or_through_an_ssh_tunnel_to_e_c5532d")}</p>}
        </div>
      )}
      <div className="flex gap-1.5 pt-1">
        <button className="tool-btn primary flex-1 justify-center !text-[10px]" disabled={training}
          onClick={startTraining}><Play size={12} fill="currentColor" /> {tr("ui.start_train_82f729")}</button>
        <button className="tool-btn outline !text-[10px]" disabled={training}
          onClick={() => void train(true)} title={tr("ui.check_local_resources_without_starting_training_f6b191")}>{tr("ui.preflight_8016a4")}</button>
      </div>
    </section>
  );
}

function BatchSizeFields({ micro, steps, onChange }: {
  micro: number; steps: number; onChange: (micro: number, steps: number) => void;
}) {
  useLocale();
  const [globalDraft, setGlobalDraft] = useState(String(micro * steps));
  const [microDraft, setMicroDraft] = useState(String(micro));
  useEffect(() => {
    setGlobalDraft(String(micro * steps));
    setMicroDraft(String(micro));
  }, [micro, steps]);
  const computed = accumulationSteps(globalDraft, microDraft);
  const change = (global: string, local: string) => {
    setGlobalDraft(global);
    setMicroDraft(local);
    const count = accumulationSteps(global, local);
    if (count !== null) onChange(Number(local), count);
  };
  const resetInvalid = () => {
    if (computed === null) {
      setGlobalDraft(String(micro * steps));
      setMicroDraft(String(micro));
    }
  };
  return (
    <div className="space-y-2">
      <label className="field-label"> {tr("ui.global_batch_size_eb0956")} <input className="field-input" type="number" min={1} max={16777216} step={1}
          value={globalDraft} aria-invalid={computed === null}
          onChange={(event) => change(event.target.value, microDraft)} onBlur={resetInvalid} />
      </label>
      <label className="field-label"> {tr("ui.micro_batch_size_192168")} <input className="field-input" type="number" min={1} max={4096} step={1}
          value={microDraft} aria-invalid={computed === null}
          onChange={(event) => change(globalDraft, event.target.value)} onBlur={resetInvalid} />
      </label>
      <label className="field-label"> {tr("ui.accumulate_steps_d96462")} <input className="field-input !bg-gray-100 !text-gray-500 disabled:cursor-not-allowed dark:!bg-gray-800 dark:!text-gray-400"
          type="number" value={computed ?? ''} disabled readOnly />
      </label>
      {computed === null && <p className="field-help" role="alert"> {tr("ui.enter_positive_integers_global_batch_size_must_be_a__73c679")} </p>}
    </div>
  );
}

function NumberField({ label, value, min, max, decimal = false, onChange }: {
  label: string; value: number; min: number; max: number; decimal?: boolean; onChange: (value: number) => void;
}) {
  useLocale();
  const [draft, setDraft] = useState<string | null>(null);
  const shown = draft ?? String(value);
  const parsed = Number(shown);
  const valid = shown.trim() !== '' && Number.isFinite(parsed) && parsed >= min && parsed <= max
    && (decimal || Number.isInteger(parsed));
  return (
    <label className="field-label">
      {label}
      <input className="field-input" type="number" min={min} max={max} step={decimal ? 'any' : 1}
        value={shown} aria-invalid={!valid}
        onChange={(event) => {
          const raw = event.target.value;
          setDraft(raw);
          const number = Number(raw);
          if (raw.trim() && Number.isFinite(number) && number >= min && number <= max
            && (decimal || Number.isInteger(number))) onChange(number);
        }}
        onBlur={() => setDraft(null)} />
      {!valid && <span className="field-help">{tr("ui.enter_2b9ece")} {decimal ? tr("ui.a_number_c37f5d") : tr("ui.an_integer_3c3d09")} {tr("ui.between_709634")} {min} {tr("ui.and_cffa50")} {max}.</span>}
    </label>
  );
}
