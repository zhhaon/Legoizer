"""Local resource inspection and background acquisition shared with training."""
import hashlib
import json
import os
import threading
from pathlib import Path
from studio.backend.executor import provision as p
from studio.backend.registry.registry import get_catalog, get_component, _safe_read_yaml
from studio.backend.resource_catalog import item_checkpoint, dataset_processor

_jobs = {}
_guard = threading.Lock()


def model_source(checkpoint):
    from utils.config_init import ModelInit
    return ModelInit.get(checkpoint)


def snapshot(source, download=False):
    from huggingface_hub import snapshot_download
    local = Path(source).expanduser()
    if local.is_dir():
        root = local
    else:
        root = Path(snapshot_download(source, local_files_only=not download, cache_dir=str(p.CACHE / 'huggingface/hub')))
    if not (root / 'config.json').is_file():
        raise ValueError('Missing model configuration')
    weights = list(root.glob('*.safetensors')) + list(root.glob('pytorch_model*.bin'))
    if not weights or any(not f.is_file() or not f.stat().st_size for f in weights):
        raise ValueError('Missing model weights')
    for index in root.glob('*.index.json'):
        for name in set(json.loads(index.read_text()).get('weight_map', {}).values()):
            if not (root / name).is_file():
                raise ValueError('Incomplete model shards')
    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained(str(root), local_files_only=True, trust_remote_code=False)
    return root


def dataset_config(spec, params):
    from refconfig import RefConfig, CType
    from studio.backend.resource_catalog import resource_args
    path = params.get('data_config') or spec.source['yaml']
    allowed = next(x for x in spec.parameters if x.name == 'data_config').choices
    if path not in allowed:
        raise ValueError('Unknown dataset configuration')
    variables = resource_args(path, params.get('lm'), params.get('lang'))
    parsed = RefConfig().add(CType.SMART, data=path, **variables).parse()
    data = parsed['data']
    if params.get('item_inputs'):
        data['item']['inputs'] = [{x['column']: x.get('truncate')} if isinstance(x, dict) and 'column' in x else x for x in params['item_inputs']]
    return path, data


def dataset_embeddings(spec, params):
    from refconfig import RefConfig, CType
    allowed = next(x for x in spec.parameters if x.name == 'embed_config').choices
    embed = params.get('embed_config')
    if embed not in allowed: raise ValueError('Unknown embedding configuration')
    return RefConfig().add(CType.SMART, embed=embed, lm=params.get('lm', 'glove')).parse()['embed'].get('embeddings') or []


def inspect(component, params=None):
    spec = get_component(component)
    if spec is None: raise ValueError('Unknown component')
    params = {**{x.name:x.default for x in spec.parameters}, **(params or {})}
    key = hashlib.sha256(json.dumps([component, params], sort_keys=True).encode()).hexdigest()
    with _guard: job = dict(_jobs.get(key, {}))
    if job.get('state') == 'downloading': return dict(key=key, **job)
    try:
        if spec.category == 'dataset':
            path, data = dataset_config(spec, params)
            ready = p.tables_available(data, dataset_processor(path), params.get('lang', 'cmn'))
            ready = ready and all((p.REPO_ROOT / x['path']).is_file() for x in dataset_embeddings(spec, params))
            state = 'ready' if ready else 'missing'
        elif component == 'item.externalhf':
            from studio.backend.executor.external_hf import cached_external
            if not params.get('model_id'): return dict(key=key, state='configure', message='Set the model repository in Properties first')
            state = 'ready' if cached_external(params) else 'missing'
        else:
            checkpoint = item_checkpoint(spec.meta_value or '')
            if not checkpoint: return dict(key=key, state='unnecessary')
            snapshot(model_source(checkpoint))
            state = 'ready'
        if state == 'missing' and job.get('state') == 'failed': return dict(key=key, **job)
        return dict(key=key, state=state)
    except Exception:
        return dict(key=key, state='missing')


def acquire(component, params=None, token=None):
    spec = get_component(component)
    if spec is None: raise ValueError('Unknown component')
    params = {**{x.name:x.default for x in spec.parameters}, **(params or {})}
    status = inspect(component, params)
    if status['state'] in ('ready', 'unnecessary', 'downloading', 'configure'): return status
    key = status['key']
    with _guard:
        if _jobs.get(key, {}).get('state') == 'downloading': return dict(key=key, **_jobs[key])
        _jobs[key] = dict(state='downloading')
    def work():
        try:
            # Run in a separate process: cache environment and credentials must not leak between jobs.
            import subprocess, sys
            env = p.resource_env()
            if token: env['HF_TOKEN'] = token
            result = subprocess.run([sys.executable, '-m', 'studio.backend.resources'],
                input=json.dumps(dict(component=component, params=params)), text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
            if result.returncode: raise RuntimeError(result.stdout[-3000:])
            with _guard: _jobs.pop(key, None)
        except Exception as exc:
            with _guard: _jobs[key] = dict(state='failed', message=str(exc))
    threading.Thread(target=work, daemon=True).start()
    return dict(key=key, state='downloading')


def prepare_component(component, params):
    spec = get_component(component)
    if spec.category == 'dataset':
        path, data = dataset_config(spec, params)
        p.processed_data(data, dataset_processor(path), params.get('lang', 'cmn'))
        for entry in dataset_embeddings(spec, params):
            if not (p.REPO_ROOT / entry['path']).is_file():
                target = Path(entry['path'])
                if target.parent != Path('data/embeddings') or target.stem not in {*p.LM_MODELS, 'glove'}:
                    raise ValueError('Missing dataset-specific embedding matrix')
                p.embedding(target.stem, target)
    elif component == 'item.externalhf':
        from studio.backend.executor.external_hf import prepare_external_hf
        prepare_external_hf(params)
    else:
        checkpoint = item_checkpoint(spec.meta_value or '')
        if checkpoint: snapshot(model_source(checkpoint), download=True)

def _remove_owned(path):
    """Only remove concrete dataset/model assets under server-owned roots."""
    import shutil
    path = Path(path).absolute()
    roots = [(p.REPO_ROOT / 'data').resolve(), p.CACHE.resolve()]
    resolved = path.resolve()
    if not any(resolved != root and resolved.is_relative_to(root) for root in roots):
        raise ValueError('Only project-managed resources can be deleted')
    if path.is_symlink():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def remove(component, params=None):
    from studio.backend.executor import executor
    from studio.backend.executor.external_hf import cached_external, external_record
    spec = get_component(component)
    if spec is None: raise ValueError('Unknown component')
    params = {**{x.name:x.default for x in spec.parameters}, **(params or {})}
    # Shared resources can serve multiple components and runs.
    with _guard:
        if any(job.get('state') == 'downloading' for job in _jobs.values()):
            raise ValueError('Wait for resource downloads to finish before deleting')
        if any(run.status in ('pending', 'preparing', 'running') and not run.error.startswith('Preflight only') for run in executor.list_runs()):
            raise ValueError('Stop active training or evaluation before deleting resources')
        if spec.category == 'dataset':
            config, data = dataset_config(spec, params)
            processor = dataset_processor(config)
            paths = [p.REPO_ROOT / x for x in p.dataset_paths(data)]
            paths.append(p.prepared_directory(data, processor, params.get('lang', 'cmn')))
            # Remove every prepared variant of this dataset, leaving shared embeddings intact.
            paths.extend((p.CACHE / 'datasets').glob(processor + '-*'))
            for path in paths:
                if path.exists(): _remove_owned(path)
        else:
            record = cached_external(params) if component == 'item.externalhf' else None
            checkpoint = item_checkpoint(spec.meta_value or '')
            if record:
                source = p.REPO_ROOT / record['prepared']['checkpoint_dir']
            elif checkpoint:
                source = snapshot(model_source(checkpoint))
            else:
                raise ValueError('No downloaded model resources to delete')
            hub = (p.CACHE / 'huggingface/hub').resolve()
            if source.resolve().is_relative_to(hub):
                from huggingface_hub import scan_cache_dir
                cache = scan_cache_dir(hub)
                revisions = [rev.commit_hash for repo in cache.repos for rev in repo.revisions
                             if rev.snapshot_path.resolve() == source.resolve()]
                if not revisions: raise ValueError('Model cache revision was not found')
                cache.delete_revisions(*revisions).execute()
            else:
                _remove_owned(source)
            if checkpoint:
                _remove_owned(p.REPO_ROOT / 'data/embeddings' / (checkpoint + '.npy'))
            if record:
                _remove_owned((p.REPO_ROOT / record['embedding']['path']).parent)
                _remove_owned(external_record(params))
        _jobs.clear()
    return inspect(component, params)


if __name__ == '__main__':
    import sys
    request = json.load(sys.stdin)
    prepare_component(**request)
