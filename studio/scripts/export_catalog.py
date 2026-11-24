"""Build the versioned, public browser catalog. Run from the repository root."""
import json
import subprocess
from pathlib import Path

from refconfig import CType, RefConfig
from studio.backend.registry.registry import get_catalog
from studio.backend.registry.metadata import MODEL_ARCHETYPES
from studio.backend.templates.templates import get_templates
from studio.backend.resource_catalog import LM_MODELS, TOKENIZERS, XMIND_LANGUAGES, resource_args


def build():
    root = Path(__file__).resolve().parents[2]
    catalog = get_catalog().model_dump(mode='json')
    resources = {}
    for spec in catalog['components']:
        if spec['role'] != 'dataset':
            continue
        defaults = {p['name']: p['default'] for p in spec['parameters']}
        data = defaults.get('data_config')
        if not data:
            continue
        for lm in TOKENIZERS:
            embed = ('bertbase' if lm == 'bert' else 'opt' if lm.startswith('opt') else 'llama' if lm.startswith('llama') else lm)
            for lang in (XMIND_LANGUAGES if data.endswith('/xmind.yaml') else ('cmn',)):
                variables = resource_args(data, lm, lang)
                cfg = RefConfig().add(CType.SMART, data=data, embed=f'config/embed/{embed}.yaml', **variables).parse()
                if lm.startswith('opt'):
                    for entry in cfg['embed'].get('embeddings') or []:
                        entry['vocab_name'] = lm
                resources[f'{data}|{lm}|{lang}'] = {'data': cfg['data'], 'embed': cfg['embed']}
    templates = [{k: v for k, v in t.items() if k != 'load_kwargs'} for t in get_templates()]
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    result = dict(schema_version=1, revision=revision, catalog=catalog, templates=templates,
                  resources=resources, archetypes=MODEL_ARCHETYPES, lm_models=LM_MODELS,
                  references={str(p.relative_to(root)): p.read_text() for p in (root / 'config/model').rglob('*.yaml')})
    target = root / 'studio/frontend/public/catalog.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, separators=(',', ':')) + '\n')
    from studio.backend.api.export import export_graph
    from studio.backend.schemas.graph import GraphJson
    parity = {t['id']: export_graph(GraphJson.model_validate(t['graph'])) for t in templates if t.get('available') and t.get('graph')}
    (root / 'studio/frontend/src/runtime/parity.json').write_text(json.dumps(parity, ensure_ascii=False, separators=(',', ':')) + '\n')
    print(f'Exported {len(catalog["components"])} components, {len(templates)} templates, {len(resources)} resource presets')


if __name__ == '__main__':
    build()
