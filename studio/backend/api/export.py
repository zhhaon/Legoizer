"""Portable configuration export; no training, downloads or config writes."""
import subprocess
from fastapi import APIRouter, HTTPException
from refconfig import CType, RefConfig

from studio.backend.compiler.compiler import build_model_config, build_exp_config
from studio.backend.config import REPO_ROOT
from studio.backend.dataset_inputs import dataset_settings, dataset_variables, resolve_item_inputs, dataset_external_config, item_resource_owner
from studio.backend.schemas.graph import GraphJson
from studio.backend.validator.validator import validate

router = APIRouter(prefix='/api')


@router.post('/graph/export')
def export_graph(graph: GraphJson):
    report = validate(graph)
    if not report.valid:
        raise HTTPException(422, report.model_dump(mode='json'))
    data_path, embed_path, lm = dataset_settings(graph)
    variables = dataset_variables(graph, data_path, lm)
    cfg = RefConfig().add(CType.SMART, data=data_path, embed=embed_path, **variables).parse()
    data, embed = cfg['data'], cfg['embed']
    if lm and lm.startswith('opt'):
        for entry in embed.get('embeddings') or []:
            entry['vocab_name'] = lm
    data['item']['inputs'] = resolve_item_inputs(graph, data, embed, data_path)['values']
    external, checkpoint = item_resource_owner(graph)
    model, _ = build_model_config(graph)
    return dict(model=model, exp=build_exp_config(graph), data=data, embed=embed,
                plan=dict(data=data, embed=embed, lang=variables['lang'], data_config=data_path,
                          checkpoint=checkpoint, external_hf=external, dataset_external_hf=dataset_external_config(graph)),
                revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, text=True).strip())
