"""Audit registry defaults against real config classes and dataset YAMLs (no training)."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
parser = argparse.ArgumentParser()
parser.add_argument('--output', default='/tmp/studio-audited-defaults.json')
args = parser.parse_args()
import copy,inspect,json
from pathlib import Path
from loader.class_hub import ClassHub
from model.lego_config import LegoConfig
from studio.backend.registry.registry import get_catalog,_signature_params
from studio.backend.schemas.graph import GraphJson,GraphNode
from studio.backend.dataset_inputs import resolve_dataset_inputs

catalog=get_catalog()
before=catalog.model_dump_json()
records=[]
count=0
effective_changes=[]
for component in catalog.components:
    hidden=next((p.default for p in component.parameters if p.name=='hidden_size'),256)
    effective={}
    if component.backend_class:
        hub=ClassHub.predictors() if component.role=='predictor' else ClassHub.operators()
        cls=hub(component.meta_value)
        signature=_signature_params(cls.config_class)
        kwargs={p.name:copy.deepcopy(p.default) for p in component.parameters if p.name in signature}
        kwargs.update(hidden_size=hidden,input_dim=next((p.default for p in component.parameters if p.name=='item_hidden_size' and p.default is not None),hidden))
        instance=cls.config_class(**kwargs)
        for param in component.parameters:
            if param.name not in signature:continue
            if param.source=='introspected':
                declared=signature[param.name].default
                if declared is inspect.Parameter.empty:
                    assert param.required and param.default is None,(component.id,param.name,'required default')
                else:
                    assert param.default==declared,(component.id,param.name,param.default,declared)
            value=getattr(instance,param.name,param.default)
            effective[param.name]=value
            if value!=param.default:effective_changes.append([component.id,param.name,value])
    fields={}
    for param in component.parameters:
        count+=1
        result={'kind':'value','value':effective.get(param.name,param.default)}
        if param.name=='item_inputs' and component.role=='dataset':
            node=GraphNode(id='dataset_audit',component=component.id,params={p.name:copy.deepcopy(p.default) for p in component.parameters if p.default is not None})
            rows=resolve_dataset_inputs(GraphJson(nodes=[node]),use_override=False)['defaults']
            result={'kind':'dataset','rows':rows}
        elif param.default is None:
            if param.default_hint=='Matches hidden_size':
                result={'kind':'value','value':LegoConfig(hidden_size=hidden, user_config={}).item_hidden_size}
            elif param.default_hint=='Detected from the pretrained token embedding matrix':
                result={'kind':'runtime'}
            elif param.required:result={'kind':'required'}
        fields[param.name]=result
    records.append({'id':component.id,'role':component.role,'parameters':fields,'spec':component.model_dump(mode='json')})
assert catalog.model_dump_json()==before,'Auditing must not mutate registry defaults'
Path(args.output).write_text(json.dumps(records,ensure_ascii=False))
print(json.dumps({'components':len(records),'parameters':count,'computed_defaults':effective_changes},ensure_ascii=False))
