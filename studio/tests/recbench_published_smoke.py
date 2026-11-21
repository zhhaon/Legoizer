"""Optional real-data CPU training-step check; uses published splits unchanged.
Run: PYTHONPATH=. python studio/tests/recbench_published_smoke.py books
"""
import json,yaml,torch,sys
domain=sys.argv[1] if len(sys.argv)>1 else "books"
processor=domain+"rb"
content="--content" in sys.argv
torch.set_num_threads(2)
from pathlib import Path
from studio.backend.executor.provision import processed_data
from studio.backend.compiler.compiler import build_exp_config
from studio.backend.schemas.graph import GraphJson
from trainer import Trainer,get_configurations
from loader.env import Env
from loader.symbols import Symbols
folder=Path('studio/backend/workspace/generated') / ('published-smoke-'+domain+('-content' if content else '')); folder.mkdir(exist_ok=True)
data=yaml.safe_load(Path('config/recbench/base.yaml').read_text());data['name']=processor+'-published-smoke'
raw_config = yaml.safe_load(Path(f'config/recbench/{domain}.yaml').read_text())
content_inputs = raw_config.get('item', {}).get('inputs', [{'title@${lm}': 50}])
content_inputs = [({key.replace('${lm}', 'glove'): value for key, value in entry.items()} if isinstance(entry, dict) else entry.replace('${lm}', 'glove')) for entry in content_inputs]
base='data/recbench/'+processor; data['item']={'ut':base+'/items','inputs': content_inputs if content else ['item_id']}; data['user']={'ut':base+'/users','truncate':50}; data['inter'].update(train=base+'/train',dev=base+'/valid',test=base+'/test')
processed_data(data,processor,'cmn')
# The saved interactions must be identical to the official exports, not resplit.
import pandas as pd
from studio.backend.executor.provision import DOWNLOADS
for split in ('train', 'valid', 'test'):
    expected = pd.read_parquet(DOWNLOADS / 'legommenders' / processor / f'{split}.parquet')
    actual = pd.read_parquet(Path(data['item']['ut']).parent / f'{split}.parquet')
    pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected.reset_index(drop=True))
(folder/'data.yaml').write_text(yaml.safe_dump(data))
g=GraphJson(graph_id='smoke',name='smoke',nodes=[],edges=[],task='ranking')
g.train.device='cpu';g.train.hidden_size=16;g.train.batch_size=8
exp=build_exp_config(g);exp['policy']['disable_tqdm']=True
(folder/'exp.yaml').write_text(yaml.safe_dump(exp))
cfg=get_configurations(dict(data=str(folder/'data.yaml'),model='config/model/naml.yaml' if content else 'config/model/din_id.yaml',embed='config/embed/glove.yaml' if content else 'config/embed/null.yaml',exp=str(folder/'exp.yaml'),hidden_size=16,batch_size=8,lr=0.001,cuda='-1'))
torch.set_num_threads(2)
t=Trainer(cfg);t.init_optimizer(); t.manager.setup(Symbols.train);t.legommender.train()
batch=next(iter(t.manager.get_train_loader()));loss=t.legommender(batch=batch)
assert torch.isfinite(loss)
loss.backward();t.optimizer.step()
print('TRAIN_STEP_OK '+domain,loss.item(),flush=True)
