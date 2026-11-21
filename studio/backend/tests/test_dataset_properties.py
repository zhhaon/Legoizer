"""Dataset attributes stay independent of tokenizer selection."""
import pytest
from studio.backend.schemas.graph import GraphJson
from studio.backend.dataset_inputs import dataset_settings, resolve_dataset_inputs


def graph(lm='glove', item=None, inputs=None):
    nodes = [{'id': 'data', 'component': 'dataset.mind-lm-prompt', 'params': {'lm': lm}}]
    if inputs is not None:
        nodes[0]['params']['item_inputs'] = inputs
    if item:
        nodes.append({'id': 'encoder', 'component': item, 'params': {}})
    return GraphJson.model_validate({'nodes': nodes, 'edges': []})


def test_prompt_defaults_and_all_attributes():
    result = resolve_dataset_inputs(graph())
    assert result['defaults'] == [
        {'column': 'prompt@glove', 'truncate': None},
        {'column': 'prompt_title@glove', 'truncate': None},
        {'column': 'title@glove', 'truncate': 30},
        {'column': 'prompt_category@glove', 'truncate': None},
        {'column': 'category@glove', 'truncate': None},
    ]
    for field in ('prompt', 'prompt_title', 'title', 'abstract', 'prompt_category', 'category', 'prompt_subcategory', 'subcategory'):
        assert field + '@glove' in result['columns']


@pytest.mark.parametrize('lm,config', [('glove', 'glove'), ('bert', 'bertbase'), ('llama2', 'llama'), ('optlarge', 'opt'), ('glm', 'glm')])
def test_embedding_alignment(lm, config):
    assert dataset_settings(graph(lm))[1] == f'config/embed/{config}.yaml'
    result = resolve_dataset_inputs(graph(lm, inputs=[{'column': 'abstract@glove', 'truncate': 99}]))
    assert result['values'] == [{f'abstract@{lm}': 99}]


def test_encoder_owns_resources():
    g = graph('glove', 'item.bertbase', [{'column': 'prompt_subcategory', 'truncate': 12}])
    assert dataset_settings(g)[2] == 'bert'
    assert resolve_dataset_inputs(g)['values'] == [{'prompt_subcategory@bert': 12}]
