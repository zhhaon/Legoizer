"""Exercise cached built-in language operators without downloading checkpoints."""

from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("family", ["bert", "llama", "opt"])
def test_cached_layers_resume_the_pretrained_forward(family):
    from model.operators.bert_operator import BertBaseOperator
    from model.operators.llama_operator import Llama1Operator
    from model.operators.opt_operator import OPTBaseOperator
    from transformers import AutoConfig, AutoModel

    classes = {"bert": BertBaseOperator, "llama": Llama1Operator, "opt": OPTBaseOperator}
    config = AutoConfig.for_model(
        family, hidden_size=16, num_hidden_layers=3, num_attention_heads=2,
        vocab_size=24, max_position_embeddings=32,
        **({"word_embed_proj_dim": 12, "ffn_dim": 24, "layerdrop": 0.0} if family == "opt" else {}),
    )
    model = AutoModel.from_config(config).eval()
    input_ids = torch.tensor([[2, 3, 4, 0], [5, 6, 7, 8]])
    mask = input_ids.ne(0).long()
    with torch.no_grad():
        full = model(input_ids=input_ids, attention_mask=mask, output_hidden_states=True, return_dict=True)
        operator = classes[family].__new__(classes[family])
        torch.nn.Module.__init__(operator)
        operator.transformer = model
        operator.config = SimpleNamespace(tune_from=1)
        operator._slice_transformer_layers()
        resumed = operator._loop_forward(full.hidden_states[2], mask)
    torch.testing.assert_close(resumed, full.last_hidden_state)
