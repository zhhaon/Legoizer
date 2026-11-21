"""External encoders accept the CPU masks supplied by ConcatInputer."""

from types import SimpleNamespace

import pytest
import torch
from transformers import BertConfig, BertModel

from model.operators.external_hf_operator import ExternalHFOperator


DEVICES = ["cpu"] + [f"cuda:{index}" for index in range(torch.cuda.device_count())]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", ["token_embeddings", "contextual"])
def test_external_hf_aligns_masks_with_embeddings(device, mode):
    operator = ExternalHFOperator.__new__(ExternalHFOperator)
    torch.nn.Module.__init__(operator)
    operator.config = SimpleNamespace(context_limit=3, freeze_backbone=False)
    operator.backbone = None if mode == "token_embeddings" else BertModel(BertConfig(
        hidden_size=8, intermediate_size=16, num_hidden_layers=1,
        num_attention_heads=2, vocab_size=16, max_position_embeddings=16,
    ))
    operator.projection = torch.nn.Linear(8, 4)
    operator.to(device)
    embeddings = torch.randn(2, 4, 8, device=device, requires_grad=True)
    mask = torch.tensor([[1, 1, 0, 0], [0, 0, 0, 0]], device="cpu")
    result = operator(embeddings, mask=mask)
    assert result.device == embeddings.device
    assert result.shape == (2, 4)
    assert torch.isfinite(result).all()
    result.sum().backward()
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()
