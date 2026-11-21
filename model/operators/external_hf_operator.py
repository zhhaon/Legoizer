"""Use a Transformers text checkpoint as a Legommenders item encoder."""

import torch
from torch import nn

from model.inputer.concat_inputer import ConcatInputer
from model.operators.base_operator import BaseOperator, BaseOperatorConfig


class ExternalHFOperatorConfig(BaseOperatorConfig):
    def __init__(
        self,
        model_id: str,
        revision: str = "main",
        tokenizer_id: str = "",
        tokenizer_revision: str = "",
        encoding_mode: str = "contextual",
        freeze_backbone: bool = True,
        freeze_token_embeddings: bool = True,
        **kwargs,
    ):
        self.checkpoint_dir = kwargs.pop("checkpoint_dir", None)
        self.context_width = kwargs.pop("context_width", None)
        self.context_limit = kwargs.pop("context_limit", None)
        # UniTok stores the checkpoint's actual special token IDs with each text field.
        kwargs["inputer_config"] = {"use_cls_token": False, "use_sep_token": False}
        super().__init__(**kwargs)
        self.model_id = model_id
        self.revision = revision
        self.tokenizer_id = tokenizer_id
        self.tokenizer_revision = tokenizer_revision
        self.encoding_mode = encoding_mode
        self.freeze_backbone = freeze_backbone
        self.freeze_token_embeddings = freeze_token_embeddings


class ExternalHFOperator(BaseOperator):
    config_class = ExternalHFOperatorConfig
    inputer_class = ConcatInputer

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.backbone = None
        config = self.config
        width = config.input_dim
        if config.encoding_mode == "contextual":
            if not config.checkpoint_dir or config.context_width is None:
                raise ValueError("Prepare the external Hugging Face checkpoint before training.")
            from transformers import AutoModel

            self.backbone = AutoModel.from_pretrained(
                config.checkpoint_dir, local_files_only=True,
                trust_remote_code=False, use_safetensors=True,
            )
            if hasattr(self.backbone.config, "use_cache"):
                self.backbone.config.use_cache = False
            self.backbone.requires_grad_(not config.freeze_backbone)
            # The recommendation inputer supplies the separate, optionally trainable token embeddings.
            self.backbone.get_input_embeddings().requires_grad_(False)
            width = config.context_width
        self.projection = nn.Linear(width, config.hidden_size) if width != config.hidden_size else nn.Identity()

    def forward(self, embeddings, mask=None, **kwargs):
        config = self.config
        # ConcatInputer returns a CPU mask while token embeddings follow the selected device.
        mask = mask.to(device=embeddings.device)
        output_dtype = embeddings.dtype
        if self.backbone is not None:
            if config.context_limit is not None:
                embeddings = embeddings[:, :config.context_limit]
                mask = mask[:, :config.context_limit]
            if config.freeze_backbone:
                self.backbone.eval()
            attention_mask = mask.clone()
            attention_mask[:, 0] |= ~mask.bool().any(dim=1)
            token_dtype = self.backbone.get_input_embeddings().weight.dtype
            embeddings = self.backbone(
                inputs_embeds=embeddings.to(token_dtype), attention_mask=attention_mask,
                return_dict=True,
            ).last_hidden_state
        weights = mask.unsqueeze(-1).to(embeddings.dtype)
        pooled = (embeddings * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        return self.projection(pooled.to(output_dtype))
