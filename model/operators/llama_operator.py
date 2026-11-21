import abc

import torch
from transformers import LlamaModel

from model.operators.once_operator import OnceOperator


class LlamaOperator(OnceOperator, abc.ABC):
    dtype = torch.bfloat16

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.transformer: LlamaModel
        self.transformer.embed_tokens = None

        if self.transformer.config.hidden_size != self.config.input_dim:
            raise ValueError(f'In {self.classname}, hidden_size of transformer ({self.transformer.config.hidden_size}) '
                             f'does not match input_dim ({self.config.input_dim})')

        self._prepare_network()

    def _slice_transformer_layers(self):
        self.transformer.layers = self.transformer.layers[self.config.tune_from + 1:]

    def _loop_forward(
            self,
            hidden_states: torch.FloatTensor,
            attention_mask: torch.Tensor,
    ):
        # LLaMA applies rotary positions in its decoder, so cached layer
        # states can resume through the sliced model's public forward method.
        return self.transformer(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state



class Llama1Operator(LlamaOperator):
    pass


class Llama2Operator(LlamaOperator):
    pass


class Llama3Operator(LlamaOperator):
    pass
