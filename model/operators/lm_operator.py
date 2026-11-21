from typing import Optional

from torch import nn

from model.operators.base_operator import BaseOperator


class LMOperator(BaseOperator):
    transformer: Optional[nn.Module]

    def load_transformer(self):
        from transformers import AutoModel

        return AutoModel.from_pretrained(self.transformer_key, trust_remote_code=True)

    def use_lm_cache(self):
        raise NotImplementedError

    @property
    def transformer_key(self):
        raise NotImplementedError

    @property
    def operator_name(self):
        return self.__class__.__name__.replace('Operator', '').lower()

    def get_layer_nums(self):
        raise NotImplementedError
