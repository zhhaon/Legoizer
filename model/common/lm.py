from torch import nn
from transformers import AutoConfig, AutoModel


class BaseLM(nn.Module):
    def __init__(self, key):
        super().__init__()

        self.transformer = AutoModel.from_pretrained(key, trust_remote_code=True)

    def get_layer_nums(self):
        return self.transformer.config.num_hidden_layers


def legacy_chatglm_config(checkpoint):
    config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
    # The repository's custom model reads the old generation default during init.
    if not hasattr(config, "max_length"):
        config.max_length = config.seq_length
    if not hasattr(config, "use_cache"):
        config.use_cache = False
    return config


def load_legacy_chatglm(checkpoint, auto_class=AutoModel):
    return auto_class.from_pretrained(
        checkpoint, config=legacy_chatglm_config(checkpoint), trust_remote_code=True
    )
