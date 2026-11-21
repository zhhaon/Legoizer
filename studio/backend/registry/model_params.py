"""Constraints from the model operators and the layers they construct."""

from studio.backend.schemas.components import ParamSpec

# Only use these layer counts for the repository's stock checkpoint aliases.
STOCK_CHECKPOINTS = {
    "bertbase": ("bert-base-uncased", 12),
    "bertlarge": ("bert-large-uncased", 24),
    "optbase": ("facebook/opt-350m", 24),
    "optlarge": ("facebook/opt-1.3b", 24),
    "llama1": ("huggyllama/llama-7b", 32),
    "llama2": ("meta-llama/Llama-2-7b-hf", 32),
    "llama3": ("meta-llama/Meta-Llama-3-8B", 32),
    "glm4th9b": ("THUDM/glm-4-9b", 40),
}

# Parameters retained in older YAMLs or base configs but unused by these implementations.
UNUSED_PARAMETERS = {
    "transformer": {"additive_hidden_size"},
    "flattentransformer": {"additive_hidden_size"},
    "fastformer": {"additive_hidden_size", "attention_dropout"},
    "flattenfastformer": {"additive_hidden_size", "attention_dropout"},
    "finalmlp": {"use_fs", "fs1_context", "fs2_context", "fs_hidden_units", "embedding_regularizer", "net_regularizer"},
    "din": {"embedding_regularizer", "net_regularizer"},
}

POSITIVE_INTS = {
    "kernel_size", "num_layers", "num_attention_heads", "num_heads", "attention_dim",
    "additive_hidden_size", "context_code_dim", "num_context_codes", "num_attention_layers",
    "num_blocks", "num_experts", "block_dim", "low_rank",
    "lora_r", "layer_selection_step", "global_proj_size", "local_proj_size",
}
LAYER_DROPOUT = {"dnn_dropout", "net_dropout", "mlp1_dropout", "mlp2_dropout"}
LAYER_ACTIVATIONS = {"dnn_activations", "activations", "mlp1_hidden_activations", "mlp2_hidden_activations"}
ACTIVATIONS = {
    "ReLU", "Sigmoid", "Tanh", "ReLU6", "LeakyReLU", "ELU", "CELU", "SELU", "GELU",
    "PReLU", "RReLU", "SiLU", "Mish", "Softplus", "Softsign", "Tanhshrink", "Hardtanh",
    "Hardsigmoid", "Hardswish", "Hardshrink", "Softshrink", "LogSigmoid", "Identity",
}
CHOICES = {
    "model_structure": ["crossnet_only", "stacked", "parallel", "stacked_parallel"],
    "score_type": ["weighted", "max", "mean"],
}


def checkpoint_layers(checkpoint: str | None) -> int | None:
    from utils.config_init import ModelInit

    if checkpoint not in STOCK_CHECKPOINTS:
        return None
    repo_id, layers = STOCK_CHECKPOINTS[checkpoint]
    return layers if ModelInit.get(checkpoint, default=repo_id) == repo_id else None


def apply_model_constraints(spec: ParamSpec, key: str, checkpoint: str | None) -> None:
    name = spec.name
    # Model parameters have no Studio-imposed upper limit unless the implementation does.
    spec.minimum = spec.maximum = None
    if name in POSITIVE_INTS:
        spec.type = "integer"
        spec.minimum = 1
    if name == "reduction_ratio":
        spec.type = "number"
        spec.minimum = 0
    if name in {"cross_num", "num_hidden_layers"}:
        spec.type = "integer"
        spec.minimum = 0
    if name.endswith("_hidden_units") or name == "hidden_units":
        spec.type = "list"
    if name.endswith("_config"):
        spec.type = "object"
    if name in CHOICES:
        spec.type = "enum"
        spec.choices = CHOICES[name]
    if name in LAYER_ACTIVATIONS:
        spec.type = "string_or_list"
    if "dropout" in name:
        per_layer = name in LAYER_DROPOUT or (name == "dropout" and key == "masknet") or (
            name == "attention_dropout" and key == "din"
        )
        spec.type = "number_or_list" if per_layer else "number"
        spec.minimum, spec.maximum = 0, 1

    if name == "tune_from":
        spec.description = "0 or null trains all layers. A positive index uses cached states; negative indices count back from the end."
    layers = checkpoint_layers(checkpoint) if name in {"tune_from", "layer_selection_step"} else None
    if layers is not None:
        if name == "tune_from":
            spec.minimum = -layers
        spec.maximum = layers

    if spec.default is None and spec.type == "list":
        if name == "attention_hidden_units":
            spec.default_hint = "[hidden_size]"
        elif name in {"stacked_dnn_hidden_units", "parallel_dnn_hidden_units", "mlp1_hidden_units", "mlp2_hidden_units"}:
            spec.default_hint = "Three layers of hidden_size"
        elif name == "dnn_hidden_units" and key == "din":
            spec.default_hint = "[8 × hidden_size, 2 × hidden_size, hidden_size]"
