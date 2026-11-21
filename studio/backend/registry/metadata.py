"""Curated *semantic* metadata for Legoizer components.

This module deliberately contains **no parameter schemas for Python classes**:
those are introspected from ``config_class.__init__`` signatures and from the
real YAML files (see :mod:`studio.backend.registry.registry`).  What lives here
is the information that cannot be derived from source code:

* port types and their compatibility rules,
* human descriptions / icons / visual categories,
* parameter descriptions and safe numeric ranges,
* which blocks are merely conceptual (i.e. not executable Legommenders code).

Adding a new operator to ``model/operators/`` therefore makes it show up in the
Studio automatically; only its prose here is optional.
"""
from __future__ import annotations

from typing import Any

# --------------------------------------------------------------------------- #
# Semantic port types.  ``accepts`` lists the *output* types that may be wired
# into an *input* port of this type.  A port never matches "anything".
# --------------------------------------------------------------------------- #
PORT_DEFINITIONS: dict[str, dict[str, Any]] = {
    "Dataset": {
        "label": "Dataset",
        "color": "#3b82f6",
        "accepts": ["Dataset"],
        "description": "A configured Legommenders dataset (item/user/interaction UniTok tables).",
    },
    "ItemContent": {
        "label": "Item Content",
        "color": "#3b82f6",
        "accepts": ["ItemContent"],
        "description": "Raw item content tokens supplied by the item UniTok table.",
    },
    "ItemEmbedding": {
        "label": "Item Embedding",
        "color": "#16a34a",
        "accepts": ["ItemEmbedding", "ItemEmbeddingSequence", "CandidateItemEmbedding"],
        "description": "A single item representation produced by an Item Operator.",
    },
    "ItemEmbeddingSequence": {
        "label": "Item Embedding Sequence",
        "color": "#16a34a",
        "accepts": ["ItemEmbeddingSequence"],
        "description": "The click-history items encoded by the Item Operator.",
    },
    "CandidateItemEmbedding": {
        "label": "Candidate Item Embedding",
        "color": "#16a34a",
        "accepts": ["ItemEmbedding", "CandidateItemEmbedding", "FeatureEmbedding"],
        "description": "Candidate item representation consumed by a Click Predictor.",
    },
    "UserEmbedding": {
        "label": "User Embedding",
        "color": "#ea580c",
        "accepts": ["UserEmbedding"],
        "description": "User representation produced by a Behavior/User Operator.",
    },
    "FeatureEmbedding": {
        "label": "Feature Embedding",
        "color": "#0d9488",
        "accepts": ["FeatureEmbedding", "ItemEmbedding"],
        "description": "ID / categorical feature embedding from the EmbeddingHub (ID-based ranking).",
    },
    "PredictionScore": {
        "label": "Prediction Score",
        "color": "#7c3aed",
        "accepts": ["PredictionScore"],
        "description": "Raw user-item score returned by the Click Predictor.",
    },
    "TrainingConfig": {
        "label": "Training Config",
        "color": "#e11d48",
        "accepts": ["TrainingConfig"],
        "description": "Objective / sampling / loss settings for the run.",
    },
    "TrainingData": {
        "label": "Training Run",
        "color": "#e11d48",
        "accepts": ["TrainingData", "Dataset"],
        "description": "A completed (or configured) training run.",
    },
    "EvaluationResult": {
        "label": "Evaluation Result",
        "color": "#0891b2",
        "accepts": ["EvaluationResult", "PredictionScore", "TrainingData"],
        "description": "Metric values computed by tester.py / MetricPool.",
    },
    "MetricsConfig": {
        "label": "Metrics Config",
        "color": "#0891b2",
        "accepts": ["MetricsConfig"],
        "description": "A metric requested from the real utils.metrics.MetricPool.",
    },
}

CATEGORY_META: list[dict[str, Any]] = [
    {
        "id": "dataset",
        "label": "Dataset",
        "color": "#3b82f6",
        "icon": "Database",
        "description": "Real trainer data configurations under config/data and config/recbench.",
    },
    {
        "id": "item_encoder",
        "label": "Item Encoder",
        "color": "#16a34a",
        "icon": "FileText",
        "description": "Content/Item Operators (model.operators.*).",
    },
    {
        "id": "user_encoder",
        "label": "User Encoder",
        "color": "#ea580c",
        "icon": "Users",
        "description": "Behavior/User Operators (model.operators.*).",
    },
    {
        "id": "predictor",
        "label": "Predictor",
        "color": "#7c3aed",
        "icon": "Target",
        "description": "Click Predictors (model.predictors.*).",
    },
    {
        "id": "training",
        "label": "Training",
        "color": "#e11d48",
        "icon": "Zap",
        "description": "Objective blocks. They drive config.use_neg_sampling / the loss function.",
    },
    {
        "id": "evaluation",
        "label": "Evaluation",
        "color": "#0891b2",
        "icon": "BarChart3",
        "description": "Real metrics registered in utils.metrics.MetricPool.",
    },
]

# --------------------------------------------------------------------------- #
# Prose for the components that appear in the built-in model templates.  Keys
# are ClassHub keys (lower-cased class names without the "Operator"/"Predictor"
# suffix).  Unknown keys get a generated, honest description instead.
# --------------------------------------------------------------------------- #
COMPONENT_NOTES: dict[str, dict[str, Any]] = {
    "externalhf": {
        "name": "External Hugging Face",
        "icon": "FileText",
        "description": "Use an external Hugging Face text model, its tokenizer and pretrained token embeddings.",
        "detail": "Enter a Hugging Face repository ID and optional revisions. Train prepares the tokenizer and embeddings. Contextual mode also runs the pretrained backbone.",
        "params": {
            "model_id": {"description": "Hugging Face model repository ID, for example hf-internal-testing/tiny-random-bert."},
            "revision": {"description": "Model branch, tag, or commit hash. Train resolves and records the exact commit."},
            "tokenizer_id": {"description": "Optional tokenizer repository. Its vocabulary must match the model token IDs."},
            "tokenizer_revision": {"description": "Optional tokenizer branch, tag, or commit hash. Defaults to the model revision for the same repository, or main for another repository."},
            "encoding_mode": {"description": "Contextual runs the backbone. Token embeddings pools pretrained input vectors only."},
            "freeze_backbone": {"description": "Keep the backbone weights fixed in contextual mode."},
            "freeze_token_embeddings": {"description": "Keep the imported token embeddings fixed during training."},
        },
    },
    "cnn": {
        "name": "CNN",
        "description": "NAML-style convolutional news encoder with additive attention.",
        "detail": "model/operators/cnn_operator.py — CNNOperator. Conv1d over token embeddings followed by additive attention over each content column. Used by config/model/naml.yaml.",
        "icon": "FileText",
        "params": {
            "kernel_size": {"description": "Conv1d kernel width (config.item_config.kernel_size).", "minimum": 1, "maximum": 31},
            "dropout": {"description": "Dropout applied to the CNN activations.", "minimum": 0.0, "maximum": 1.0},
            "additive_hidden_size": {"description": "Hidden size of the additive attention layer.", "minimum": 1, "maximum": 8192},
        },
    },
    "cnncat": {
        "name": "CNN (Cat)",
        "description": "LSTUR's CNN variant: concatenates per-column representations.",
        "detail": "model/operators/cnn_cat_operator.py — CNNCatOperator. output_dim = hidden_size × number of item columns. Used by config/model/lstur.yaml.",
        "icon": "FileText",
        "params": {
            "kernel_size": {"description": "Conv1d kernel width.", "minimum": 1, "maximum": 31},
            "dropout": {"description": "Dropout applied to CNN activations.", "minimum": 0.0, "maximum": 1.0},
            "additive_hidden_size": {"description": "Hidden size of additive attention.", "minimum": 1, "maximum": 8192},
        },
    },
    "attention": {
        "name": "Attention",
        "description": "Multi-head self-attention encoder (NRMS item/user encoder).",
        "detail": "model/operators/attention_operator.py — AttentionOperator. Used by config/model/nrms.yaml.",
        "icon": "Sparkles",
        "params": {
            "num_attention_heads": {"description": "Number of attention heads.", "minimum": 1, "maximum": 64},
            "attention_dropout": {"description": "Dropout on attention weights.", "minimum": 0.0, "maximum": 1.0},
            "additive_hidden_size": {"description": "Hidden size of the final additive attention.", "minimum": 1, "maximum": 8192},
        },
    },
    "ada": {
        "name": "Additive Attention",
        "description": "NAML user encoder: additive attention over clicked news.",
        "detail": "model/operators/ada_operator.py — AdaOperator. Selected by config/model/common/operators/user-ada.yaml (meta.user: Ada).",
        "icon": "Sparkles",
        "params": {
            "additive_hidden_size": {"description": "Hidden size of the additive attention query.", "minimum": 1, "maximum": 8192},
        },
    },
    "gru": {
        "name": "GRU",
        "description": "LSTUR user encoder: GRU over the click sequence.",
        "detail": "model/operators/gru_operator.py — GRUOperator. Selected by config/model/common/operators/user-gru.yaml (meta.user: GRU).",
        "icon": "Activity",
        "params": {
            "num_layers": {"description": "Number of stacked GRU layers (config.user_config.num_layers).", "minimum": 1, "maximum": 8},
        },
    },
    "pooling": {
        "name": "Pooling",
        "description": "Mean / max pooling over a behaviour sequence; used by ID-based ranking models.",
        "detail": "model/operators/pooling_operator.py — PoolingOperator. Selected by common/operators/user-pooling.yaml and item-pooling.yaml.",
        "icon": "Layers",
        "params": {
            "flatten": {"description": "Flatten the sequence before pooling.", "advanced": True},
            "max_pooling": {"description": "Use max pooling instead of mean pooling.", "advanced": True},
        },
    },
    "transformer": {
        "name": "Transformer",
        "description": "Transformer encoder over item tokens or user behaviour.",
        "detail": "model/operators/transformer_operator.py — TransformerOperator.",
        "icon": "Boxes",
        "params": {
            "num_hidden_layers": {"description": "Number of Transformer blocks.", "minimum": 1, "maximum": 24},
        },
    },
    "fastformer": {
        "name": "Fastformer",
        "description": "Additive-attention Fastformer encoder (item or user side).",
        "detail": "model/operators/fastformer_operator.py — FastformerOperator. Used by config/model/fastformer.yaml.",
        "icon": "Zap",
        "params": {
            "num_hidden_layers": {"description": "Number of Fastformer layers.", "minimum": 1, "maximum": 24},
            "num_attention_heads": {"description": "Number of attention heads.", "minimum": 1, "maximum": 64},
            "hidden_dropout_prob": {"description": "Hidden dropout probability.", "minimum": 0.0, "maximum": 1.0},
        },
    },
    "polyattention": {
        "name": "PolyAttention",
        "description": "MINER user encoder with multiple context codes.",
        "detail": "model/operators/poly_attention_operator.py — PolyAttentionOperator.",
        "icon": "GitBranch",
        "params": {
            "num_context_codes": {"description": "Number of global context codes.", "minimum": 1, "maximum": 1024},
            "context_code_dim": {"description": "Dimension of each context code.", "minimum": 1, "maximum": 8192},
        },
    },
    "bertbase": {
        "name": "BERT",
        "description": "Pre-trained BERT item encoder (requires cached BERT embeddings).",
        "detail": "model/operators/bert_operator.py — BertBaseOperator. Requires config/embed/bertbase.yaml and data/embeddings/bertbase.npy.",
        "icon": "Brain",
        "params": {
            "tune_from": {"description": "Fine-tune from layer N (0 = train all).", "minimum": -100, "maximum": 100},
            "use_lora": {"description": "Enable LoRA adapters (peft).", "advanced": True},
            "lora_r": {"description": "LoRA rank.", "minimum": 1, "maximum": 256, "advanced": True},
            "lora_alpha": {"description": "LoRA alpha.", "minimum": 1, "maximum": 1024, "advanced": True},
            "lora_dropout": {"description": "LoRA dropout.", "minimum": 0.0, "maximum": 1.0, "advanced": True},
        },
    },
    "llama1": {
        "name": "Llama",
        "description": "Llama-based item encoder (ONCE / IISAN family).",
        "detail": "model/operators/llama_operator.py — requires .model entries and cached Llama embeddings.",
        "icon": "Brain",
        "params": {
            "tune_from": {"description": "Fine-tune from layer N.", "minimum": -100, "maximum": 100},
            "use_lora": {"description": "Enable LoRA adapters.", "advanced": True},
            "lora_r": {"description": "LoRA rank.", "minimum": 1, "maximum": 256, "advanced": True},
            "lora_alpha": {"description": "LoRA alpha.", "minimum": 1, "maximum": 1024, "advanced": True},
        },
    },
    "nullconcat": {
        "name": "Null Concat",
        "description": "DIN user operator: passes ID features through without content encoding.",
        "detail": "model/operators/null_operator.py — NullConcatOperator. meta.user: NullConcat in config/model/din_id.yaml.",
        "icon": "Minus",
        "params": {},
    },
    "dot": {
        "name": "Dot Product",
        "description": "Dot-product click predictor with negative sampling (matching).",
        "detail": "model/predictors/dot_predictor.py — DotPredictor. common/predictors/dot.yaml sets use_neg_sampling: true, neg_count: 4.",
        "icon": "CircleDot",
        "params": {},
    },
    "dnn": {
        "name": "MLP / DNN",
        "description": "Deep neural network click predictor.",
        "detail": "model/predictors/dnn_predictor.py — DNNPredictor.",
        "icon": "Network",
        "params": {
            "dnn_hidden_units": {"description": "Hidden units of the DNN (YAML list).", "advanced": True},
            "dnn_activations": {"description": "Activation function name.", "advanced": True},
            "dnn_dropout": {"description": "Dropout rate.", "minimum": 0.0, "maximum": 1.0, "advanced": True},
            "dnn_batch_norm": {"description": "Apply batch normalisation.", "advanced": True},
        },
    },
    "pnn": {"name": "PNN", "description": "Product-based Neural Network predictor.", "detail": "model/predictors/pnn_predictor.py — config/model/pnn_id.yaml.", "icon": "Network", "params": {}},
    "deepfm": {"name": "DeepFM", "description": "Factorization-Machine + DNN ranking predictor.", "detail": "model/predictors/deepfm_predictor.py — config/model/deepfm_id.yaml.", "icon": "Network", "params": {}},
    "dcn": {"name": "DCN", "description": "Deep & Cross Network predictor.", "detail": "model/predictors/dcn_predictor.py — config/model/dcn_id.yaml.", "icon": "Network", "params": {}},
    "dcnv2": {"name": "DCNv2", "description": "Improved Deep & Cross Network (matrix/low-rank variants).", "detail": "model/predictors/dcnv2_predictor.py — config/model/dcnv2_id.yaml.", "icon": "Network", "params": {}},
    "din": {"name": "DIN", "description": "Deep Interest Network (ranking only, attention over history).", "detail": "model/predictors/din_predictor.py — config/model/din_id.yaml. allow_matching=False.", "icon": "Network", "params": {}},
    "autoint": {"name": "AutoInt", "description": "Automatic feature interaction via self-attention.", "detail": "model/predictors/autoint_predictor.py — config/model/autoint_id.yaml.", "icon": "Network", "params": {}},
    "masknet": {"name": "MaskNet", "description": "MaskNet predictor with instance-guided masks.", "detail": "model/predictors/mask_net_predictor.py — config/model/masknet_id.yaml.", "icon": "Network", "params": {}},
    "gdcn": {"name": "GDCN", "description": "Gated Deep & Cross Network predictor.", "detail": "model/predictors/gdcn_predictor.py — config/model/gdcn_id.yaml.", "icon": "Network", "params": {}},
    "finalmlp": {"name": "FinalMLP", "description": "Two-stream MLP with feature selection.", "detail": "model/predictors/final_mlp_predictor.py — config/model/finalmlp_id.yaml.", "icon": "Network", "params": {}},
    "miner": {"name": "MINER", "description": "Matching-only predictor with poly attention.", "detail": "model/predictors/miner_predictor.py — config/model/miner.yaml. allow_ranking=False.", "icon": "CircleDot", "params": {}},
}

COMMON_PARAM_NOTES: dict[str, dict[str, Any]] = {
    "hidden_size": {"description": "Global hidden dimension (LegoConfig.hidden_size). Set in Training settings.", "minimum": 8, "maximum": 8192},
    "item_hidden_size": {"description": "Input embedding dimension of the Item Operator (LegoConfig.item_hidden_size).", "minimum": 8, "maximum": 16384},
    "item_page_size": {"description": "Page size for LM item encoding (LegoConfig.item_page_size). 0 disables paging.", "minimum": 0, "maximum": 4096},
    "use_fast_eval": {"description": "Enable cached fast evaluation (LegoConfig.use_fast_eval).", "advanced": True},
    "neg_count": {"description": "Number of negative samples for matching (LegoConfig.neg_count).", "minimum": 1, "maximum": 100},
    "inputer_config.use_cls_token": {"description": "Prepend a CLS token in the inputer.", "advanced": True},
    "inputer_config.use_sep_token": {"description": "Append a SEP token in the inputer.", "advanced": True},
    "dnn_hidden_units": {"description": "Hidden units of the predictor DNN (YAML list, e.g. [1000, 1000, 1000]).", "advanced": True},
    "dnn_activations": {"description": "Predictor activation (ReLU / etc.).", "advanced": True},
    "dnn_dropout": {"description": "Predictor dropout.", "minimum": 0.0, "maximum": 1.0, "advanced": True},
    "dnn_batch_norm": {"description": "Use batch normalisation in the predictor.", "advanced": True},
    "cross_num": {"description": "Number of cross layers.", "minimum": 1, "maximum": 20, "advanced": True},
    "num_attention_layers": {"description": "Number of interacting layers (AutoInt).", "minimum": 1, "maximum": 20, "advanced": True},
    "num_attention_heads": {"description": "Number of attention heads.", "minimum": 1, "maximum": 64, "advanced": True},
    "attention_dim": {"description": "Attention embedding dimension.", "minimum": 1, "maximum": 4096, "advanced": True},
    "attention_dropout": {"description": "Attention dropout.", "minimum": 0.0, "maximum": 1.0, "advanced": True},
    "use_residual": {"description": "Use residual connections.", "advanced": True},
    "use_wide": {"description": "Use the wide (linear) part.", "advanced": True},
    "mlp1_hidden_units": {"description": "Hidden units of the first MLP stream.", "advanced": True},
    "mlp2_hidden_units": {"description": "Hidden units of the second MLP stream.", "advanced": True},
    "hidden_units": {"description": "Hidden units of MaskNet blocks.", "advanced": True},
    "attention_hidden_units": {"description": "Hidden units of the DIN attention net.", "advanced": True},
    "din_use_softmax": {"description": "Apply softmax to DIN attention weights.", "advanced": True},
    "model_structure": {"description": "DCNv2 structure: parallel / stacked / stacked_parallel.", "advanced": True, "choices": ["parallel", "stacked", "stacked_parallel"]},
    "use_low_rank_mixture": {"description": "Use low-rank mixture of experts in DCNv2.", "advanced": True},
    "num_context_codes": {"description": "Number of context codes.", "minimum": 1, "maximum": 1024},
    "context_code_dim": {"description": "Dimension of context codes.", "minimum": 1, "maximum": 8192},
    "num_hidden_layers": {"description": "Number of stacked layers.", "minimum": 1, "maximum": 24},
    "num_layers": {"description": "Number of stacked RNN layers.", "minimum": 1, "maximum": 8},
    "kernel_size": {"description": "CNN kernel width.", "minimum": 1, "maximum": 31},
    "dropout": {"description": "Dropout rate.", "minimum": 0.0, "maximum": 1.0},
    "additive_hidden_size": {"description": "Hidden size of additive attention.", "minimum": 1, "maximum": 8192},
}

# --------------------------------------------------------------------------- #
# Training objective blocks.  These ARE executable: they map onto
# config.use_neg_sampling and the loss chosen in model/legommender.py.
# --------------------------------------------------------------------------- #
TRAINING_COMPONENTS: list[dict[str, Any]] = [
    {
        "key": "matching_task",
        "name": "Matching Task",
        "description": "Negative-sampling training with CrossEntropyLoss.",
        "detail": "Sets config.use_neg_sampling: true. model/legommender.py selects nn.CrossEntropyLoss over B×(K+1) scores. Requires a predictor with allow_matching=True.",
        "task": "matching",
    },
    {
        "key": "ranking_task",
        "name": "Ranking Task",
        "description": "Point-wise ranking with BCEWithLogitsLoss.",
        "detail": "Sets config.use_neg_sampling: false. model/legommender.py selects nn.BCEWithLogitsLoss. Requires a predictor with allow_ranking=True.",
        "task": "ranking",
    },

]

METRIC_NOTES: dict[str, str] = {
    "GAUC": "Grouped AUC (per-user AUC averaged). Default dev metric in config/exp/default.yaml.",
    "AUC": "Area under the ROC curve (point-wise).",
    "MRR": "Mean reciprocal rank. NOTE: this repository's MRR follows common RecSys implementations, not the original definition (use MRR0 instead).",
    "MRR0": "Original mean reciprocal rank definition.",
    "NDCG@1": "Normalised discounted cumulative gain at 1.",
    "NDCG@5": "Normalised discounted cumulative gain at 5.",
    "NDCG@10": "Normalised discounted cumulative gain at 10.",
    "Recall@K": "Recall at K, e.g. Recall@10. Implemented by utils.metrics.Recall.",
    "LogLoss": "Binary log loss (lower is better).",
    "HitRatio@K": "Hit ratio at K, e.g. HitRatio@10.",
    "LRAP": "Label ranking average precision.",
    "F1": "F1 at a configurable threshold, e.g. F1@0.5.",
}

DATASET_LABELS: dict[str, str] = {
    "mind": "MIND (small)",
    "mind-lm": "MIND (LM)",
    "mind-lm-prompt": "MIND (LM prompts)",
    "oncemind": "ONCE-MIND",
    "mindrb-with-id": "MIND RecBench + ID",
    "mindrb-dual-emb": "MIND RecBench dual embedding",
    "mindrb": "MIND (RecBench)",
    "xmind": "xMIND",
    "movielensrb": "MovieLens (RecBench bridge)",
    "microlensrb": "MicroLens (RecBench)",
    "netflixrb": "Netflix (RecBench)",
    "booksrb": "Amazon Books (RecBench)",
    "automotiverb": "Amazon Automotive (RecBench)",
    "cdsrb": "Amazon CDs (RecBench)",
    "yelprb": "Yelp (RecBench)",
    "goodreadsrb": "Goodreads (RecBench)",
    "pensrb": "PENS (RecBench)",
    "ebnerdrb": "EB-NeRD (RecBench)",
    "hmrb": "H&M (RecBench)",
    "lastfmrb": "LastFM (RecBench)",
    "pogrb": "POG (RecBench)",
}

# Model archetypes taken from README.md's model table.  ``components`` uses
# registry component ids.  ``reference`` is the real YAML in the repository.
MODEL_ARCHETYPES: list[dict[str, Any]] = [
    {"key": "naml", "name": "NAML", "type": "matching", "components": ("item.cnn", "user.ada", "predictor.dot"), "reference": "config/model/naml.yaml"},
    {"key": "nrms", "name": "NRMS", "type": "matching", "components": ("item.attention", "user.attention", "predictor.dot"), "reference": "config/model/nrms.yaml"},
    {"key": "lstur", "name": "LSTUR", "type": "matching", "components": ("item.cnn", "user.gru", "predictor.dot"), "reference": "config/model/lstur.yaml", "exact": False},
    {"key": "lstur_cat", "name": "LSTUR", "type": "matching", "components": ("item.cnncat", "user.gru", "predictor.dot"), "reference": "config/model/lstur.yaml", "exact": True},
    {"key": "fastformer", "name": "Fastformer", "type": "matching", "components": ("item.fastformer", "user.fastformer", "predictor.dot"), "reference": "config/model/fastformer.yaml", "exact": False},
    {"key": "miner", "name": "MINER", "type": "matching", "components": ("item.transformer", "user.polyattention", "predictor.miner"), "reference": "config/model/miner.yaml"},
    {"key": "pnn", "name": "PNN", "type": "ranking", "components": ("user.pooling", "predictor.pnn"), "reference": "config/model/pnn_id.yaml"},
    {"key": "deepfm", "name": "DeepFM", "type": "ranking", "components": ("user.pooling", "predictor.deepfm"), "reference": "config/model/deepfm_id.yaml"},
    {"key": "dcn", "name": "DCN", "type": "ranking", "components": ("user.pooling", "predictor.dcn"), "reference": "config/model/dcn_id.yaml"},
    {"key": "din", "name": "DIN", "type": "ranking", "components": ("user.nullconcat", "predictor.din"), "reference": "config/model/din_id.yaml"},
    {"key": "autoint", "name": "AutoInt", "type": "ranking", "components": ("user.pooling", "predictor.autoint"), "reference": "config/model/autoint_id.yaml"},
    {"key": "dcnv2", "name": "DCNv2", "type": "ranking", "components": ("user.pooling", "predictor.dcnv2"), "reference": "config/model/dcnv2_id.yaml"},
    {"key": "masknet", "name": "MaskNet", "type": "ranking", "components": ("user.pooling", "predictor.masknet"), "reference": "config/model/masknet_id.yaml"},
    {"key": "gdcn", "name": "GDCN", "type": "ranking", "components": ("user.pooling", "predictor.gdcn"), "reference": "config/model/gdcn_id.yaml"},
    {"key": "finalmlp", "name": "FinalMLP", "type": "ranking", "components": ("user.pooling", "predictor.finalmlp"), "reference": "config/model/finalmlp_id.yaml"},
]
