"""Resource identities shared by graph compilation and the preparation job."""

import re
from pathlib import Path

# The dimension is the checkpoint's token embedding width.
LM_MODELS = {
    "bertbase": ("bert", 768),
    "bertlarge": ("bert", 1024),
    "llama1": ("llama1", 4096),
    "llama2": ("llama2", 4096),
    "llama3": ("llama3", 4096),
    "optbase": ("optbase", 512),
    "optlarge": ("optlarge", 2048),
    "glm4th9b": ("glm", 4096),
}
TOKENIZERS = ("glove", "bert", "llama1", "llama2", "llama3", "optbase", "optlarge", "glm")
XMIND_LANGUAGES = ("cmn", "fin", "grn", "hat", "ind", "jpn", "ron", "som", "swh", "tam", "tha", "tur", "vie")
LM_BASE_CLASSES = {"lm", "once", "iisan", "bert", "llama", "glm", "opt",
                   "bertiisan", "llamaiisan", "glmiisan", "optiisan"}


def item_checkpoint(meta_item: str) -> str | None:
    name = meta_item.lower().removesuffix("iisan")
    return name if name in LM_MODELS else None


def dataset_processor(data_config: str) -> str:
    path = Path(data_config)
    if path.parent.name == "recbench":
        return path.stem.split("-")[0] + "rb"
    if path.stem.startswith("mindrb"):
        return "mindrb"
    if path.stem.startswith("mind-lm"):
        return "mind"
    return path.stem


def resource_args(data_config: str, lm: str | None, lang: str | None = None) -> dict[str, str]:
    lm = {"opt": "optbase"}.get(lm, lm) or "glove"
    if lm not in TOKENIZERS:
        raise ValueError(f"Unknown tokenizer: {lm}")
    lang = lang or "cmn"
    if lang not in XMIND_LANGUAGES:
        raise ValueError(f"Unsupported xMIND language: {lang}")
    processor = dataset_processor(data_config)
    return {
        "lm": lm,
        "lang": lang,
        "rbname": processor if processor.endswith("rb") else "mindrb",
        "llama": lm[-1] if lm.startswith("llama") else "1",
        "opt": lm.removeprefix("opt") if lm.startswith("opt") else "base",
    }


EXTERNAL_HF_COMPONENT = "item.externalhf"
EXTERNAL_HF_VOCAB = "external_hf"
EXTERNAL_HF_MODES = ("contextual", "token_embeddings")


def validate_hf_source(repo_id: str, revision: str) -> None:
    from huggingface_hub.utils import validate_repo_id

    if not isinstance(repo_id, str) or not repo_id:
        raise ValueError("Enter a Hugging Face model repository ID.")
    validate_repo_id(repo_id)
    if (not isinstance(revision, str) or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_./-]{0,127}", revision)
            or any(part in {"", ".", ".."} for part in revision.split("/"))):
        raise ValueError("Enter a Hugging Face branch, tag, or commit hash.")


def external_hf_sources(config: dict) -> tuple[str, str, str, str]:
    model = config.get("model_id", "")
    revision = config.get("revision", "main")
    tokenizer = config.get("tokenizer_id") or model
    tokenizer_revision = config.get("tokenizer_revision") or (revision if tokenizer == model else "main")
    validate_hf_source(model, revision)
    validate_hf_source(tokenizer, tokenizer_revision)
    return model, revision, tokenizer, tokenizer_revision
