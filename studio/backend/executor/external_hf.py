"""Prepare a Hub text model and its aligned token embeddings for one run."""

import hashlib
import inspect
import json
from pathlib import Path

from unitok import BaseTokenizer


class ExternalHFTokenizer(BaseTokenizer):
    """Persist the token ID vocabulary; prepared tables already hold encoded text."""

    return_list = True
    param_list = []

    def __init__(self, vocab, tokenizer, rows, max_length):
        super().__init__(vocab=vocab, tokenizer_id=vocab)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.vocab.extend(range(rows))

    def __call__(self, text):
        token_ids = self.tokenizer(str(text), truncation=True, max_length=self.max_length)["input_ids"]
        for token in token_ids:
            self.vocab.counter(token)
        return token_ids


def external_record(config):
    from studio.backend.executor.provision import CACHE
    from studio.backend.resource_catalog import external_hf_sources
    identity = [*external_hf_sources(config), config.get("encoding_mode", "contextual")]
    return CACHE / "external-models" / (hashlib.sha256(json.dumps(identity).encode()).hexdigest() + ".json")


def cached_external(config):
    from studio.backend.config import REPO_ROOT
    try:
        record = json.loads(external_record(config).read_text())
        if not (REPO_ROOT / record["embedding"]["path"]).is_file(): return None
        if not (REPO_ROOT / record["prepared"]["checkpoint_dir"] / "config.json").is_file(): return None
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(record["tokenizer_dir"], local_files_only=True, trust_remote_code=False)
        return record
    except (OSError, ValueError, KeyError):
        # Recognize assets prepared before resource status tracking was introduced.
        try:
            from huggingface_hub import hf_hub_download
            from studio.backend.executor.provision import CACHE
            from studio.backend.resource_catalog import external_hf_sources
            repo, revision, tok, tok_revision = external_hf_sources(config)
            checkpoint = Path(hf_hub_download(repo, "config.json", revision=revision,
                cache_dir=str(CACHE / "huggingface/hub"), local_files_only=True)).parent
            token_dir = checkpoint if (repo, revision) == (tok, tok_revision) else Path(hf_hub_download(
                tok, "tokenizer_config.json", revision=tok_revision,
                cache_dir=str(CACHE / "huggingface/hub"), local_files_only=True)).parent
            identity = json.dumps([repo, checkpoint.name, tok, token_dir.name])
            vocab = "hf_" + hashlib.sha256(identity.encode()).hexdigest()[:24]
            root = CACHE / "external-models" / vocab
            details = json.loads((root / "asset.json").read_text())
            if config.get("encoding_mode", "contextual") == "contextual" and not details.get("context_width"):
                return None
            import numpy as np
            matrix = np.load(root / "tokens.npy", mmap_mode="r", allow_pickle=False)
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(token_dir), local_files_only=True, trust_remote_code=False)
            return dict(embedding=dict(path=str((root / "tokens.npy").relative_to(REPO_ROOT)), vocab_name=vocab),
                prepared=dict(checkpoint_dir=str(checkpoint.relative_to(REPO_ROOT)), context_width=details.get("context_width"),
                    context_limit=min(tokenizer.model_max_length, details.get("limit") or 4096, 4096),
                    revision=checkpoint.name, tokenizer_revision=token_dir.name),
                rows=matrix.shape[0], width=matrix.shape[1], tokenizer_dir=str(token_dir))
        except (OSError, ValueError, KeyError):
            return None


def prepare_external_hf(config: dict):
    import numpy as np
    import torch
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoModel, AutoTokenizer

    from studio.backend.config import REPO_ROOT
    from studio.backend.executor.provision import CACHE, resource_lock, within_repo
    from studio.backend.resource_catalog import external_hf_sources

    cached = cached_external(config)
    if cached:
        tokenizer = AutoTokenizer.from_pretrained(cached["tokenizer_dir"], local_files_only=True, trust_remote_code=False)
        embedding = dict(cached["embedding"], frozen=config.get("freeze_token_embeddings", True))
        print("[prepare] Reusing local Hugging Face resources", flush=True)
        return ExternalHFTokenizer(embedding["vocab_name"], tokenizer, cached["rows"], cached["prepared"]["context_limit"]), embedding, cached["prepared"], cached["width"]
    repo, revision, tokenizer_repo, tokenizer_revision = external_hf_sources(config)
    print(f"[prepare] Resolving Hugging Face checkpoint {repo}@{revision}", flush=True)
    config_path = Path(hf_hub_download(repo, "config.json", revision=revision))
    checkpoint_dir = within_repo(config_path.parent)
    commit = config_path.parent.name
    same_source = (tokenizer_repo, tokenizer_revision) == (repo, revision)
    tokenizer_commit = commit if same_source else HfApi().model_info(tokenizer_repo, revision=tokenizer_revision).sha
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo, revision=tokenizer_commit, trust_remote_code=False)
    vocab = tokenizer.get_vocab()
    if not same_source:
        native = AutoTokenizer.from_pretrained(repo, revision=commit, trust_remote_code=False)
        special_keys = set(tokenizer.special_tokens_map) | set(native.special_tokens_map)
        if vocab != native.get_vocab() or any(
            getattr(tokenizer, key, None) != getattr(native, key, None) for key in special_keys
        ):
            raise ValueError("The selected tokenizer does not match the checkpoint's token IDs or special tokens.")
    if not vocab or min(vocab.values()) < 0:
        raise ValueError("The tokenizer has no valid token ID vocabulary.")

    identity = json.dumps([repo, commit, tokenizer_repo, tokenizer_commit])
    vocab_name = "hf_" + hashlib.sha256(identity.encode()).hexdigest()[:24]
    root = within_repo(CACHE / "external-models" / vocab_name)
    matrix = root / "tokens.npy"
    manifest = root / "asset.json"
    contextual = config.get("encoding_mode", "contextual") == "contextual"
    rows = max(vocab.values()) + 1
    with resource_lock(f"external-models/{vocab_name}.lock"):
        details = json.loads(manifest.read_text()) if manifest.is_file() and matrix.is_file() else {}
        if not details or (contextual and "context_width" not in details):
            print(f"[prepare] Loading pretrained token weights from {repo}@{commit}", flush=True)
            backbone, loading = AutoModel.from_pretrained(
                repo, revision=commit, trust_remote_code=False, use_safetensors=True, output_loading_info=True,
            )
            token_embedding = backbone.get_input_embeddings()
            if not isinstance(token_embedding, torch.nn.Embedding) or rows > token_embedding.num_embeddings:
                raise ValueError("The tokenizer emits IDs outside the model's pretrained input embedding table.")
            names = {name for name, param in backbone.named_parameters(remove_duplicate=False) if param is token_embedding.weight}
            if names.intersection(loading["missing_keys"]):
                raise ValueError("The checkpoint is missing its pretrained input embedding weights.")
            limit = getattr(backbone.config, "max_position_embeddings", None)
            details = {"width": token_embedding.embedding_dim, "limit": limit}
            if contextual:
                details["context_width"] = _context_width(backbone, token_embedding.weight)
            root.mkdir(parents=True, exist_ok=True)
            with (root / "tokens.tmp").open("wb") as stream:
                np.save(stream, token_embedding.weight[:rows].detach().float().cpu().numpy(), allow_pickle=False)
            (root / "tokens.tmp").replace(matrix)
            manifest.write_text(json.dumps(details))

    max_length = min(tokenizer.model_max_length, details["limit"] or 4096, 4096)
    prepared = {
        "checkpoint_dir": str(checkpoint_dir.relative_to(REPO_ROOT)),
        "context_width": details.get("context_width"),
        "context_limit": max_length,
        "revision": commit,
        "tokenizer_revision": tokenizer_commit,
    }
    embedding = {
        "path": str(matrix.relative_to(REPO_ROOT)),
        "vocab_name": vocab_name,
        "frozen": config.get("freeze_token_embeddings", True),
    }
    print(f"[prepare] Tokenizer ready, {rows} token IDs × {details['width']} pretrained dimensions", flush=True)
    tokenizer_dir = root / "tokenizer"
    tokenizer.save_pretrained(tokenizer_dir)
    record = external_record(config)
    record.parent.mkdir(parents=True, exist_ok=True)
    temporary = record.with_suffix(".tmp")
    temporary.write_text(json.dumps(dict(embedding=embedding, prepared=prepared, rows=rows, width=details["width"], tokenizer_dir=str(tokenizer_dir))))
    temporary.replace(record)
    return ExternalHFTokenizer(vocab_name, tokenizer, rows, max_length), embedding, prepared, details["width"]


def _context_width(backbone, weights) -> int:
    import torch

    params = inspect.signature(backbone.forward).parameters
    if not {"inputs_embeds", "attention_mask"}.issubset(params) or getattr(backbone.config, "is_encoder_decoder", False):
        raise ValueError("This checkpoint cannot run as a contextual text encoder. Select Token embeddings mode.")
    with torch.no_grad():
        output = backbone.eval()(
            inputs_embeds=weights[:1].unsqueeze(0), attention_mask=torch.ones((1, 1), dtype=torch.long),
            return_dict=True,
        )
    hidden = getattr(output, "last_hidden_state", None)
    if hidden is None or hidden.ndim != 3:
        raise ValueError("This checkpoint does not return token-level hidden states. Select Token embeddings mode.")
    return hidden.shape[-1]
