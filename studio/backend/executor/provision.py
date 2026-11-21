"""Prepare run inputs using Legommenders' original processors."""

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import yaml

from studio.backend.config import REPO_ROOT, WORKSPACE
from studio.backend.executor.recbench_sources import PUBLISHED_PARQUET
from studio.backend.dataset_inputs import source_column
from studio.backend.resource_catalog import EXTERNAL_HF_VOCAB, LM_MODELS, dataset_processor

CACHE = WORKSPACE / "resources"
DOWNLOADS = CACHE / "downloads"
MIND_SOURCE = "https://mind201910small.blob.core.windows.net/release/"
# Current source used by recommenders-team/recommenders/datasets/mind.py.
MIND_HF_SOURCE = "https://huggingface.co/datasets/Recommenders/MIND/resolve/main/"
RECBENCH_SOURCE = "https://github.com/Jyonn/RecBench/releases/download/dataset/RecBench.zip"
XMIND_SOURCE = "https://raw.githubusercontent.com/andreeaiana/xMIND/main/xMIND"
RECBENCH_FILES = ("items.parquet", "users.parquet", "finetune.parquet", "test.parquet", "valid_user_set_0.1.txt")


def within_repo(path: str | Path) -> Path:
    path = (REPO_ROOT / path).resolve()
    if not path.is_relative_to(REPO_ROOT):
        raise ValueError(f"Output path is outside the project: {path}")
    return path


@contextmanager
def resource_lock(name: str):
    lock = within_repo(CACHE / "locks" / name)
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def resource_env() -> dict[str, str]:
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    for key, suffix in {
        "XDG_CACHE_HOME": "xdg", "HF_HOME": "huggingface",
        "HF_HUB_CACHE": "huggingface/hub", "HUGGINGFACE_HUB_CACHE": "huggingface/hub",
        "TRANSFORMERS_CACHE": "huggingface/hub", "HF_MODULES_CACHE": "huggingface/modules",
        "TORCH_HOME": "torch", "TMPDIR": "tmp",
        "NLTK_DATA": "nltk",
    }.items():
        folder = within_repo(CACHE / suffix)
        folder.mkdir(parents=True, exist_ok=True)
        if key == "NLTK_DATA":
            folder.chmod(0o700)
        env[key] = str(folder)
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    env.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    env.setdefault("HF_TOKEN_PATH", str(Path.home() / ".cache/huggingface/token"))
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("MKL_NUM_THREADS", "4")
    env.setdefault("OPENBLAS_NUM_THREADS", "4")
    return env


def table_ready(path: str | Path) -> bool:
    path = REPO_ROOT / path
    return (path / "meta.json").is_file() and (path / "data.pkl").is_file()


def dataset_paths(data: dict) -> list[str]:
    return [data["item"]["ut"], data["user"]["ut"], *(data["inter"][key] for key in ("train", "dev", "test"))]


def download(name: str, urls: tuple[str, ...]) -> Path:
    with resource_lock(f"download/{name}.lock"):
        return _download(name, urls)


def _download(name: str, urls: tuple[str, ...]) -> Path:
    output = within_repo(DOWNLOADS / name)
    if output.is_file():
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    part = output.with_suffix(f"{output.suffix}.{os.getpid()}.part")
    for url in urls:
        print(f"[prepare] Downloading {name} from {url}", flush=True)
        try:
            subprocess.run(
                ["curl", "--fail", "--location", "--progress-bar", "--show-error", "--retry", "2",
                 "--connect-timeout", "15", "--max-time", "1800", "--output", str(part), url],
                check=True,
            )
            if output.suffix == ".parquet":
                with part.open("rb") as stream:
                    valid = stream.read(4) == b"PAR1"
            else:
                valid = zipfile.is_zipfile(part)
            if not valid:
                raise ValueError(f"{name} did not contain the expected file format")
            part.replace(output)
            return output
        except (subprocess.CalledProcessError, ValueError):
            part.unlink(missing_ok=True)
    raise RuntimeError(f"Unable to download {name}. Check publisher access or configure its raw files in .data.")


def extract_members(archive: Path, members: dict[str, Path]) -> None:
    with zipfile.ZipFile(archive) as source:
        for member, destination in members.items():
            destination = within_repo(destination)
            if destination.is_file() and destination.stat().st_size == source.getinfo(member).file_size:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            part = destination.with_suffix(f"{destination.suffix}.{os.getpid()}.part")
            with source.open(member) as stream, part.open("wb") as output:
                shutil.copyfileobj(stream, output)
            part.replace(destination)


def raw_mind() -> Path:
    from utils.config_init import DataInit

    configured = DataInit.get("mind", default="")
    names = ("news.tsv", "behaviors.tsv")
    splits = ("train", "dev")
    if configured:
        local = Path(configured).expanduser()
        if all((local / split / name).is_file() for split in splits for name in names):
            return local
        print('[prepare] Configured MIND path is incomplete; trying automatic download.', flush=True)
    root = DOWNLOADS / "mind"
    for split in splits:
        filename = f"MINDsmall_{split}.zip"
        archive = DOWNLOADS / filename
        if not archive.is_file():
            archive = download(filename, (MIND_HF_SOURCE + filename, MIND_SOURCE + filename))
        extract_members(archive, {name: root / split / name for name in names})
    return root


def raw_recbench(processor: str) -> tuple[Path, bool]:
    from utils.config_init import DataInit

    domain = processor.removesuffix("rb")
    configured = DataInit.get(processor, default=DataInit.get(domain, default=""))
    if configured:
        local = Path(configured).expanduser()
        if all((local / f"{name}.parquet").is_file() for name in ("items", "users", "train", "valid", "test")):
            return local, True
        if all((local / name).is_file() for name in RECBENCH_FILES):
            return local, False
        print(f'[prepare] Configured {processor} path is incomplete; trying automatic download.', flush=True)
    if processor in PUBLISHED_PARQUET:
        root = DOWNLOADS / "legommenders" / processor
        print(f"[prepare] Using official Legommenders {processor} train/valid/test splits (no resplitting)", flush=True)
        for name, file_id in PUBLISHED_PARQUET[processor].items():
            url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
            download(f"legommenders/{processor}/{name}.parquet", (url,))
        return root, True
    root = DOWNLOADS / "recbench" / domain
    if all((root / name).is_file() for name in RECBENCH_FILES):
        return root, False
    archive = download("recbench.zip", (RECBENCH_SOURCE,))
    with zipfile.ZipFile(archive) as source:
        members = {
            Path(member).name: member for member in source.namelist()
            if Path(member).parent.name.lower() in {domain, processor}
        }
    if not all(name in members for name in RECBENCH_FILES):
        missing = ", ".join(name for name in RECBENCH_FILES if name not in members)
        raise RuntimeError(f"RecBench {domain} source is incomplete (missing: {missing}). No complete official Legommenders export is configured for {processor}. Provide its published train/valid/test Parquet files via .data; Studio will not create new splits.")
    extract_members(archive, {members[name]: root / name for name in RECBENCH_FILES})
    return root, False


def raw_xmind(lang: str) -> Path:
    from utils.config_init import DataInit

    configured = DataInit.get(f"xmind{lang}", default="")
    if configured:
        local = Path(configured).expanduser()
        if all((local / f"xMINDsmall_{split}/news.tsv").is_file() for split in ("train", "dev")):
            return local
        print(f'[prepare] Configured xMIND {lang} path is incomplete; trying automatic download.', flush=True)
    root = DOWNLOADS / "xmind" / lang
    source_lang = "zho" if lang == "cmn" else lang
    for split in ("train", "dev"):
        folder = f"xMINDsmall_{split}"
        filename = f"{folder}.zip"
        archive = download(f"xmind-{lang}-{split}.zip", (f"{XMIND_SOURCE}/{source_lang}/{filename}",))
        with zipfile.ZipFile(archive) as source:
            news = next(name for name in source.namelist() if Path(name).name == "news.tsv")
        extract_members(archive, {news: root / folder / "news.tsv"})
    return root


def embedding(checkpoint: str, target: str | Path) -> Path:
    with resource_lock(f"embedding/{checkpoint}.lock"):
        return _export_embedding(checkpoint, target)


def _export_embedding(checkpoint: str, target: str | Path) -> Path:
    import numpy as np
    from loader.class_hub import ClassHub

    target = within_repo(target)
    if target.is_file():
        return target
    print(f"[prepare] Exporting pretrained token embeddings for {checkpoint}", flush=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint == "glove":
        from embedder.glove_embedder import GloVeEmbedder
        if not within_repo(GloVeEmbedder.FILE_PATH).is_file():
            archive = within_repo(GloVeEmbedder.ZIP_PATH)
            if not zipfile.is_zipfile(archive):
                archive = download(GloVeEmbedder.ZIP_NAME, (GloVeEmbedder.DL_URI, GloVeEmbedder.FALLBACK_DL_URI))
            extract_members(archive, {GloVeEmbedder.FILE_NAME: Path(GloVeEmbedder.FILE_PATH)})
            # The original exporter checks the archive before opening the text file.
            if not zipfile.is_zipfile(GloVeEmbedder.ZIP_PATH):
                zip_path = within_repo(GloVeEmbedder.ZIP_PATH)
                zip_path.unlink(missing_ok=True)
                os.link(archive, zip_path)
    weights = ClassHub.embedders()[checkpoint]().get_embeddings()
    part = target.with_suffix(".part.npy")
    np.save(part, weights)
    part.replace(target)
    return target


def tokenizer(vocab: str, external_tokenizer=None):
    if external_tokenizer is not None and vocab == external_tokenizer.vocab.name:
        return external_tokenizer
    from unitok import GloVeTokenizer, TransformersTokenizer
    from utils.config_init import ModelInit

    if vocab == "glove":
        import nltk

        nltk_cache = within_repo(CACHE / "nltk")
        nltk_cache.mkdir(parents=True, exist_ok=True)
        nltk_cache.chmod(0o700)
        if str(nltk_cache) not in nltk.data.path:
            nltk.data.path.insert(0, str(nltk_cache))
        try:
            nltk.data.find("tokenizers/punkt_tab/english/")
        except LookupError:
            print("[prepare] Downloading NLTK English tokenization data", flush=True)
            if not nltk.download("punkt_tab", download_dir=str(nltk_cache), quiet=True, raise_on_error=True):
                raise RuntimeError("Unable to prepare NLTK tokenization data. Check network access to nltk_data.")
        from embedder.glove_embedder import GloVeEmbedder
        if not within_repo(Path(GloVeEmbedder.FILE_DIR) / "glove.vocab").is_file():
            from loader.class_hub import ClassHub
            if within_repo("data/embeddings/glove.npy").is_file():
                ClassHub.embedders()["glove"]().get_embeddings()
            else:
                embedding("glove", "data/embeddings/glove.npy")
        return GloVeTokenizer(vocab=GloVeEmbedder.get_glove_vocab())
    checkpoint = {"bert": "bertbase", "glm": "glm4th9b"}.get(vocab, vocab)
    return TransformersTokenizer(
        vocab=vocab, key=ModelInit.get(checkpoint), **({"trust_remote_code": True} if vocab == "glm" else {})
    )


def item_columns(data: dict) -> list[str]:
    return [
        source_column(next(iter(value)) if isinstance(value, dict) else value)
        for value in data["item"]["inputs"]
    ]


def item_vocabularies(data: dict) -> list[str]:
    return sorted({column.split("@", 1)[1] for column in item_columns(data) if "@" in column})


def prepared_directory(data: dict, processor: str, lang: str) -> Path:
    parts = [processor, lang] if processor == "xmind" else [processor]
    signature = hashlib.sha256(json.dumps(data["item"]["inputs"], sort_keys=True).encode()).hexdigest()[:12]
    version = "published-v1" if processor in PUBLISHED_PARQUET else "v2"
    key = "-".join([*parts, *item_vocabularies(data), version, signature])
    return within_repo(CACHE / "datasets" / key)


def item_table_ready(path: str | Path, columns: list[str]) -> bool:
    if not table_ready(path):
        return False
    features = json.loads((REPO_ROOT / path / "meta.json").read_text())["features"]
    return set(columns).issubset(feature["name"] for feature in features)


def tables_available(data: dict, processor: str, lang: str) -> bool:
    item, *other_paths = dataset_paths(data)
    columns = item_columns(data)
    if item_table_ready(item, columns) and all(table_ready(path) for path in other_paths):
        return True
    directory = prepared_directory(data, processor, lang)
    if not item_table_ready(directory / "items", columns):
        return False
    if processor == "xmind":
        mind = REPO_ROOT / "data/mind"
        alternate = CACHE / "datasets/mind-v1"
        return any(
            all(table_ready(root / part) for part in ("items", "users", "train", "valid", "test"))
            for root in (mind, alternate)
        )
    return all(table_ready(directory / part) for part in ("users", "train", "valid", "test"))


def processed_data(data: dict, processor: str, lang: str, external_tokenizer=None) -> None:
    with resource_lock(f"dataset/{processor}.lock"):
        _process_data(data, processor, lang, external_tokenizer)


def _process_data(data: dict, processor: str, lang: str, external_tokenizer=None) -> None:
    from process import get_processor
    from processor.base_processor import BaseProcessor
    from unitok import EntityTokenizer, UniTok
    from utils import function
    from utils.config_init import DataInit

    columns = item_columns(data)
    paths = [within_repo(path) for path in dataset_paths(data)]
    expanded = any(
        isinstance(entry, dict) and any(
            length and length > {"title": 50, "abstract": 200}.get(source_column(column).split("@", 1)[0], float("inf"))
            for column, length in entry.items()
        ) for entry in data["item"]["inputs"]
    )
    if not expanded and item_table_ready(paths[0], columns) and all(table_ready(path) for path in paths[1:]):
        print(f"[prepare] Using existing UniTok tables for {data['name']}", flush=True)
        return

    vocabularies = item_vocabularies(data)
    directory = prepared_directory(data, processor, lang)
    item_dir = directory / "items"
    base = directory
    if processor == "xmind":
        mind = {
            "name": "mind", "item": {"ut": "data/mind/items", "inputs": ["item_id"]},
            "user": {"ut": "data/mind/users"},
            "inter": {key: f"data/mind/{folder}" for key, folder in (("train", "train"), ("dev", "valid"), ("test", "test"))},
        }
        processed_data(mind, "mind", lang)
        base = (REPO_ROOT / mind["user"]["ut"]).parent

    if not item_table_ready(item_dir, columns):
        processor_name = {"xmind": f"xmind{lang}", "mind-demo": "mind"}.get(processor, processor)
        cls = get_processor(processor_name)

        class StudioProcessor(cls):
            @classmethod
            def get_name(cls):
                return "tables"

            @property
            def attrs(self):
                attrs = dict(super().attrs)
                for entry in data["item"]["inputs"]:
                    if isinstance(entry, dict):
                        column, length = next(iter(entry.items()))
                        attr = source_column(column).split("@", 1)[0]
                        if attr in attrs and length and attrs[attr]:
                            attrs[attr] = max(attrs[attr], length)
                return attrs

            def config_item_tokenization(self):
                if processor == "xmind":
                    return super().config_item_tokenization()
                for vocab in vocabularies:
                    self.add_item_tokenizer(tokenizer(vocab, external_tokenizer))
                if processor in {"mind", "mind-demo", "oncemind"}:
                    for column in ("category", "subcategory"):
                        self.item.add_feature(tokenizer=EntityTokenizer(vocab=column), column=column)

            def load_users(self):
                users = super().load_users()
                if processor == "mind-demo":
                    users = users.sample(n=min(650, len(users)), random_state=2023)
                return users

        directory.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=directory.parent, prefix=".preparing-") as stage:
            StudioProcessor.BASE_STORE_DIR = stage
            if processor == "xmind":
                instance = StudioProcessor.__new__(StudioProcessor)
                BaseProcessor.__init__(instance, data_dir=str(raw_xmind(lang)))
                instance.tokenizers = [tokenizer(vocab, external_tokenizer) for vocab in vocabularies]
                instance.load_base_item = lambda: UniTok.load(str(base / "items"))
            elif processor.endswith("rb"):
                raw, published_splits = raw_recbench(processor)
                if published_splits:
                    import pandas as pd
                    instance = StudioProcessor.__new__(StudioProcessor)
                    BaseProcessor.__init__(instance, data_dir=str(raw))
                    instance.item_df = pd.read_parquet(raw / "items.parquet")
                    instance.user_df = pd.read_parquet(raw / "users.parquet").drop(columns=[instance.NEG_COL], errors="ignore")
                    instance.user_df[instance.HIS_COL] = instance.user_df[instance.HIS_COL].map(list)
                    for mode in ("train", "valid", "test"):
                        setattr(instance, f"{mode}_df", pd.read_parquet(raw / f"{mode}.parquet"))
                else:
                    instance = StudioProcessor(data_dir=str(raw))
            elif processor == "oncemind":
                configured = DataInit.get("oncemind", default="")
                if not configured:
                    import random
                    raw = raw_mind()
                    with (raw / "dev" / "behaviors.tsv").open(encoding="utf-8") as behaviors:
                        impressions = [int(line.split("\t", 1)[0]) for line in behaviors]
                    random.Random(2023).shuffle(impressions)
                    impression_file = Path(stage) / "impressions.json"
                    impression_file.write_text(json.dumps(impressions), encoding="utf-8")
                    configured = f"{raw}${impression_file}"
                    print("[prepare] Using a reproducible 50/50 split of MIND dev impressions for ONCE-MIND", flush=True)
                instance = StudioProcessor(data_dir=configured)
            else:
                instance = StudioProcessor(data_dir=str(raw_mind()))
            print(f"[prepare] Tokenizing {data['name']} with {', '.join(vocabularies) or 'ID features'}", flush=True)
            function.seeding(2023)
            if processor == "xmind":
                instance.load()
            else:
                # Skip RecBench's wrapper, which writes a config/data YAML.
                BaseProcessor.load(instance)
            Path(instance.save_dir).rename(directory)

    data["item"]["ut"] = str(item_dir.relative_to(REPO_ROOT))
    data["user"]["ut"] = str((base / "users").relative_to(REPO_ROOT))
    for split, name in (("train", "train"), ("dev", "valid"), ("test", "test")):
        data["inter"][split] = str((base / name).relative_to(REPO_ROOT))
    if table_ready(item_dir):
        features = json.loads((item_dir / "meta.json").read_text())["features"]
        missing = set(columns) - {feature["name"] for feature in features}
        if missing:
            raise RuntimeError(f"Prepared item table is missing columns: {', '.join(sorted(missing))}.")
    if not all(table_ready(path) for path in dataset_paths(data)):
        raise RuntimeError(f"Preprocessing produced incomplete UniTok tables for {data['name']}.")


def prepare(plan: dict, output: Path) -> None:
    data, embed = plan["data"], plan["embed"]
    checkpoint = plan.get("checkpoint")
    processor = dataset_processor(plan["data_config"])
    if checkpoint:
        from studio.backend.resources import snapshot, model_source
        try:
            local_checkpoint = snapshot(model_source(checkpoint))
            print(f"[prepare] Reusing local {checkpoint} resources", flush=True)
        except (OSError, ValueError):
            local_checkpoint = snapshot(model_source(checkpoint), download=True)
        from utils.config_init import ModelInit
        ModelInit.parse()[checkpoint] = str(local_checkpoint)
        vocab = LM_MODELS[checkpoint][0]
        embed["embeddings"] = [{
            "path": f"data/embeddings/{checkpoint}.npy", "vocab_name": vocab, "frozen": True,
        }]

    external_tokenizer = None
    external_item = plan.get("external_hf")
    external_config = plan.get("dataset_external_hf")
    if external_item or external_config:
        from studio.backend.executor.external_hf import prepare_external_hf

        if external_item:
            model = yaml.safe_load((output / "model.yaml").read_text())
            external_config = model["config"]["item_config"]
        external_tokenizer, embedding_config, settings, width = prepare_external_hf(external_config)
        if external_item:
            external_config.update(settings)
            model["config"]["item_hidden_size"] = width
            within_repo(output / "prepared-model.yaml").write_text(yaml.safe_dump(model, sort_keys=False), encoding="utf-8")
        embed = {"name": "external_hf", "embeddings": [embedding_config],
                 "transformation": "auto", "transformation_dropout": 0.0}
        suffix = "@" + embedding_config["vocab_name"]
        data["item"]["inputs"] = [
            {column.replace("@" + EXTERNAL_HF_VOCAB, suffix): truncate for column, truncate in entry.items()}
            if isinstance(entry, dict) else entry.replace("@" + EXTERNAL_HF_VOCAB, suffix)
            for entry in data["item"]["inputs"]
        ]
        resource_info = {
            "owner": "item_encoder" if external_item else "dataset",
            "model_id": external_config["model_id"],
            "revision": settings["revision"],
            "tokenizer_id": external_config.get("tokenizer_id") or external_config["model_id"],
            "tokenizer_revision": settings["tokenizer_revision"],
            "vocab_name": embedding_config["vocab_name"],
            "embedding_width": width,
        }
        within_repo(output / "token-resources.json").write_text(json.dumps(resource_info), encoding="utf-8")

    for target in ("cache", "checkpoints"):
        within_repo(target)
    processed_data(data, processor, plan["lang"], external_tokenizer)
    for entry in embed.get("embeddings") or []:
        path = entry["path"]
        if not (REPO_ROOT / path).is_file():
            target = Path(path)
            if target.parent != Path("data/embeddings") or target.stem not in {*LM_MODELS, "glove"}:
                raise RuntimeError(f"Missing published dataset-specific embeddings: {path}. Install its matching matrix or select token embeddings.")
            embedding(target.stem, target)
    if checkpoint:
        print(f"[prepare] {checkpoint} inputs ready. Legommenders will load its checkpoint and prepare any hidden-state cache.", flush=True)
        # The framework keys transformer hidden-state caches by data.name.
        cache_key = json.dumps(data["item"], sort_keys=True).encode()
        data["name"] += "-" + hashlib.sha256(cache_key).hexdigest()[:12]

    for filename, config in (("data.yaml", data), ("embed.yaml", embed)):
        within_repo(output / filename).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print("[prepare] Resources ready", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    print("[prepare] Checking dataset and model resources", flush=True)
    prepare(plan, args.plan.parent)


if __name__ == "__main__":
    main()
