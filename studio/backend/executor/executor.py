"""Training / evaluation executor.

Uses ``subprocess.Popen`` with an argument **array** and server-controlled
paths only.  No shell, no string concatenation of user input into a command,
no arbitrary file access.  Every result displayed in the UI comes from the real
process's stdout or from the real files written by ``trainer.py``/``tester.py``.
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import yaml

from studio.backend.compiler.compiler import build_exp_config, compile_graph
from studio.backend.executor.provision import dataset_paths, resource_env, resource_lock, table_ready, tables_available
from studio.backend.dataset_inputs import dataset_external_config, dataset_settings, dataset_variables, resolve_item_inputs
from studio.backend.resource_catalog import EXTERNAL_HF_COMPONENT, dataset_processor, item_checkpoint, resource_args
from studio.backend.config import (
    EMBED_CONFIG_DIR,
    REPO_ROOT,
    RUNS_DIR,
    TESTER_SCRIPT,
    TRAINER_SCRIPT,
)
from studio.backend.executor.progress_log import ProgressLog
from studio.backend.schemas.graph import GraphJson
from studio.backend.schemas.runs import CurvePoint, RunRecord

_LOCK = threading.RLock()
_RUNS: dict[str, RunRecord] = {}
_PROCESSES: dict[str, subprocess.Popen] = {}
_LOG_LOCK = threading.Lock()
_DEVICE_LOCKS: dict[str, threading.Lock] = {}

_METRIC_LINE = re.compile(r"\b([A-Za-z][A-Za-z0-9_]*(?:@\d+(?:\.\d+)?)?):\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\b")
_LOSS_LINE = re.compile(r"\[epoch\s+(\d+)\]\s+step\s+(\d+),\s+loss\s+([-+]?\d*\.?\d+)")
_EPOCH_METRIC = re.compile(r"\[epoch\s+(\d+)\]\s+([A-Za-z][A-Za-z0-9_]*(?:@\d+(?:\.\d+)?)?)\s+([-+]?\d*\.?\d+)")
_SIGNATURE = re.compile(r"SIGNATURE:\s*([A-Za-z0-9_.-]+)")


def _load_persisted_runs() -> None:
    for path in sorted(RUNS_DIR.glob("*.json")):
        try:
            record = RunRecord.model_validate_json(path.read_text(encoding="utf-8"))
            if record.status in ("pending", "running") and not record.error:
                record.status = "failed"
                record.error = "Backend restarted while the run was active; the process no longer exists."
            _RUNS[record.run_id] = record
        except Exception:
            continue


_load_persisted_runs()


def _persist(record: RunRecord) -> None:
    try:
        (RUNS_DIR / f"{record.run_id}.json").write_text(
            json.dumps(record.model_dump(mode="json"), indent=2), encoding="utf-8"
        )
    except OSError:
        pass


def _append_log(record: RunRecord, line: str, replace: bool = False) -> None:
    with _LOG_LOCK:
        clean = line.rstrip("\r\n")
        if replace and record.log_lines:
            record.log_lines[-1] = clean
        else:
            record.log_lines.append(clean)
        if len(record.log_lines) > 12000:
            del record.log_lines[:2000]
            record.log_offset += 2000
        if record.log_file:
            try:
                with open(REPO_ROOT / record.log_file, "a", encoding="utf-8") as handle:
                    handle.write(line)
            except OSError:
                pass


def environment_report() -> dict[str, Any]:
    deps: dict[str, str] = {}
    for name in ("torch", "transformers", "tokenizers", "huggingface_hub", "peft", "accelerate", "unitok", "refconfig", "oba", "smartdict", "pigmento", "yaml", "sklearn"):
        try:
            module = __import__(name)
            deps[name] = str(getattr(module, "__version__", "installed"))
        except Exception as exc:  # noqa: BLE001
            deps[name] = f"MISSING ({type(exc).__name__})"
    cuda_devices = []
    cuda_available, device_count, device_name = False, 0, None
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0) if cuda_available else None
        for index in range(device_count):
            cuda_devices.append({"index": index, "name": torch.cuda.get_device_name(index)})
    except Exception:
        pass

    data_root = REPO_ROOT / "data"
    warnings: list[str] = []
    if not data_root.exists():
        warnings.append(
            "No local data/ directory yet. Studio will download and prepare missing supported "
            "datasets when you click Train; .data is an optional local override."
        )
    if not cuda_available:
        warnings.append("CUDA is unavailable; use device=cpu (--cuda -1) for small-scale runs.")

    from refconfig import CType, RefConfig
    from studio.backend.registry.registry import get_catalog

    datasets: list[dict[str, Any]] = []
    for spec in get_catalog().components:
        if spec.role != "dataset":
            continue
        relative = spec.source["yaml"]
        params = {p.name: p.default for p in spec.parameters}
        try:
            variables = resource_args(relative, params.get("lm"), params.get("lang"))
            data = RefConfig().add(CType.SMART, data=relative, **variables).parse()["data"]
            present = tables_available(data, dataset_processor(relative), variables["lang"])
        except (KeyError, TypeError, ValueError):
            present = False
        datasets.append({"config": relative, "config_exists": True, "files_present": present})
    embeddings: list[dict[str, Any]] = []
    for path in sorted(EMBED_CONFIG_DIR.glob("*.yaml")):
        import yaml

        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            data = {}
        for entry in data.get("embeddings") or []:
            if isinstance(entry, dict) and isinstance(entry.get("path"), str):
                target = entry["path"]
                embeddings.append({"config": str(path.relative_to(REPO_ROOT)), "path": target,
                                   "present": "${" not in target and (REPO_ROOT / target).exists()})
    return {
        "python": sys.version.split()[0],
        "https_port": int(os.environ["STUDIO_HTTPS_PORT"]) if os.environ.get("STUDIO_HTTPS_PORT") else None,
        "executable": sys.executable,
        "torch": deps.get("torch", "MISSING"),
        "cuda_available": cuda_available,
        "cuda_device_count": device_count,
        "cuda_devices": cuda_devices,
        "device_name": device_name,
        "dependencies": deps,
        "trainer_script": TRAINER_SCRIPT.is_file(),
        "data_root_exists": data_root.exists(),
        "datasets": datasets,
        "embeddings": embeddings,
        "warnings": warnings,
    }


def _resolve_data_and_embed(graph: GraphJson, compile_result) -> tuple[dict[str, Any], dict[str, Any]]:
    """Use the real config loader to obtain concrete (placeholder-free) paths."""
    from trainer import get_configurations  # noqa: WPS433

    kwargs: dict[str, Any] = dict(
        data=compile_result.data_config,
        model=compile_result.generated_file,
        embed=compile_result.embed_config,
        exp=compile_result.exp_config,
        hidden_size=graph.train.hidden_size,
        item_hidden_size=graph.train.item_hidden_size or graph.train.hidden_size,
        batch_size=graph.train.batch_size,
        lr=graph.train.lr,
        cuda="-1",
    )
    _, _, lm = dataset_settings(graph)
    kwargs.update(dataset_variables(graph, compile_result.data_config, lm))
    cfg = get_configurations(kwargs)
    data, embed = cfg.data(), cfg.embed()
    if lm and lm.startswith("opt"):
        for entry in embed.get("embeddings") or []:
            entry["vocab_name"] = lm
    data["item"]["inputs"] = resolve_item_inputs(graph, data, embed, compile_result.data_config)["values"]
    return data, embed


def preflight(graph: GraphJson, compile_result) -> list[str]:
    blockers: list[str] = []
    if dataset_external_config(graph) is not None or any(node.component == EXTERNAL_HF_COMPONENT for node in graph.nodes):
        blockers.append("Train resolves the external Hugging Face checkpoint and tokenizer and validates their compatibility.")
        return blockers
    if not TRAINER_SCRIPT.is_file():
        blockers.append(f"trainer.py not found at {TRAINER_SCRIPT}")
    try:
        data, embed = _resolve_data_and_embed(graph, compile_result)
    except Exception as exc:  # noqa: BLE001
        return [f"Cannot resolve data/embed configuration with Legommenders: {type(exc).__name__}: {exc}"]

    _, _, lm = dataset_settings(graph)
    variables = dataset_variables(graph, compile_result.data_config, lm)
    if not tables_available(data, dataset_processor(compile_result.data_config), variables["lang"]):
        missing = [str(path) for path in dataset_paths(data) if not table_ready(path)]
        reason = "Dataset files are missing: " + ", ".join(missing) if missing else "Dataset item columns need preprocessing"
        blockers.append(reason + ". Train prepares supported datasets when their source is available.")
    for entry in embed.get("embeddings") or []:
        if isinstance(entry, dict) and isinstance(entry.get("path"), str):
            target = entry["path"]
            if "${" in target:
                blockers.append(f"Embedding path still contains an unresolved placeholder: {target}")
            elif not (REPO_ROOT / target).exists():
                blockers.append(
                    f"Required pre-trained embedding `{target}` is missing. "
                    "Generate it with `python embed.py --model <name>` or choose config/embed/null.yaml."
                )
    if graph.train.device == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                blockers.append("CUDA was requested but torch.cuda.is_available() is False. Use device=cpu.")
        except Exception as exc:  # noqa: BLE001
            blockers.append(f"PyTorch is not usable: {exc}")
    return blockers


def _automatic_cuda_index() -> int:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        raise ValueError("CUDA is unavailable.")
    available = []
    for index in range(torch.cuda.device_count()):
        try:
            free, _ = torch.cuda.mem_get_info(index)
            available.append((free, -index))
        except Exception:
            continue
    return -max(available)[1] if available else 0


def _build_command(graph: GraphJson, compile_result, *, script: Path) -> list[str]:
    cuda = "-1"
    if graph.train.device == "cuda":
        cuda = str(graph.train.cuda_index if graph.train.cuda_index is not None else _automatic_cuda_index())
    elif graph.train.device == "auto":
        try:
            import torch

            cuda = "0" if torch.cuda.is_available() else "-1"
        except Exception:
            cuda = "-1"
    command = [
        sys.executable, "-u", script.name,
        "--data", str(compile_result.data_config),
        "--model", str(compile_result.generated_file),
        "--embed", str(compile_result.embed_config),
        "--exp", str(compile_result.exp_config),
        "--hidden_size", str(graph.train.hidden_size),
        "--item_hidden_size", str(graph.train.item_hidden_size or graph.train.hidden_size),
        "--batch_size", str(graph.train.batch_size),
        "--lr", str(graph.train.lr),
        "--cuda", cuda,
        "--seed", str(graph.train.seed),
    ]
    _, _, lm = dataset_settings(graph)
    for key, value in dataset_variables(graph, compile_result.data_config, lm).items():
        command += [f"--{key}", value]
    return command


def _parse_results(record: RunRecord) -> None:
    known = set(record.config.get("metrics") or []) | {
        "GAUC", "AUC", "MRR", "MRR0", "LogLoss", "LRAP", "NDCG@1", "NDCG@5", "NDCG@10"
    }
    text = "\n".join(record.log_lines)
    for match in _SIGNATURE.finditer(text):
        record.signature = match.group(1)
    for match in _METRIC_LINE.finditer(text):
        name, raw = match.group(1), match.group(2)
        if name in known:
            try:
                record.metrics[name] = float(raw)
            except ValueError:
                continue
    for match in _LOSS_LINE.finditer(text):
        record.loss_curve.append(CurvePoint(epoch=int(match.group(1)), step=int(match.group(2)),
                                            value=float(match.group(3)), metric="loss"))
    for match in _EPOCH_METRIC.finditer(text):
        name = match.group(2)
        if name.upper() in {m.upper() for m in known}:
            record.validation_curve.append(CurvePoint(epoch=int(match.group(1)), value=float(match.group(3)), metric=name))
    # The real tester.py also writes ``metric: value`` lines to <signature>.csv.
    if record.signature and record.config.get("data_name"):
        candidate = REPO_ROOT / "checkpoints" / record.config["data_name"] / record.config.get("model_name", "") / f"{record.signature}.csv"
        if candidate.is_file():
            record.result_file = str(candidate.relative_to(REPO_ROOT))
            for line in candidate.read_text(encoding="utf-8", errors="replace").splitlines():
                match = _METRIC_LINE.search(line)
                if match and match.group(1) in known:
                    try:
                        record.metrics[match.group(1)] = float(match.group(2))
                    except ValueError:
                        pass
    record.loss_curve = record.loss_curve[-2000:]
    record.validation_curve = record.validation_curve[-2000:]


def _run_step(record: RunRecord, command: list[str], env: dict[str, str]) -> int | None:
    with _LOCK:
        if record.status == "cancelled":
            return None
        _append_log(record, f"[studio] {' '.join(command)}\n")
        process = subprocess.Popen(
            command, cwd=REPO_ROOT, env=env, start_new_session=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=False, bufsize=0,
        )
        _PROCESSES[record.run_id] = process
    assert process.stdout is not None
    import codecs

    token = env.get("HF_TOKEN")
    def emit(line: str, replace: bool) -> None:
        safe = line.replace(token, "[redacted]") if token else line
        _append_log(record, safe + "\n", replace=replace)

    parser = ProgressLog(emit)
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    with process.stdout:
        while chunk := process.stdout.read(4096):
            parser.feed(decoder.decode(chunk))
        parser.feed(decoder.decode(b"", final=True))
    parser.finish()
    code = process.wait()
    with _LOCK:
        _PROCESSES.pop(record.run_id, None)
    return code


def _run_worker(record: RunRecord, plan_path: Path | None, hf_token: str | None) -> None:
    log_dir = RUNS_DIR / record.run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        if record.status == "cancelled":
            return
        record.log_file = str((log_dir / "stdout.log").relative_to(REPO_ROOT))
        record.status = "running"
        record.started_at = time.time()
        _persist(record)

    code = None
    error = ""
    try:
        env = resource_env()
        if hf_token:
            env.update(HF_TOKEN=hf_token, HF_HUB_DISABLE_IMPLICIT_TOKEN="0", HF_HUB_VERBOSITY="warning")
            env.pop("HF_DEBUG", None)
        if plan_path:
            code = _run_step(record, [sys.executable, "-u", "-m", "studio.backend.executor.provision", str(plan_path)], env)
            if code is None:
                return
            if code:
                raise RuntimeError("Resource preparation failed. See the preparation log for the missing resource or download error.")
            token_resources = plan_path.with_name("token-resources.json")
            if token_resources.is_file():
                with _LOCK:
                    record.config["token_resources"] = json.loads(token_resources.read_text())
                    _persist(record)
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        device = record.command[record.command.index("--cuda") + 1]
        with _LOCK:
            device_lock = _DEVICE_LOCKS.setdefault(device, threading.Lock())
        if device_lock.locked():
            _append_log(record, "[studio] Waiting for the selected device\n")
        checkpoint = record.config.get("checkpoint")
        data_config = REPO_ROOT / record.config["prepared_data"]
        data_name = yaml.safe_load(data_config.read_text())["name"]
        cache_lock = resource_lock(f"model/{data_name}-{checkpoint}.lock") if checkpoint else nullcontext()
        with device_lock, cache_lock:
            code = _run_step(record, record.command, env)
        if code is None:
            return
        if code:
            raise RuntimeError(f"Legommenders exited with code {code}.")
    except Exception as exc:
        message = str(exc)
        if hf_token:
            message = message.replace(hf_token, "[redacted]")
        error = f"{message}\n" + "\n".join(record.log_lines[-16:])
        _append_log(record, f"[studio] {message}\n")
    finally:
        with _LOCK:
            if record.status != "cancelled":
                record.finished_at = time.time()
                record.exit_code = code
                record.error = error
                try:
                    data_config = REPO_ROOT / record.config["prepared_data"]
                    if data_config.is_file():
                        record.config["data_name"] = yaml.safe_load(data_config.read_text())["name"]
                    _parse_results(record)
                except (OSError, ValueError, KeyError):
                    pass
                if not error and not record.metrics:
                    record.error = "The process did not emit any evaluation metrics. See the run log."
                record.status = "failed" if record.error else "completed"
            _persist(record)


def _start(graph: GraphJson, kind: str, *, source_run: RunRecord | None = None,
           preflight_only: bool = False, hf_token: str | None = None) -> RunRecord:
    run_id = uuid.uuid4().hex[:12]
    compile_result = compile_graph(graph, artifact_id=run_id)
    if not compile_result.compatible_with_reference:
        details = [note for note in compile_result.notes if note.startswith("Verification error:")]
        raise ValueError("The compiled graph could not be loaded by Legommenders. " + " ".join(details))
    exp = build_exp_config(graph)
    record = RunRecord(
        run_id=run_id, kind=kind, experiment_name=graph.name,
        model_label=compile_result.model_label, dataset=compile_result.data_config or "",
        task=graph.task, created_at=time.time(),
        config={
            "model_name": compile_result.model_dict.get("name"),
            "metrics": exp["metrics"],
            "hidden_size": graph.train.hidden_size,
            "batch_size": graph.train.batch_size,
            "accumulate_batch": graph.train.accumulate_batch,
            "lr": graph.train.lr,
            "epochs": graph.train.epochs,
            "device": graph.train.device,
            "model_yaml": compile_result.generated_file,
            "exp_yaml": compile_result.exp_config,
            "data_config": compile_result.data_config,
            "embed_config": compile_result.embed_config,
            "reference_yaml": compile_result.reference_yaml,
        },
        params=graph.train.model_dump(mode="json"), graph=graph.model_dump(mode="json"),
        model_yaml=compile_result.yaml_text,
    )
    if preflight_only:
        record.blockers = preflight(graph, compile_result)
        record.status = "failed" if record.blockers else "pending"
        record.error = " | ".join(record.blockers) if record.blockers else "Preflight only; no process started."
        with _LOCK:
            _RUNS[run_id] = record
            _persist(record)
        return record

    if graph.train.device == "cuda":
        import torch
        if graph.train.cuda_index is not None and graph.train.cuda_index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {graph.train.cuda_index} is unavailable.")

    folder = (REPO_ROOT / compile_result.generated_file).parent
    checkpoint = item_checkpoint(compile_result.model_dict["meta"].get("item", ""))
    if source_run and source_run.config.get("prepared_data"):
        checkpoint = source_run.config.get("checkpoint")
    record.config["checkpoint"] = checkpoint
    plan_path = None
    exp["studio_run_id"] = run_id  # distinct checkpoint signatures for repeated or concurrent runs
    if source_run and source_run.config.get("prepared_data"):
        if not source_run.signature:
            raise ValueError("The source run has no saved checkpoint signature.")
        exp["load"]["sign"] = source_run.signature
        compile_result.data_config = source_run.config["prepared_data"]
        compile_result.embed_config = source_run.config["prepared_embed"]
        compile_result.generated_file = source_run.config["model_yaml"]
        record.config["model_name"] = source_run.config["model_name"]
        if source_run.config.get("token_resources"):
            record.config["token_resources"] = source_run.config["token_resources"]
    else:
        data, embed = _resolve_data_and_embed(graph, compile_result)
        _, _, lm = dataset_settings(graph)
        variables = dataset_variables(graph, compile_result.data_config, lm)
        external_hf = compile_result.model_dict["meta"].get("item") == "ExternalHF"
        plan_path = folder / "prepare.json"
        plan_path.write_text(json.dumps({
            "data": data, "embed": embed, "lang": variables["lang"],
            "data_config": compile_result.data_config,
            "checkpoint": checkpoint,
            "external_hf": external_hf,
            "dataset_external_hf": dataset_external_config(graph),
        }), encoding="utf-8")
        if external_hf:
            compile_result.generated_file = str((folder / "prepared-model.yaml").relative_to(REPO_ROOT))
            record.config["model_yaml"] = compile_result.generated_file
        compile_result.data_config = str((folder / "data.yaml").relative_to(REPO_ROOT))
        compile_result.embed_config = str((folder / "embed.yaml").relative_to(REPO_ROOT))
        if source_run:
            exp["load"]["sign"] = source_run.signature

    (REPO_ROOT / compile_result.exp_config).write_text(yaml.safe_dump(exp, sort_keys=False), encoding="utf-8")
    record.config.update(prepared_data=compile_result.data_config, prepared_embed=compile_result.embed_config)
    script = TESTER_SCRIPT if kind == "evaluate" else TRAINER_SCRIPT
    record.command = _build_command(graph, compile_result, script=script)
    with _LOCK:
        _RUNS[run_id] = record
        _persist(record)
    threading.Thread(target=_run_worker, args=(record, plan_path, hf_token), daemon=True).start()
    return record


def start_training(graph: GraphJson, *, preflight_only: bool = False, hf_token: str | None = None) -> RunRecord:
    return _start(graph, "train", preflight_only=preflight_only, hf_token=hf_token)


def start_evaluation(graph: GraphJson, source_run: RunRecord | None = None) -> RunRecord:
    return _start(graph, "evaluate", source_run=source_run)


def get_run(run_id: str) -> RunRecord | None:
    with _LOCK:
        return _RUNS.get(run_id)


def list_runs() -> list[RunRecord]:
    with _LOCK:
        return sorted(_RUNS.values(), key=lambda r: r.created_at, reverse=True)


def logs_since(run_id: str, after: int = 0) -> dict[str, Any]:
    record = get_run(run_id)
    if not record:
        raise KeyError(run_id)
    with _LOG_LOCK:
        # The last observed row may have changed without advancing the cursor.
        start = max(0, min(after - 1, record.log_offset + len(record.log_lines)) - record.log_offset)
        replace_from = record.log_offset + start
        lines = list(record.log_lines[start:])
        next_index = record.log_offset + len(record.log_lines)
    return {
        "run_id": run_id,
        "status": record.status,
        "lines": lines,
        "replace_from": replace_from,
        "next": next_index,
        "error": record.error,
        "metrics": record.metrics,
    }


def cancel(run_id: str) -> RunRecord:
    record = get_run(run_id)
    if not record:
        raise KeyError(run_id)
    with _LOCK:
        process = _PROCESSES.get(run_id)
        if record.status in ("completed", "failed", "cancelled"):
            return record
        record.status = "cancelled"
        record.finished_at = time.time()
        record.error = "Cancelled by user."
        if process and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        _persist(record)
    _append_log(record, "[studio] run cancelled by user\n")
    return record


def delete_run(run_id: str) -> None:
    with _LOCK:
        record = _RUNS.get(run_id)
        if record is None: raise KeyError(run_id)
        if record.status not in ('completed', 'failed', 'cancelled'):
            raise ValueError('Only finished runs can be deleted')
        (RUNS_DIR / f"{record.run_id}.json").unlink(missing_ok=True)
        del _RUNS[run_id]
