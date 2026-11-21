"""Verify a compiled configuration through the *original* Legommenders code.

Nothing here re-implements a model.  We only exercise the real config loader
(``trainer.get_configurations``), the real dynamic class registry
(``loader.class_hub.ClassHub``) and the real configuration classes attached to
each operator/predictor.
"""
from __future__ import annotations
from studio.backend.i18n import localized


import inspect
from typing import Any

from studio.backend.config import REPO_ROOT


def verify_with_legommenders(
    *,
    model_path: str,
    data_config: str,
    exp_config: str,
    embed_config: str,
    hidden_size: int,
    batch_size: int,
    lr: float,
    lm: str | None = None,
    variables: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Return a structured report; never raises for a well-formed HTTP request."""
    report: dict[str, Any] = {
        "config_loaded": False,
        "classes_resolved": False,
        "configs_instantiated": False,
        "checks": [],
        "errors": [],
    }
    try:
        from trainer import get_configurations  # noqa: WPS433
        from utils.function import combine_config  # noqa: WPS433
        from loader.class_hub import ClassHub  # noqa: WPS433
        from model.lego_config import LegoConfig  # noqa: WPS433

        kwargs: dict[str, Any] = dict(
            data=data_config,
            model=model_path,
            embed=embed_config,
            exp=exp_config,
            hidden_size=hidden_size,
            item_hidden_size=hidden_size,
            batch_size=batch_size,
            lr=lr,
            cuda="-1",
        )
        if lm:
            kwargs["lm"] = lm
        kwargs.update(variables or {})
        cfg = get_configurations(kwargs)
        resolved_model = cfg.model()
        resolved_data = cfg.data()
        resolved_exp = cfg.exp()
        report["config_loaded"] = True
        report["checks"].append(localized('backend.00d994cb4755', "trainer.get_configurations() parsed the generated YAML", {}))
        report["resolved_model"] = resolved_model
        report["data_name"] = resolved_data.get("name")

        meta = resolved_model.get("meta") or {}
        config = resolved_model.get("config") or {}
        operators = ClassHub.operators()
        predictors = ClassHub.predictors()
        selected: dict[str, type] = {}
        for key, hub in (("item", operators), ("user", operators), ("predictor", predictors)):
            value = meta.get(key)
            if key == "item" and not value:
                continue
            if not isinstance(value, str) or value.lower() not in hub.class_dict:
                report["errors"].append(localized('backend.0875dcc2ac3b', f"meta.{key}={value!r} cannot be resolved by ClassHub", {'p0': f'{key}', 'p1': f'{value!r}'}))
                continue
            selected[key] = hub(value)
        report["classes_resolved"] = len(selected) == (3 if meta.get("item") else 2)
        report["checks"].append(
            "ClassHub resolved: " + ", ".join(f"{k}={v.__name__}" for k, v in selected.items())
        )

        # Reject keys that LegoConfig does not declare (no invented YAML keys).
        lego_params = set(inspect.signature(LegoConfig.__init__).parameters.keys())
        unknown = [key for key in config if key not in lego_params and key != "kwargs"]
        if unknown:
            report["errors"].append(localized('backend.bfd6fc8f9966', f"Unknown LegoConfig keys: {unknown}", {'p0': f'{unknown}'}))
        else:
            lego = LegoConfig(**config)
            report["checks"].append(
                localized('backend.1e24f1314d27', f"LegoConfig instantiated (hidden_size={lego.hidden_size}, "
                f"use_neg_sampling={lego.use_neg_sampling})", {'p0': f'{lego.hidden_size}', 'p1': f'{lego.use_neg_sampling}'})
            )

        # Real task compatibility, exactly as build_components() evaluates it.
        predictor_cls = selected.get("predictor")
        if predictor_cls:
            if config.get("use_neg_sampling") and not predictor_cls.allow_matching:
                report["errors"].append(localized('backend.1d7854414f94', f"{predictor_cls.__name__} does not support negative sampling", {'p0': f'{predictor_cls.__name__}'}))
            if not config.get("use_neg_sampling") and not predictor_cls.allow_ranking:
                report["errors"].append(localized('backend.1134bf3b9e35', f"{predictor_cls.__name__} only supports negative sampling", {'p0': f'{predictor_cls.__name__}'}))

        for key, section in (("item", "item_config"), ("user", "user_config"), ("predictor", "predictor_config")):
            cls = selected.get(key)
            if cls is None or key == "item" and not config.get("use_item_content"):
                continue
            raw = dict(config.get(section) or {})
            input_dim = config.get("item_hidden_size", hidden_size) if key == "item" else hidden_size
            try:
                cfg_obj = cls.config_class(**combine_config(config=raw, hidden_size=hidden_size, input_dim=input_dim))
                report["checks"].append(localized('backend.90200ff51f2d', f"{cls.__name__}.config_class instantiated: {type(cfg_obj).__name__}", {'p0': f'{cls.__name__}', 'p1': f'{type(cfg_obj).__name__}'}))
            except Exception as exc:  # noqa: BLE001
                report["errors"].append(localized('backend.6783513ab946', f"{cls.__name__}.config_class rejected config: {exc}", {'p0': f'{cls.__name__}', 'p1': f'{exc}'}))
        report["configs_instantiated"] = not report["errors"]
        report["exp_metrics"] = resolved_exp.get("metrics")
    except Exception as exc:  # noqa: BLE001
        report["errors"].append(localized('backend.43639a4c8b51', f"{type(exc).__name__}: {exc}", {'p0': f'{type(exc).__name__}', 'p1': f'{exc}'}))
    return report
