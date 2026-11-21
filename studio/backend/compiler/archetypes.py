"""Recognise model graphs that correspond to real Legommenders configurations.

The signatures come from the model table in ``README.md`` and the shipped
YAML files.  A graph that does not match any signature is *always* labelled
``Custom Model`` — the Studio never pretends a novel combination is a paper.
"""
from __future__ import annotations
from studio.backend.i18n import localized


from dataclasses import dataclass, field

from studio.backend.registry.metadata import MODEL_ARCHETYPES


@dataclass
class Archetype:
    key: str | None
    label: str
    reference_yaml: str | None
    exact: bool
    notes: list[str] = field(default_factory=list)


def detect(
    *,
    task: str,
    item_component: str | None,
    user_component: str | None,
    predictor_component: str | None,
) -> Archetype:
    signature: tuple[str, ...]
    if item_component:
        signature = (item_component, user_component or "", predictor_component or "")
    else:
        signature = (user_component or "", predictor_component or "")

    matches = [
        entry
        for entry in MODEL_ARCHETYPES
        if tuple(entry["components"]) == signature and entry["type"] == task
    ]
    if not matches:
        return Archetype(
            key=None,
            label="Custom Model",
            reference_yaml=None,
            exact=False,
            notes=[
                localized('backend.0d9e8f11e409', "This combination is not one of the models listed in README.md; "
                "it is compiled as a genuine custom Legommenders configuration.", {})
            ],
        )

    exact = next((entry for entry in matches if entry.get("exact", True)), None)
    chosen = exact or matches[0]
    notes: list[str] = []
    if exact is None:
        notes.append(
            localized('backend.88f69647be80', "The compiled structure follows the README model table but is not an "
            "exact copy of the reference YAML. Check the YAML preview for the real "
            "meta.* values before claiming a paper reproduction.", {})
        )
    if chosen["key"] in ("lstur", "lstur_cat"):
        if item_component == "item.cnn":
            notes.append(
                localized('backend.a790a483e725', "config/model/lstur.yaml uses the CNNCat Item Encoder (category "
                "concatenation). This graph uses CNN. Both are valid Legommenders "
                "CNN+GRU+Dot configurations; README.md lists LSTUR as CNN + GRU + Dot.", {})
            )
    return Archetype(
        key=str(chosen["key"]),
        label=str(chosen["name"]),
        reference_yaml=str(chosen["reference"]),
        exact=exact is not None,
        notes=notes,
    )
