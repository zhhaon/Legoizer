"""Language-neutral message references; English strings remain wire-compatible."""
from typing import Any

class LocalizedString(str):
    def __new__(cls, key: str, default: str, params: dict[str, Any]):
        value = super().__new__(cls, default)
        value.reference = {"key": key, "params": params}
        return value

def localized(key: str, default: str, params: dict[str, Any] | None = None) -> str:
    return LocalizedString(key, default, params or {})

def references(**fields: str) -> dict[str, Any]:
    return {name: value.reference for name, value in fields.items() if isinstance(value, LocalizedString)}


def error_reference(detail: Any) -> dict[str, Any]:
    if isinstance(detail, LocalizedString):
        return detail.reference
    value = detail.get('message', str(detail)) if isinstance(detail, dict) else str(detail)
    return {'key': 'api.requestFailed', 'params': {'detail': value}}
