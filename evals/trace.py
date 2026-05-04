from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any

SECRET_KEYS = ("api_key", "apikey", "authorization", "token", "secret", "password")
SAFE_TOKEN_COUNT_KEYS = {"input_tokens", "output_tokens", "total_tokens"}


def redact_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            key_lower = str(key).lower()
            if key_lower in SAFE_TOKEN_COUNT_KEYS:
                out[key] = redact_secrets(item)
            elif any(secret in key_lower for secret in SECRET_KEYS):
                out[key] = "[REDACTED]"
            else:
                out[key] = redact_secrets(item)
        return out
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    return value


class TraceCollector:
    def __init__(self, *, run_id: str, suite: str, case_id: str, trial: int) -> None:
        self.data: dict[str, Any] = {
            "run_id": run_id,
            "suite": suite,
            "case_id": case_id,
            "trial": trial,
            "created_at": datetime.utcnow().isoformat(),
        }

    def record(self, section: str, value: Any) -> None:
        self.data[section] = deepcopy(value)

    def append(self, section: str, value: Any) -> None:
        self.data.setdefault(section, []).append(deepcopy(value))

    def to_dict(self) -> dict[str, Any]:
        return redact_secrets(self.data)
