from __future__ import annotations

from pathlib import Path

import yaml

from evals.types import QASuite


def load_suite(path: Path) -> QASuite:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    suite = QASuite.model_validate(raw)

    seen: set[str] = set()
    for case in suite.cases:
        if case.id in seen:
            raise ValueError(f"Duplicate eval case id: {case.id}")
        seen.add(case.id)
    return suite
