from __future__ import annotations

import traceback
from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class ExecutionRecord:
    step: str
    success: bool
    output_summary: str
    error: str = ''

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SafeExecutor:
    """Guarded executor for callable units with lightweight retry/fallback."""

    def __init__(self):
        self.records: list[ExecutionRecord] = []

    def run(self, step: str, fn, *args, retries: int = 1, **kwargs):
        last_error = ''
        for attempt in range(retries + 1):
            try:
                output = fn(*args, **kwargs)
                self.records.append(ExecutionRecord(step=step, success=True, output_summary=f'attempt={attempt+1}'))
                return output
            except Exception:  # noqa: BLE001
                last_error = traceback.format_exc(limit=3)
        self.records.append(ExecutionRecord(step=step, success=False, output_summary='failed', error=last_error))
        raise RuntimeError(f'Step {step} failed after retries')

    def as_table(self):
        try:
            import pandas as pd
        except ImportError:
            return [r.to_dict() for r in self.records]
        return pd.DataFrame([r.to_dict() for r in self.records])
