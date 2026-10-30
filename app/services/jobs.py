"""Toolkit-independent background job execution.

The Qt worker in ``app/ui/workers.py`` is a thin adapter around ``run_job`` so
that error capture, cancellation and event forwarding are unit-testable
without a display.
"""
from __future__ import annotations

import threading
import traceback
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class JobOutcome:
    ok: bool
    result: Any = None
    error: str = ""
    details: str = ""
    cancelled: bool = False


def run_job(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> JobOutcome:
    """Run ``fn`` and convert any exception into a structured, displayable outcome."""
    try:
        return JobOutcome(ok=True, result=fn(*args, **kwargs))
    except Exception as exc:  # noqa: BLE001 - every failure must reach the UI
        return JobOutcome(ok=False, error=f"{type(exc).__name__}: {exc}",
                          details=traceback.format_exc())


class CancelToken:
    def __init__(self) -> None:
        self.event = threading.Event()

    def cancel(self) -> None:
        self.event.set()

    @property
    def cancelled(self) -> bool:
        return self.event.is_set()
