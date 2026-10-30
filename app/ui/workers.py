"""Qt adapters that run backend jobs off the GUI thread."""
from __future__ import annotations

import threading
from typing import Any, Callable

from PyQt6.QtCore import QObject, QRunnable, QThread, QThreadPool, pyqtSignal

from app.services.jobs import run_job


class _Signals(QObject):
    # NOTE: never name a signal after a QObject/QWidget method. In particular a signal
    # called ``event`` shadows the virtual QObject.event(QEvent) that Qt calls for every
    # delivered event, which raises "TypeError: native Qt signal is not callable".
    succeeded = pyqtSignal(object)  # result
    failed = pyqtSignal(str, str)  # message, details


class FunctionJob(QRunnable):
    """Run ``fn(*args, **kwargs)`` in the global thread pool."""

    def __init__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.signals = _Signals()
        self._fn, self._args, self._kwargs = fn, args, kwargs
        self.setAutoDelete(True)

    def run(self) -> None:  # executed in a worker thread
        outcome = run_job(self._fn, *self._args, **self._kwargs)
        if outcome.ok:
            self.signals.succeeded.emit(outcome.result)
        else:
            self.signals.failed.emit(outcome.error, outcome.details)


_ACTIVE: set[FunctionJob] = set()


def run_in_background(fn: Callable[..., Any], *args: Any,
                      on_done: Callable[[Any], None] | None = None,
                      on_error: Callable[[str, str], None] | None = None, **kwargs: Any) -> FunctionJob:
    job = FunctionJob(fn, *args, **kwargs)
    _ACTIVE.add(job)  # keep the Python wrapper (and its signals) alive until it reports

    def _done(result: Any) -> None:
        _ACTIVE.discard(job)
        if on_done:
            on_done(result)

    def _err(msg: str, details: str) -> None:
        _ACTIVE.discard(job)
        if on_error:
            on_error(msg, details)

    job.signals.succeeded.connect(_done)
    job.signals.failed.connect(_err)
    QThreadPool.globalInstance().start(job)
    return job


class ExperimentWorker(QObject):
    """Runs one federated experiment (or sweep) in a dedicated QThread.

    ``fn`` receives ``on_event`` and ``stop_event`` keyword arguments. Signals:

    * ``progress(dict)``   - every progress event from the backend
    * ``succeeded(object)`` - the job's return value
    * ``failed(str, str)``  - error message and traceback
    * ``done()``            - always emitted last (after succeeded/failed)

    Connect these only to methods of QObjects living in the GUI thread (e.g. page
    methods) so Qt queues the calls into the GUI thread; never touch widgets from
    ``fn``. Call ``wait()`` in the ``done`` slot before releasing the worker.
    """

    progress = pyqtSignal(object)
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str, str)
    done = pyqtSignal()

    def __init__(self, fn: Callable[..., Any]) -> None:
        super().__init__()
        self._fn = fn
        self.stop_event = threading.Event()
        self._qthread = QThread()
        self.moveToThread(self._qthread)
        self._qthread.started.connect(self._run)

    def start(self) -> None:
        self._qthread.start()

    def stop(self) -> None:
        self.stop_event.set()

    def is_running(self) -> bool:
        return self._qthread.isRunning()

    def _run(self) -> None:  # executes in the worker thread
        try:
            outcome = run_job(self._fn, on_event=self.progress.emit, stop_event=self.stop_event)
            if outcome.ok:
                self.succeeded.emit(outcome.result)
            else:
                self.failed.emit(outcome.error, outcome.details)
        finally:
            self.done.emit()
            self._qthread.quit()

    def wait(self, ms: int = 10000) -> bool:
        return self._qthread.wait(ms)
