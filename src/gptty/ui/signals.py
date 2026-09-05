from __future__ import annotations

import signal
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class TurnControlSignals:
    """Signal-backed local control state for one enhanced TUI turn."""

    stop_requested: threading.Event = field(default_factory=threading.Event)
    quit_requested: threading.Event = field(default_factory=threading.Event)
    wake: threading.Event = field(default_factory=threading.Event)

    def request_stop(self) -> None:
        self.stop_requested.set()
        self.wake.set()

    def request_quit(self) -> None:
        self.quit_requested.set()
        self.wake.set()

    def consume_stop(self) -> bool:
        if not self.stop_requested.is_set():
            return False
        self.stop_requested.clear()
        self.wake.clear()
        return True


@contextmanager
def turn_control_signals(*, enabled: bool) -> Iterator[TurnControlSignals]:
    """Map Ctrl-C/Ctrl-\\ to turn-control flags and restore prior handlers."""
    controls = TurnControlSignals()
    sigquit = getattr(signal, "SIGQUIT", None)
    if not enabled or threading.current_thread() is not threading.main_thread():
        yield controls
        return

    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigquit = signal.getsignal(sigquit) if sigquit is not None else None

    def request_stop(_signum: int, _frame: object) -> None:
        controls.request_stop()

    def request_quit(_signum: int, _frame: object) -> None:
        controls.request_quit()

    signal.signal(signal.SIGINT, request_stop)
    if sigquit is not None:
        signal.signal(sigquit, request_quit)
    try:
        yield controls
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        if sigquit is not None and previous_sigquit is not None:
            signal.signal(sigquit, previous_sigquit)
