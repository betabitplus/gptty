from __future__ import annotations

import signal

import pytest

from gptty.ui.signals import turn_control_signals


def test_turn_control_signals_map_ctrl_keys_and_restore_handlers() -> None:
    sigquit = getattr(signal, "SIGQUIT", None)
    if sigquit is None:
        pytest.skip("SIGQUIT is unavailable on this platform")

    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigquit = signal.getsignal(sigquit)
    with turn_control_signals(enabled=True) as controls:
        signal.raise_signal(signal.SIGINT)
        assert controls.stop_requested.is_set()
        assert controls.wake.is_set()
        assert controls.consume_stop() is True
        assert not controls.stop_requested.is_set()

        signal.raise_signal(sigquit)
        assert controls.quit_requested.is_set()
        assert controls.wake.is_set()

    assert signal.getsignal(signal.SIGINT) == previous_sigint
    assert signal.getsignal(sigquit) == previous_sigquit
