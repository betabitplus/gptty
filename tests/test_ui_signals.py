from __future__ import annotations

import signal

from gptty.ui.signals import (
    TurnControlSignals,
    routed_turn_control_signals,
)


def test_routed_turn_control_signals_routes_real_sigint_to_active_turn() -> None:
    previous_sigint = signal.getsignal(signal.SIGINT)
    controls = TurnControlSignals()
    active: TurnControlSignals | None = controls

    with routed_turn_control_signals(enabled=True, controls=lambda: active):
        signal.raise_signal(signal.SIGINT)
        assert controls.stop_requested.is_set()
        assert controls.stop_intent.is_set()
        assert controls.wake.is_set()

    assert signal.getsignal(signal.SIGINT) == previous_sigint
