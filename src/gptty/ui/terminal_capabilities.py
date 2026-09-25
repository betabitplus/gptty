from __future__ import annotations

from typing import Any

_ALTERNATE_SCROLL_ENABLE = "\x1b[?1007h"
_ALTERNATE_SCROLL_DISABLE = "\x1b[?1007l"


class TerminalCapabilities:
    """Own terminal-private mode changes used by the persistent TUI."""

    def __init__(self, output: Any) -> None:
        self._output = output
        self._alternate_scroll_enabled = False

    @property
    def alternate_scroll_enabled(self) -> bool:
        return self._alternate_scroll_enabled

    def enter_persistent_tui(self) -> None:
        if self._alternate_scroll_enabled:
            return
        try:
            self._output.write_raw(_ALTERNATE_SCROLL_ENABLE)
            # The enable may already have reached the terminal even if flush
            # fails, so restoration must remain armed.
            self._alternate_scroll_enabled = True
            self._output.flush()
        except Exception:
            return

    def restore(self) -> None:
        if not self._alternate_scroll_enabled:
            return
        try:
            self._output.write_raw(_ALTERNATE_SCROLL_DISABLE)
            self._output.flush()
        except Exception:
            # Keep restoration armed so a later normal shutdown/done callback
            # can retry instead of silently declaring the terminal restored.
            return
        self._alternate_scroll_enabled = False


__all__ = ["TerminalCapabilities"]
