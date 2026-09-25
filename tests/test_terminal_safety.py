from __future__ import annotations

import random
import re

from gptty.ui.terminal_safety import sanitize_terminal_text

_VALID_SGR = re.compile(r"\x1b\[[0-9;:]*m")


def _assert_terminal_safe(value: str, *, allow_sgr: bool = False) -> None:
    remainder = _VALID_SGR.sub("", value) if allow_sgr else value
    assert "\x1b" not in remainder
    for char in remainder:
        codepoint = ord(char)
        assert char in {"\n", "\t"} or (
            codepoint >= 0x20
            and codepoint != 0x7F
            and not 0x80 <= codepoint <= 0x9F
        )


def test_plain_unicode_newlines_and_tabs_are_preserved() -> None:
    value = "Latin 中文 👩‍💻 e\u0301 שלום\nnext\tcolumn"

    assert sanitize_terminal_text(value) == value


def test_only_valid_sgr_can_survive_when_explicitly_allowed() -> None:
    value = (
        "a\x1b[1;7mstyled\x1b[0m"
        "\x1b[38;2;255;0;128mtruecolor\x1b[mz"
    )

    assert sanitize_terminal_text(value) == "astyledtruecolorz"
    kept = sanitize_terminal_text(value, allow_sgr=True)
    assert kept == value
    _assert_terminal_safe(kept, allow_sgr=True)


def test_osc_title_hyperlink_and_clipboard_sequences_are_stripped() -> None:
    value = (
        "A"
        "\x1b]0;evil title\x07"
        "B"
        "\x1b]2;evil title\x1b\\"
        "C"
        "\x1b]8;;https://evil.invalid\x1b\\linked\x1b]8;;\x1b\\"
        "D"
        "\x1b]52;c;Y2xpcGJvYXJk\x07"
        "E"
    )

    assert sanitize_terminal_text(value) == "ABClinkedDE"


def test_dcs_apc_pm_sos_and_c1_equivalents_are_stripped() -> None:
    value = (
        "A\x1bPpayload\x1b\\B"
        "\x1b_payload\x1b\\C"
        "\x1b^payload\x1b\\D"
        "\x1bXpayload\x1b\\E"
        "\x90payload\x9cF"
        "\x9fpayload\x9cG"
        "\x9epayload\x9cH"
        "\x98payload\x9cI"
        "\x9dpayload\x9cJ"
    )

    assert sanitize_terminal_text(value) == "ABCDEFGHIJ"


def test_cursor_erase_private_csi_and_single_escape_controls_are_stripped() -> None:
    value = (
        "A\x1b[2JB\x1b[HC\x1b[10CD\x1b[?25lE"
        "\x1b7F\x1b8G\x1b=H"
    )

    assert sanitize_terminal_text(value) == "ABCDEFGH"


def test_c0_del_c1_and_malformed_escape_fragments_never_reach_output() -> None:
    value = (
        "a\x00\x01\x07\x08b\x7fc\x81d"
        "\x1b[31"  # unterminated CSI: consume remainder safely
    )
    safe = sanitize_terminal_text(value)

    assert safe == "abcd"
    _assert_terminal_safe(safe)


def test_c1_sgr_is_normalized_to_seven_bit_sgr_only_when_allowed() -> None:
    value = "A\x9b31mred\x9b0mB"

    assert sanitize_terminal_text(value) == "AredB"
    assert sanitize_terminal_text(value, allow_sgr=True) == (
        "A\x1b[31mred\x1b[0mB"
    )


def test_deterministic_fuzz_never_leaves_terminal_controls() -> None:
    rng = random.Random(0x47505459)
    atoms = [
        "a",
        "中",
        "👩‍💻",
        "\n",
        "\t",
        "\x00",
        "\x07",
        "\x1b",
        "\x1b[31m",
        "\x1b[2J",
        "\x1b]52;c;AAAA\x07",
        "\x1bPabc\x1b\\",
        "\x9b32m",
        "\x9d0;title\x9c",
        "\x90payload\x9c",
        "\x7f",
    ]

    for _ in range(500):
        value = "".join(rng.choice(atoms) for _ in range(rng.randint(0, 25)))
        _assert_terminal_safe(sanitize_terminal_text(value))
        _assert_terminal_safe(
            sanitize_terminal_text(value, allow_sgr=True),
            allow_sgr=True,
        )
