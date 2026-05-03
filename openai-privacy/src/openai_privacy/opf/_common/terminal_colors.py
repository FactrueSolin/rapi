"""ANSI terminal color helpers shared by CLI and eval renderers."""

from typing import Final, Mapping, Sequence

from .label_space import BACKGROUND_CLASS_LABEL


ANSI_RESET: Final[str] = "\x1b[0m"
"""ANSI escape sequence that resets terminal styling."""

# ANSI 256-color foreground codes used for terminal label highlighting.
COLOR_PALETTE: Final[tuple[int, ...]] = (
    39,  # bright blue
    45,  # cyan
    82,  # green
    208,  # orange
    201,  # magenta
    51,  # bright cyan
    99,  # purple
    220,  # yellow
    118,  # lime
    214,  # amber
)
"""ANSI 256-color foreground codes used for label highlighting in terminals."""


def build_label_color_map(labels: Sequence[str]) -> dict[str, int]:
    """Assign a stable ANSI color code to each non-background label.

    Colors cycle deterministically when the label set exceeds the palette size.
    """

    color_map: dict[str, int] = {}
    cursor = 0
    for label in labels:
        if label == BACKGROUND_CLASS_LABEL or label in color_map:
            continue
        color_map[label] = COLOR_PALETTE[cursor % len(COLOR_PALETTE)]
        cursor += 1
    return color_map


def style_labeled_text(
    text: str,
    label: str,
    *,
    label_colors: Mapping[str, int],
) -> str:
    """Wrap text in an ANSI foreground color for the given label."""

    if label == BACKGROUND_CLASS_LABEL:
        return text
    code = label_colors.get(label)
    if code is None:
        return text
    return f"\x1b[38;5;{code}m{text}{ANSI_RESET}"
