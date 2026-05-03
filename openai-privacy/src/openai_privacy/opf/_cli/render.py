from __future__ import annotations

import argparse
import sys
from types import SimpleNamespace
from typing import Mapping, Sequence, TYPE_CHECKING

from .._common.terminal_colors import style_labeled_text

if TYPE_CHECKING:
    from .._api import OPF
    from .._core.runtime import DetectedSpan


def render_color_coded_text(
    *,
    text: str,
    spans: Sequence[DetectedSpan],
    label_colors: Mapping[str, int],
) -> str:
    """Render color-coded text for the detected spans."""
    if not spans:
        return text
    pieces: list[str] = []
    cursor = 0
    text_len = len(text)
    for span in spans:
        start = max(0, min(text_len, int(span.start)))
        end = max(start, min(text_len, int(span.end)))
        if end <= cursor:
            continue
        if start < cursor:
            start = cursor
        if start > cursor:
            pieces.append(text[cursor:start])
        pieces.append(
            style_labeled_text(
                text[start:end],
                span.label,
                label_colors=label_colors,
            )
        )
        cursor = end
    if cursor < text_len:
        pieces.append(text[cursor:])
    return "".join(pieces)


def print_session_header(
    *,
    checkpoint: str,
    device: str,
    encoding_name: str,
    n_ctx: int,
    output_mode: str,
) -> None:
    """Print a one-line CLI session header to stderr."""
    print(
        "session: "
        f"checkpoint={checkpoint} "
        f"device={device} "
        f"encoding={encoding_name} "
        f"n_ctx={n_ctx} "
        f"output_mode={output_mode}",
        file=sys.stderr,
    )


def render_color_legend(*, label_colors: Mapping[str, int]) -> str:
    """Render the CLI color legend line."""
    if not label_colors:
        return "color legend: (none)"
    items: list[str] = []
    for label in label_colors:
        items.append(
            style_labeled_text(
                label,
                label,
                label_colors=label_colors,
            )
        )
    return "color legend: " + " | ".join(items)


def run_summary_line(
    *,
    summary: Mapping[str, object],
    latency_ms: float,
) -> str:
    """Build the stderr summary line for one CLI inference result."""
    counts = summary["by_label"]
    if not isinstance(counts, dict):
        raise ValueError("summary.by_label must be a dict")
    by_label = (
        ", ".join(f"{label}:{count}" for label, count in counts.items())
        if counts
        else "-"
    )
    mismatch = "yes" if summary["decoded_mismatch"] else "no"
    return (
        f"summary: output_mode={summary['output_mode']} "
        f"spans={summary['span_count']} "
        f"by_label={by_label} "
        f"latency_ms={latency_ms:.1f} "
        f"decoded_mismatch={mismatch}"
    )


def build_redactor_from_args(
    args: argparse.Namespace, *, output_text_only: bool = False
):
    """Construct an ``OPF`` redactor from parsed CLI arguments."""
    from .._api import OPF

    redactor = OPF(
        model=args.checkpoint,
        context_window_length=args.n_ctx,
        trim_whitespace=args.trim_span_whitespace,
        device=args.device,
        output_mode=args.output_mode,
        discard_overlapping_predicted_spans=args.discard_overlapping_predicted_spans,
        output_text_only=output_text_only,
    )
    if args.decode_mode == "viterbi":
        return redactor.set_viterbi_decoder(
            calibration_path=args.viterbi_calibration_path,
        )
    return redactor.set_decode_mode("argmax")


def build_session_runtime_view(redactor: OPF) -> SimpleNamespace:
    """Extract the runtime fields needed by the CLI renderer."""
    runtime = redactor.get_runtime()
    return SimpleNamespace(
        checkpoint=runtime.checkpoint,
        device=runtime.device,
        active_encoding_name=runtime.active_encoding_name,
        n_ctx=runtime.n_ctx,
        category_version=runtime.category_version,
        bidirectional_context=runtime.bidirectional_context,
        output_mode=runtime.output_mode,
        label_info=runtime.label_info,
    )
