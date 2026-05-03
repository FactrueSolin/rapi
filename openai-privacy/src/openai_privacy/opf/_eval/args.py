import argparse

from .._cli.common import (
    CliHelpFormatter,
    add_checkpoint_arg,
    add_decode_mode_arg,
    add_device_arg,
    add_discard_overlapping_predicted_spans_arg,
    add_n_ctx_arg,
    add_trim_whitespace_args,
    add_viterbi_args,
    resolve_prog,
)


def build_parser(*, prog: str | None = None) -> argparse.ArgumentParser:
    """Build the ``opf eval`` argument parser."""
    parser = argparse.ArgumentParser(
        description="Run encoder eval on a ground-truth dataset.",
        formatter_class=CliHelpFormatter,
        prog=prog or resolve_prog("opf eval"),
    )
    input_group = parser.add_argument_group("Input / Source")
    runtime_group = parser.add_argument_group("Model / Runtime")
    decode_group = parser.add_argument_group("Decode")
    output_group = parser.add_argument_group("Output")
    add_checkpoint_arg(runtime_group)
    add_device_arg(runtime_group)
    add_n_ctx_arg(runtime_group)
    runtime_group.add_argument(
        "--window-batch-size",
        type=int,
        default=1,
        help=(
            "Maximum rows per eval forward batch. "
            "Rows are right-padded to the batch-local max length."
        ),
    )
    add_decode_mode_arg(decode_group)
    add_discard_overlapping_predicted_spans_arg(decode_group)
    decode_group.add_argument(
        "--discard-overlapping-ground-truth-spans",
        action="store_true",
        help="Discard overlapping ground-truth spans (per label)",
    )
    decode_group.add_argument(
        "--debug-decode",
        action="store_true",
        help="Print a decode alignment check for the first example",
    )
    decode_group.add_argument(
        "--skip-non-ascii-examples",
        action="store_true",
        help="Skip examples that contain non-ASCII characters",
    )
    add_trim_whitespace_args(
        parser,
        decode_group,
    )
    decode_group.add_argument(
        "--span-metrics-space",
        choices=("char", "token"),
        default="char",
        help="Evaluate spans in character or token space (default: char)",
    )
    decode_group.add_argument(
        "--eval-mode",
        choices=("typed", "untyped"),
        default="typed",
        help=(
            "typed requires matching opf categories and reports category-level metrics; "
            "untyped ignores category identity and reports span-level detection metrics."
        ),
    )
    add_viterbi_args(decode_group)
    output_group.add_argument(
        "--per-class",
        action="store_true",
        help="Print per-class metrics (typed mode only)",
    )
    output_group.add_argument(
        "--label-counts",
        action="store_true",
        help="Print ground truth vs predicted label counts (typed mode only)",
    )
    output_group.add_argument(
        "--preview",
        action="store_true",
        help="Print one ANSI color-coded preview with highlighted predictions and per-token tags.",
    )
    output_group.add_argument(
        "--preview-example-id",
        type=str,
        default=None,
        help="example_id to preview (defaults to the first decoded example).",
    )
    output_group.add_argument(
        "--preview-max-tokens",
        type=int,
        default=256,
        help="Maximum number of token rows to print in the preview section.",
    )
    output_group.add_argument(
        "--preview-max-chars",
        type=int,
        default=4000,
        help="Maximum number of text characters to print in the preview section.",
    )
    runtime_group.add_argument(
        "--attn-low-precision",
        action="store_true",
        help="Use low precision for rotary caches and attention math (lower quality).",
    )
    runtime_group.add_argument(
        "--experts-per-token",
        type=int,
        default=None,
        help="Override MoE experts-per-token at inference.",
    )
    runtime_group.add_argument(
        "--moe-triton",
        action="store_true",
        help="Use Triton kernels for MoE (experimental).",
    )
    output_group.add_argument(
        "--predictions-out",
        type=str,
        default=None,
        help="Path to write per-example predicted spans (JSONL)",
    )
    output_group.add_argument(
        "--prediction-write-workers",
        type=int,
        default=0,
        help=("CPU workers for predictions JSONL rendering. 0 = auto, 1 = disabled."),
    )
    output_group.add_argument(
        "--timings-out",
        type=str,
        default=None,
        help="Path to write key eval timings and throughput as JSON.",
    )
    output_group.add_argument(
        "--metrics-out",
        type=str,
        default=None,
        help="Path to write machine-readable eval metrics (JSON)",
    )
    output_group.add_argument(
        "--predictions-token-logprobs-topk",
        type=int,
        default=0,
        help="Include top-k logprobs per token in the predictions output",
    )
    output_group.add_argument(
        "--predictions-token-logprobs-example-id",
        type=str,
        default=None,
        help="Only include token logprobs for this example_id",
    )
    output_group.add_argument(
        "--predictions-token-logprobs-max-tokens",
        type=int,
        default=None,
        help="Limit number of tokens to include in token logprobs output",
    )
    input_group.add_argument(
        "dataset",
        type=str,
        help="Local path/glob for ground-truth JSON/JSONL(.gz) records.",
    )
    input_group.add_argument(
        "--dataset-variant",
        type=str,
        choices=("full", "message"),
        default="full",
        help="Dataset variant preprocessing (e.g. message-level extraction).",
    )
    input_group.add_argument(
        "--preprocess-workers",
        type=int,
        default=0,
        help=(
            "CPU worker processes for dataset preprocessing "
            "(parse+tokenize+label). 0 = auto, 1 = disabled."
        ),
    )
    input_group.add_argument(
        "--preprocess-chunksize",
        type=int,
        default=16,
        help="Task chunksize used by --preprocess-workers process map.",
    )
    input_group.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Limit number of examples for quick iteration",
    )
    input_group.add_argument(
        "--progress-every",
        type=int,
        default=None,
        help="Print a compact progress update every N processed examples",
    )
    return parser


def parse_args(
    argv: list[str] | None = None, *, prog: str | None = None
) -> argparse.Namespace:
    """Parse ``opf eval`` arguments."""
    return build_parser(prog=prog).parse_args(argv)
