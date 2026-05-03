from __future__ import annotations

import argparse

from .._cli.common import (
    CliHelpFormatter,
    add_checkpoint_arg,
    add_device_arg,
    add_n_ctx_arg,
    resolve_prog,
)


def build_parser(*, prog: str | None = None) -> argparse.ArgumentParser:
    """Build the ``opf train`` argument parser."""
    epilog = (
        "Examples:\n"
        "  opf train /path/to/train.jsonl "
        "--validation-dataset /path/to/validation.jsonl "
        "--output-dir /tmp/opf_finetuned\n"
        "  opf train /path/to/train.jsonl --checkpoint /path/to/base_ckpt "
        "--label-space-json /path/to/custom_label_space.json "
        "--output-dir /tmp/opf_custom_labels\n"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune an existing OPF checkpoint on a local labeled dataset. "
            "Outputs a reusable checkpoint directory with config.json and model.safetensors."
        ),
        formatter_class=CliHelpFormatter,
        prog=prog or resolve_prog("opf train"),
        epilog=epilog,
    )

    input_group = parser.add_argument_group("Input / Source")
    runtime_group = parser.add_argument_group("Model / Runtime")
    optimize_group = parser.add_argument_group("Optimization")
    output_group = parser.add_argument_group("Output")

    input_group.add_argument(
        "dataset",
        type=str,
        help="Local train dataset path or glob (JSON/JSONL(.gz))",
    )
    input_group.add_argument(
        "--dataset-variant",
        choices=("full", "message"),
        default="full",
        help="Dataset preprocessing variant (full records or per-message expansion).",
    )
    input_group.add_argument(
        "--validation-dataset",
        type=str,
        default=None,
        help="Optional explicit validation dataset path/glob.",
    )
    input_group.add_argument(
        "--validation-dataset-variant",
        choices=("full", "message"),
        default=None,
        help="Optional validation dataset variant (defaults to --dataset-variant).",
    )
    input_group.add_argument(
        "--validation-split",
        type=float,
        default=0.1,
        help=(
            "If --validation-dataset is omitted, reserve this fraction of train examples "
            "for validation (0 disables validation split)."
        ),
    )
    input_group.add_argument(
        "--shuffle-seed",
        type=int,
        default=0,
        help="Random seed used for train/validation split and epoch shuffles.",
    )
    input_group.add_argument(
        "--max-train-examples",
        type=int,
        default=None,
        help="Optional cap on loaded training examples.",
    )
    input_group.add_argument(
        "--max-validation-examples",
        type=int,
        default=None,
        help="Optional cap on loaded validation examples.",
    )
    input_group.add_argument(
        "--label-space-json",
        type=str,
        default=None,
        help=(
            "Optional JSON file defining custom labels (span_class_names and/or "
            "ner_class_names). Use this for non-default ontologies. "
            "When span_class_names is set, include 'O' as the first entry."
        ),
    )

    add_checkpoint_arg(runtime_group)
    add_device_arg(runtime_group)
    add_n_ctx_arg(runtime_group)

    optimize_group.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of finetuning epochs.",
    )
    optimize_group.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of same-length windows per optimizer micro-batch.",
    )
    optimize_group.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps before optimizer update.",
    )
    optimize_group.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="AdamW learning rate.",
    )
    optimize_group.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="AdamW weight decay.",
    )
    optimize_group.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Gradient clipping max norm (<=0 disables clipping).",
    )

    output_group.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help=(
            "Directory to write finetuned checkpoint artifacts "
            "(config.json, model.safetensors, finetune_summary.json, USAGE.txt)."
        ),
    )
    output_group.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Allow writing into a non-empty output directory.",
    )
    output_group.add_argument(
        "--output-param-dtype",
        choices=("inherit", "bf16", "fp32"),
        default="inherit",
        help="Parameter dtype used when writing model.safetensors.",
    )
    output_group.add_argument(
        "--summary-name",
        type=str,
        default="finetune_summary.json",
        help="Filename for the written finetuning summary JSON.",
    )

    return parser


def parse_args(
    argv: list[str] | None = None, *, prog: str | None = None
) -> argparse.Namespace:
    """Parse ``opf train`` arguments."""
    return build_parser(prog=prog).parse_args(argv)
