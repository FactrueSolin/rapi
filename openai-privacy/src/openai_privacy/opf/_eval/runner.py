from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence, TextIO

from .args import parse_args

import torch
import torch.nn.functional as F
import tiktoken

from .._api import OPF, resolve_checkpoint_path
from .data import (
    iter_json_records,
    iter_message_variant_records,
)
from .metrics import compute_metrics, write_predictions
from .render import build_prediction_preview
from .._common.env import get_env_bool
from .._common.label_space import resolve_checkpoint_label_space
from .._core.sequence_labeling import (
    BACKGROUND_CLASS_LABEL,
    LabelInfo,
    TokenizedExample,
    Window,
    example_to_windows,
)
from .preprocess import (
    EvalExampleAggregation,
    PreparedTokenizedExample,
    labels_from_entities,
    parse_record,
    token_char_ranges,
)


_PREPROCESS_WORKER_ENCODING_NAME: str | None = None
_PREPROCESS_WORKER_LABEL_INFO: "LabelInfo | None" = None
_PREPROCESS_WORKER_EVAL_MODE: str = "typed"
_PREPROCESS_WORKER_SKIP_NON_ASCII: bool = False


@functools.lru_cache(maxsize=8)
def _cached_encoding(requested_encoding_name: str) -> tiktoken.Encoding:
    """Return the cached tokenizer encoding for eval preprocessing."""
    return tiktoken.get_encoding(requested_encoding_name)


def _resolve_preprocess_workers(requested: int, *, device: torch.device) -> int:
    """Resolve the effective preprocess worker count."""
    if requested < 0:
        raise ValueError("--preprocess-workers must be >= 0")
    if requested == 0:
        if device.type != "cuda":
            return 1
        cpu_count = os.cpu_count() or 1
        return max(1, min(8, cpu_count - 1))
    return requested


def _resolve_prediction_write_workers(requested: int) -> int:
    """Resolve the effective predictions-output worker count."""
    if requested < 0:
        raise ValueError("--prediction-write-workers must be >= 0")
    if requested == 0:
        cpu_count = os.cpu_count() or 1
        return max(1, min(8, cpu_count // 2))
    return requested


def _prepare_tokenized_example(
    *,
    record: Mapping[str, object],
    idx: int,
    encoding: tiktoken.Encoding,
    label_info: "LabelInfo",
    eval_mode: str,
    skip_non_ascii_examples: bool,
) -> PreparedTokenizedExample | None:
    """Tokenize and label one eval record."""
    example_id, text, entities = parse_record(record, idx)
    if skip_non_ascii_examples and any(ord(ch) > 127 for ch in text):
        return None
    tokens = list(encoding.encode(text, allowed_special="all"))
    char_starts, char_ends = token_char_ranges(tokens, encoding, text)
    labels = labels_from_entities(
        tokens,
        text,
        entities,
        label_info,
        encoding,
        eval_mode=eval_mode,
        token_char_ranges_hint=(char_starts, char_ends),
    )
    return PreparedTokenizedExample(
        tokenized=TokenizedExample(
            tokens=tuple(tokens),
            labels=tuple(labels),
            example_id=example_id,
            text=text,
        ),
        char_starts=tuple(char_starts),
        char_ends=tuple(char_ends),
        original_gold_char_spans=tuple(
            (entity.label, entity.start, entity.end) for entity in entities
        ),
    )


def _init_preprocess_worker(
    encoding_name: str,
    label_info: "LabelInfo",
    eval_mode: str,
    skip_non_ascii_examples: bool,
) -> None:
    """Initialize one eval preprocess worker process."""
    global _PREPROCESS_WORKER_ENCODING_NAME
    global _PREPROCESS_WORKER_LABEL_INFO
    global _PREPROCESS_WORKER_EVAL_MODE
    global _PREPROCESS_WORKER_SKIP_NON_ASCII
    _PREPROCESS_WORKER_ENCODING_NAME = encoding_name
    _PREPROCESS_WORKER_LABEL_INFO = label_info
    _PREPROCESS_WORKER_EVAL_MODE = eval_mode
    _PREPROCESS_WORKER_SKIP_NON_ASCII = skip_non_ascii_examples


def _prepare_tokenized_example_worker(
    item: tuple[int, Mapping[str, object]],
) -> PreparedTokenizedExample | None:
    """Worker entrypoint for preprocessing one eval record."""
    if (
        _PREPROCESS_WORKER_ENCODING_NAME is None
        or _PREPROCESS_WORKER_LABEL_INFO is None
    ):
        raise RuntimeError("Preprocess worker is not initialized")
    idx, record = item
    encoding = _cached_encoding(_PREPROCESS_WORKER_ENCODING_NAME)
    return _prepare_tokenized_example(
        record=record,
        idx=idx,
        encoding=encoding,
        label_info=_PREPROCESS_WORKER_LABEL_INFO,
        eval_mode=_PREPROCESS_WORKER_EVAL_MODE,
        skip_non_ascii_examples=_PREPROCESS_WORKER_SKIP_NON_ASCII,
    )


def _first_text_mismatch(left: str, right: str) -> int | None:
    """Return the first differing character index between two strings."""
    min_len = min(len(left), len(right))
    for idx in range(min_len):
        if left[idx] != right[idx]:
            return idx
    if len(left) != len(right):
        return min_len
    return None


def _format_metric(value: float | int | None) -> str:
    """Format one metric value for terminal output."""
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def _format_seconds(value: float | None) -> str:
    """Format one optional seconds value for terminal output."""
    if value is None:
        return "-"
    return f"{float(value):.4f}s"


def _safe_throughput(tokens: int, seconds: float | None) -> float | None:
    """Return throughput when the denominator is valid."""
    if seconds is None:
        return None
    if seconds <= 0.0:
        return None
    return float(tokens) / float(seconds)


def _rate_per_second(count: int, elapsed_s: float) -> float | None:
    """Compute a simple per-second rate."""
    if elapsed_s <= 0:
        return None
    return float(count) / float(elapsed_s)


def _print_progress_line(
    *,
    processed_examples: int,
    max_examples: int | None,
    total_windows: int,
    total_window_tokens: int,
    total_padded_window_tokens: int,
    elapsed_s: float,
    output: TextIO | None = None,
) -> None:
    """Print a compact eval progress line."""
    stream = sys.stderr if output is None else output
    example_text = str(processed_examples)
    if max_examples is not None:
        example_text = f"{processed_examples}/{max_examples}"
    window_rate = _rate_per_second(total_window_tokens, elapsed_s)
    window_rate_text = "-" if window_rate is None else f"{window_rate:.1f}"
    padded_window_rate = _rate_per_second(total_padded_window_tokens, elapsed_s)
    padded_window_rate_text = (
        "-" if padded_window_rate is None else f"{padded_window_rate:.1f}"
    )
    print(
        "progress: "
        f"examples={example_text} "
        f"windows={total_windows} "
        f"window_tokens={total_window_tokens} "
        f"padded_window_tokens={total_padded_window_tokens} "
        f"elapsed_s={elapsed_s:.1f} "
        f"window_tokens_per_s={window_rate_text} "
        f"padded_window_tokens_per_s={padded_window_rate_text}",
        file=stream,
        flush=True,
    )


def _build_eval_redactor(args: argparse.Namespace) -> OPF:
    """Build the shared ``OPF`` redactor used by eval."""
    redactor = OPF(
        model=args.checkpoint,
        context_window_length=args.n_ctx,
        trim_whitespace=args.trim_span_whitespace,
        device=args.device,
        output_mode="typed",
        decode_mode=args.decode_mode,
        discard_overlapping_predicted_spans=args.discard_overlapping_predicted_spans,
    )
    if args.decode_mode == "argmax":
        redactor.set_decode_mode("argmax")
        return redactor
    redactor.set_viterbi_decoder(
        calibration_path=args.viterbi_calibration_path,
    )
    return redactor


def _jsonable_mapping(metrics: Mapping[str, object]) -> dict[str, object]:
    """Convert a metrics mapping into a JSON-safe scalar mapping."""
    result: dict[str, object] = {}
    for key, value in metrics.items():
        if isinstance(value, (str, bool)) or value is None:
            result[key] = value
        elif isinstance(value, int):
            result[key] = int(value)
        elif isinstance(value, float):
            result[key] = float(value)
    return result


def _write_metrics_payload(
    *,
    output_path: str,
    args: argparse.Namespace,
    device: torch.device,
    n_ctx: int,
    active_encoding_name: str,
    category_version: str,
    total_windows: int,
    total_window_tokens: int,
    total_padded_window_tokens: int,
    elapsed_s: float,
    metrics: Mapping[str, float],
) -> None:
    """Write the machine-readable eval metrics payload."""
    summary = {
        "examples": int(metrics["n_examples"]),
        "tokens": int(metrics["n_tokens"]),
        "windows": int(total_windows),
        "window_tokens": int(total_window_tokens),
        "padded_window_tokens": int(total_padded_window_tokens),
        "elapsed_s": float(elapsed_s),
        "tokens_per_s": _rate_per_second(int(metrics["n_tokens"]), elapsed_s),
        "window_tokens_per_s": _rate_per_second(total_window_tokens, elapsed_s),
        "padded_window_tokens_per_s": _rate_per_second(
            total_padded_window_tokens, elapsed_s
        ),
        "eval_mode": args.eval_mode,
        "loss": float(metrics["loss"]) if "loss" in metrics else None,
        "token_accuracy": (
            float(metrics["token_accuracy"]) if "token_accuracy" in metrics else None
        ),
    }
    payload = {
        "summary": summary,
        "metrics": _jsonable_mapping(metrics),
        "config": {
            "checkpoint": args.checkpoint,
            "dataset": args.dataset,
            "dataset_variant": args.dataset_variant,
            "category_version": category_version,
            "device": str(device),
            "resolved_n_ctx": int(n_ctx),
            "encoding": active_encoding_name,
            "attention_impl": "banded",
            "decode_mode": args.decode_mode,
            "viterbi_calibration_path": (
                args.viterbi_calibration_path if args.decode_mode == "viterbi" else None
            ),
        },
        "args": {
            key: value
            for key, value in vars(args).items()
            if isinstance(value, (str, bool, int, float)) or value is None
        },
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _print_key_value_table(
    *, title: str, key_header: str, value_header: str, rows: Sequence[Sequence[str]]
) -> None:
    """Print a two-column table to stdout."""
    _print_table(title=title, headers=(key_header, value_header), rows=rows)


def _print_table(
    *, title: str, headers: Sequence[str], rows: Sequence[Sequence[str]]
) -> None:
    """Print a generic fixed-width text table."""
    print(title)
    if not rows:
        print("  (none)")
        return
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))
    print(
        "  "
        + "  ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers))
    )
    print("  " + "  ".join("-" * width for width in widths))
    for row in rows:
        print("  " + "  ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row)))


def _split_per_class_metric_tables(
    metrics: Mapping[str, float],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Split flat by-class metrics into span and token metric tables."""
    span_metrics: dict[str, dict[str, float]] = {}
    token_metrics: dict[str, dict[str, float]] = {}
    for key, value in metrics.items():
        if not key.startswith("by_class."):
            continue
        suffix = key[len("by_class.") :]
        if ".span." in suffix:
            label, metric_name = suffix.split(".span.", 1)
            span_metrics.setdefault(label, {})[metric_name] = float(value)
            continue
        label, metric_name = suffix.rsplit(".", 1)
        token_metrics.setdefault(label, {})[metric_name] = float(value)
    return span_metrics, token_metrics


def _ordered_labels(labels: Sequence[str], preferred_order: Sequence[str]) -> list[str]:
    """Order labels by a preferred sequence, then append the rest sorted."""
    seen: set[str] = set()
    ordered: list[str] = []
    for label in preferred_order:
        if label in labels and label not in seen:
            ordered.append(label)
            seen.add(label)
    for label in sorted(labels):
        if label not in seen:
            ordered.append(label)
            seen.add(label)
    return ordered


def _print_per_class_metrics(
    metrics: Mapping[str, float],
    *,
    span_class_names: Sequence[str],
    ner_class_names: Sequence[str],
) -> None:
    """Print per-class span and token metric tables."""
    span_metrics, token_metrics = _split_per_class_metric_tables(metrics)
    span_labels = _ordered_labels(
        list(span_metrics),
        [label for label in span_class_names if label != BACKGROUND_CLASS_LABEL],
    )
    span_rows = [
        [
            label,
            _format_metric(span_metrics[label].get("precision")),
            _format_metric(span_metrics[label].get("recall")),
            _format_metric(span_metrics[label].get("f1")),
            _format_metric(span_metrics[label].get("f2")),
        ]
        for label in span_labels
    ]
    _print_table(
        title="per_class_span_metrics:",
        headers=("label", "precision", "recall", "f1", "f2"),
        rows=span_rows,
    )

    token_labels = _ordered_labels(
        list(token_metrics),
        [label for label in ner_class_names if label != BACKGROUND_CLASS_LABEL],
    )
    token_rows = [
        [
            label,
            _format_metric(token_metrics[label].get("precision")),
            _format_metric(token_metrics[label].get("recall")),
            _format_metric(token_metrics[label].get("f1")),
            _format_metric(token_metrics[label].get("f2")),
            _format_metric(token_metrics[label].get("loss")),
        ]
        for label in token_labels
    ]
    _print_table(
        title="per_class_token_metrics:",
        headers=("label", "precision", "recall", "f1", "f2", "loss"),
        rows=token_rows,
    )


def _print_label_counts(
    label_counts: Mapping[str, Mapping[str, int]],
    *,
    ner_class_names: Sequence[str],
) -> None:
    """Print ground-truth versus predicted label counts."""
    ground_truth = label_counts.get("gold", {})
    pred = label_counts.get("pred", {})
    labels = _ordered_labels(
        list(set(ground_truth) | set(pred)),
        list(ner_class_names),
    )
    rows: list[list[str]] = []
    for label in labels:
        ground_truth_count = int(ground_truth.get(label, 0))
        pred_count = int(pred.get(label, 0))
        if ground_truth_count == 0 and pred_count == 0:
            continue
        delta = pred_count - ground_truth_count
        rows.append([label, str(ground_truth_count), str(pred_count), f"{delta:+d}"])
    _print_table(
        title="label_counts:",
        headers=("label", "ground_truth", "pred", "delta(pred-ground_truth)"),
        rows=rows,
    )


def _print_ground_truth_label_recall(metrics: Mapping[str, float]) -> None:
    """Print the ground-truth label recall table for untyped eval."""
    recalled_prefix = "ground_truth_label_recall.recalled_chars."
    total_prefix = "ground_truth_label_recall.ground_truth_chars."
    recall_prefix = "ground_truth_label_recall.recall."
    recalled_chars: dict[str, int] = {}
    ground_truth_chars: dict[str, int] = {}
    recall_by_label: dict[str, float] = {}
    labels: set[str] = set()
    for key, value in metrics.items():
        if key.startswith(recalled_prefix):
            label = key[len(recalled_prefix) :]
            labels.add(label)
            recalled_chars[label] = int(round(float(value)))
            continue
        if key.startswith(total_prefix):
            label = key[len(total_prefix) :]
            labels.add(label)
            ground_truth_chars[label] = int(round(float(value)))
            continue
        if key.startswith(recall_prefix):
            label = key[len(recall_prefix) :]
            labels.add(label)
            recall_by_label[label] = float(value)
    rows = [
        [
            label,
            str(recalled_chars.get(label, 0)),
            str(ground_truth_chars.get(label, 0)),
            _format_metric(recall_by_label.get(label)),
        ]
        for label in sorted(labels)
    ]
    _print_table(
        title="ground_truth_label_recall:",
        headers=("label", "recalled_chars", "ground_truth_chars", "recall"),
        rows=rows,
    )


def main(argv: Sequence[str] | None = None, *, prog: str | None = None) -> None:
    """Run the ``opf eval`` command."""
    overall_start = time.perf_counter()
    args = parse_args(list(argv) if argv is not None else None, prog=prog)
    args.checkpoint = resolve_checkpoint_path(args.checkpoint)
    if args.attn_low_precision:
        os.environ["OPF_ATTN_LOW_PRECISION"] = "1"
    if args.experts_per_token is not None:
        if args.experts_per_token <= 0:
            raise ValueError("--experts-per-token must be > 0")
        os.environ["OPF_EXPERTS_PER_TOKEN"] = str(args.experts_per_token)
    if args.moe_triton:
        os.environ["OPF_MOE_TRITON"] = "1"
    start_time = time.perf_counter()
    is_untyped_eval = args.eval_mode == "untyped"
    if args.predictions_token_logprobs_topk < 0:
        raise ValueError("predictions_token_logprobs_topk must be >= 0")
    if (
        args.predictions_token_logprobs_max_tokens is not None
        and args.predictions_token_logprobs_max_tokens <= 0
    ):
        raise ValueError(
            "predictions_token_logprobs_max_tokens must be > 0 when provided"
        )
    if args.preprocess_chunksize <= 0:
        raise ValueError("--preprocess-chunksize must be > 0")
    if args.window_batch_size <= 0:
        raise ValueError("--window-batch-size must be > 0")
    if args.progress_every is not None and args.progress_every <= 0:
        raise ValueError("progress_every must be > 0 when provided")
    if args.preview_max_tokens <= 0:
        raise ValueError("preview_max_tokens must be > 0")
    if args.preview_max_chars <= 0:
        raise ValueError("preview_max_chars must be > 0")

    redactor = _build_eval_redactor(args)
    runtime, decoder = redactor.get_prediction_components()
    (
        resolved_category_version,
        resolved_span_class_names,
        resolved_ner_class_names,
    ) = resolve_checkpoint_label_space(args.checkpoint)
    device = runtime.device
    encoding = runtime.encoding
    active_encoding_name = runtime.active_encoding_name
    pad_token_id = runtime.pad_token_id
    debug_decode = args.debug_decode
    debug_decode_done = False

    n_ctx = int(runtime.n_ctx)
    startup_seconds = time.perf_counter() - overall_start

    model = runtime.model
    label_info = runtime.label_info
    preprocess_workers = _resolve_preprocess_workers(
        args.preprocess_workers, device=device
    )
    prediction_write_workers = _resolve_prediction_write_workers(
        args.prediction_write_workers
    )
    window_batch_size = args.window_batch_size
    store_scores_on_cuda = bool(
        device.type == "cuda"
        and get_env_bool("OPF_STORE_EVAL_SCORES_ON_CUDA", default=True)
    )
    score_storage_device = device if store_scores_on_cuda else torch.device("cpu")
    n_label_classes = len(resolved_ner_class_names)
    aggregated: dict[str, EvalExampleAggregation] = {}
    example_texts: dict[str, str] = {}
    total_windows = 0
    total_window_tokens = 0
    total_padded_window_tokens = 0
    model_forward_seconds = 0.0
    score_stitch_seconds = 0.0
    window_stage_wall_seconds = 0.0
    compute_metrics_wall_seconds = 0.0
    prediction_write_seconds = 0.0
    model_forward_cuda_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

    def build_window_batch_tensors(
        windows: Sequence[Window],
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        if not windows:
            raise ValueError("Cannot build tensors for an empty window batch")
        max_window_len = max(len(window.tokens) for window in windows)
        if max_window_len <= 0:
            raise ValueError("Window batch contains only empty windows")
        token_rows: list[list[int]] = []
        mask_rows: list[list[bool]] = []
        real_window_tokens = 0
        for window in windows:
            if not window.tokens:
                raise ValueError("Window batch contains an empty window")
            window_tokens = list(window.tokens)
            pad_count = max_window_len - len(window_tokens)
            token_rows.append(window_tokens + ([pad_token_id] * pad_count))
            mask_rows.append(([True] * len(window_tokens)) + ([False] * pad_count))
            real_window_tokens += len(window_tokens)
        tokens_t = torch.tensor(token_rows, device=device, dtype=torch.int32)
        attention_mask_t = torch.tensor(mask_rows, device=device, dtype=torch.bool)
        padded_window_tokens = max_window_len * len(windows)
        return tokens_t, attention_mask_t, real_window_tokens, padded_window_tokens

    def stitch_window_scores(
        windows: Sequence[Window],
        *,
        log_probs: torch.Tensor,
    ) -> None:
        if log_probs.dim() != 3:
            raise ValueError("Expected 3D logprobs for batched window forward")
        for batch_idx, window in enumerate(windows):
            window_log_probs = log_probs[batch_idx]
            positions_by_example: defaultdict[str, list[int]] = defaultdict(list)
            offsets_by_example: defaultdict[str, list[int]] = defaultdict(list)
            for token_pos, (is_valid, token_example_id, token_idx) in enumerate(
                zip(window.mask, window.token_example_ids, window.offsets)
            ):
                if (not is_valid) or token_example_id is None or token_idx < 0:
                    continue
                positions_by_example[token_example_id].append(token_pos)
                offsets_by_example[token_example_id].append(token_idx)

            for token_example_id, positions in positions_by_example.items():
                example_agg = aggregated.get(token_example_id)
                if example_agg is None:
                    raise ValueError(
                        f"Missing aggregation state for example {token_example_id}"
                    )
                offsets = offsets_by_example[token_example_id]
                pos_t = torch.tensor(
                    positions, device=score_storage_device, dtype=torch.long
                )
                offset_t = torch.tensor(
                    offsets, device=score_storage_device, dtype=torch.long
                )
                if int(offset_t.max().item()) >= example_agg.length:
                    raise ValueError(
                        "Token index out of range for example %s (max_offset=%d length=%d)"
                        % (
                            token_example_id,
                            int(offset_t.max().item()),
                            example_agg.length,
                        )
                    )
                already_written = example_agg.written.index_select(0, offset_t)
                if bool(already_written.any().item()):
                    bad_pos = int(already_written.nonzero(as_tuple=False)[0].item())
                    raise ValueError(
                        "Token %d for example %s was seen multiple times; overlapping windows are not supported."
                        % (int(offset_t[bad_pos].item()), token_example_id)
                    )
                score_slice = window_log_probs.index_select(0, pos_t)
                example_agg.score_matrix.index_copy_(0, offset_t, score_slice)
                example_agg.written.index_fill_(0, offset_t, True)

    def process_window_batch(windows: Sequence[Window]) -> None:
        nonlocal total_windows, total_window_tokens, total_padded_window_tokens
        nonlocal model_forward_seconds, score_stitch_seconds
        if not windows:
            return
        batch_size = len(windows)
        tokens_t, attention_mask_t, real_window_tokens, padded_window_tokens = (
            build_window_batch_tensors(windows)
        )

        if device.type == "cuda":
            forward_start_event = torch.cuda.Event(enable_timing=True)
            forward_end_event = torch.cuda.Event(enable_timing=True)
            forward_start_event.record()
            with torch.inference_mode():
                logits = model(tokens_t, attention_mask=attention_mask_t)
                log_probs = F.log_softmax(logits.float(), dim=-1)
            forward_end_event.record()
            model_forward_cuda_events.append((forward_start_event, forward_end_event))
        else:
            forward_start = time.perf_counter()
            with torch.inference_mode():
                logits = model(tokens_t, attention_mask=attention_mask_t)
                log_probs = F.log_softmax(logits.float(), dim=-1)
            model_forward_seconds += time.perf_counter() - forward_start

        stitch_start = time.perf_counter()
        if log_probs.device != score_storage_device:
            log_probs = log_probs.to(score_storage_device)

        expected_window_len = int(tokens_t.shape[1])
        if log_probs.dim() != 3 or int(log_probs.shape[0]) != batch_size:
            raise ValueError(
                "Batched logprob output shape mismatch: got %s expected (%d,%d,*)"
                % (tuple(log_probs.shape), batch_size, expected_window_len)
            )
        if int(log_probs.shape[1]) != expected_window_len:
            raise ValueError(
                "Batched logprob output shape mismatch: got %s expected (%d,%d,*)"
                % (tuple(log_probs.shape), batch_size, expected_window_len)
            )
        if int(log_probs.shape[2]) != n_label_classes:
            raise ValueError(
                "Unexpected class dimension from model: %d (expected %d)"
                % (int(log_probs.shape[2]), n_label_classes)
            )
        total_windows += batch_size
        total_window_tokens += real_window_tokens
        total_padded_window_tokens += padded_window_tokens
        stitch_window_scores(windows, log_probs=log_probs)
        score_stitch_seconds += time.perf_counter() - stitch_start

    pending_windows: list[Window] = []

    def enqueue_window(window: Window) -> None:
        if not window.tokens:
            return
        pending_windows.append(window)
        if len(pending_windows) >= window_batch_size:
            process_window_batch(tuple(pending_windows))
            pending_windows.clear()

    def flush_batched_windows() -> None:
        if not pending_windows:
            return
        process_window_batch(tuple(pending_windows))
        pending_windows.clear()

    max_examples = args.max_examples
    record_iter: Iterable[Mapping[str, object]] = iter_json_records(args.dataset)
    if args.dataset_variant == "message":
        record_iter = iter_message_variant_records(record_iter)
    original_gold_char_spans_by_example: (
        dict[str, tuple[tuple[str, int, int], ...]] | None
    ) = {} if is_untyped_eval else None
    token_char_ranges_by_example: dict[
        str, tuple[tuple[int, ...], tuple[int, ...]]
    ] = {}

    def iter_prepared_examples() -> Iterator[PreparedTokenizedExample]:
        if preprocess_workers <= 1:
            for idx, record in enumerate(record_iter):
                prepared = _prepare_tokenized_example(
                    record=record,
                    idx=idx,
                    encoding=encoding,
                    label_info=label_info,
                    eval_mode=args.eval_mode,
                    skip_non_ascii_examples=args.skip_non_ascii_examples,
                )
                if prepared is None:
                    continue
                yield prepared
            return

        with ProcessPoolExecutor(
            max_workers=preprocess_workers,
            initializer=_init_preprocess_worker,
            initargs=(
                active_encoding_name,
                label_info,
                args.eval_mode,
                args.skip_non_ascii_examples,
            ),
        ) as pool:
            for prepared in pool.map(
                _prepare_tokenized_example_worker,
                enumerate(record_iter),
                chunksize=args.preprocess_chunksize,
            ):
                if prepared is None:
                    continue
                yield prepared

    window_stage_start = time.perf_counter()
    processed_examples = 0
    for prepared in iter_prepared_examples():
        example = prepared.tokenized
        tokens = list(example.tokens)
        labels = list(example.labels)
        if len(tokens) != len(labels):
            raise ValueError(
                "Tokenized example %s has mismatched token/label lengths (%d != %d)"
                % (example.example_id, len(tokens), len(labels))
            )
        if debug_decode and not debug_decode_done:
            decoded_text = encoding.decode(tokens, errors="replace")
            mismatch_idx = _first_text_mismatch(decoded_text, example.text)
            print(f"debug_decode.encoding: {active_encoding_name}")
            if mismatch_idx is None:
                print(
                    "debug_decode.match: True (decoded_len=%d text_len=%d)"
                    % (len(decoded_text), len(example.text))
                )
            else:
                start = max(0, mismatch_idx - 40)
                end = min(len(example.text), mismatch_idx + 40)
                print(
                    "debug_decode.match: False (decoded_len=%d text_len=%d mismatch=%d)"
                    % (len(decoded_text), len(example.text), mismatch_idx)
                )
                print(
                    f"debug_decode.decoded[{start}:{end}]: {decoded_text[start:end]!r}"
                )
                print(f"debug_decode.text[{start}:{end}]: {example.text[start:end]!r}")
            debug_decode_done = True
        if example.example_id in aggregated:
            raise ValueError(f"Duplicate example_id encountered: {example.example_id}")
        if original_gold_char_spans_by_example is not None:
            original_gold_char_spans_by_example[example.example_id] = (
                prepared.original_gold_char_spans
            )
        token_char_ranges_by_example[example.example_id] = (
            prepared.char_starts,
            prepared.char_ends,
        )
        aggregated[example.example_id] = EvalExampleAggregation(
            score_matrix=torch.empty(
                (len(tokens), n_label_classes),
                device=score_storage_device,
                dtype=torch.float32,
            ),
            written=torch.zeros(
                (len(tokens),),
                device=score_storage_device,
                dtype=torch.bool,
            ),
            labels=torch.tensor(labels, dtype=torch.long),
            token_ids=tuple(tokens),
            length=len(tokens),
        )
        example_texts[example.example_id] = example.text

        for window in example_to_windows(
            example,
            n_ctx,
        ):
            enqueue_window(window)

        processed_examples += 1
        if (
            args.progress_every is not None
            and processed_examples % args.progress_every == 0
        ):
            _print_progress_line(
                processed_examples=processed_examples,
                max_examples=max_examples,
                total_windows=total_windows,
                total_window_tokens=total_window_tokens,
                total_padded_window_tokens=total_padded_window_tokens,
                elapsed_s=time.perf_counter() - start_time,
            )
        if max_examples is not None and processed_examples >= max_examples:
            break

    flush_batched_windows()
    if model_forward_cuda_events:
        torch.cuda.synchronize(device)
        model_forward_seconds = (
            sum(
                start_event.elapsed_time(end_event)
                for start_event, end_event in model_forward_cuda_events
            )
            / 1000.0
        )
    window_stage_wall_seconds = time.perf_counter() - window_stage_start

    if (
        args.progress_every is not None
        and processed_examples > 0
        and processed_examples % args.progress_every != 0
    ):
        _print_progress_line(
            processed_examples=processed_examples,
            max_examples=max_examples,
            total_windows=total_windows,
            total_window_tokens=total_window_tokens,
            total_padded_window_tokens=total_padded_window_tokens,
            elapsed_s=time.perf_counter() - start_time,
        )

    for example_id, example_agg in aggregated.items():
        if not bool(example_agg.written.all().item()):
            missing_idx = int((~example_agg.written).nonzero(as_tuple=False)[0].item())
            raise ValueError(
                "Missing score assignment for example %s token %d"
                % (example_id, missing_idx)
            )

    metrics_timings: dict[str, float] = {}
    compute_metrics_start = time.perf_counter()
    metrics, predictions, label_counts = compute_metrics(
        aggregated,
        label_info,
        decoder=decoder,
        example_texts=example_texts,
        encoding=encoding,
        span_metrics_space=args.span_metrics_space,
        trim_span_whitespace=args.trim_span_whitespace,
        discard_overlapping_predicted_spans=args.discard_overlapping_predicted_spans,
        discard_overlapping_ground_truth_spans=args.discard_overlapping_ground_truth_spans,
        ner_class_names=resolved_ner_class_names,
        background_class_label=BACKGROUND_CLASS_LABEL,
        token_char_ranges_fn=token_char_ranges,
        token_char_ranges_by_example=token_char_ranges_by_example,
        predictions_token_logprobs_topk=args.predictions_token_logprobs_topk,
        predictions_token_logprobs_example_id=args.predictions_token_logprobs_example_id,
        predictions_token_logprobs_max_tokens=args.predictions_token_logprobs_max_tokens,
        span_match_ignore_label=is_untyped_eval,
        original_gold_char_spans_by_example=original_gold_char_spans_by_example,
        timings=metrics_timings,
    )
    compute_metrics_wall_seconds = time.perf_counter() - compute_metrics_start
    decode_seconds = float(metrics_timings.get("decode_seconds", 0.0))
    prediction_assembly_seconds = float(metrics_timings.get("prediction_seconds", 0.0))
    metrics_prepare_seconds = float(metrics_timings.get("prepare_seconds", 0.0))
    metrics_accuracy_seconds = float(metrics_timings.get("accuracy_seconds", 0.0))
    metrics_finalize_seconds = float(metrics_timings.get("finalize_seconds", 0.0))
    metrics_total_seconds = float(
        metrics_timings.get("total_seconds", compute_metrics_wall_seconds)
    )
    input_pipeline_seconds = max(
        0.0,
        window_stage_wall_seconds - model_forward_seconds - score_stitch_seconds,
    )
    n_tokens = int(metrics["n_tokens"])
    inference_prediction_seconds = (
        model_forward_seconds
        + score_stitch_seconds
        + decode_seconds
        + prediction_assembly_seconds
    )
    inference_tokens_per_second = _safe_throughput(
        n_tokens,
        inference_prediction_seconds,
    )
    model_forward_tokens_per_second = _safe_throughput(n_tokens, model_forward_seconds)
    elapsed_s = time.perf_counter() - start_time
    summary_rows = [
        ["examples", str(int(metrics["n_examples"]))],
        ["tokens", str(n_tokens)],
        ["windows", str(total_windows)],
        ["window_tokens", str(total_window_tokens)],
        ["padded_window_tokens", str(total_padded_window_tokens)],
        ["elapsed_s", _format_metric(elapsed_s)],
        [
            "tokens_per_s",
            _format_metric(_rate_per_second(int(metrics["n_tokens"]), elapsed_s)),
        ],
        [
            "window_tokens_per_s",
            _format_metric(_rate_per_second(total_window_tokens, elapsed_s)),
        ],
        [
            "padded_window_tokens_per_s",
            _format_metric(_rate_per_second(total_padded_window_tokens, elapsed_s)),
        ],
        ["eval_mode", args.eval_mode],
        [
            "inference_tokens_per_second",
            _format_metric(inference_tokens_per_second),
        ],
    ]
    if "loss" in metrics and not is_untyped_eval:
        summary_rows.append(["loss", _format_metric(metrics["loss"])])
        summary_rows.append(
            ["token_accuracy", _format_metric(metrics["token_accuracy"])]
        )
    elif is_untyped_eval:
        summary_rows.append(
            ["note", "category-level loss/token_accuracy hidden in untyped mode"]
        )
    _print_key_value_table(
        title="summary:",
        key_header="field",
        value_header="value",
        rows=summary_rows,
    )
    print()
    detection_rows = [
        [key, _format_metric(metrics.get(key))]
        for key in (
            "detection.precision",
            "detection.recall",
            "detection.f1",
            "detection.f2",
            "detection.span.precision",
            "detection.span.recall",
            "detection.span.f1",
            "detection.span.f2",
        )
        if key in metrics
    ]
    _print_key_value_table(
        title="detection_metrics:",
        key_header="metric",
        value_header="value",
        rows=detection_rows,
    )

    if is_untyped_eval:
        print()
        _print_ground_truth_label_recall(metrics)
        if args.label_counts:
            print()
            print(
                "note: --label-counts output suppressed in untyped mode because "
                "category names do not need to match the model ontology."
            )
    elif args.per_class:
        print()
        _print_per_class_metrics(
            metrics,
            span_class_names=resolved_span_class_names,
            ner_class_names=resolved_ner_class_names,
        )

    if args.label_counts and not is_untyped_eval:
        print()
        _print_label_counts(label_counts, ner_class_names=resolved_ner_class_names)

    if args.preview:
        print()
        print("preview:")
        if not predictions:
            print("  no predictions available")
        else:
            preview_example_id = args.preview_example_id or next(iter(predictions))
            prediction = predictions.get(preview_example_id)
            if prediction is None:
                raise ValueError(
                    f"preview_example_id {preview_example_id!r} not found in predictions"
                )
            preview_text = example_texts.get(preview_example_id, "")
            preview_output = build_prediction_preview(
                example_id=preview_example_id,
                text=preview_text,
                token_ids=prediction.token_ids,
                predicted_spans=prediction.predicted_spans,
                span_class_names=label_info.span_class_names,
                encoding=encoding,
                max_tokens=args.preview_max_tokens,
                max_chars=args.preview_max_chars,
            )
            print(preview_output)

    if args.predictions_out:
        prediction_write_start = time.perf_counter()
        write_predictions(
            output_path=args.predictions_out,
            encoding=encoding,
            label_info=label_info,
            example_texts=example_texts,
            predictions=predictions,
            trim_span_whitespace=args.trim_span_whitespace,
            discard_overlapping_predicted_spans=args.discard_overlapping_predicted_spans,
            workers=prediction_write_workers,
            token_char_ranges_by_example=token_char_ranges_by_example,
        )
        prediction_write_seconds = time.perf_counter() - prediction_write_start
        print(f"predictions_out: {args.predictions_out}")

    eval_core_seconds = window_stage_wall_seconds + compute_metrics_wall_seconds
    full_eval_seconds = eval_core_seconds + prediction_write_seconds
    full_eval_tokens_per_second = _safe_throughput(n_tokens, full_eval_seconds)
    overall_wall_seconds = time.perf_counter() - overall_start

    print()
    timing_rows = [
        ["startup", _format_seconds(startup_seconds)],
        ["window_stage_wall", _format_seconds(window_stage_wall_seconds)],
        ["input_pipeline", _format_seconds(input_pipeline_seconds)],
        ["model_forward", _format_seconds(model_forward_seconds)],
        ["score_stitch", _format_seconds(score_stitch_seconds)],
        ["metrics_prepare", _format_seconds(metrics_prepare_seconds)],
        ["decode", _format_seconds(decode_seconds)],
        ["prediction_assembly", _format_seconds(prediction_assembly_seconds)],
        ["metrics_accuracy", _format_seconds(metrics_accuracy_seconds)],
        ["metrics_finalize", _format_seconds(metrics_finalize_seconds)],
        ["metrics_total", _format_seconds(metrics_total_seconds)],
        ["prediction_write", _format_seconds(prediction_write_seconds)],
        ["eval_core_total", _format_seconds(eval_core_seconds)],
        ["eval_total_with_write", _format_seconds(full_eval_seconds)],
        ["overall_wall", _format_seconds(overall_wall_seconds)],
    ]
    _print_table(
        title="timings:",
        headers=("stage", "seconds"),
        rows=timing_rows,
    )

    print()
    throughput_rows = [
        [
            "inference_only_tokens_per_second",
            _format_metric(inference_tokens_per_second),
        ],
        [
            "model_forward_tokens_per_second",
            _format_metric(model_forward_tokens_per_second),
        ],
        [
            "eval_total_tokens_per_second",
            _format_metric(full_eval_tokens_per_second),
        ],
    ]
    _print_table(
        title="throughput:",
        headers=("metric", "tokens_per_second"),
        rows=throughput_rows,
    )
    print(
        "throughput_note: inference_only_tokens_per_second uses "
        "model_forward + score_stitch + decode + prediction_assembly."
    )

    if args.timings_out:
        timings_payload = {
            "schema_version": "opf.eval.timings.v1",
            "counts": {
                "examples": int(metrics["n_examples"]),
                "tokens": n_tokens,
                "windows": total_windows,
                "window_tokens": total_window_tokens,
                "padded_window_tokens": total_padded_window_tokens,
            },
            "config": {
                "checkpoint": args.checkpoint,
                "dataset": args.dataset,
                "dataset_variant": args.dataset_variant,
                "category_version": resolved_category_version,
                "device": str(device),
                "n_ctx": int(n_ctx),
                "decode_mode": args.decode_mode,
                "eval_mode": args.eval_mode,
                "preprocess_workers": preprocess_workers,
                "prediction_write_workers": prediction_write_workers,
                "window_batch_size": window_batch_size,
            },
            "timings_seconds": {
                "startup": startup_seconds,
                "window_stage_wall": window_stage_wall_seconds,
                "input_pipeline": input_pipeline_seconds,
                "model_forward": model_forward_seconds,
                "score_stitch": score_stitch_seconds,
                "metrics_prepare": metrics_prepare_seconds,
                "decode": decode_seconds,
                "prediction_assembly": prediction_assembly_seconds,
                "metrics_accuracy": metrics_accuracy_seconds,
                "metrics_finalize": metrics_finalize_seconds,
                "metrics_total": metrics_total_seconds,
                "prediction_write": prediction_write_seconds,
                "eval_core_total": eval_core_seconds,
                "eval_total_with_write": full_eval_seconds,
                "overall_wall": overall_wall_seconds,
            },
            "throughput_tokens_per_second": {
                "inference_only": inference_tokens_per_second,
                "model_forward_only": model_forward_tokens_per_second,
                "eval_total_with_write": full_eval_tokens_per_second,
            },
            "throughput_notes": {
                "inference_only_formula": (
                    "tokens / (model_forward + score_stitch + decode + prediction_assembly)"
                ),
            },
        }
        timings_path = Path(args.timings_out)
        if timings_path.parent != Path():
            timings_path.parent.mkdir(parents=True, exist_ok=True)
        timings_path.write_text(
            json.dumps(timings_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"timings_out: {args.timings_out}")

    if args.metrics_out:
        _write_metrics_payload(
            output_path=args.metrics_out,
            args=args,
            device=device,
            n_ctx=n_ctx,
            active_encoding_name=active_encoding_name,
            category_version=resolved_category_version,
            total_windows=total_windows,
            total_window_tokens=total_window_tokens,
            total_padded_window_tokens=total_padded_window_tokens,
            elapsed_s=elapsed_s,
            metrics=metrics,
        )


if __name__ == "__main__":
    main()
