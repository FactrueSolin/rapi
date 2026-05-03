import os
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
import tiktoken

from .._common.env import get_env_bool
from .._core.spans import (
    decode_text_with_offsets,
    discard_overlapping_spans_by_label,
    labels_to_spans,
    token_spans_to_char_spans,
    trim_char_spans_whitespace,
)


@dataclass(frozen=True)
class ExamplePrediction:
    """Predicted token ids and decoded spans for one eval example."""

    token_ids: tuple[int, ...]
    predicted_spans: tuple[tuple[int, int, int], ...]
    token_logprobs_topk: tuple[dict[str, object], ...] | None = None


@dataclass
class _ExampleDecodeState:
    """Intermediate decode state accumulated for one eval example."""

    example_id: str
    gold_labels_by_index: dict[int, int]
    token_positions: list[int]
    stacked_scores: torch.Tensor | None
    token_ids: list[int]


def _match_spans_containment(
    predicted_spans: Sequence[tuple[int, int, int]],
    gold_spans: Sequence[tuple[int, int, int]],
    *,
    mode: str,
    ignore_label: bool = False,
) -> tuple[set[int], set[int]]:
    """Match spans by containment in the requested direction."""
    matched_predicted: set[int] = set()
    matched_gold: set[int] = set()
    for p_idx, (p_lbl, p_s, p_e) in enumerate(predicted_spans):
        for g_idx, (g_lbl, g_s, g_e) in enumerate(gold_spans):
            if not ignore_label and p_lbl != g_lbl:
                continue
            if mode == "pred_in_gold":
                if g_s <= p_s and g_e >= p_e:
                    matched_predicted.add(p_idx)
                    matched_gold.add(g_idx)
            else:
                if p_s <= g_s and p_e >= g_e:
                    matched_predicted.add(p_idx)
                    matched_gold.add(g_idx)
    return matched_predicted, matched_gold


def _covered_overlap_length(
    target_start: int,
    target_end: int,
    spans: Sequence[tuple[int, int]],
) -> int:
    """Return the covered overlap length between a target span and many spans."""
    clipped: list[tuple[int, int]] = []
    for span_start, span_end in spans:
        overlap_start = max(target_start, span_start)
        overlap_end = min(target_end, span_end)
        if overlap_end <= overlap_start:
            continue
        clipped.append((overlap_start, overlap_end))
    if not clipped:
        return 0
    clipped.sort(key=lambda span: (span[0], span[1]))
    covered = 0
    current_start, current_end = clipped[0]
    for start, end in clipped[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        covered += current_end - current_start
        current_start, current_end = start, end
    covered += current_end - current_start
    return covered


def _should_collect_token_logprobs(
    *,
    example_id: str,
    topk: int,
    example_filter: str | None,
) -> bool:
    """Return whether token logprob diagnostics should be collected."""
    if topk <= 0:
        return False
    if example_filter is not None and example_filter != example_id:
        return False
    return True


def _viterbi_cuda_batch_size() -> int:
    """Read and validate the CUDA Viterbi batch size from the environment."""
    raw = os.environ.get("OPF_VITERBI_CUDA_BATCH_SIZE", "512").strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"OPF_VITERBI_CUDA_BATCH_SIZE must be an integer (got {raw!r})"
        ) from exc
    if value <= 0:
        raise ValueError("OPF_VITERBI_CUDA_BATCH_SIZE must be positive")
    return value


def _collect_token_logprobs_topk(
    *,
    example_id: str,
    token_ids: Sequence[int],
    token_positions: Sequence[int],
    token_score_tensor: torch.Tensor | None,
    gold_labels_by_index: Mapping[int, int],
    predicted_labels_by_index: Mapping[int, int],
    ner_class_names: Sequence[str],
    topk: int,
    example_filter: str | None,
    max_tokens: int | None,
) -> tuple[dict[str, object], ...] | None:
    """Collect per-token top-k logprob diagnostics for one example."""
    if not _should_collect_token_logprobs(
        example_id=example_id,
        topk=topk,
        example_filter=example_filter,
    ):
        return None
    if token_score_tensor is None:
        return None

    diagnostics: list[dict[str, object]] = []
    for row_idx, token_idx in enumerate(token_positions):
        if max_tokens is not None and len(diagnostics) >= max_tokens:
            break
        score_vec = token_score_tensor[row_idx]
        token_id = token_ids[token_idx] if token_idx < len(token_ids) else None
        pred_label_id = predicted_labels_by_index.get(token_idx)
        gold_label_id = gold_labels_by_index.get(token_idx)
        entry: dict[str, object] = {
            "token_idx": token_idx,
            "token_id": token_id,
            "label_space": "token",
            "pred_label_id": pred_label_id,
            "gold_label_id": gold_label_id,
            "pred_label": ner_class_names[pred_label_id]
            if pred_label_id is not None
            else None,
            "gold_label": ner_class_names[gold_label_id]
            if gold_label_id is not None
            else None,
        }
        k = min(topk, score_vec.shape[0])
        if k > 0:
            top_vals, top_idx = torch.topk(score_vec, k)
            topk_entries: list[dict[str, object]] = []
            for idx, val in zip(top_idx.tolist(), top_vals.tolist()):
                topk_entries.append(
                    {
                        "label_id": idx,
                        "label": ner_class_names[idx],
                        "logprob": float(val),
                    }
                )
            entry["topk_logprobs"] = topk_entries
        diagnostics.append(entry)

    return tuple(diagnostics)


def compute_metrics(
    aggregated_examples: Mapping[str, Any],
    label_info: Any,
    *,
    decoder: Any | None,
    example_texts: Mapping[str, str] | None,
    encoding: tiktoken.Encoding | None,
    span_metrics_space: str,
    trim_span_whitespace: bool,
    discard_overlapping_predicted_spans: bool,
    discard_overlapping_ground_truth_spans: bool,
    ner_class_names: Sequence[str],
    background_class_label: str,
    token_char_ranges_fn: Callable[
        [Sequence[int], tiktoken.Encoding, str],
        tuple[list[int], list[int]],
    ],
    token_char_ranges_by_example: (
        Mapping[str, tuple[Sequence[int], Sequence[int]]] | None
    ) = None,
    predictions_token_logprobs_topk: int = 0,
    predictions_token_logprobs_example_id: str | None = None,
    predictions_token_logprobs_max_tokens: int | None = None,
    span_match_ignore_label: bool = False,
    original_gold_char_spans_by_example: (
        Mapping[str, Sequence[tuple[str, int, int]]] | None
    ) = None,
    timings: dict[str, float] | None = None,
) -> tuple[dict[str, float], dict[str, ExamplePrediction], dict[str, dict[str, int]]]:
    """Compute eval metrics and per-example predictions from aggregated scores."""
    total_start = time.perf_counter()
    prepare_seconds = 0.0
    decode_seconds = 0.0
    prediction_seconds = 0.0
    accuracy_seconds = 0.0
    finalize_seconds = 0.0

    if original_gold_char_spans_by_example is not None and (
        example_texts is None or encoding is None
    ):
        raise ValueError(
            "original_gold_char_spans_by_example requires example_texts and encoding"
        )
    total_examples = len(aggregated_examples)
    total_tokens = 0
    total_correct_tokens = 0
    total_logprob_sum = 0.0
    per_class_logprob: defaultdict[int, float] = defaultdict(float)
    per_class_counts: defaultdict[int, int] = defaultdict(int)
    per_class_correct: defaultdict[int, int] = defaultdict(int)
    per_class_predicted: defaultdict[int, int] = defaultdict(int)
    detection_true_positive = 0
    detection_false_positive = 0
    detection_false_negative = 0

    span_true_positive_precision = 0
    span_false_positive = 0
    span_true_positive_recall = 0
    span_false_negative = 0

    per_class_span_tp_precision: defaultdict[int, int] = defaultdict(int)
    per_class_span_tp_recall: defaultdict[int, int] = defaultdict(int)
    per_class_span_fp: defaultdict[int, int] = defaultdict(int)
    per_class_span_fn: defaultdict[int, int] = defaultdict(int)

    background_class_idx = label_info.background_token_label
    example_predictions: dict[str, ExamplePrediction] = {}
    label_counts: dict[str, dict[str, int]] = {
        "gold": defaultdict(int),
        "pred": defaultdict(int),
    }
    original_label_covered_chars: defaultdict[str, int] = defaultdict(int)
    original_label_total_chars: defaultdict[str, int] = defaultdict(int)
    decode_device: torch.device | None = None
    decode_batch_size = 128
    if (
        decoder is not None
        and get_env_bool("OPF_VITERBI_ON_CUDA", default=True)
        and torch.cuda.is_available()
    ):
        decode_device = torch.device("cuda")
        decode_batch_size = _viterbi_cuda_batch_size()

    n_token_classes = len(ner_class_names)
    example_states: list[_ExampleDecodeState] = []
    prepare_start = time.perf_counter()
    for example_id, example_agg in aggregated_examples.items():
        gold_labels_by_index: dict[int, int] = {}
        token_positions: list[int] = []
        stacked_scores: torch.Tensor | None = None
        if hasattr(example_agg, "score_matrix") and hasattr(example_agg, "labels"):
            length = int(example_agg.length)
            stacked_scores = example_agg.score_matrix[:length]
            labels_t = example_agg.labels[:length].to(dtype=torch.long, device="cpu")
            if length > 0:
                token_positions = list(range(length))
                labels_list = labels_t.tolist()
                gold_labels_by_index = {
                    token_idx: int(label) for token_idx, label in enumerate(labels_list)
                }
                labels_on_scores = labels_t.to(device=stacked_scores.device)
                label_logprobs = stacked_scores.gather(
                    1, labels_on_scores.unsqueeze(1)
                ).squeeze(1)
                total_logprob_sum += float(label_logprobs.sum().item())
                total_tokens += length
                class_counts = torch.bincount(labels_t, minlength=n_token_classes)
                per_class_logprob_t = torch.zeros(
                    (n_token_classes,),
                    device=stacked_scores.device,
                    dtype=stacked_scores.dtype,
                )
                per_class_logprob_t.scatter_add_(0, labels_on_scores, label_logprobs)
                per_class_logprob_cpu = per_class_logprob_t.to(device="cpu")
                for class_idx in (
                    class_counts.nonzero(as_tuple=False).flatten().tolist()
                ):
                    count = int(class_counts[class_idx].item())
                    label_counts["gold"][ner_class_names[class_idx]] += count
                    per_class_counts[class_idx] += count
                    per_class_logprob[class_idx] += float(
                        per_class_logprob_cpu[class_idx].item()
                    )
        else:
            token_score_vectors: list[torch.Tensor] = []
            for token_idx in range(example_agg.length):
                count = example_agg.counts[token_idx]
                if count <= 0:
                    continue
                label = example_agg.labels[token_idx]
                score_vec = example_agg.logprob_logsumexp[token_idx]
                if label is None or score_vec is None:
                    continue
                if count == 1:
                    avg_logprob_vec = score_vec
                else:
                    avg_logprob_vec = score_vec - math.log(count)
                label_logprob = float(avg_logprob_vec[label].item())
                total_logprob_sum += label_logprob
                total_tokens += 1
                gold_labels_by_index[token_idx] = label
                label_counts["gold"][ner_class_names[label]] += 1
                per_class_logprob[label] += label_logprob
                per_class_counts[label] += 1
                token_positions.append(token_idx)
                token_score_vectors.append(avg_logprob_vec)
            stacked_scores = (
                torch.stack(token_score_vectors, dim=0) if token_score_vectors else None
            )

        token_ids: list[int] = []
        for token_idx in range(example_agg.length):
            token_id = example_agg.token_ids[token_idx]
            if token_id is None:
                raise ValueError(
                    "Missing token id for example %s token %d" % (example_id, token_idx)
                )
            token_ids.append(int(token_id))

        example_states.append(
            _ExampleDecodeState(
                example_id=example_id,
                gold_labels_by_index=gold_labels_by_index,
                token_positions=token_positions,
                stacked_scores=stacked_scores,
                token_ids=token_ids,
            )
        )
    prepare_seconds += time.perf_counter() - prepare_start

    decoded_labels_by_state: list[list[int]] = [[] for _ in example_states]
    decode_start = time.perf_counter()
    states_with_scores = [
        idx
        for idx, state in enumerate(example_states)
        if state.stacked_scores is not None and state.stacked_scores.numel() > 0
    ]
    if decoder is not None:
        if states_with_scores:
            score_tensors = [
                example_states[idx].stacked_scores for idx in states_with_scores
            ]
            if decode_device is not None and decode_device.type == "cuda":
                torch.cuda.synchronize(decode_device)
            decoded_many = decoder.decode_many(
                score_tensors,
                device=decode_device,
                max_batch_size=decode_batch_size,
            )
            if decode_device is not None and decode_device.type == "cuda":
                torch.cuda.synchronize(decode_device)
            if len(decoded_many) != len(states_with_scores):
                raise RuntimeError(
                    "Decoder returned unexpected number of decoded sequences: "
                    f"{len(decoded_many)} != {len(states_with_scores)}"
                )
            for state_idx, decoded in zip(states_with_scores, decoded_many):
                decoded_labels_by_state[state_idx] = list(decoded)
    else:
        for state_idx in states_with_scores:
            stacked_scores = example_states[state_idx].stacked_scores
            if stacked_scores is None:
                continue
            decoded_labels_by_state[state_idx] = stacked_scores.argmax(dim=1).tolist()
    decode_seconds += time.perf_counter() - decode_start

    for state_idx, state in enumerate(example_states):
        example_id = state.example_id
        token_positions = state.token_positions
        stacked_scores = state.stacked_scores
        token_ids = state.token_ids
        gold_labels_by_index = state.gold_labels_by_index
        predicted_labels_by_index: dict[int, int] = {}

        if stacked_scores is None:
            decoded_labels: list[int] = []
        else:
            decoded_labels = decoded_labels_by_state[state_idx]
            if len(decoded_labels) != len(token_positions):
                fallback_start = time.perf_counter()
                decoded_labels = stacked_scores.argmax(dim=1).tolist()
                decode_seconds += time.perf_counter() - fallback_start

        prediction_start = time.perf_counter()
        for token_idx, predicted_class in zip(token_positions, decoded_labels):
            predicted_labels_by_index[token_idx] = predicted_class
            label_counts["pred"][ner_class_names[predicted_class]] += 1
            per_class_predicted[predicted_class] += 1

        token_logprobs_topk = _collect_token_logprobs_topk(
            example_id=example_id,
            token_ids=token_ids,
            token_positions=token_positions,
            token_score_tensor=stacked_scores,
            gold_labels_by_index=gold_labels_by_index,
            predicted_labels_by_index=predicted_labels_by_index,
            ner_class_names=ner_class_names,
            topk=predictions_token_logprobs_topk,
            example_filter=predictions_token_logprobs_example_id,
            max_tokens=predictions_token_logprobs_max_tokens,
        )

        predicted_spans = labels_to_spans(predicted_labels_by_index, label_info)
        example_predictions[example_id] = ExamplePrediction(
            token_ids=tuple(token_ids),
            predicted_spans=tuple(predicted_spans),
            token_logprobs_topk=token_logprobs_topk,
        )
        prediction_seconds += time.perf_counter() - prediction_start

        accuracy_start = time.perf_counter()
        for token_idx, predicted_class in predicted_labels_by_index.items():
            gold_label = gold_labels_by_index.get(token_idx)
            if gold_label is None:
                continue
            if predicted_class == gold_label:
                total_correct_tokens += 1
                per_class_correct[gold_label] += 1
            if predicted_class != background_class_idx:
                if gold_label != background_class_idx:
                    detection_true_positive += 1
                else:
                    detection_false_positive += 1
            elif gold_label != background_class_idx:
                detection_false_negative += 1

        gold_spans = labels_to_spans(gold_labels_by_index, label_info)

        gold_spans_metrics = gold_spans
        predicted_spans_metrics = predicted_spans
        text = example_texts.get(example_id, "") if example_texts is not None else ""
        char_starts: Sequence[int] | None = None
        char_ends: Sequence[int] | None = None
        if (
            text
            and encoding is not None
            and (
                span_metrics_space == "char"
                or original_gold_char_spans_by_example is not None
            )
        ):
            if token_char_ranges_by_example is not None:
                cached_char_ranges = token_char_ranges_by_example.get(example_id)
                if cached_char_ranges is not None:
                    char_starts, char_ends = cached_char_ranges
            if char_starts is None or char_ends is None:
                char_starts, char_ends = token_char_ranges_fn(token_ids, encoding, text)
        if (
            span_metrics_space == "char"
            and char_starts is not None
            and char_ends is not None
        ):
            gold_spans_metrics = token_spans_to_char_spans(
                gold_spans_metrics, char_starts, char_ends
            )
            predicted_spans_metrics = token_spans_to_char_spans(
                predicted_spans_metrics, char_starts, char_ends
            )
            if trim_span_whitespace:
                gold_spans_metrics = trim_char_spans_whitespace(
                    gold_spans_metrics, text
                )
                predicted_spans_metrics = trim_char_spans_whitespace(
                    predicted_spans_metrics, text
                )

        if discard_overlapping_predicted_spans:
            predicted_spans_metrics = discard_overlapping_spans_by_label(
                predicted_spans_metrics
            )
        if discard_overlapping_ground_truth_spans:
            gold_spans_metrics = discard_overlapping_spans_by_label(gold_spans_metrics)

        if original_gold_char_spans_by_example is not None:
            predicted_spans_char: list[tuple[int, int, int]] = []
            if char_starts is not None and char_ends is not None:
                predicted_spans_char = token_spans_to_char_spans(
                    predicted_spans, char_starts, char_ends
                )
                if trim_span_whitespace:
                    predicted_spans_char = trim_char_spans_whitespace(
                        predicted_spans_char, text
                    )
                if discard_overlapping_predicted_spans:
                    predicted_spans_char = discard_overlapping_spans_by_label(
                        predicted_spans_char
                    )
            predicted_char_ranges = [
                (char_start, char_end)
                for _pred_label, char_start, char_end in predicted_spans_char
            ]
            for (
                original_label,
                gold_start,
                gold_end,
            ) in original_gold_char_spans_by_example.get(example_id, ()):
                if gold_end <= gold_start:
                    continue
                original_label_total_chars[original_label] += gold_end - gold_start
                original_label_covered_chars[original_label] += _covered_overlap_length(
                    gold_start,
                    gold_end,
                    predicted_char_ranges,
                )
        matched_pred_for_precision, _ = _match_spans_containment(
            predicted_spans_metrics,
            gold_spans_metrics,
            mode="pred_in_gold",
            ignore_label=span_match_ignore_label,
        )
        _, matched_gold_for_recall = _match_spans_containment(
            predicted_spans_metrics,
            gold_spans_metrics,
            mode="gold_in_pred",
            ignore_label=span_match_ignore_label,
        )

        span_true_positive_precision += len(matched_pred_for_precision)
        span_false_positive += len(predicted_spans_metrics) - len(
            matched_pred_for_precision
        )
        span_true_positive_recall += len(matched_gold_for_recall)
        span_false_negative += len(gold_spans_metrics) - len(matched_gold_for_recall)

        for pred_idx, (pred_label, _ps, _pe) in enumerate(predicted_spans_metrics):
            if pred_idx in matched_pred_for_precision:
                per_class_span_tp_precision[pred_label] += 1
            else:
                per_class_span_fp[pred_label] += 1
        for gold_idx, (gold_label, _gs, _ge) in enumerate(gold_spans_metrics):
            if gold_idx in matched_gold_for_recall:
                per_class_span_tp_recall[gold_label] += 1
            else:
                per_class_span_fn[gold_label] += 1
        accuracy_seconds += time.perf_counter() - accuracy_start

    metrics: dict[str, float] = {}
    finalize_start = time.perf_counter()
    metrics["n_examples"] = float(total_examples)
    metrics["n_tokens"] = float(total_tokens)
    if total_tokens > 0:
        metrics["loss"] = float(-total_logprob_sum / total_tokens)
        metrics["token_accuracy"] = float(total_correct_tokens / total_tokens)

    def emit_token_f_scores(prefix: str, tp: int, fp: int, fn: int) -> None:
        f1_denom = 2 * tp + fp + fn
        if f1_denom > 0:
            metrics[f"{prefix}.f1"] = float(2 * tp / f1_denom)
        f2_denom = 5 * tp + 4 * fn + fp
        if f2_denom > 0:
            metrics[f"{prefix}.f2"] = float(5 * tp / f2_denom)

    def emit_span_f_scores(
        prefix: str, p_num: int, p_den: int, r_num: int, r_den: int
    ) -> None:
        p = (p_num / p_den) if p_den > 0 else None
        r = (r_num / r_den) if r_den > 0 else None
        if p is not None:
            metrics[f"{prefix}.precision"] = float(p)
        if r is not None:
            metrics[f"{prefix}.recall"] = float(r)
        if p is not None and r is not None and (p + r) > 0:
            metrics[f"{prefix}.f1"] = float(2 * p * r / (p + r))
            denom = 4 * p + r
            if denom > 0:
                metrics[f"{prefix}.f2"] = float(5 * p * r / denom)

    positive_predictions = detection_true_positive + detection_false_positive
    positive_targets = detection_true_positive + detection_false_negative
    if positive_predictions > 0:
        metrics["detection.precision"] = float(
            detection_true_positive / positive_predictions
        )
    if positive_targets > 0:
        metrics["detection.recall"] = float(detection_true_positive / positive_targets)
    emit_token_f_scores(
        "detection",
        detection_true_positive,
        detection_false_positive,
        detection_false_negative,
    )

    emit_span_f_scores(
        "detection.span",
        span_true_positive_precision,
        span_true_positive_precision + span_false_positive,
        span_true_positive_recall,
        span_true_positive_recall + span_false_negative,
    )
    for label in sorted(original_label_total_chars):
        total_chars = original_label_total_chars[label]
        if total_chars <= 0:
            continue
        recalled_chars = original_label_covered_chars.get(label, 0)
        metrics[f"ground_truth_label_recall.recalled_chars.{label}"] = float(
            recalled_chars
        )
        metrics[f"ground_truth_label_recall.ground_truth_chars.{label}"] = float(
            total_chars
        )
        metrics[f"ground_truth_label_recall.recall.{label}"] = float(
            recalled_chars / total_chars
        )

    for class_idx, count in per_class_counts.items():
        if count == 0:
            continue
        label = ner_class_names[class_idx]
        if label == background_class_label:
            continue
        metrics[f"by_class.{label}.loss"] = float(-per_class_logprob[class_idx] / count)
        predicted_count = per_class_predicted.get(class_idx, 0)
        true_positive = per_class_correct.get(class_idx, 0)
        if predicted_count > 0:
            metrics[f"by_class.{label}.precision"] = float(
                true_positive / predicted_count
            )
        if count > 0:
            metrics[f"by_class.{label}.recall"] = float(true_positive / count)
        false_positive = max(predicted_count - true_positive, 0)
        false_negative = max(count - true_positive, 0)
        emit_token_f_scores(
            f"by_class.{label}",
            true_positive,
            false_positive,
            false_negative,
        )

    span_class_indices = set(per_class_span_tp_precision)
    span_class_indices |= set(per_class_span_fp)
    span_class_indices |= set(per_class_span_tp_recall)
    span_class_indices |= set(per_class_span_fn)
    for class_idx in span_class_indices:
        label = label_info.span_class_names[class_idx]
        if label == background_class_label:
            continue
        c_tp_p = per_class_span_tp_precision.get(class_idx, 0)
        c_fp = per_class_span_fp.get(class_idx, 0)
        c_tp_r = per_class_span_tp_recall.get(class_idx, 0)
        c_fn = per_class_span_fn.get(class_idx, 0)
        total_counts = c_tp_p + c_fp + c_tp_r + c_fn
        if total_counts == 0:
            continue
        emit_span_f_scores(
            f"by_class.{label}.span",
            c_tp_p,
            c_tp_p + c_fp,
            c_tp_r,
            c_tp_r + c_fn,
        )
    finalize_seconds += time.perf_counter() - finalize_start

    if timings is not None:
        timings.clear()
        timings["prepare_seconds"] = float(prepare_seconds)
        timings["decode_seconds"] = float(decode_seconds)
        timings["prediction_seconds"] = float(prediction_seconds)
        timings["accuracy_seconds"] = float(accuracy_seconds)
        timings["finalize_seconds"] = float(finalize_seconds)
        timings["total_seconds"] = float(time.perf_counter() - total_start)

    return metrics, example_predictions, label_counts


def write_predictions(
    *,
    output_path: str,
    encoding: tiktoken.Encoding,
    label_info: Any,
    example_texts: Mapping[str, str],
    predictions: Mapping[str, ExamplePrediction],
    trim_span_whitespace: bool,
    discard_overlapping_predicted_spans: bool,
    workers: int = 1,
    token_char_ranges_by_example: (
        Mapping[str, tuple[Sequence[int], Sequence[int]]] | None
    ) = None,
) -> None:
    """Write per-example predictions as JSONL."""
    if workers <= 0:
        raise ValueError("workers must be positive")

    def _build_prediction_line(item: tuple[str, ExamplePrediction]) -> str:
        example_id, prediction = item
        text = example_texts.get(example_id, "")
        cached_char_ranges = (
            token_char_ranges_by_example.get(example_id)
            if token_char_ranges_by_example is not None
            else None
        )
        if cached_char_ranges is not None:
            decoded_text = text
            char_starts, char_ends = cached_char_ranges
        else:
            decoded_text, char_starts, char_ends = decode_text_with_offsets(
                prediction.token_ids, encoding
            )
            if decoded_text != text:
                raise ValueError(
                    "Decoded tokens for example %s do not match golden text"
                    % example_id
                )
        spans = token_spans_to_char_spans(
            prediction.predicted_spans,
            char_starts,
            char_ends,
        )
        if trim_span_whitespace:
            spans = trim_char_spans_whitespace(spans, text)

        if discard_overlapping_predicted_spans:
            spans = discard_overlapping_spans_by_label(spans)

        spans_map: OrderedDict[str, list[list[int]]] = OrderedDict()
        for label_idx, char_start, char_end in spans:
            if not (0 <= char_start < char_end <= len(text)):
                continue
            label = label_info.span_class_names[label_idx]
            substring = text[char_start:char_end]
            key = f"{label}: {substring}"
            spans_map.setdefault(key, []).append([char_start, char_end])
        record = OrderedDict(
            [
                ("example_id", example_id),
                ("text", text),
                ("predicted_spans", spans_map),
            ]
        )
        if prediction.token_logprobs_topk is not None:
            record["token_logprobs_topk"] = list(prediction.token_logprobs_topk)
        return json.dumps(record, ensure_ascii=False)

    lines: list[str] = []
    prediction_items = list(predictions.items())
    if workers == 1 or len(prediction_items) <= 1:
        for item in prediction_items:
            lines.append(_build_prediction_line(item))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for line in executor.map(_build_prediction_line, prediction_items):
                lines.append(line)
    payload = "\n".join(lines)
    if payload:
        payload = f"{payload}\n"
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        handle.write(payload)
