from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from typing import Mapping, Sequence

import torch
import tiktoken

from .._core.sequence_labeling import LabelInfo, TokenizedExample
from .._core.spans import token_char_ranges_for_text
from .._common.label_space import BACKGROUND_CLASS_LABEL


@dataclass(frozen=True)
class NamedEntity:
    """Ground-truth named-entity span from an eval dataset."""

    start: int
    end: int
    label: str


@dataclass(frozen=True)
class PreparedTokenizedExample:
    """Tokenized eval example with cached character-offset metadata."""

    tokenized: TokenizedExample
    char_starts: tuple[int, ...]
    char_ends: tuple[int, ...]
    original_gold_char_spans: tuple[tuple[str, int, int], ...]


@dataclass
class EvalExampleAggregation:
    """Accumulated logits, labels, and token ids for one eval example."""

    score_matrix: torch.Tensor
    written: torch.Tensor
    labels: torch.Tensor
    token_ids: tuple[int, ...]
    length: int


def _stable_example_id_for_record(record: Mapping[str, object], idx: int) -> str:
    """Build a stable example id from one eval record."""
    text = str(record.get("text", ""))
    spans = record.get("spans")
    labels = record.get("label")
    payload: dict[str, object] = {"text": text, "index": idx}
    if spans is not None:
        payload["spans"] = spans
    if labels is not None:
        payload["label"] = labels
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_integer_offset(
    value: object, *, idx: int, field: str, label: str | None = None
) -> int:
    """Validate that one dataset offset is a real integer character index."""
    if isinstance(value, bool) or not isinstance(value, int):
        label_text = "" if label is None else f" for {label!r}"
        raise ValueError(
            f"Example {idx} {field}{label_text} must be an integer character offset"
        )
    return value


def parse_record(
    record: Mapping[str, object], idx: int
) -> tuple[str, str, list[NamedEntity]]:
    """Parse one eval record into text and normalized entity spans."""
    text = str(record.get("text", ""))
    example_id = _stable_example_id_for_record(record, idx)

    spans = record.get("spans") or {}
    entities: list[NamedEntity] = []

    if spans:
        if not isinstance(spans, Mapping):
            raise ValueError(
                f"Example {idx} spans field is not a mapping: {type(spans)!r}"
            )
        for key, offsets in spans.items():
            label_key = str(key)
            label = label_key.split(": ", 1)[0] if ": " in label_key else label_key
            if not label:
                raise ValueError(
                    f"Example {idx} span label {label_key!r} missing category"
                )
            if not isinstance(offsets, Sequence):
                raise ValueError(
                    f"Example {idx} span list for {label_key!r} is not a sequence"
                )
            for span_idx, span in enumerate(offsets):
                if not isinstance(span, Sequence) or isinstance(span, (bytes, str)):
                    raise ValueError(
                        f"Example {idx} span {span_idx} for {label_key!r} is not [start, end]"
                    )
                if len(span) != 2:
                    raise ValueError(
                        f"Example {idx} span {span_idx} for {label_key!r} does not have 2 elements"
                    )
                start, end = span
                entities.append(
                    NamedEntity(
                        start=_require_integer_offset(
                            start,
                            idx=idx,
                            field=f"span {span_idx} start",
                            label=label_key,
                        ),
                        end=_require_integer_offset(
                            end,
                            idx=idx,
                            field=f"span {span_idx} end",
                            label=label_key,
                        ),
                        label=label,
                    )
                )
        return example_id, text, entities

    labels = record.get("label") or []
    if not isinstance(labels, Sequence) or isinstance(labels, (bytes, str)):
        raise ValueError(
            f"Example {idx} label field is not a sequence: {type(labels)!r}"
        )
    for label_idx, entry in enumerate(labels):
        if not isinstance(entry, Mapping):
            raise ValueError(f"Example {idx} label {label_idx} is not an object")
        category = entry.get("category")
        if not isinstance(category, str) or not category:
            raise ValueError(f"Example {idx} label {label_idx} missing category")
        start = entry.get("start")
        end = entry.get("end")
        entities.append(
            NamedEntity(
                start=_require_integer_offset(
                    start,
                    idx=idx,
                    field=f"label {label_idx} start",
                ),
                end=_require_integer_offset(
                    end,
                    idx=idx,
                    field=f"label {label_idx} end",
                ),
                label=category,
            )
        )
    return example_id, text, entities


def _discard_overlapping_entity_spans(
    spans: Sequence[tuple[int, int, int]],
) -> list[tuple[int, int, int]]:
    """Drop overlapping entity spans after sorting by start and length."""
    sorted_spans = sorted(spans, key=lambda span: (span[0], -(span[1] - span[0])))
    kept: list[tuple[int, int, int]] = []
    for candidate_start, candidate_end, candidate_label in sorted_spans:
        if kept and candidate_start < kept[-1][1]:
            continue
        kept.append((candidate_start, candidate_end, candidate_label))
    return kept


def token_char_ranges(
    tokens: Sequence[int], encoding: tiktoken.Encoding, text: str
) -> tuple[list[int], list[int]]:
    """Compute token-to-character ranges for one known source text."""
    char_starts, char_ends = token_char_ranges_for_text(tokens, encoding, text)
    if len(char_starts) != len(tokens) or len(char_ends) != len(tokens):
        raise ValueError("Failed to compute character offsets for all tokens")
    return char_starts, char_ends


def labels_from_entities(
    tokens: Sequence[int],
    text: str,
    entities: Sequence[NamedEntity],
    label_info: LabelInfo,
    encoding: tiktoken.Encoding,
    eval_mode: str = "typed",
    token_char_ranges_hint: tuple[Sequence[int], Sequence[int]] | None = None,
) -> list[int]:
    """Project character-span entities onto the token label sequence."""
    if not entities:
        return [label_info.background_token_label for _ in tokens]
    if eval_mode not in {"typed", "untyped"}:
        raise ValueError(f"Unsupported eval_mode: {eval_mode!r}")
    is_untyped = eval_mode == "untyped"
    span_label_lookup = label_info.span_label_lookup
    fallback_untyped_label = next(
        (
            label
            for label in label_info.span_class_names
            if label != BACKGROUND_CLASS_LABEL
        ),
        None,
    )
    if is_untyped and fallback_untyped_label is None:
        raise ValueError(
            "No untyped fallback span label available for --eval-mode untyped"
        )
    for entity in entities:
        if is_untyped:
            continue
        if entity.label not in span_label_lookup:
            raise ValueError(
                f"Unknown entity label: {entity.label}. "
                "If your dataset uses a different label scheme, rerun with "
                "--eval-mode untyped."
            )

    normalized: list[tuple[int, int, int]] = []
    text_length = len(text)
    for entity in entities:
        start_idx, end_idx = entity.start, entity.end
        if not (0 <= start_idx <= end_idx <= text_length):
            raise ValueError(
                f"Entity span ({entity.start}, {entity.end}) invalid for text '{text}'"
            )
        resolved_label = fallback_untyped_label if is_untyped else entity.label
        if resolved_label is None:
            raise ValueError(
                "No untyped fallback span label available for --eval-mode untyped"
            )
        label_id = span_label_lookup[resolved_label]
        normalized.append((start_idx, end_idx, label_id))
    normalized = _discard_overlapping_entity_spans(normalized)

    if token_char_ranges_hint is None:
        char_starts, char_ends = token_char_ranges(tokens, encoding, text)
    else:
        char_starts, char_ends = token_char_ranges_hint
        if len(char_starts) != len(tokens) or len(char_ends) != len(tokens):
            raise ValueError("Token char ranges must align to token length")
    labels: list[int] = [label_info.background_token_label] * len(tokens)
    span_to_tokens: defaultdict[int, list[int]] = defaultdict(list)
    best_span_by_token = [-1] * len(tokens)
    best_overlap_by_token = [0] * len(tokens)

    for span_idx, (span_start, span_end, _span_label) in enumerate(normalized):
        token_idx = bisect_right(char_ends, span_start)
        while token_idx < len(tokens) and char_starts[token_idx] < span_end:
            token_start = char_starts[token_idx]
            token_end = char_ends[token_idx]
            overlap = min(token_end, span_end) - max(token_start, span_start)
            if overlap > best_overlap_by_token[token_idx]:
                best_overlap_by_token[token_idx] = overlap
                best_span_by_token[token_idx] = span_idx
            token_idx += 1

    for token_idx, span_idx in enumerate(best_span_by_token):
        if span_idx >= 0 and best_overlap_by_token[token_idx] > 0:
            span_to_tokens[span_idx].append(token_idx)

    for span_idx, token_indices in span_to_tokens.items():
        if not token_indices:
            continue
        span_label_id = normalized[span_idx][2]
        span_label = label_info.span_class_names[span_label_id]
        boundary_lookup = label_info.boundary_label_lookup.get(span_label)
        if boundary_lookup is None:
            raise ValueError(f"No boundary labels registered for class {span_label!r}")
        if len(token_indices) == 1:
            labels[token_indices[0]] = boundary_lookup["S"]
            continue
        labels[token_indices[0]] = boundary_lookup["B"]
        labels[token_indices[-1]] = boundary_lookup["E"]
        for mid_idx in token_indices[1:-1]:
            labels[mid_idx] = boundary_lookup["I"]

    if len(labels) != len(tokens):
        raise ValueError(
            f"Example '{text}' produced {len(tokens)} tokens but {len(labels)} labels."
        )
    return labels
