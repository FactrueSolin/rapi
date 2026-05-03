"""Sequence-labeling data structures and window construction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Mapping, Sequence

import torch
from .._common.label_space import (
    BACKGROUND_CLASS_LABEL,
    BOUNDARY_PREFIXES,
)


@dataclass(frozen=True)
class LabelInfo:
    """Resolved label-space mappings used for inference and span decoding."""

    boundary_label_lookup: Mapping[str, Mapping[str, int]]
    token_to_span_label: Mapping[int, int]
    token_boundary_tags: Mapping[int, str | None]
    span_class_names: tuple[str, ...]
    span_label_lookup: Mapping[str, int]
    background_token_label: int
    background_span_label: int


@dataclass(frozen=True)
class TokenizedExample:
    """One tokenized example with aligned token ids and label ids."""

    tokens: tuple[int, ...]
    labels: tuple[int, ...]
    example_id: str
    text: str


@dataclass(frozen=True)
class Window:
    """One inference or eval window derived from a tokenized example."""

    example_id: str | None
    tokens: tuple[int, ...]
    labels: tuple[int, ...]
    offsets: tuple[int, ...]
    token_example_ids: tuple[str | None, ...]
    mask: tuple[int, ...]


@dataclass
class ExampleAggregation:
    """Accumulates per-token scores and metadata across overlapping windows."""

    logprob_logsumexp: list[torch.Tensor | None]
    counts: list[int]
    labels: list[int | None]
    token_ids: list[int | None]
    length: int = 0

    def ensure_capacity(self, index: int) -> None:
        """Grow internal buffers so they can store data for `index`."""
        needed = index + 1 - len(self.logprob_logsumexp)
        if needed > 0:
            self.logprob_logsumexp.extend([None] * needed)
            self.counts.extend([0] * needed)
            self.labels.extend([None] * needed)
            self.token_ids.extend([None] * needed)

    def record_token_id(self, index: int, token_id: int, example_id: str) -> None:
        """Record a token id and reject conflicting observations for one position."""
        existing = self.token_ids[index]
        if existing is None:
            self.token_ids[index] = token_id
        elif existing != token_id:
            raise ValueError(
                f"Conflicting tokens for example {example_id} token {index}: "
                f"{existing} vs {token_id}"
            )


def build_label_info(class_names: Sequence[str]) -> LabelInfo:
    """Build label-space lookup tables from the checkpoint class-name list."""
    span_class_names: list[str] = [BACKGROUND_CLASS_LABEL]
    span_label_lookup: dict[str, int] = {BACKGROUND_CLASS_LABEL: 0}
    boundary_label_lookup: dict[str, dict[str, int]] = {}
    token_to_span_label: dict[int, int] = {}
    token_boundary_tags: dict[int, str | None] = {}
    background_idx: int | None = None

    for idx, name in enumerate(class_names):
        if name == BACKGROUND_CLASS_LABEL:
            background_idx = idx
            token_to_span_label[idx] = span_label_lookup[BACKGROUND_CLASS_LABEL]
            token_boundary_tags[idx] = None
            continue
        boundary, base_label = name.split("-", 1)
        span_idx = span_label_lookup.get(base_label)
        if span_idx is None:
            span_idx = len(span_class_names)
            span_class_names.append(base_label)
            span_label_lookup[base_label] = span_idx
        token_to_span_label[idx] = span_idx
        token_boundary_tags[idx] = boundary
        mapping = boundary_label_lookup.setdefault(base_label, {})
        mapping[boundary] = idx

    if background_idx is None:
        raise ValueError("Class names must include background label 'O'")

    for base_label, mapping in boundary_label_lookup.items():
        missing = set(BOUNDARY_PREFIXES) - set(mapping)
        if missing:
            raise ValueError(
                f"Missing boundary classes {sorted(missing)} for base label {base_label}"
            )

    return LabelInfo(
        boundary_label_lookup={
            key: dict(value) for key, value in boundary_label_lookup.items()
        },
        token_to_span_label=dict(token_to_span_label),
        token_boundary_tags=dict(token_boundary_tags),
        span_class_names=tuple(span_class_names),
        span_label_lookup=dict(span_label_lookup),
        background_token_label=background_idx,
        background_span_label=span_label_lookup[BACKGROUND_CLASS_LABEL],
    )


def example_to_windows(
    example: TokenizedExample,
    window_size: int,
) -> Iterator[Window]:
    """Split a tokenized example into fixed-size windows."""
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    tokens = example.tokens
    labels = example.labels
    if len(tokens) != len(labels):
        raise ValueError(
            "Tokenized example contains mismatched token and label lengths"
        )
    total_tokens = len(tokens)
    if total_tokens == 0:
        yield Window(
            example_id=example.example_id,
            tokens=(),
            labels=(),
            offsets=(),
            token_example_ids=(),
            mask=(),
        )
        return
    stride = window_size
    for start in range(0, total_tokens, stride):
        end = min(start + window_size, total_tokens)
        window_tokens = tokens[start:end]
        window_labels = labels[start:end]
        offsets = tuple(range(start, start + len(window_tokens)))
        token_example_ids = tuple(example.example_id for _ in window_tokens)
        mask = tuple(1 for _ in window_tokens)
        yield Window(
            example_id=example.example_id,
            tokens=tuple(window_tokens),
            labels=tuple(window_labels),
            offsets=offsets,
            token_example_ids=token_example_ids,
            mask=mask,
        )
