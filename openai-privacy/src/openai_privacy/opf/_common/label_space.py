"""Versioned internal label-space definitions used across OPF subsystems."""

import json
from pathlib import Path
from typing import Final, Mapping, Sequence

BACKGROUND_CLASS_LABEL: Final[str] = "O"
"""Background token label used for non-entity positions."""

BOUNDARY_PREFIXES: Final[tuple[str, ...]] = ("B", "I", "E", "S")
"""Boundary markers used to expand span classes into token-level labels."""

DEFAULT_CATEGORY_VERSION: Final[str] = "v2"
"""Default category taxonomy used when a checkpoint provides no label-space hint."""

SPAN_CLASS_NAMES_BY_CATEGORY_VERSION: Final[dict[str, tuple[str, ...]]] = {
    "v2": (
        BACKGROUND_CLASS_LABEL,
        "account_number",
        "private_address",
        "private_date",
        "private_email",
        "private_person",
        "private_phone",
        "private_url",
        "secret",
    ),
    "v4": (
        BACKGROUND_CLASS_LABEL,
        "private_person",
        "other_person",
        "personal_url",
        "other_url",
        "personal_location",
        "other_location",
        "personal_email",
        "other_email",
        "personal_phone",
        "other_phone",
        "personal_date",
        "other_date",
        "personal_id",
        "secret",
    ),
    "v7": (
        BACKGROUND_CLASS_LABEL,
        "personal_name",
        "personal_handle",
        "other_person",
        "personal_email",
        "other_email",
        "personal_phone",
        "other_phone",
        "personal_location",
        "other_location",
        "personal_url",
        "other_url",
        "personal_org",
        "personal_gov_id",
        "personal_fin_id",
        "personal_health_id",
        "personal_device_id",
        "personal_vehicle_id",
        "personal_property_id",
        "personal_edu_id",
        "personal_emp_id",
        "personal_membership_id",
        "personal_registry_id",
        "personal_date",
        "secret",
        "secret_url",
    ),
}
"""Span-level label taxonomy for each supported category version."""


def _expand_with_boundary_markers(span_class_names: Sequence[str]) -> tuple[str, ...]:
    """Expand span labels into token-level BIESO labels."""
    expanded: list[str] = [BACKGROUND_CLASS_LABEL]
    for base_label in span_class_names:
        if base_label == BACKGROUND_CLASS_LABEL:
            continue
        for prefix in BOUNDARY_PREFIXES:
            expanded.append(f"{prefix}-{base_label}")
    return tuple(expanded)


NER_CLASS_NAMES_BY_CATEGORY_VERSION: Final[dict[str, tuple[str, ...]]] = {
    category_version: _expand_with_boundary_markers(span_class_names)
    for category_version, span_class_names in SPAN_CLASS_NAMES_BY_CATEGORY_VERSION.items()
}
"""Token-level BIESO label vocabulary for each supported category version."""

_CATEGORY_VERSION_BY_NUM_LABELS: dict[int, str] = {
    len(ner_class_names): category_version
    for category_version, ner_class_names in NER_CLASS_NAMES_BY_CATEGORY_VERSION.items()
}


def _parse_string_sequence(
    value: object,
    *,
    field_name: str,
    context: str,
) -> list[str]:
    """Parse and validate one non-empty string sequence config field."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{context} {field_name} must be a sequence of strings")
    parsed: list[str] = []
    for idx, item in enumerate(value):
        if not isinstance(item, str):
            raise ValueError(
                f"{context} {field_name}[{idx}] must be a string (got {type(item)!r})"
            )
        normalized = item.strip()
        if not normalized:
            raise ValueError(f"{context} {field_name}[{idx}] must be non-empty")
        parsed.append(normalized)
    if not parsed:
        raise ValueError(f"{context} {field_name} must not be empty")
    return parsed


def _ensure_unique(
    values: Sequence[str],
    *,
    field_name: str,
    context: str,
) -> None:
    """Validate one string sequence has no duplicates."""
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        else:
            seen.add(value)
    if duplicates:
        shown = ", ".join(sorted(duplicates))
        raise ValueError(f"{context} {field_name} has duplicate values: {shown}")


def _normalize_span_class_names(
    span_class_names: Sequence[str],
    *,
    context: str,
) -> tuple[str, ...]:
    """Normalize span class names with ``O`` forced into index 0."""
    _ensure_unique(span_class_names, field_name="span_class_names", context=context)
    non_background = [
        name for name in span_class_names if name != BACKGROUND_CLASS_LABEL
    ]
    if len(non_background) == len(span_class_names):
        raise ValueError(
            f"{context} span_class_names must include background label {BACKGROUND_CLASS_LABEL!r}"
        )
    return (BACKGROUND_CLASS_LABEL, *non_background)


def _parse_ner_class_names(
    ner_class_names: Sequence[str],
    *,
    context: str,
) -> tuple[str, ...]:
    """Parse and validate token-level BIESO class names."""
    _ensure_unique(ner_class_names, field_name="ner_class_names", context=context)
    if BACKGROUND_CLASS_LABEL not in ner_class_names:
        raise ValueError(
            f"{context} ner_class_names must include background label {BACKGROUND_CLASS_LABEL!r}"
        )
    boundary_map: dict[str, set[str]] = {}
    for name in ner_class_names:
        if name == BACKGROUND_CLASS_LABEL:
            continue
        if "-" not in name:
            raise ValueError(
                f"{context} ner_class_names entry {name!r} must use '<B|I|E|S>-<label>' format"
            )
        boundary, base_label = name.split("-", 1)
        if boundary not in BOUNDARY_PREFIXES or not base_label:
            raise ValueError(
                f"{context} ner_class_names entry {name!r} must use '<B|I|E|S>-<label>' format"
            )
        boundary_map.setdefault(base_label, set()).add(boundary)
    for base_label, boundaries in boundary_map.items():
        missing = set(BOUNDARY_PREFIXES) - boundaries
        if missing:
            shown = ", ".join(sorted(missing))
            raise ValueError(
                f"{context} ner_class_names missing boundaries [{shown}] for label {base_label!r}"
            )
    return tuple(ner_class_names)


def _resolve_custom_label_space_from_config(
    checkpoint_config: Mapping[str, object],
    *,
    context: str,
) -> tuple[str, tuple[str, ...], tuple[str, ...]] | None:
    """Resolve one custom label-space payload from checkpoint config fields."""
    custom_span_raw = checkpoint_config.get("span_class_names")
    custom_ner_raw = checkpoint_config.get("ner_class_names")
    has_custom = custom_span_raw is not None or custom_ner_raw is not None
    if not has_custom:
        return None

    span_class_names: tuple[str, ...]
    if custom_span_raw is None:
        span_class_names = ()
    else:
        parsed_span = _parse_string_sequence(
            custom_span_raw,
            field_name="span_class_names",
            context=context,
        )
        span_class_names = _normalize_span_class_names(parsed_span, context=context)

    ner_class_names: tuple[str, ...]
    if custom_ner_raw is None:
        if not span_class_names:
            raise ValueError(
                f"{context} custom label space requires span_class_names or ner_class_names"
            )
        ner_class_names = _expand_with_boundary_markers(span_class_names)
    else:
        parsed_ner = _parse_string_sequence(
            custom_ner_raw,
            field_name="ner_class_names",
            context=context,
        )
        ner_class_names = _parse_ner_class_names(parsed_ner, context=context)

    if not span_class_names:
        # Derive span labels from the provided token-level labels.
        derived: list[str] = [BACKGROUND_CLASS_LABEL]
        seen_span: set[str] = {BACKGROUND_CLASS_LABEL}
        for name in ner_class_names:
            if name == BACKGROUND_CLASS_LABEL:
                continue
            _boundary, base_label = name.split("-", 1)
            if base_label not in seen_span:
                seen_span.add(base_label)
                derived.append(base_label)
        span_class_names = tuple(derived)
    else:
        expected_ner = _expand_with_boundary_markers(span_class_names)
        if tuple(ner_class_names) != tuple(expected_ner):
            raise ValueError(
                f"{context} custom ner_class_names do not match span_class_names expansion"
            )

    configured_category_version_raw = checkpoint_config.get("category_version")
    if configured_category_version_raw is None:
        resolved_category_version = "custom"
    else:
        resolved_category_version = str(configured_category_version_raw).strip().lower()
        if not resolved_category_version:
            raise ValueError(
                f"{context} category_version must be non-empty when provided"
            )
        if resolved_category_version in SPAN_CLASS_NAMES_BY_CATEGORY_VERSION:
            builtin_span = SPAN_CLASS_NAMES_BY_CATEGORY_VERSION[
                resolved_category_version
            ]
            builtin_ner = NER_CLASS_NAMES_BY_CATEGORY_VERSION[resolved_category_version]
            if tuple(span_class_names) != tuple(builtin_span) or tuple(
                ner_class_names
            ) != tuple(builtin_ner):
                raise ValueError(
                    f"{context} custom label names conflict with built-in "
                    f"category_version={resolved_category_version!r}"
                )

    num_labels_raw = checkpoint_config.get("num_labels")
    if num_labels_raw is not None:
        if isinstance(num_labels_raw, bool):
            raise ValueError(f"{context} num_labels must be an integer, got bool")
        try:
            num_labels = int(num_labels_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{context} num_labels must be an integer, got {num_labels_raw!r}"
            ) from exc
        if num_labels != len(ner_class_names):
            raise ValueError(
                f"{context} num_labels={num_labels} does not match ner_class_names "
                f"length={len(ner_class_names)}"
            )

    return (
        resolved_category_version,
        span_class_names,
        ner_class_names,
    )


def resolve_label_space_from_config(
    checkpoint_config: Mapping[str, object],
    *,
    context: str,
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """Resolve the category version and label sets described by a checkpoint config."""
    custom_resolution = _resolve_custom_label_space_from_config(
        checkpoint_config,
        context=context,
    )
    if custom_resolution is not None:
        return custom_resolution

    configured_category_version_raw = checkpoint_config.get("category_version")
    configured_category_version: str | None = None
    if configured_category_version_raw is not None:
        configured_category_version = (
            str(configured_category_version_raw).strip().lower()
        )
        if configured_category_version not in SPAN_CLASS_NAMES_BY_CATEGORY_VERSION:
            known = ", ".join(sorted(SPAN_CLASS_NAMES_BY_CATEGORY_VERSION))
            raise ValueError(
                f"{context} has unsupported category_version={configured_category_version_raw!r}. "
                f"Known values: {known}"
            )

    inferred_category_version: str | None = None
    num_labels_raw = checkpoint_config.get("num_labels")
    if num_labels_raw is not None:
        if isinstance(num_labels_raw, bool):
            raise ValueError(f"{context} num_labels must be an integer, got bool")
        try:
            num_labels = int(num_labels_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{context} num_labels must be an integer, got {num_labels_raw!r}"
            ) from exc
        inferred_category_version = _CATEGORY_VERSION_BY_NUM_LABELS.get(num_labels)
        if inferred_category_version is None:
            known_counts = ", ".join(
                f"{version}:{count}"
                for count, version in sorted(_CATEGORY_VERSION_BY_NUM_LABELS.items())
            )
            raise ValueError(
                f"{context} num_labels={num_labels} does not match known encoder label spaces "
                f"({known_counts})"
            )

    if (
        configured_category_version is not None
        and inferred_category_version is not None
        and configured_category_version != inferred_category_version
    ):
        raise ValueError(
            f"{context} has inconsistent label-space hints: "
            f"category_version={configured_category_version!r}, "
            f"num_labels={checkpoint_config.get('num_labels')!r} "
            f"(infers {inferred_category_version!r})"
        )

    resolved_category_version = (
        configured_category_version
        or inferred_category_version
        or DEFAULT_CATEGORY_VERSION
    )
    return (
        resolved_category_version,
        SPAN_CLASS_NAMES_BY_CATEGORY_VERSION[resolved_category_version],
        NER_CLASS_NAMES_BY_CATEGORY_VERSION[resolved_category_version],
    )


def resolve_checkpoint_label_space(
    checkpoint_dir: str | Path,
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """Load `config.json` from a checkpoint directory and resolve its label space."""
    config_path = Path(checkpoint_dir) / "config.json"
    with config_path.open("r", encoding="utf-8") as handle:
        checkpoint_config = json.load(handle)
    if not isinstance(checkpoint_config, Mapping):
        raise ValueError(f"Invalid checkpoint config payload at {config_path}")
    return resolve_label_space_from_config(
        checkpoint_config,
        context=str(config_path),
    )
