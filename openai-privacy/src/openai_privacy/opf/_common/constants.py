"""Cross-cutting internal constants shared across OPF subsystems."""

from pathlib import Path
from typing import Final


DEFAULT_MODEL_ENV_VAR: Final[str] = "OPF_CHECKPOINT"
"""Environment variable used to override the default checkpoint directory."""

DEFAULT_MODEL_PATH: Final[Path] = Path.home() / ".opf" / "privacy_filter"
"""Default local checkpoint directory used when no override is provided."""

OUTPUT_MODES: Final[tuple[str, str]] = ("typed", "redacted")
"""Supported structured output modes."""

SCHEMA_VERSION: Final[int] = 1
"""Current structured output schema version emitted by OPF."""

REDACTED_OUTPUT_LABEL: Final[str] = "redacted"
"""Collapsed label used when output mode hides model-specific categories."""

REDACTED_OUTPUT_PLACEHOLDER: Final[str] = "<REDACTED>"
"""Replacement marker inserted for spans in fully redacted output mode."""
