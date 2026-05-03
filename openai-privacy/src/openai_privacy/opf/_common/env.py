"""Environment variable parsing helpers."""

from __future__ import annotations

import os

TRUE_ENV_VALUES = frozenset({"1", "true", "yes", "on"})


def get_env_bool(name: str, *, default: bool = False) -> bool:
    """Return whether one environment variable is set to a truthy value."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in TRUE_ENV_VALUES
