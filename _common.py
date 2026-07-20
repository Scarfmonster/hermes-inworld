"""Shared utilities for the hermes-inworld plugin.

Constants, retry helpers, and config-reader used by both the TTS and STT
provider modules (see :mod:`.tts` and :mod:`.stt`).
"""

import os
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

try:
    import yaml as _yaml
except ImportError:
    _yaml = None  # type: ignore[assignment]  # PyYAML is a declared dependency; this guards import-time robustness

INWORLD_TTS_API_URL = "https://api.inworld.ai/tts/v1/voice"
INWORLD_STT_API_URL = "https://api.inworld.ai/stt/v1/transcribe"


def api_key() -> str:
    key = os.environ.get("INWORLD_API_KEY", "")
    if not key:
        raise RuntimeError(
            "INWORLD_API_KEY is not set. "
            "Add it to ~/.hermes/.env or export it in your shell."
        )
    return key


# ── Retry policy helpers ──────────────────────────────────────────
# Shared by tts.py and stt.py. Each provider reads its own config
# section (the ``retry:`` block) and resolves a policy dict; the two
# transport loops live in their respective modules so the error
# contracts (TTS raises, STT returns envelope) stay separate.


def is_retryable_http_status(code: int, retry_on_5xx: bool, retry_on_429: bool) -> bool:
    """Return True when an HTTP status is transient and worth retrying."""
    if retry_on_429 and code == 429:
        return True
    if retry_on_5xx and 500 <= code < 600:
        return True
    return False


def parse_retry_after_seconds(error: Exception) -> float | None:
    """Read the ``Retry-After`` header from an HTTPError, if present.

    Returns the wait duration in seconds (integer seconds or HTTP-date),
    or ``None`` when the header is missing/unparseable so the caller can
    fall back to the policy's exponential backoff.
    """
    headers = getattr(error, "headers", None)
    if headers is None:
        return None
    raw = headers.get("Retry-After") if hasattr(headers, "get") else None
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        pass
    try:
        target = parsedate_to_datetime(raw)
        if target is None:
            return None
        if target.tzinfo is None:
            target = target.replace(tzinfo=timezone.utc)
        delta = (target - datetime.now(tz=timezone.utc)).total_seconds()
        return max(0.0, delta)
    except (TypeError, ValueError):
        return None


def compute_retry_backoff(
    attempt: int,
    initial_backoff: float,
    backoff_multiplier: float,
    max_backoff: float,
) -> float:
    """Exponential backoff capped at ``max_backoff`` (seconds).

    ``attempt`` is zero-indexed: the first retry uses ``initial_backoff``;
    each subsequent retry multiplies the wait by ``backoff_multiplier``.
    """
    if attempt < 0:
        attempt = 0
    return min(initial_backoff * (backoff_multiplier**attempt), max_backoff)


# Defaults for the per-provider ``retry:`` config block. Each provider
# falls back to these when ``config.yaml`` does not specify a value.
RETRY_POLICY_DEFAULTS: dict[str, float | bool | int] = {
    "max_retries": 3,
    "initial_backoff": 1.0,
    "max_backoff": 30.0,
    "backoff_multiplier": 2.0,
    "retry_on_5xx": True,
    "retry_on_429": True,
    "retry_on_network_error": True,
}


# ── Hermes config reader (uncached) ────────────────────────────────
# Read fresh on every call so a gateway-time change to ``tts.provider`` /
# ``stt.provider`` (or any other selection switch) propagates on the
# next hook invocation. Failing closed — a missing file, missing
# PyYAML, or an invalid YAML returns ``{}``, which downstream checks
# interpret as "this provider is not active" and skip the hook.

_HERMES_CONFIG_PATH_DEFAULT = "/opt/data/config.yaml"


def read_hermes_config() -> dict[str, Any]:
    """Read ``$HERMES_HOME/config.yaml`` fresh.

    Returns ``{}`` on any failure (missing PyYAML, missing file,
    invalid YAML, wrong root type). Intentionally **non-caching** —
    callers that need an ``==`` comparison against a configured value
    rely on seeing the latest file contents.
    """
    if _yaml is None:
        return {}
    config_path = (
        os.environ.get("HERMES_HOME", _HERMES_CONFIG_PATH_DEFAULT) + "/config.yaml"
    )
    try:
        with open(config_path) as f:
            loaded = _yaml.safe_load(f)
    except Exception:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def is_provider_active(provider_section: str, expected_name: str) -> bool:
    """Return True when ``expected_name`` is the configured section provider.

    Args:
        provider_section: ``"tts"`` or ``"stt"``.
        expected_name: e.g. ``"inworld-tts"`` or ``"inworld-stt"``.

    Reads the YAML config each invocation (no caching) so a runtime
    change of ``tts.provider`` propagates immediately. A failing read
    returns False, which makes the hook skip — failing closed is
    safer than injecting stale provider-specific guidance.
    """
    cfg = read_hermes_config()
    return cfg.get(provider_section, {}).get("provider") == expected_name
