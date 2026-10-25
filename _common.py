"""Shared utilities for the hermes-inworld plugin.

Constants, retry helpers, the shared HTTP opener, and config-reader used by
the TTS and STT provider modules (see :mod:`.tts`, :mod:`.stt`, and
:mod:`._tts_streaming`).
"""

import logging
import os
import time
import urllib.error
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

try:
    from ruamel.yaml import YAML as _yaml
except ImportError:
    print("Failed to import ruamel.yaml; falling back to config-less defaults.")
    _yaml = None  # type: ignore[assignment]

INWORLD_TTS_API_URL = "https://api.inworld.ai/tts/v1/voice"
INWORLD_TTS_STREAM_API_URL = "https://api.inworld.ai/tts/v1/voice:stream"
INWORLD_STT_API_URL = "https://api.inworld.ai/stt/v1/transcribe"

logger = logging.getLogger(__name__)


def api_key() -> str:
    key = os.environ.get("INWORLD_API_KEY", "")
    if not key:
        raise RuntimeError(
            "INWORLD_API_KEY is not set. "
            "Add it to ~/.hermes/.env or export it in your shell."
        )
    return key


# ── Retry policy helpers ──────────────────────────────────────────
# Shared by tts.py, _tts_streaming.py, and stt.py. Each provider reads
# its own config section (the ``retry:`` block) and resolves a policy
# dict. TTS sync synthesis and TTS streaming share ``open_with_retries``
# below; STT keeps its own loop in stt.py because its AUTO_DETECT
# fallback and result-envelope error contract are different.


def is_retryable_http_status(code: int, retry_on_5xx: bool, retry_on_429: bool) -> bool:
    """Return True when an HTTP status is transient and worth retrying."""
    if retry_on_429 and code == 429:
        return True
    if retry_on_5xx and 500 <= code < 600:
        return True
    return False


def _exception_chain(error: BaseException) -> Iterator[BaseException]:
    """Yield ``error`` and its ``__cause__``/``__context__`` chain.

    Cycle protection is required because re-raised exceptions can form a
    loop (``__context__`` pointing back at an ancestor).
    """
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _header_retry_after(headers: object) -> Any:
    """Return the raw ``Retry-After`` value from a header mapping, if any.

    Tries both casings so plain ``dict`` headers (which are
    case-sensitive) and ``HTTPMessage``/``httpx.Headers`` both work.
    """
    if headers is None:
        return None
    getter = getattr(headers, "get", None)
    if getter is None:
        return None
    for name in ("Retry-After", "retry-after"):
        try:
            raw = getter(name)
        except Exception:
            return None
        if raw is not None:
            return raw
    return None


def _parse_retry_after_value(raw: Any) -> float | None:
    """Parse an integer-seconds or HTTP-date ``Retry-After`` value."""
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


def parse_retry_after_seconds(error: Exception) -> float | None:
    """Read the ``Retry-After`` header from an exception, if present.

    Checks the exception's own ``headers`` first (the ``urllib.error``
    shape), then ``error.response.headers`` (the OpenAI/httpx shape), then
    walks the ``__cause__``/``__context__`` chain. Returns the wait
    duration in seconds (integer seconds or HTTP-date), or ``None`` when
    no header is found/unparseable so the caller can fall back to the
    policy's exponential backoff.
    """
    for exc in _exception_chain(error):
        raw = _header_retry_after(getattr(exc, "headers", None))
        if raw is None:
            response = getattr(exc, "response", None)
            raw = _header_retry_after(getattr(response, "headers", None))
        if raw is not None:
            return _parse_retry_after_value(raw)
    return None


_HTTP_STATUS_ATTRS = ("status_code", "status", "http_status", "code")

_NETWORK_ERROR_CLASS_NAMES: frozenset[str] = frozenset(
    {
        "APIConnectionError",
        "APITimeoutError",
        "TimeoutException",
        "TimeoutError",
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "ReadError",
        "WriteTimeout",
        "PoolTimeout",
        "NetworkError",
        "TransportError",
        "ConnectionError",
        "ConnectionResetError",
        "ConnectionAbortedError",
        "ConnectionRefusedError",
        "RemoteDisconnected",
        "IncompleteRead",
        "URLError",
        "SSLError",
    }
)

_NETWORK_ERROR_MESSAGE_MARKERS: tuple[str, ...] = (
    "connection reset",
    "connection refused",
    "timed out",
    "timeout",
    "temporary failure in name resolution",
    "incomplete chunked read",
    "peer closed connection",
    "response ended prematurely",
    "remote end closed connection",
)


def get_exception_http_status(error: BaseException) -> int | None:
    """Return an HTTP status code carried by ``error`` or its chain.

    Supports OpenAI SDK ``status_code`` attributes, ``response`` objects,
    and ``urllib.error.HTTPError`` (``code``). Only integer values are
    accepted because some exception classes use ``code`` for a string
    error identifier.
    """
    for exc in _exception_chain(error):
        for attr in _HTTP_STATUS_ATTRS:
            value = getattr(exc, attr, None)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
        response = getattr(exc, "response", None)
        if response is None:
            continue
        for attr in ("status_code", "status"):
            value = getattr(response, attr, None)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
    return None


def is_network_error(error: BaseException) -> bool:
    """Return True when ``error`` or its chain looks like a transport error.

    Uses class-name matching (so OpenAI/httpx exception classes are
    recognized without importing them) plus a conservative set of message
    markers. Unknown failures return False.
    """
    for exc in _exception_chain(error):
        names = {cls.__name__ for cls in type(exc).__mro__}
        if names & _NETWORK_ERROR_CLASS_NAMES:
            return True
        message = str(exc).lower()
        if any(marker in message for marker in _NETWORK_ERROR_MESSAGE_MARKERS):
            return True
    return False


def is_retryable_llm_error(error: BaseException, policy: dict[str, Any]) -> bool:
    """Return True when an LLM client error is transient per ``policy``.

    A known HTTP status is authoritative: only 429/5xx are retried (per
    the ``retry_on_429``/``retry_on_5xx`` flags). Without a status, only
    connection/timeout failures are retried, and only when
    ``retry_on_network_error`` is enabled. Anything unrecognized is not
    retried.
    """
    status = get_exception_http_status(error)
    if isinstance(status, int):
        return is_retryable_http_status(
            status,
            bool(policy.get("retry_on_5xx", False)),
            bool(policy.get("retry_on_429", False)),
        )
    return bool(policy.get("retry_on_network_error", False)) and is_network_error(error)


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


def compute_retry_delay(
    error: Exception,
    attempt: int,
    policy: dict[str, Any],
) -> float:
    """Prefer a server-provided ``Retry-After`` over exponential backoff."""
    retry_after = parse_retry_after_seconds(error)
    if retry_after is not None:
        return retry_after
    return compute_retry_backoff(
        attempt,
        policy["initial_backoff"],
        policy["backoff_multiplier"],
        policy["max_backoff"],
    )


def open_with_retries(
    open_request: Callable[[], Any],
    policy: dict[str, Any],
    *,
    service: str = "Inworld TTS",
    log: logging.Logger | None = None,
) -> Any:
    """Open an HTTP request, retrying transient failures per ``policy``.

    ``open_request`` is called once per attempt and must perform only the
    open (for example ``urllib.request.urlopen``); reading and decoding
    stay with the caller. 5xx/429 responses are retried when the policy
    enables them, network errors are retried when
    ``retry_on_network_error`` is set, and a server ``Retry-After`` header
    wins over the policy's exponential backoff. ``service`` prefixes the
    warning lines and ``log`` selects the caller's module logger.

    This helper never wraps or replaces the final failure: the last
    ``HTTPError``/``URLError`` is re-raised untouched so callers can keep
    their own ``RuntimeError`` wording and ``from e`` chaining.
    """
    target_log = log if log is not None else logger
    for attempt in range(policy["max_retries"] + 1):
        try:
            return open_request()
        except urllib.error.HTTPError as e:
            if not is_retryable_http_status(
                e.code, policy["retry_on_5xx"], policy["retry_on_429"]
            ):
                raise
            if attempt >= policy["max_retries"]:
                raise
            backoff = compute_retry_delay(e, attempt, policy)
            target_log.warning(
                "%s HTTP %d — retry %d/%d in %.1fs",
                service,
                e.code,
                attempt + 1,
                policy["max_retries"],
                backoff,
            )
            time.sleep(backoff)
        except urllib.error.URLError as e:
            if not policy["retry_on_network_error"]:
                raise
            if attempt >= policy["max_retries"]:
                raise
            backoff = compute_retry_backoff(
                attempt,
                policy["initial_backoff"],
                policy["backoff_multiplier"],
                policy["max_backoff"],
            )
            target_log.warning(
                "%s network error (%s) — retry %d/%d in %.1fs",
                service,
                e.reason,
                attempt + 1,
                policy["max_retries"],
                backoff,
            )
            time.sleep(backoff)
    # Unreachable: the loop returns on success or re-raises on the final
    # attempt. Defensive raise to keep the type checker happy.
    raise RuntimeError(f"{service} retry loop exited unexpectedly")


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
            yaml = _yaml(typ="safe")
            loaded = yaml.load(f)
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
    safer than injecting stale provider-specific guidance. A non-dict
    section (for example ``tts: null``) is treated as inactive.
    """
    cfg = read_hermes_config()
    section = cfg.get(provider_section)
    return isinstance(section, dict) and section.get("provider") == expected_name
