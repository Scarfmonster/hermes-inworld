"""HTTP streaming transport for the Inworld TTS provider.

Implements the plugin side of Hermes' raw-PCM streaming contract against
``POST https://api.inworld.ai/tts/v1/voice:stream``: it opens the request
through the shared retry opener in :mod:`._common`, parses the NDJSON response
line by line, and yields base64-decoded PCM16LE mono chunks at
:data:`STREAM_SAMPLE_RATE`.

The module imports only :mod:`._common`; it must never import :mod:`tts`
(that would create an import cycle). It receives a built payload and an
already-resolved API key, timeout, and retry policy.
"""

from __future__ import annotations

import base64
import json
import logging
import urllib.error
import urllib.request
from collections.abc import Iterable, Iterator
from typing import IO, Any

from . import _common

logger = logging.getLogger(__name__)

# Raw int16 little-endian mono PCM at 24,000 Hz. Matches the sync request
# rate and stays inside Inworld's documented 8,000-48,000 Hz range.
STREAM_SAMPLE_RATE = 24000

# Streaming requests accept 4,000 UTF-16 code units per request.
STREAM_MAX_UTF16_UNITS = 4000

# Thinking is disabled for the per-sentence streaming rewrite. Copied before
# being passed to ``call_llm`` so the module constant stays immutable.
STREAM_REWRITE_REASONING_CONFIG: dict[str, bool] = {"enabled": False}


def utf16_len(text: str) -> int:
    """Return the number of UTF-16 code units in ``text``.

    Astral characters (code points above U+FFFF) count as two units, matching
    the API's request limit.
    """
    return sum(2 if ord(char) > 0xFFFF else 1 for char in text)


def truncate_utf16(text: str, limit: int) -> str:
    """Truncate ``text`` to at most ``limit`` UTF-16 code units.

    Astral characters count as two units and are never split. A non-positive
    ``limit`` returns an empty string.
    """
    if limit <= 0:
        return ""
    units = 0
    for index, char in enumerate(text):
        units += 2 if ord(char) > 0xFFFF else 1
        if units > limit:
            return text[:index]
    return text


def open_stream_request(
    payload: dict[str, Any], key: str, timeout: float | int
) -> IO[bytes]:
    """Open the streaming endpoint and return the unread ``HTTPResponse``.

    Sends the built payload as JSON with HTTP Basic auth. The caller owns
    reading and closing the response; retries around this call live in
    :func:`stream_synthesis`.
    """
    request = urllib.request.Request(
        _common.INWORLD_TTS_STREAM_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Basic {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    return urllib.request.urlopen(request, timeout=timeout)


def _error_detail(error: object) -> str:
    """Format a streamed error payload with its code and message."""
    if isinstance(error, dict):
        parts: list[str] = []
        code = error.get("code")
        message = error.get("message")
        if code is not None:
            parts.append(f"code {code}")
        if message is not None:
            parts.append(str(message))
        if parts:
            return ": ".join(parts)
        return json.dumps(error, ensure_ascii=False)
    return str(error)


def _decode_stream_message(raw_line: bytes) -> bytes | None:
    """Return audio bytes for one NDJSON line, or ``None`` when it has none.

    Blank lines and timestamp-only messages return ``None``. Protocol errors
    raise ``RuntimeError`` with the wording used by :func:`iter_stream_audio`.
    """
    try:
        line = raw_line.decode("utf-8")
    except UnicodeDecodeError as e:
        raise RuntimeError(f"Invalid UTF-8 from Inworld streaming API: {e}") from e
    line = line.strip()
    if not line:
        return None
    try:
        message = json.loads(line)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON from Inworld streaming API: {e}") from e
    if not isinstance(message, dict):
        raise RuntimeError(f"Unexpected Inworld streaming response: {line!r}")

    error = message.get("error")
    if error:
        raise RuntimeError(f"Inworld streaming API error: {_error_detail(error)}")
    if "code" in message and "message" in message:
        raise RuntimeError(f"Inworld streaming API error: {_error_detail(message)}")

    if "result" not in message:
        return None
    result = message["result"]
    if not isinstance(result, dict):
        raise RuntimeError(f"Unexpected Inworld streaming result: {line!r}")

    if "audioContent" not in result:
        return None
    audio_b64 = result["audioContent"]
    if not isinstance(audio_b64, str):
        raise RuntimeError(f"Unexpected Inworld streaming audioContent: {line!r}")
    if not audio_b64:
        return None
    try:
        return base64.b64decode(audio_b64, validate=True)
    except Exception as e:
        raise RuntimeError(
            f"Failed to decode base64 audio from Inworld streaming API: {e}"
        ) from e


def iter_stream_audio(response: Iterable[bytes]) -> Iterator[bytes]:
    """Yield decoded PCM chunks from an Inworld NDJSON streaming response.

    Iterates the response line by line so partial network reads are buffered
    and an unterminated final line is still parsed. Each line is parsed by
    :func:`_decode_stream_message`; blank and timestamp-only messages are
    skipped. Malformed JSON, streamed error envelopes, invalid payload shapes,
    and invalid base64 raise ``RuntimeError``.

    Raises ``RuntimeError`` at EOF when the stream produced no audio at all.
    """
    yielded = False
    for raw_line in response:
        audio = _decode_stream_message(raw_line)
        if audio is None:
            continue
        yielded = True
        yield audio
    if not yielded:
        raise RuntimeError("Inworld streaming API returned no audio.")


def stream_synthesis(
    payload: dict[str, Any],
    key: str,
    timeout: float | int,
    policy: dict[str, Any],
) -> Iterator[bytes]:
    """Stream PCM for ``payload``, retrying only the initial HTTP open.

    The open goes through :func:`_common.open_with_retries`, so 5xx/429 and
    network failures follow the configured policy. An exhausted open is
    translated to ``RuntimeError`` carrying the HTTP status and body or the
    URL reason, with the original exception chained.

    Once the response is open, audio is parsed incrementally and the response
    is always closed in ``finally`` (including an early generator close). A
    mid-stream transport failure propagates without a retry because retrying
    would duplicate audio; protocol errors use the ``RuntimeError`` wording
    from :func:`iter_stream_audio`.
    """
    try:
        response = _common.open_with_retries(
            lambda: open_stream_request(payload, key, timeout),
            policy,
            service="Inworld TTS streaming",
            log=logger,
        )
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Inworld streaming API returned HTTP {e.code}: {body}"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Could not reach Inworld streaming API: {e.reason}") from e
    try:
        yield from iter_stream_audio(response)
    finally:
        response.close()
