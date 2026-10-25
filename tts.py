"""Inworld TTS provider for Hermes Agent.

Supports two models: ``inworld-tts-2`` (instruction tags such as
``[say excitedly]``, non-verbal sounds, emphasis via capitalization) and
``inworld-tts-2-flash`` (non-verbal sounds only; instruction tags are
ignored server-side). Both support multi-language output via a
``{language}`` prefix and configurable delivery modes.

When ``tts.inworld-tts.steering_rewrite`` is enabled, the text is first
rewritten by an auxiliary LLM (task ``tts_audio_tags``) to add expressive
tags and formatting appropriate to the resolved model before synthesis. The
legacy first-turn ``pre_llm_call`` prompt injection has been removed.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from agent.tts_provider import TTSProvider

from ._common import (
    INWORLD_TTS_API_URL,
    RETRY_POLICY_DEFAULTS,
    api_key,
    compute_retry_backoff,
    compute_retry_delay,
    is_retryable_http_status,
    is_retryable_llm_error,
    parse_retry_after_seconds,
    read_hermes_config,
)
from ._tts_prompts import TTS_REDUCED_REWRITE_PROMPT, TTS_REWRITE_PROMPT

logger = logging.getLogger(__name__)

_PROVIDER_NAME = "inworld-tts"

_MODEL_TTS2 = "inworld-tts-2"
_MODEL_TTS2_FLASH = "inworld-tts-2-flash"

DEFAULT_MODEL = _MODEL_TTS2


def _nonblank_str(value: object) -> str | None:
    """Return a stripped string, or ``None`` when blank or non-string."""
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return stripped
    return None


def _as_dict(value: object) -> dict[str, Any]:
    """Return ``value`` when it is a dict, otherwise an empty dict."""
    return value if isinstance(value, dict) else {}


def _clean_model_id(model_id: str | None) -> str:
    """Strip a caller/config model id; fall back on empty or non-string."""
    return _nonblank_str(model_id) or DEFAULT_MODEL


# Allowed values for the ``delivery_mode`` ``tts.inworld-tts`` config key.
# Validated against this set in ``InworldTTSProvider._delivery_mode_config``.
_VALID_DELIVERY_MODES: frozenset[str] = frozenset({"STABLE", "BALANCED", "CREATIVE"})

# Default applied when ``delivery_mode`` is unset or invalid. Preserves
# the pre-existing hardcoded payload value.
_DEFAULT_DELIVERY_MODE = "CREATIVE"

# Default HTTP timeout (seconds) for the synchronous TTS endpoint. The
# value in ``tts.inworld-tts.timeout`` overrides this at runtime.
_TTS_HTTP_TIMEOUT_SECONDS = 60

# Auxiliary LLM task used to rewrite TTS text when ``steering_rewrite`` is
# enabled. Shared with the Gemini/xAI tag rewriting task in Hermes.
_TTS_REWRITE_TASK = "tts_audio_tags"
_TTS_REWRITE_TEMPERATURE = 0.7

# ── Language override ────────────────────────────────────────────
# Full-word language names → ISO codes for GA Inworld TTS-2 languages.
# Prefix any TTS text with {language_name} to set the language explicitly.
# Example: "{polish} Witaj, świecie!" → language="pl"
_LANG_MAP: dict[str, str] = {
    name: code
    for name, code in (
        ("arabic", "ar"),
        ("chinese", "zh"),
        ("dutch", "nl"),
        ("english", "en"),
        ("french", "fr"),
        ("german", "de"),
        ("hebrew", "he"),
        ("hindi", "hi"),
        ("italian", "it"),
        ("japanese", "ja"),
        ("korean", "ko"),
        ("polish", "pl"),
        ("portuguese", "pt"),
        ("russian", "ru"),
        ("spanish", "es"),
    )
}
# Also accept ISO codes directly
for _code in (
    "ar",
    "zh",
    "nl",
    "en",
    "fr",
    "de",
    "he",
    "hi",
    "it",
    "ja",
    "ko",
    "pl",
    "pt",
    "ru",
    "es",
):
    _LANG_MAP[_code] = _code


_LANG_RE = re.compile(r"^\{([a-z]{2,20})\}\s*")

# ── Protocol constants ────────────────────────────────────────────
# Map incoming format strings to Inworld's documented audioEncoding values.
# Insertion order is significant — earlier keys are matched first.
_FORMAT_TO_ENCODING: dict[str, str] = {
    "MP3": "MP3",
    "WAV": "WAV",
    "OGG": "OGG_OPUS",
    "OGG_OPUS": "OGG_OPUS",
    "FLAC": "FLAC",
    "LINEAR16": "LINEAR16",
    "PCM": "LINEAR16",
}


def _extract_language(text: str) -> tuple[str, str | None]:
    """Extract a language override from the beginning of TTS text.

    If text starts with ``{language}`` (e.g. ``{polish}``, ``{pl}``),
    strips the prefix and returns (cleaned_text, iso_code).

    Returns (original_text, None) when no valid prefix is found.
    """
    m = _LANG_RE.match(text)
    if not m:
        return text, None
    raw = m.group(1).lower()
    code = _LANG_MAP.get(raw)
    if not code:
        return text, None
    return text[m.end() :].lstrip(), code


# ── Model classification ──────────────────────────────────────────
# Exact-ID matching is deliberate: the fallback for any future or renamed
# model is "no prompt context" rather than a guess. Add new IDs to these
# sets when Inworld ships them.
_STEERING_MODEL_IDS: frozenset[str] = frozenset({_MODEL_TTS2})
_FLASH_MODEL_IDS: frozenset[str] = frozenset({_MODEL_TTS2_FLASH})


def _normalized_model_id(model_id: str | None) -> str:
    return (_nonblank_str(model_id) or "").lower()


def _supports_inworld_steering(model_id: str | None) -> bool:
    """True only for models that accept instruction tags."""
    return _normalized_model_id(model_id) in _STEERING_MODEL_IDS


def _is_inworld_flash_model(model_id: str | None) -> bool:
    """True only for the Flash model (non-verbals only)."""
    return _normalized_model_id(model_id) in _FLASH_MODEL_IDS


_warned_skipped_tts_rewrite_models: set[str] = set()


def _warn_tts_rewrite_skipped_once(model_id: str) -> None:
    key = _normalized_model_id(model_id)
    if key in _warned_skipped_tts_rewrite_models:
        return
    _warned_skipped_tts_rewrite_models.add(key)
    logger.warning(
        "Inworld model %r is not a current TTS-2 model; skipping TTS rewrite. "
        "Use %s or %s.",
        model_id,
        _MODEL_TTS2,
        _MODEL_TTS2_FLASH,
    )


def _extract_auxiliary_message_content(response: Any) -> str:
    """Return the message content string from an OpenAI-style response."""
    try:
        content = response.choices[0].message.content
    except Exception:
        return ""
    return content if isinstance(content, str) else ""


def _clean_rewrite_output(content: str) -> str:
    """Strip surrounding whitespace and a fence wrapping the whole output."""
    clean = (content or "").strip()
    fence = re.fullmatch(
        r"```(?:[A-Za-z0-9_-]+)?\s*(.*?)\s*```", clean, flags=re.DOTALL
    )
    return fence.group(1).strip() if fence else clean


def _call_llm_rewrite(transcript: str, system_prompt: str) -> str:
    """Make one auxiliary LLM rewrite attempt and return cleaned content."""
    from agent.auxiliary_client import call_llm

    response = call_llm(
        task=_TTS_REWRITE_TASK,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"TRANSCRIPT TO REWRITE:\n{transcript}"},
        ],
        temperature=_TTS_REWRITE_TEMPERATURE,
    )
    return _clean_rewrite_output(_extract_auxiliary_message_content(response))


def _rewrite_tts_text(text: str, system_prompt: str, policy: dict[str, Any]) -> str:
    """Rewrite ``text`` with the auxiliary LLM, retrying transient failures.

    Retries follow ``policy`` (the shared TTS ``retry`` block) and fall
    back to the original text once the policy is exhausted or a
    non-transient error occurs. Only ``Exception`` is caught so
    cancellation signals propagate.
    """
    transcript = text.strip()
    if not transcript:
        return text
    for attempt in range(policy["max_retries"] + 1):
        try:
            rewritten = _call_llm_rewrite(transcript, system_prompt)
        except Exception as exc:
            is_last_attempt = attempt >= policy["max_retries"]
            if is_last_attempt or not is_retryable_llm_error(exc, policy):
                logger.warning(
                    "Inworld TTS rewrite failed; using original text: %s", exc
                )
                return text
            backoff = compute_retry_delay(exc, attempt, policy)
            logger.warning(
                "Inworld TTS rewrite failed (%s) — retry %d/%d in %.1fs",
                exc,
                attempt + 1,
                policy["max_retries"],
                backoff,
            )
            time.sleep(backoff)
        else:
            if not rewritten:
                logger.warning(
                    "Inworld TTS rewrite returned empty text; using original."
                )
                return text
            logger.debug(
                "Inworld TTS rewrite applied (%d -> %d chars)",
                len(transcript),
                len(rewritten),
            )
            return rewritten
    return text


def _rewrite_prompt_for_model(model_id: str | None) -> str | None:
    """Return the rewrite prompt for ``model_id``, or ``None`` to skip."""
    if _supports_inworld_steering(model_id):
        return TTS_REWRITE_PROMPT
    if _is_inworld_flash_model(model_id):
        return TTS_REDUCED_REWRITE_PROMPT
    return None


def _apply_steering_rewrite(text: str, model_id: str, policy: dict[str, Any]) -> str:
    """Rewrite text for the resolved model, or return it unchanged."""
    rewrite_prompt = _rewrite_prompt_for_model(model_id)
    if rewrite_prompt is None:
        _warn_tts_rewrite_skipped_once(model_id)
        return text
    return _rewrite_tts_text(text, rewrite_prompt, policy)


def _build_synthesis_payload(
    text: str,
    voice_id: str,
    model_id: str,
    speed: float | None,
    output_format: str,
    delivery_mode: str = _DEFAULT_DELIVERY_MODE,
) -> dict[str, Any]:
    """Build the request payload for the synchronous TTS endpoint.

    Performs language-prefix extraction (``{polish} Witaj!``) and returns
    the request dict. ``speakingRate`` lives inside ``audioConfig`` per
    the documented Inworld v1AudioConfig schema. Note: voice outputs may
    differ from earlier placements of this field at non-1.0 speeds.
    """
    # Extract language override from text prefix
    # e.g. "{polish} Witaj!" → text="Witaj!", language="pl"
    text, lang_code = _extract_language(text)

    payload: dict[str, Any] = {
        "modelId": model_id,
        "voiceId": voice_id,
        "text": text,
        "audioConfig": {
            "audioEncoding": "MP3",
            "sampleRateHertz": 24000,
        },
        "deliveryMode": delivery_mode,
        "applyTextNormalization": "ON",
    }

    # Set language if overridden
    if lang_code:
        payload["language"] = lang_code

    # Add speed if overridden — placed inside ``audioConfig`` per the
    # current Inworld v1AudioConfig schema.
    if speed is not None and speed > 0:
        payload["audioConfig"]["speakingRate"] = speed

    # Map format string to Inworld's encoding
    format_upper = output_format.upper()
    payload["audioConfig"]["audioEncoding"] = _FORMAT_TO_ENCODING.get(
        format_upper, "MP3"
    )

    return payload


def _open_synthesis_request(payload: dict[str, Any], key: str, timeout: float | int):
    """Open the synchronous TTS endpoint with the given payload + key.

    No retry — that lives in ``InworldTTSProvider._synthesize_with_retries``.
    HTTP/URL translation stays in the caller so the original RuntimeError
    messages and ``from e`` chaining are preserved.
    """
    req = urllib.request.Request(
        INWORLD_TTS_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Basic {key}",
            "Content-Type": "application/json",
        },
    )
    return urllib.request.urlopen(req, timeout=timeout)


def _decode_synthesis_response(response) -> bytes:
    """Decode a TTS response into raw audio bytes.

    Raises RuntimeError on JSON parse failure, missing audioContent,
    or base64 decode failure.
    """
    try:
        data: dict[str, Any] = json.loads(response.read())
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON from Inworld API: {e}") from e

    audio_b64 = data.get("audioContent")
    if not audio_b64:
        raise RuntimeError(
            f"Inworld API returned no audioContent. "
            f"Response: {json.dumps(data, indent=2)}"
        )

    try:
        return base64.b64decode(audio_b64)
    except Exception as e:
        raise RuntimeError(f"Failed to decode base64 audio: {e}") from e


# ── Provider ────────────────────────────────────────────────────────


class InworldTTSProvider(TTSProvider):
    """Inworld TTS-2 provider with optional LLM steering rewrite."""

    @property
    def name(self) -> str:
        return _PROVIDER_NAME

    @property
    def display_name(self) -> str:
        return "Inworld TTS-2"

    def default_voice(self) -> str:
        return "Sarah"

    @property
    def voice_compatible(self) -> bool:
        """Return True when ffmpeg is available (needed for Opus conversion).

        Inworld returns MP3. For Telegram voice bubbles we need Opus/OGG.
        """
        return shutil.which("ffmpeg") is not None

    def is_available(self) -> bool:
        return bool(os.environ.get("INWORLD_API_KEY"))

    def list_voices(self) -> list[dict]:
        return []

    def list_models(self) -> list[dict]:
        return [
            {
                "id": _MODEL_TTS2,
                "display": "Inworld Realtime TTS-2 (flagship, steering)",
            },
            {
                "id": _MODEL_TTS2_FLASH,
                "display": "Inworld Realtime TTS-2 Flash (fastest, lowest cost)",
            },
        ]

    # ── Config-driven settings ────────────────────────────────────────
    # Each getter caches the parsed config block once per process so the
    # cost of reading ``HERMES_HOME/config.yaml`` is paid only on the
    # first call.

    _HERMES_CONFIG_CACHE: dict[str, Any] | None = None

    def _get_provider_config(self) -> dict[str, Any]:
        """Return the ``tts.inworld-tts`` block from Hermes config.yaml.

        Returns ``{}`` on any error (missing file, invalid YAML, wrong
        type). A non-dict ``tts`` section is treated as empty. The result
        is cached for the lifetime of this provider instance — config
        files are not expected to change at runtime for the per-provider
        section (delivery_mode, steering_rewrite, timeout, retry, etc.).
        Restart Hermes/gateway after changing these values.
        """
        if self._HERMES_CONFIG_CACHE is not None:
            return self._HERMES_CONFIG_CACHE
        cfg = read_hermes_config()
        tts_cfg = _as_dict(cfg.get("tts"))
        provider_cfg = _as_dict(tts_cfg.get(_PROVIDER_NAME))
        self._HERMES_CONFIG_CACHE = provider_cfg
        return provider_cfg

    def _configured_model(self, forwarded: str | None = None) -> str:
        """Return the effective model for this provider.

        Reads ``tts.inworld-tts.model`` (alongside ``voice``) fresh so a
        gateway-time change applies on the next synthesis. Precedence:
        the plugin-scoped ``model`` key, then the top-level ``tts.model``
        Hermes forwards, then the ``forwarded`` argument, then
        ``DEFAULT_MODEL``. Blank, non-string, or malformed values are
        skipped.
        """
        cfg = read_hermes_config()
        tts_cfg = _as_dict(cfg.get("tts"))
        provider_cfg = _as_dict(tts_cfg.get(_PROVIDER_NAME))
        model = _nonblank_str(provider_cfg.get("model")) or _nonblank_str(
            tts_cfg.get("model")
        )
        return model or _clean_model_id(forwarded)

    def _delivery_mode_config(self) -> str:
        """Return the validated ``delivery_mode`` value.

        Configures the voice style/expressiveness tradeoff. Accepts
        ``STABLE`` (consistent), ``BALANCED`` (default), or ``CREATIVE``
        (expressive). Falls back to ``_DEFAULT_DELIVERY_MODE`` when unset
        or set to anything other than the documented three values.
        """
        cfg = self._get_provider_config()
        raw = cfg.get("delivery_mode")
        if not isinstance(raw, str):
            return _DEFAULT_DELIVERY_MODE
        upper = raw.strip().upper()
        if upper in _VALID_DELIVERY_MODES:
            return upper
        logger.warning(
            "Ignoring invalid tts.inworld-tts.delivery_mode=%r; expected "
            "one of %s. Falling back to %s.",
            raw,
            sorted(_VALID_DELIVERY_MODES),
            _DEFAULT_DELIVERY_MODE,
        )
        return _DEFAULT_DELIVERY_MODE

    def _steering_rewrite_config(self) -> bool:
        """Return whether the auxiliary TTS rewrite pass is enabled.

        Reads ``tts.inworld-tts.steering_rewrite`` (default ``false``).
        Accepts booleans and common string spellings. The provider config
        block is cached per instance, so changing this value requires a
        Hermes/gateway restart.
        """
        cfg = self._get_provider_config()
        raw = cfg.get("steering_rewrite", False)
        if isinstance(raw, str):
            return raw.strip().lower() in {"1", "true", "yes", "on"}
        return bool(raw)

    def _tts_timeout_config(self) -> int:
        """Return the HTTP request timeout (seconds) for TTS.

        Falls back to ``_TTS_HTTP_TIMEOUT_SECONDS`` when unset, negative,
        or non-numeric. Values smaller than 1 are clamped to 1.
        """
        cfg = self._get_provider_config()
        raw = cfg.get("timeout", _TTS_HTTP_TIMEOUT_SECONDS)
        if raw is None:
            return _TTS_HTTP_TIMEOUT_SECONDS
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return _TTS_HTTP_TIMEOUT_SECONDS
        if value < 1:
            return 1
        return value

    def _retry_policy_config(self) -> dict[str, Any]:
        """Return the resolved retry policy for TTS HTTP and rewrite calls.

        Falls back to ``RETRY_POLICY_DEFAULTS`` for any key that is
        missing, non-numeric, or out of range. The same policy drives the
        TTS HTTP requests and the optional steering-rewrite LLM call.
        Per-policy keys: ``max_retries``, ``initial_backoff``,
        ``max_backoff``, ``backoff_multiplier``, ``retry_on_5xx``,
        ``retry_on_429``, ``retry_on_network_error``.
        """
        cfg = self._get_provider_config()
        retry_cfg = cfg.get("retry", {})
        if not isinstance(retry_cfg, dict):
            retry_cfg = {}
        policy: dict[str, Any] = {}
        for key, default in RETRY_POLICY_DEFAULTS.items():
            raw = (
                retry_cfg.get(key, default) if isinstance(retry_cfg, dict) else default
            )
            if raw is None:
                raw = default
            if isinstance(default, bool):
                policy[key] = bool(raw)
            elif isinstance(default, int):
                try:
                    policy[key] = max(0, int(raw))
                except (TypeError, ValueError):
                    policy[key] = default
            else:
                try:
                    policy[key] = max(0.0, float(raw))
                except (TypeError, ValueError):
                    policy[key] = default
        return policy

    def _voice_config(self) -> str:
        """Return the configured voice ID, or ``default_voice()`` when unset.

        Reads ``tts.inworld-tts.voice`` from Hermes ``config.yaml``. Non-string
        or whitespace-only values fall back to ``default_voice()`` so a
        typo like ``voice: 123`` doesn't crash synthesis.
        """
        cfg = self._get_provider_config()
        raw = cfg.get("voice")
        if isinstance(raw, str):
            stripped = raw.strip()
            if stripped:
                return stripped
        return self.default_voice()

    def _synthesize_with_retries(
        self,
        payload: dict[str, Any],
        key: str,
        timeout: int,
        policy: dict[str, Any],
    ):
        """Open the TTS endpoint, retrying transient errors per policy.

        The helper raises the last ``HTTPError``/``URLError`` when the
        policy is exhausted so ``synthesize`` can preserve the original
        ``from e`` chaining and RuntimeError wording.
        """
        last_error: Exception | None = None
        for attempt in range(policy["max_retries"] + 1):
            try:
                return _open_synthesis_request(payload, key, timeout)
            except urllib.error.HTTPError as e:
                last_error = e
                if not is_retryable_http_status(
                    e.code, policy["retry_on_5xx"], policy["retry_on_429"]
                ):
                    raise
                if attempt >= policy["max_retries"]:
                    raise
                retry_after = parse_retry_after_seconds(e)
                backoff = (
                    retry_after
                    if retry_after is not None
                    else compute_retry_backoff(
                        attempt,
                        policy["initial_backoff"],
                        policy["backoff_multiplier"],
                        policy["max_backoff"],
                    )
                )
                logger.warning(
                    "Inworld TTS HTTP %d — retry %d/%d in %.1fs",
                    e.code,
                    attempt + 1,
                    policy["max_retries"],
                    backoff,
                )
                time.sleep(backoff)
            except urllib.error.URLError as e:
                last_error = e
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
                logger.warning(
                    "Inworld TTS network error (%s) — retry %d/%d in %.1fs",
                    e.reason,
                    attempt + 1,
                    policy["max_retries"],
                    backoff,
                )
                time.sleep(backoff)
        # Unreachable: the loop returns on success or re-raises on the
        # final attempt. Defensive raise to keep the type checker happy.
        if last_error is not None:
            raise last_error
        raise RuntimeError("Inworld TTS retry loop exited unexpectedly")

    def synthesize(
        self,
        text: str,
        output_path: str,
        *,
        voice: str | None = None,
        model: str | None = None,
        speed: float | None = None,
        format: str = "mp3",
        **extra,
    ) -> str:
        """Synthesize speech via Inworld TTS-2 API.

        When ``tts.inworld-tts.steering_rewrite`` is enabled, the text is
        first rewritten by the auxiliary LLM (task ``tts_audio_tags``) using
        the prompt appropriate for the resolved model. Any rewrite failure
        or empty response falls back to the original text.

        Args:
            text: Text to speak (can include steering tags like [say excitedly])
            output_path: Where to write the audio file
            voice: Voice ID from tts.inworld-tts.voice in config.yaml
            model: Model ID forwarded by Hermes; ``tts.inworld-tts.model``
                wins when set, otherwise this value is used, falling back
                to ``DEFAULT_MODEL``
            speed: Speaking rate 0.5-1.5 (default: 1.0). The field is
                serialized inside ``audioConfig.speakingRate`` per the
                documented Inworld v1AudioConfig schema; existing voice
                outputs at non-1.0 speeds may sound different from prior
                plugin versions which placed this field at the top level.
            format: Audio format — mp3, wav, ogg_opus, flac, linear16

        Returns:
            The output_path that was written to.

        Raises:
            RuntimeError: On API errors, auth failures, or empty responses.
        """
        key = api_key()
        voice_id = voice or self._voice_config()
        model_id = self._configured_model(model)
        retry_policy = self._retry_policy_config()
        if self._steering_rewrite_config():
            text = _apply_steering_rewrite(text, model_id, retry_policy)
        delivery_mode = self._delivery_mode_config()
        timeout = self._tts_timeout_config()

        payload = _build_synthesis_payload(
            text, voice_id, model_id, speed, format, delivery_mode
        )

        # Make the API call. The retry loop honors the configured retry
        # policy and re-raises the last HTTPError/URLError when retries
        # are exhausted so the original ``from e`` chaining stays intact.
        try:
            resp = self._synthesize_with_retries(payload, key, timeout, retry_policy)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Inworld API returned HTTP {e.code}: {body}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Could not reach Inworld API: {e.reason}") from e

        audio_bytes = _decode_synthesis_response(resp)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(audio_bytes)

        return output_path

    def get_setup_schema(self) -> dict:
        """Show setup info in hermes tools / hermes setup UI."""
        return {
            "name": self.display_name,
            "badge": "Inworld AI",
            "tag": "TTS-2 voice synthesis",
            "env_vars": [
                {
                    "key": "INWORLD_API_KEY",
                    "prompt": "Inworld API key",
                    "url": "https://inworld.ai",
                },
            ],
        }
