"""Inworld TTS-2 (Text-to-Speech) provider for Hermes Agent.

Supports expressive steering tags (``[say excitedly]``), non-verbal sounds
(``[laugh]``, ``[sigh]``), emphasis via capitalization, multi-language output
via ``{language}`` prefix, and configurable delivery modes.
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
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from agent.tts_provider import TTSProvider

from ._common import (
    INWORLD_TTS_API_URL,
    RETRY_POLICY_DEFAULTS,
    api_key,
    compute_retry_backoff,
    is_provider_active,
    is_retryable_http_status,
    parse_retry_after_seconds,
    read_hermes_config,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.environ.get("INWORLD_MODEL_ID", "inworld-tts-2")

# Allowed values for the ``delivery_mode`` ``tts.inworld-tts`` config key.
# Validated against this set in ``InworldTTSProvider._delivery_mode_config``.
_VALID_DELIVERY_MODES: frozenset[str] = frozenset({"STABLE", "BALANCED", "CREATIVE"})

# Default applied when ``delivery_mode`` is unset or invalid. Preserves
# the pre-existing hardcoded payload value.
_DEFAULT_DELIVERY_MODE = "CREATIVE"

# Default HTTP timeout (seconds) for the synchronous TTS endpoint. The
# value in ``tts.inworld-tts.timeout`` overrides this at runtime.
_TTS_HTTP_TIMEOUT_SECONDS = 60

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
    """Inworld TTS-2 provider with steering support."""

    @property
    def name(self) -> str:
        return "inworld-tts"

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
                "id": "inworld-tts-2",
                "display": "Inworld Realtime TTS-2 (flagship, steering)",
            },
            {
                "id": "inworld-tts-1.5-max",
                "display": "Inworld TTS 1.5 Max (stable, high quality)",
            },
            {
                "id": "inworld-tts-1.5-mini",
                "display": "Inworld TTS 1.5 Mini (ultra-fast, lowest cost)",
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
        type). The result is cached for the lifetime of this provider
        instance — config files are not expected to change at runtime
        for the per-provider section (delivery_mode, timeout, retry,
        etc.); only the top-level ``tts.provider`` selection is
        re-checked on every hook call via ``is_provider_active`` from
        ``_common``.
        """
        if self._HERMES_CONFIG_CACHE is not None:
            return self._HERMES_CONFIG_CACHE
        cfg = read_hermes_config()
        tts_cfg = cfg.get("tts", {}) if cfg else {}
        provider_cfg = tts_cfg.get("inworld-tts", {})
        if not isinstance(provider_cfg, dict):
            provider_cfg = {}
        self._HERMES_CONFIG_CACHE = provider_cfg
        return provider_cfg

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
        """Return the resolved retry policy for TTS HTTP requests.

        Falls back to ``RETRY_POLICY_DEFAULTS`` for any key that is
        missing, non-numeric, or out of range. Per-policy keys:
        ``max_retries``, ``initial_backoff``, ``max_backoff``,
        ``backoff_multiplier``, ``retry_on_5xx``, ``retry_on_429``,
        ``retry_on_network_error``.
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

        Args:
            text: Text to speak (can include steering tags like [say excitedly])
            output_path: Where to write the audio file
            voice: Voice ID from tts.inworld-tts.voice in config.yaml
            model: Model ID (default: inworld-tts-2)
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
        model_id = model or DEFAULT_MODEL
        delivery_mode = self._delivery_mode_config()
        timeout = self._tts_timeout_config()
        retry_policy = self._retry_policy_config()

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
            "tag": "TTS-2 steering",
            "env_vars": [
                {
                    "key": "INWORLD_API_KEY",
                    "prompt": "Inworld API key",
                    "url": "https://inworld.ai",
                },
            ],
        }

    # ── Hooks ────────────────────────────────────────────────────────

    def register_hooks(self, ctx):
        """Register lifecycle hooks for steering context injection."""
        ctx.register_hook("pre_llm_call", self._on_pre_llm_call)

    def _on_pre_llm_call(self, is_first_turn, **kwargs):
        """Inject steering instructions on the first turn only.

        Skipped entirely when ``inworld-tts`` is not the configured TTS
        provider — the steering guidance is specific to this provider's
        tag syntax and would mislead the LLM when a different TTS is
        selected. The ``is_provider_active`` helper re-reads the YAML
        config each invocation so a gateway-time change of
        ``tts.provider`` propagates immediately.
        """
        if not is_first_turn:
            return None
        if not is_provider_active("tts", "inworld-tts"):
            return None

        return {"context": _TTS_HOOK_CONTEXT}


# ── First-turn TTS hook context ───────────────────────────────────
# Concatenated for byte-for-byte preservation of the original literal
# returned from ``InworldTTSProvider._on_pre_llm_call``. Wording,
# punctuation, and pre-existing typos are intentional.
_TTS_HOOK_CONTEXT = (
    "## How to Talk When Using TTS (`text_to_speech`)\n\n"
    "When your response gets spoken aloud, drop a bracketed "
    "steering tag at the start of the sentence it applies to.\n"
    "Natural language only. No commas or capitals inside the brackets.\n\n"
    "### Instruction tags\n"
    "Use instruction tags to match your delivery to the content. The following are "
    "suggestions; natural language instructions can be used to describe the "
    "appropriate delivery:\n"
    "- Emotion: [say excitedly], [sound sad], [sound concerned], [sound terrified]\n"
    "- Articulation: [say with force], [articulate clearly], [say with deliberate pauses]\n"
    "- Intonation: [say with a falling pitch], [say with a rising pitch]\n"
    "- Volume: [very quiet], [very loud]\n"
    "- Pitch: [say in a low tone], [say in a high pitch]\n"
    "- Range: [say playfully], [say with no pitch variation]\n"
    "- Speed: [very fast], [very slow]\n"
    "- Vocal style: [whisper in a hushed style], [give a nasal quality]\n"
    "- Non-verbals: [laugh], [sigh], [clear throat], [breathe], [cough], [yawn]\n"
    "### Combining dimensions for richer direction\n"
    "Layer mood, rhythm, pitch and mode in one tag:\n"
    "- [say sadly with deliberate pauses in a low voice and hushed style]\n"
    "- [say excitedly with a high pitch and fast pace]\n"
    "- [sound concerned with a measured pace and low tone]\n"
    "- [quietly with a warm and gentle tone]\n\n"
    "For maximum control, combine qualities from multiple categories "
    "in a single natural language instruction.\n"
    "A single tag applies across multiple subsequent sentences; "
    "repeat or change tags only when the delivery should change.\n\n"
    "### Non-verbals (inline anywhere)\n"
    "[laugh] [sigh] [breathe] [clear throat] [cough] [yawn]\n\n"
    "### Emphasis via capitalization\n"
    'Full word: "I told you NOT to do that"\n'
    'Syllable: "AbsoLUTEly"\n'
    "Use sparingly or it loses impact.\n\n"
    "### Rules\n"
    "- One steering tag per passage. Change it only when the "
    "delivery shifts.\n"
    "- Match the tag to the content. Don't [say sadly] on "
    "celebratory lines.\n"
    "- Avoid opposites in the same tag — [whisper] and [very loud] "
    "together produce unpredictable results.\n"
    "- Non-verbals can go inline wherever natural.\n\n"
    "### Formatting for speech\n"
    '- Write numbers as words: "twenty-three" not "23"\n'
    '- Write dates as spoken: "march fifteenth"\n'
    "- Use contractions: don't, can't, I'm, we're\n"
    "- No markdown, no bullet points, no emoji in speech\n"
    "- Include filler words where natural: uh, um, well, like, you know\n"
    "- Vary sentence length for natural rhythm\n\n"
    "### The goal\n"
    "Don't force it. Use steering when it fits the moment — "
    "hype when something's exciting, quiet when something matters, "
    "[laugh] when something's funny. If it's a simple reply, "
    "just talk. The tags are there to make the spoken delivery sound natural, "
    "not like a robot reading a script.\n"
    "### When to use\n"
    "Use this steering in the input to text_to_speech tool only. "
    "Don't use it in the normal text output to the user.\n\n"
    "### Language override\n"
    "To speak in a language, put the language "
    "name in curly braces at the very start of the text. "
    "Supported: {polish}, {spanish}, {french}, {german}, {italian}, "
    "{portuguese}, {dutch}, {russian}, {japanese}, {chinese}, "
    "{korean}, {arabic}, {hebrew}, {hindi}. "
    'Example: "{polish} Witaj, świecie!" will speak Polish.\n'
)
