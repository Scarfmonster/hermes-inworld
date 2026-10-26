"""Inworld STT (Speech-to-Text) provider for Hermes Agent.

Transcribes audio files and optionally appends Voice Profile metadata
(emotion, age, accent, pitch, vocal style) with confidence scores to the
transcript when ``voice_profile`` is enabled.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import struct
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict, cast

from agent.transcription_provider import TranscriptionProvider

from ._common import (
    INWORLD_STT_API_URL,
    RETRY_POLICY_DEFAULTS,
    api_key,
    compute_retry_backoff,
    is_provider_active,
    is_retryable_http_status,
    parse_retry_after_seconds,
    read_hermes_config,
)

logger = logging.getLogger(__name__)


class _ModelMeta(TypedDict):
    id: str
    display: str
    languages: list[str]


class _ProbeResult(TypedDict, total=False):
    sample_rate: int | None
    channels: int | None
    encoding: str


# Models supported through Inworld's STT router
MODELS: list[_ModelMeta] = [
    {
        "id": "inworld/inworld-stt-1",
        "display": "Inworld Realtime STT-1 — voice profile (emotion, age, accent, pitch, vocal style) + configurable turn-taking",
        "languages": ["en"],
    },
    {
        "id": "groq/whisper-large-v3",
        "display": "Groq Whisper Large v3 — 99+ languages, fast",
        "languages": [],
    },
]

DEFAULT_MODEL: str = MODELS[0]["id"]
DEFAULT_LANGUAGE = ""  # empty = auto-detect


# ── Protocol constants ─────────────────────────────────────────────
# Extension → Inworld audioEncoding mapping for sync STT.
_EXT_TO_ENCODING: dict[str, str] = {
    ".wav": "LINEAR16",
    ".mp3": "MP3",
    ".ogg": "OGG_OPUS",
    ".opus": "OGG_OPUS",
    ".flac": "FLAC",
    ".pcm": "LINEAR16",
    ".raw": "LINEAR16",
}

# File extensions Inworld's sync STT endpoint accepts verbatim.
_SUPPORTED_EXTS = frozenset({".wav", ".mp3", ".ogg", ".opus", ".flac", ".pcm", ".raw"})

# Voice Profile fields stripped from the transcript suffix.
_VOICE_PROFILE_SKIP_KEYS = frozenset({"age", "gender", "accent"})

# Per-label confidence floor for the Voice Profile suffix. Labels below
# this threshold are dropped so the model is not distracted by weak
# detections. Configurable via ``stt.inworld-stt.voice_profile_min_confidence``.
_VOICE_PROFILE_MIN_CONFIDENCE = 0.2

# Native voice surfaces that synthesize assistant replies themselves.
# On these platforms the plugin does not instruct the model to call
# ``text_to_speech`` unless the user explicitly asks for audio.
# Configurable via ``stt.inworld-stt.voice_reply.platforms``.
_DEFAULT_VOICE_REPLY_PLATFORMS = frozenset({"desktop", "tui", "cli"})

# STT sources that belong to a native voice surface. Recordings from these
# sources omit the ``voice_prefix`` marker because the surrounding client
# already presents the turn as voice. Configurable via
# ``stt.inworld-stt.voice_reply.sources``.
_DEFAULT_VOICE_REPLY_SOURCES = frozenset({"voice_mode"})

# Transcription source reported by the host for the current dispatch.
# ``pre_transcription`` runs synchronously before the provider call, so a
# value set there is visible here and to every chunk dispatched by
# ``transcribe_oversized()``.
_current_transcription_source: ContextVar[str | None] = ContextVar(
    "inworld_stt_source", default=None
)


@dataclass(frozen=True)
class _VoiceReplyConfig:
    """Resolved ``voice_reply`` policy for native voice surfaces.

    Attributes:
        platforms: Lower-cased platform names whose replies are spoken by
            the client, so the model needs no ``text_to_speech`` guidance.
        sources: Lower-cased STT sources whose transcripts omit the
            ``voice_prefix`` marker.
        suppress_voice_prefix: Whether ``sources`` transcripts omit the
            marker.
        suppress_reply_prompt: Whether ``platforms`` first turns skip the
            ``text_to_speech`` instruction.
    """

    platforms: frozenset[str] = _DEFAULT_VOICE_REPLY_PLATFORMS
    sources: frozenset[str] = _DEFAULT_VOICE_REPLY_SOURCES
    suppress_voice_prefix: bool = True
    suppress_reply_prompt: bool = True

    def omits_prefix_for(self, source: str | None) -> bool:
        """Return whether ``source`` transcripts omit the voice prefix."""
        return self.suppress_voice_prefix and bool(source) and source in self.sources

    def skips_reply_prompt_for(self, platform: str) -> bool:
        """Return whether ``platform`` first turns skip the TTS instruction."""
        return self.suppress_reply_prompt and platform in self.platforms


_VOICE_REPLY_DEFAULTS = _VoiceReplyConfig()
_VOICE_REPLY_DISABLED = _VoiceReplyConfig(
    platforms=frozenset(),
    sources=frozenset(),
    suppress_voice_prefix=False,
    suppress_reply_prompt=False,
)


# Reference glossary for the Voice Profile suffix injected into the LLM
# prompt. Each top-level key is a Voice Profile category returned at the
# response root (``voiceProfile.<category>``); the inner mapping gives
# every label the API may emit alongside a one-line description of what
# it means. Sourced from https://docs.inworld.ai/stt/voice-profiles.
#
# Categories whose labels are stripped via ``voice_profile_skip_keys``
# are filtered out of the prompt glossary at injection time so the model
# does not see guidance for values it will never observe.
_VOICE_PROFILE_LABELS: dict[str, dict[str, str]] = {
    "age": {
        "young": "Young adult / teenager",
        "adult": "Adult speaker",
        "kid": "Child speaker",
        "old": "Elderly speaker",
        "unclear": "Age could not be determined",
    },
    "emotion": {
        "tender": "Soft, gentle, caring tone",
        "sad": "Sorrowful or melancholy tone",
        "calm": "Relaxed, even-tempered delivery",
        "neutral": "No strong emotional signal",
        "happy": "Cheerful, upbeat tone",
        "angry": "Frustrated, aggressive tone",
        "fearful": "Anxious or frightened tone",
        "surprised": "Startled or astonished tone",
        "disgusted": "Revulsion or strong disapproval",
        "unclear": "Emotion could not be determined",
    },
    "pitch": {
        "low": "Low-pitched voice",
        "medium": "Medium-pitched voice",
        "high": "High-pitched voice",
    },
    "vocalStyle": {
        "whispering": "Hushed, breathy delivery",
        "normal": "Standard conversational speech",
        "singing": "Melodic or musical delivery",
        "mumbling": "Unclear, low-articulation speech",
        "crying": "Speech accompanied by crying",
        "laughing": "Speech accompanied by laughter",
        "shouting": "Loud, raised-voice delivery",
        "monotone": "Flat, unvaried pitch delivery",
        "unclear": "Vocal style could not be determined",
    },
    "accent": {
        "en-US": "American English",
        "en-GB": "British English",
        "en-AU": "Australian English",
        "zh-CN": "Mandarin Chinese",
        "fr-FR": "French (France)",
        "es-ES": "Spanish (Spain)",
        "es-419": "Spanish (Latin America)",
        "es-MX": "Spanish (Mexico)",
        "ar-EG": "Arabic (Egypt)",
    },
}

# Process / HTTP timeouts (seconds).
_FFMPEG_CONVERSION_TIMEOUT_SECONDS = 120
_FFPROBE_TIMEOUT_SECONDS = 10
_STT_HTTP_TIMEOUT_SECONDS = 120


# ── Helpers ─────────────────────────────────────────────────────────


def _normalize_voice_reply_set(
    raw: Any,
    default: frozenset[str],
) -> frozenset[str]:
    """Normalize a ``voice_reply`` list value to a lower-cased set.

    Accepts a list of strings or a single string (treated as a
    one-element list). Blanks are dropped, so ``[]`` yields the empty
    set (suppression disabled for that axis) while missing/``null``/other
    non-list values fall back to ``default``.
    """
    if isinstance(raw, str):
        values: list[Any] = [raw]
    elif isinstance(raw, list):
        values = raw
    else:
        return default
    normalized = {str(value).strip().lower() for value in values}
    normalized.discard("")
    return frozenset(normalized)


def _voice_reply_flag(raw: Any, default: bool) -> bool:
    """Return a boolean ``voice_reply`` toggle, falling back to ``default``.

    Only real booleans are honored; malformed values (``null``, numbers,
    strings) keep the default so a typo cannot silently flip behavior.
    """
    return raw if isinstance(raw, bool) else default


def _parse_voice_reply_config(raw: Any) -> _VoiceReplyConfig:
    """Parse the ``voice_reply`` config block into a resolved policy.

    Behavior:

    - missing, ``null``, or ``true``: use the defaults;
    - literal ``false`` or ``{enabled: false}``: disable all handling;
    - any other non-dict value (for example the string ``"false"``):
      use the defaults;
    - a dict: normalize ``platforms``/``sources`` and read the two
      suppression toggles, each falling back to its default.

    Never raises; always returns a fully populated
    :class:`_VoiceReplyConfig`.
    """
    if raw is False:
        return _VOICE_REPLY_DISABLED
    if isinstance(raw, dict):
        if raw.get("enabled") is False:
            return _VOICE_REPLY_DISABLED
        return _VoiceReplyConfig(
            platforms=_normalize_voice_reply_set(
                raw.get("platforms"), _DEFAULT_VOICE_REPLY_PLATFORMS
            ),
            sources=_normalize_voice_reply_set(
                raw.get("sources"), _DEFAULT_VOICE_REPLY_SOURCES
            ),
            suppress_voice_prefix=_voice_reply_flag(
                raw.get("suppress_voice_prefix"),
                _VOICE_REPLY_DEFAULTS.suppress_voice_prefix,
            ),
            suppress_reply_prompt=_voice_reply_flag(
                raw.get("suppress_reply_prompt"),
                _VOICE_REPLY_DEFAULTS.suppress_reply_prompt,
            ),
        )
    return _VOICE_REPLY_DEFAULTS


def _detect_format(file_path: str) -> str:
    """Detect the audio format from the file extension.

    Maps to Inworld's encoding types: LINEAR16, MP3, OGG_OPUS, FLAC.
    Returns AUTO_DETECT for unknown extensions (API handles it).
    """
    ext = Path(file_path).suffix.lower()
    return _EXT_TO_ENCODING.get(ext, "AUTO_DETECT")


def _ensure_supported_format(
    file_path: str,
    ffmpeg_timeout: int = _FFMPEG_CONVERSION_TIMEOUT_SECONDS,
) -> str:
    """Convert audio to a format Inworld's API supports, if needed.

    Inworld's sync API supports: LINEAR16 (WAV/PCM), MP3, OGG_OPUS, FLAC.
    WEBM (used by the Hermes desktop app) is NOT supported and causes a
    500 proxy error. We convert unsupported formats to OGG_OPUS via ffmpeg.

    Returns the path to a supported-format file (may be the original).
    """
    ext = Path(file_path).suffix.lower()
    if ext in _SUPPORTED_EXTS:
        return file_path

    # Convert to OGG_OPUS — smallest size, works reliably.
    # Reserve the path via NamedTemporaryFile(delete=False) so ffmpeg can
    # write to it; close the handle immediately because ffmpeg re-opens
    # the file by path. The path is returned so the caller can clean it up.

    out_handle = tempfile.NamedTemporaryFile(suffix=".ogg", delete=False)
    out_path: str = out_handle.name
    out_handle.close()
    try:
        proc = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                file_path,
                "-c:a",
                "libopus",
                "-b:a",
                "24k",
                "-ar",
                "48000",
                "-ac",
                "1",
                out_path,
            ],
            capture_output=True,
            text=True,
            timeout=ffmpeg_timeout,
        )
        if proc.returncode == 0 and Path(out_path).stat().st_size > 0:
            logger.info(
                "Converted %s (%s) -> %s (OGG_OPUS, %d bytes)",
                file_path,
                ext,
                out_path,
                Path(out_path).stat().st_size,
            )
            return out_path
        else:
            logger.warning(
                "ffmpeg conversion failed for %s (returncode=%d, stderr=%s)",
                file_path,
                proc.returncode,
                proc.stderr[:200],
            )
            return file_path  # fall back to original
    except Exception as exc:
        logger.warning("ffmpeg conversion error for %s: %s", file_path, exc)
        return file_path


def _probe_with_ffprobe(
    file_path: str,
    initial_probe: _ProbeResult,
    ffprobe_timeout: int = _FFPROBE_TIMEOUT_SECONDS,
) -> _ProbeResult | None:
    """Probe via ffprobe (handles all formats).

    Returns the (mutated) probe dict when ffprobe ran successfully, even
    if the file contained no audio stream — only an ``ffprobe`` failure
    (nonzero returncode or caught exception) allows fall-through to the
    WAV header parser.
    """
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_streams",
                file_path,
            ],
            capture_output=True,
            text=True,
            timeout=ffprobe_timeout,
        )
        if proc.returncode == 0:
            info = json.loads(proc.stdout)
            streams = info.get("streams", [])
            for s in streams:
                if s.get("codec_type") == "audio":
                    sample_rate = s.get("sample_rate")
                    initial_probe["sample_rate"] = (
                        int(sample_rate) if sample_rate else None
                    )
                    initial_probe["channels"] = s.get("channels")
                    # Map codec to Inworld encoding
                    codec = (s.get("codec_name") or "").lower()
                    if codec in ("pcm_s16le", "pcm_s16be", "pcm_f32le", "pcm_u8"):
                        initial_probe["encoding"] = "LINEAR16"
                    elif codec in ("mp3", "libmp3lame"):
                        initial_probe["encoding"] = "MP3"
                    elif codec == "opus":
                        initial_probe["encoding"] = "OGG_OPUS"
                    elif codec == "flac":
                        initial_probe["encoding"] = "FLAC"
                    break
            return initial_probe
    except FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError:
        pass
    return None


def _probe_wav_header(file_path: str, initial_probe: _ProbeResult) -> None:
    """Probe a WAV file by parsing its header directly."""
    try:
        with open(file_path, "rb") as f:
            header = f.read(44)
        if len(header) >= 44 and header[:4] == b"RIFF" and header[8:12] == b"WAVE":
            channels = struct.unpack_from("<H", header, 22)[0]
            sample_rate = struct.unpack_from("<I", header, 24)[0]
            initial_probe["channels"] = channels
            initial_probe["sample_rate"] = sample_rate
            initial_probe["encoding"] = "LINEAR16"
    except OSError, struct.error:
        pass


def _probe_audio(
    file_path: str,
    ffprobe_timeout: int = _FFPROBE_TIMEOUT_SECONDS,
) -> _ProbeResult:
    """Probe the audio file for sample rate, channels, and encoding.

    Uses ffprobe when available (most accurate, supports all formats).
    Falls back to WAV header parsing for .wav files.
    Returns a dict with sample_rate, channels, and format_hint.
    """
    result: _ProbeResult = {
        "sample_rate": None,
        "channels": None,
        "encoding": _detect_format(file_path),
    }

    # Try ffprobe first — a successful run returns immediately even
    # when the file contains no audio stream.
    ffprobe_result = _probe_with_ffprobe(file_path, result, ffprobe_timeout)
    if ffprobe_result is not None:
        return ffprobe_result

    # Fall back to WAV header parsing.
    ext = Path(file_path).suffix.lower()
    if ext == ".wav":
        _probe_wav_header(file_path, result)

    return result


# ── Provider ────────────────────────────────────────────────────────


class InworldSTTProvider(TranscriptionProvider):
    """Inworld STT provider with Voice Profile support."""

    @property
    def name(self) -> str:
        return "inworld-stt"

    @property
    def display_name(self) -> str:
        return "Inworld STT"

    def is_available(self) -> bool:
        return bool(os.environ.get("INWORLD_API_KEY"))

    def list_models(self) -> list[dict[str, Any]]:
        return cast(list[dict[str, Any]], MODELS)

    def default_model(self) -> str | None:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> dict[str, Any]:
        return {
            "name": self.display_name,
            "badge": "Inworld AI",
            "tag": "STT with Voice Profile",
            "env_vars": [
                {
                    "key": "INWORLD_API_KEY",
                    "prompt": "Inworld API key (same as TTS)",
                    "url": "https://inworld.ai",
                },
            ],
        }

    _HERMES_CONFIG_CACHE: dict[str, Any] | None = None
    _VOICE_REPLY_CONFIG_CACHE: _VoiceReplyConfig | None = None

    def _get_provider_config(self) -> dict[str, Any]:
        """Read the per-provider config from Hermes config.yaml.

        Reads ``stt.inworld-stt`` section which can contain arbitrary
        provider-specific keys like ``prompts``, ``model``, ``language``.

        Caches the parsed section for the lifetime of this provider
        instance — only the top-level ``stt.provider`` selection is
        re-checked on every hook call via ``is_provider_active`` from
        ``_common``.
        """
        if self._HERMES_CONFIG_CACHE is not None:
            return self._HERMES_CONFIG_CACHE
        cfg = read_hermes_config()
        stt_cfg = cfg.get("stt", {}) if cfg else {}
        provider_cfg = stt_cfg.get("inworld-stt", {})
        if not isinstance(provider_cfg, dict):
            provider_cfg = {}
        self._HERMES_CONFIG_CACHE = provider_cfg
        return provider_cfg

    def _get_prompts_config(self) -> list[str]:
        """Return custom vocabulary prompts from config, if set.

        Expects YAML like::

            stt:
              inworld-stt:
                prompts:
                  - "Project Aurora"
                  - "Kane"

        Returns empty list if not configured or not a list.
        """
        cfg = self._get_provider_config()
        prompts = cfg.get("prompts", [])
        if isinstance(prompts, list):
            return [str(p) for p in prompts if p]
        return []

    def _get_voice_prefix_config(self) -> str:
        """Return the voice marker prefix, or empty string to disable.

        When set, the prefix is prepended to every transcribed message
        so the agent can tell the text came from a voice recording.

        Expects YAML like::

            stt:
              inworld-stt:
                voice_prefix: "🎤"

        Set to empty string or omit to disable.
        """
        cfg = self._get_provider_config()
        prefix = cfg.get("voice_prefix", "🎤")
        if not isinstance(prefix, str) or not prefix.strip():
            return ""
        return prefix.strip()

    def _voice_reply_config(self) -> _VoiceReplyConfig:
        """Return the resolved native-voice reply policy.

        On native voice surfaces the client owns voice output, so the
        plugin can omit both the ``🎤`` marker and the first-turn
        ``text_to_speech`` instruction. Expects YAML like::

            stt:
              inworld-stt:
                voice_reply:
                  platforms: ["desktop", "tui", "cli"]
                  sources: ["voice_mode"]
                  suppress_voice_prefix: true
                  suppress_reply_prompt: true

        Missing/``null``/``true`` use the defaults; ``false`` or
        ``{enabled: false}`` disables all handling. The parsed policy is
        memoized for the lifetime of this provider instance so chunked
        transcription does not re-normalize the lists per segment.
        """
        cached = self._VOICE_REPLY_CONFIG_CACHE
        if cached is not None:
            return cached
        cfg = self._get_provider_config()
        voice_reply = _parse_voice_reply_config(cfg.get("voice_reply"))
        self._VOICE_REPLY_CONFIG_CACHE = voice_reply
        if voice_reply.suppress_voice_prefix and not voice_reply.sources:
            logger.warning(
                "voice_reply.suppress_voice_prefix is enabled but "
                "voice_reply.sources is empty — no transcript will omit "
                "the voice prefix."
            )
        if voice_reply.suppress_reply_prompt and not voice_reply.platforms:
            logger.warning(
                "voice_reply.suppress_reply_prompt is enabled but "
                "voice_reply.platforms is empty — no platform will skip "
                "the text_to_speech instruction."
            )
        return voice_reply

    def _get_voice_profile_config(self) -> bool:
        """Return whether Voice Profile analysis is enabled.

        When True, emotion, pitch, and vocal style are appended as a
        suffix tag on the transcript for any model that returns a
        ``voiceProfile`` block at the top level of its response.

        Expects YAML::

            stt:
              inworld-stt:
                voice_profile: true
        """
        cfg = self._get_provider_config()
        return bool(cfg.get("voice_profile", False))

    def _voice_profile_skip_keys_config(self) -> frozenset[str]:
        """Return the set of Voice Profile keys stripped from the suffix.

        Configurable via ``stt.inworld-stt.voice_profile_skip_keys`` (a
        list of strings). Falls back to ``_VOICE_PROFILE_SKIP_KEYS``
        (``{"age", "gender", "accent"}``) when unset or non-list.
        """
        cfg = self._get_provider_config()
        raw = cfg.get("voice_profile_skip_keys")
        if not isinstance(raw, list):
            return _VOICE_PROFILE_SKIP_KEYS
        keys = {str(k).strip() for k in raw if str(k).strip()}
        return frozenset(keys) if keys else _VOICE_PROFILE_SKIP_KEYS

    def _voice_profile_min_confidence_config(self) -> float:
        """Return the per-label confidence floor for the suffix.

        Configurable via ``stt.inworld-stt.voice_profile_min_confidence``
        (a number in ``[0.0, 1.0]``). Falls back to
        ``_VOICE_PROFILE_MIN_CONFIDENCE`` (``0.20``) when unset,
        non-numeric, or out of range. Labels whose confidence falls
        below this threshold are dropped from the suffix so the model
        is not distracted by weak detections.
        """
        cfg = self._get_provider_config()
        raw = cfg.get("voice_profile_min_confidence")
        if raw is None:
            return _VOICE_PROFILE_MIN_CONFIDENCE
        try:
            value = float(raw)
        except TypeError, ValueError:
            return _VOICE_PROFILE_MIN_CONFIDENCE
        if not 0.0 <= value <= 1.0:
            return _VOICE_PROFILE_MIN_CONFIDENCE
        return value

    def _voice_profile_glossary(self) -> str:
        """Build the Voice Profile label glossary for the hook prompt.

        Iterates ``_VOICE_PROFILE_LABELS`` (sourced from
        https://docs.inworld.ai/stt/voice-profiles) and skips any
        category listed in ``voice_profile_skip_keys`` so the LLM never
        sees guidance for labels the user has opted out of observing.

        Returns an empty string when every category is skipped so the
        caller can decide whether to emit any voice-profile guidance at
        all.
        """
        skip = self._voice_profile_skip_keys_config()
        sections: list[str] = []
        for category, labels in _VOICE_PROFILE_LABELS.items():
            if category in skip:
                continue
            if not labels:
                continue
            lines = [f"  - `{label}` — {desc}" for label, desc in labels.items()]
            sections.append(f"- **{category}**:\n" + "\n".join(lines))
        if not sections:
            return ""
        return (
            "Each label below corresponds to a possible value the API "
            "may emit in the Voice Profile suffix; use this glossary to "
            "interpret what each detection means.\n\n" + "\n\n".join(sections)
        )

    def _http_timeout_config(self) -> int:
        """HTTP request timeout (seconds) for the synchronous STT endpoint.

        Falls back to ``_STT_HTTP_TIMEOUT_SECONDS`` when unset or invalid.
        """
        cfg = self._get_provider_config()
        timeouts = cfg.get("timeouts")
        if isinstance(timeouts, dict):
            raw = timeouts.get("http", _STT_HTTP_TIMEOUT_SECONDS)
        else:
            raw = cfg.get("timeout", _STT_HTTP_TIMEOUT_SECONDS)
        if raw is None:
            return _STT_HTTP_TIMEOUT_SECONDS
        try:
            value = int(raw)
        except TypeError, ValueError:
            return _STT_HTTP_TIMEOUT_SECONDS
        if value < 1:
            return 1
        return value

    def _ffmpeg_timeout_config(self) -> int:
        """ffmpeg conversion timeout (seconds) for unsupported formats."""
        cfg = self._get_provider_config()
        timeouts = cfg.get("timeouts")
        raw = (
            timeouts.get("ffmpeg", _FFMPEG_CONVERSION_TIMEOUT_SECONDS)
            if isinstance(timeouts, dict)
            else cfg.get("ffmpeg_timeout", _FFMPEG_CONVERSION_TIMEOUT_SECONDS)
        )
        if raw is None:
            return _FFMPEG_CONVERSION_TIMEOUT_SECONDS
        try:
            value = int(raw)
        except TypeError, ValueError:
            return _FFMPEG_CONVERSION_TIMEOUT_SECONDS
        if value < 1:
            return 1
        return value

    def _ffprobe_timeout_config(self) -> int:
        """ffprobe inspection timeout (seconds) for unknown audio files."""
        cfg = self._get_provider_config()
        timeouts = cfg.get("timeouts")
        raw = (
            timeouts.get("ffprobe", _FFPROBE_TIMEOUT_SECONDS)
            if isinstance(timeouts, dict)
            else cfg.get("ffprobe_timeout", _FFPROBE_TIMEOUT_SECONDS)
        )
        if raw is None:
            return _FFPROBE_TIMEOUT_SECONDS
        try:
            value = int(raw)
        except TypeError, ValueError:
            return _FFPROBE_TIMEOUT_SECONDS
        if value < 1:
            return 1
        return value

    def _retry_policy_config(self) -> dict[str, Any]:
        """Return the resolved retry policy for STT HTTP requests.

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
                except TypeError, ValueError:
                    policy[key] = default
            else:
                try:
                    policy[key] = max(0.0, float(raw))
                except TypeError, ValueError:
                    policy[key] = default
        return policy

    def _format_voice_profile_suffix(
        self,
        voice_profile: dict[str, Any],
    ) -> str:
        """Format voice profile data into a short suffix string.

        Includes everything from the API except:

        - categories listed in ``stt.inworld-stt.voice_profile_skip_keys``
          (defaults to ``{"age", "gender", "accent"}``);
        - per-label entries whose confidence falls below
          ``stt.inworld-stt.voice_profile_min_confidence`` (defaults to
          ``0.20``);
        - the literal ``unclear`` label.

        Returns something like::

            [emotion -> calm:0.85, sad:0.10 | vocalStyle -> normal:0.92]
        """
        # API may return snake_case keys (voice_profile, vocal_style)
        # Normalize to camelCase for consistent skip/display handling.
        _VP_KEY_ALIASES = {"vocal_style": "vocalStyle"}
        skip_keys = self._voice_profile_skip_keys_config()
        min_confidence = self._voice_profile_min_confidence_config()
        parts: list[str] = []
        for raw_key, entries in voice_profile.items():
            normalized = _VP_KEY_ALIASES.get(raw_key, raw_key)
            if normalized in skip_keys:
                continue
            if not isinstance(entries, list) or not entries:
                continue
            labels = [
                f"{e['label']}:{e['confidence']:.2f}"
                for e in entries
                if e.get("label")
                and e.get("label") != "unclear"
                and float(e.get("confidence") or 0.0) >= min_confidence
            ]
            if not labels:
                continue
            label_str = ", ".join(labels)
            parts.append(f"{normalized} -> {label_str}")
        if parts:
            return "[" + " | ".join(parts) + "]"
        return ""

    def transcribe(
        self,
        file_path: str,
        *,
        model: str | None = None,
        language: str | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        """Transcribe audio via the Inworld STT API (sync).

        Returns the standard envelope:

        .. code-block:: python

            {
                "success": True,
                "transcript": "the transcribed text",
                "provider": "inworld-stt",
            }

        When ``voice_profile`` is enabled in config, any model whose
        response carries a top-level ``voiceProfile`` block has its
        emotion, pitch, and vocal style appended as a suffix on the
        transcript on success.
        """
        model_id = model or DEFAULT_MODEL
        lang = language or DEFAULT_LANGUAGE

        # Native voice surfaces already present the turn as voice, so the
        # marker is omitted for their STT sources. Computed once per call:
        # chunked transcription dispatches every segment through this
        # method with the same context value.
        voice_reply = self._voice_reply_config()
        source = _current_transcription_source.get()
        suppress_prefix = voice_reply.omits_prefix_for(source)
        if suppress_prefix:
            logger.debug("Omitting voice prefix for native voice source '%s'", source)

        # Validate the audio file exists
        audio_path = Path(file_path)
        if not audio_path.is_file():
            return self._failure_result(f"Audio file not found: {file_path}")

        # Convert unsupported formats (WEBM, etc.) to OGG_OPUS.
        # _ensure_supported_format returns the original path when the file
        # already has a supported extension, when ffmpeg is unavailable, or
        # when conversion fails — in those cases there is nothing to clean up.
        converted_path = _ensure_supported_format(
            file_path, ffmpeg_timeout=self._ffmpeg_timeout_config()
        )
        needs_cleanup = converted_path != file_path
        audio_path = Path(converted_path)

        # Read and base64-encode the audio
        try:
            audio_bytes = audio_path.read_bytes()
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        except Exception as e:
            if needs_cleanup:
                try:
                    Path(converted_path).unlink(missing_ok=True)
                except OSError:
                    pass
            return self._failure_result(f"Failed to read audio file: {e}")

        # ── Cleanup ──────────────────────────────────────────────────
        # Remove the converted-audio temp file (when one was produced)
        # after the API call returns, regardless of success/failure.
        # The finally block runs only when a converted file was produced;
        # otherwise we leave the original input untouched.
        try:
            # Probe audio file for actual sample rate, channels, and encoding
            probe = _probe_audio(
                file_path, ffprobe_timeout=self._ffprobe_timeout_config()
            )
            audio_encoding = probe.get("encoding", "AUTO_DETECT")
            sample_rate = probe.get("sample_rate")
            channels = probe.get("channels")

            payload = self._build_transcription_payload(
                model_id, lang, audio_b64, probe
            )

            logger.info(
                "Inworld STT request: model=%s encoding=%s size=%d bytes lang=%s sr=%s ch=%s",
                model_id,
                audio_encoding,
                len(audio_bytes),
                lang or "auto",
                sample_rate or "auto",
                channels or "auto",
            )

            retry_policy = self._retry_policy_config()
            http_timeout = self._http_timeout_config()

            try:
                resp, fell_back_to_auto_detect = self._open_transcription_with_retries(
                    payload,
                    retry_policy,
                    http_timeout,
                    attempt_zero_is_500_retry=True,
                )
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace")
                logger.warning(
                    "Inworld STT API returned HTTP %d (after retries): %s",
                    e.code,
                    body,
                )
                return {
                    "success": False,
                    "transcript": "",
                    "error": f"Inworld STT API returned HTTP {e.code}: {body}",
                    "provider": self.name,
                }
            except urllib.error.URLError as e:
                return self._failure_result(
                    f"Could not reach Inworld STT API: {e.reason}"
                )
            except Exception as e:
                return self._failure_result(f"Unexpected error during STT request: {e}")

            # Parse the response
            try:
                data: dict[str, Any] = json.loads(resp.read())
            except json.JSONDecodeError as e:
                return self._failure_result(f"Invalid JSON from Inworld STT API: {e}")

            # When the retry path produced a 500-then-AUTO_DETECT attempt,
            # the Voice Profile suffix is suppressed to match the
            # pre-existing STT behavior (the AUTO_DETECT fallback exists
            # for codec/server edge cases where Voice Profile is
            # unreliable).
            return self._build_transcription_result(
                data,
                include_voice_profile=not fell_back_to_auto_detect,
                empty_error="Empty transcript returned by API",
                include_voice_prefix=not suppress_prefix,
            )
        finally:
            if needs_cleanup:
                try:
                    Path(converted_path).unlink(missing_ok=True)
                except OSError:
                    pass

    def _failure_result(self, error: str) -> dict[str, Any]:
        """Build the standard STT failure envelope.

        Key insertion order matches the equivalent literals used by the
        early-return paths in ``transcribe`` and the HTTP/URL retry
        failures. Primary and retry *success* paths must not be routed
        through this helper — their wording and ordering differ.
        """
        return {
            "success": False,
            "transcript": "",
            "error": error,
            "provider": self.name,
        }

    def _build_transcription_payload(
        self,
        model_id: str,
        language: str,
        audio_b64: str,
        probe: _ProbeResult,
    ) -> dict[str, Any]:
        """Build the request payload for the synchronous STT endpoint.

        Performs config lookup for prompts and Voice Profile (still
        cached via ``_get_provider_config``) and returns the full payload
        with ``transcribeConfig`` and ``audioData``. Voice Profile is
        enabled for any model when ``voice_profile`` is set in config.
        """
        audio_encoding = probe.get("encoding", "AUTO_DETECT")
        sample_rate = probe.get("sample_rate")
        channels = probe.get("channels")

        # Build the request — only send language if it's a real code (not "auto")
        transcribe_config: dict[str, Any] = {
            "modelId": model_id,
            "audioEncoding": audio_encoding,
        }

        # Only set sample rate / channels for raw LINEAR16 where the API can't
        # infer from headers. For container formats (MP3, OGG, FLAC) the API
        # reads them from the file headers automatically.
        if audio_encoding == "LINEAR16" and sample_rate is not None:
            transcribe_config["sampleRateHertz"] = sample_rate
        if audio_encoding == "LINEAR16" and channels is not None:
            transcribe_config["numberOfChannels"] = channels
        if language and language.lower() != "auto":
            transcribe_config["language"] = language

        # Include custom vocabulary prompts from config, if configured
        # Configure via: stt.inworld-stt.prompts in config.yaml
        prompts = self._get_prompts_config()
        if prompts:
            transcribe_config["prompts"] = prompts

        # Voice Profile — any model when enabled. The documented Inworld
        # v1/v2 schema only exposes ``enableVoiceProfile`` and ``topN``;
        # the legacy ``inworldConfig.voiceProfileThreshold`` block is
        # intentionally omitted.
        if self._get_voice_profile_config() and model_id:
            transcribe_config["voiceProfileConfig"] = {
                "enableVoiceProfile": True,
                "topN": 3,
            }
        return {
            "transcribeConfig": transcribe_config,
            "audioData": {
                "content": audio_b64,
            },
        }

    def _open_transcription_request(
        self,
        payload: dict[str, Any],
        timeout: int = _STT_HTTP_TIMEOUT_SECONDS,
    ):
        """Open the synchronous STT endpoint with the given payload.

        Calls ``api_key()`` on every invocation to preserve the previous
        retry lookup behavior. No retries/backoff — those live in
        ``_open_transcription_with_retries`` (a separate method so the
        per-attempt encoding fallback can stay explicit in ``transcribe``).
        """
        req = urllib.request.Request(
            INWORLD_STT_API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Basic {api_key()}",
                "Content-Type": "application/json",
            },
        )
        return urllib.request.urlopen(req, timeout=timeout)

    def _open_transcription_with_retries(
        self,
        payload: dict[str, Any],
        policy: dict[str, Any],
        timeout: int,
        attempt_zero_is_500_retry: bool = False,
    ) -> tuple[Any, bool]:
        """Open the STT endpoint, retrying transient errors per policy.

        Returns ``(response, fell_back_to_auto_detect)``. The second
        element is True when the very first 500 was answered by a
        single AUTO_DETECT retry (the pre-existing STT fallback for
        codec edge cases). Callers use that flag to suppress the Voice
        Profile suffix — the AUTO_DETECT path is otherwise silent for
        Voice Profile.

        ``attempt_zero_is_500_retry`` enables that first 500 → AUTO_DETECT
        rewrite. After that one transformed attempt, retries follow the
        configured policy.
        """
        last_error: Exception | None = None
        last_was_500 = False
        fell_back_to_auto_detect = False
        for attempt in range(policy["max_retries"] + 1):
            transform_now = (
                attempt_zero_is_500_retry
                and last_was_500
                and not fell_back_to_auto_detect
            )
            if transform_now:
                attempt_payload: dict[str, Any] = {
                    "transcribeConfig": dict(payload["transcribeConfig"]),
                    "audioData": dict(payload["audioData"]),
                }
                attempt_payload["transcribeConfig"]["audioEncoding"] = "AUTO_DETECT"
                existing_lang = payload["transcribeConfig"].get("language")
                if existing_lang and str(existing_lang).lower() != "auto":
                    attempt_payload["transcribeConfig"]["language"] = existing_lang
                fell_back_to_auto_detect = True
            else:
                attempt_payload = payload
            try:
                resp = self._open_transcription_request(
                    attempt_payload, timeout=timeout
                )
                return resp, fell_back_to_auto_detect
            except urllib.error.HTTPError as e:
                last_error = e
                last_was_500 = e.code == 500
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
                    "Inworld STT HTTP %d — retry %d/%d in %.1fs",
                    e.code,
                    attempt + 1,
                    policy["max_retries"],
                    backoff,
                )
                time.sleep(backoff)
            except urllib.error.URLError as e:
                last_error = e
                last_was_500 = False
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
                    "Inworld STT network error (%s) — retry %d/%d in %.1fs",
                    e.reason,
                    attempt + 1,
                    policy["max_retries"],
                    backoff,
                )
                time.sleep(backoff)
        if last_error is not None:
            raise last_error
        raise RuntimeError("Inworld STT retry loop exited unexpectedly")

    def _build_transcription_result(
        self,
        data: dict[str, Any],
        include_voice_profile: bool,
        empty_error: str,
        include_voice_prefix: bool = True,
    ) -> dict[str, Any]:
        """Build the success-side envelope with optional transcript decorations.

        ``include_voice_prefix`` prepends the configured marker; callers
        pass ``False`` for native voice sources whose client already
        presents the turn as voice. ``include_voice_profile`` appends the
        Voice Profile suffix; the retry path passes ``False`` after the
        AUTO_DETECT fallback to deliberately omit it. The two flags are
        independent.

        ``voiceProfile`` lives at the top level of the documented
        Inworld response (sibling of ``transcription`` and ``usage``).
        """
        transcription = data.get("transcription", {})
        transcript: str = transcription.get("transcript", "")

        # Prepend voice marker if configured and not suppressed
        voice_prefix = self._get_voice_prefix_config() if include_voice_prefix else ""
        if voice_prefix and transcript:
            transcript = f"{voice_prefix} {transcript}"

        if include_voice_profile and transcript:
            transcription_data = data.get("transcription", {})
            voice_profile = transcription_data.get("voiceProfile") or data.get(
                "voiceProfile"
            )
            if voice_profile:
                suffix = self._format_voice_profile_suffix(voice_profile)
                if suffix:
                    transcript = f"{transcript} {suffix}"

        if not transcript:
            return self._failure_result(empty_error)
        return {
            "success": True,
            "transcript": transcript,
            "provider": self.name,
        }

    # ── Hooks ────────────────────────────────────────────────────

    def register_hooks(self, ctx):
        """Register lifecycle hooks for voice-reply handling.

        ``pre_llm_call`` injects first-turn guidance (the marker and the
        text-plus-``text_to_speech`` instruction plus the optional Voice
        Profile glossary). ``pre_transcription`` records the STT source so
        ``transcribe`` can omit the marker on native voice surfaces.

        No tool-interception hook is registered: ``text_to_speech``
        remains available on every platform so explicit audio requests
        keep working.
        """
        ctx.register_hook("pre_llm_call", self._on_pre_llm_call)
        ctx.register_hook("pre_transcription", self._on_pre_transcription)

    def _on_pre_transcription(
        self,
        file_path: str,
        provider: str,
        source: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Record the STT source for the dispatch already in flight.

        The host fires this hook synchronously on the caller thread
        before the provider runs, so the value is visible to
        ``transcribe()`` for the original file, any re-encoded file, and
        every segment dispatched by ``transcribe_oversized()``. It is
        deliberately not cleared by ``transcribe()``; the next
        invocation overwrites it.

        The source is recorded only when ``provider`` is this provider.
        A dispatch for another provider clears any value left in the
        same context. Non-string or blank sources also clear it. The
        callback never raises and returns ``None`` so the STT prompt,
        language, and model stay untouched.
        """
        try:
            normalized: str | None = None
            if (
                isinstance(provider, str)
                and provider.strip().lower() == self.name
                and isinstance(source, str)
                and source.strip()
            ):
                normalized = source.strip().lower()
            _current_transcription_source.set(normalized)
        except Exception:
            logger.debug("pre_transcription hook failed", exc_info=True)
            _current_transcription_source.set(None)
        return None

    def _on_pre_llm_call(
        self, is_first_turn: bool, **kwargs: Any
    ) -> dict[str, Any] | None:
        """Inject first-turn voice guidance for the active provider.

        Skipped entirely when ``inworld-stt`` is not the configured STT
        provider — both the voice marker and Voice Profile suffix are
        specific to this provider's output format and would mislead the
        LLM when a different transcription provider is selected. The
        ``is_provider_active`` helper re-reads the YAML config each
        invocation so a gateway-time change of ``stt.provider``
        propagates immediately.

        Native voice platforms (``voice_reply.platforms``) omit the
        ``text_to_speech`` instruction because their client speaks the
        reply itself. Gateway platforms keep it so voice notes keep
        replying in voice. The Voice Profile glossary is injected
        whenever ``voice_profile`` is enabled.
        """
        try:
            if not is_first_turn:
                return None
            if not is_provider_active("stt", "inworld-stt"):
                return None

            prefix = self._get_voice_prefix_config()
            vp_enabled = self._get_voice_profile_config()

            if not prefix and not vp_enabled:
                return None

            voice_reply = self._voice_reply_config()
            platform = str(kwargs.get("platform") or "").strip().lower()
            teach_marker = bool(prefix) and not voice_reply.skips_reply_prompt_for(
                platform
            )

            parts: list[str] = []

            if teach_marker:
                parts.append(
                    f"When a user message starts with **{prefix}** it means they "
                    "sent a voice recording that was transcribed to text. "
                    "Reply with both text and speech: write the complete "
                    "response as visible text, then call `text_to_speech` to "
                    "speak that same response. Keep the two versions "
                    "essentially identical — compose the text you intend to "
                    "speak rather than a summary of it, so what is displayed "
                    "matches what is heard. When the response includes "
                    "visual content (tables, code, charts, images), have the "
                    "speech describe the same content instead of adding a "
                    "different answer."
                )

            if vp_enabled:
                vp_intro = (
                    "A suffix in square brackets like "
                    "`[emotion -> calm:0.85 | vocalStyle -> normal:0.92]` "
                    "describes the speaker's detected voice characteristics "
                    "with confidence scores (0.00-1.00).\n"
                    "**Interpreting confidence scores** — each label is "
                    "paired with a score between 0.00 and 1.00. Use these "
                    "thresholds when deciding how strongly to weight a "
                    "detection.\n"
                    "A very low score might be a false positive; in such cases, "
                    "always match the detected characteristic with the "
                    "content of the transcription to verify."
                )
                glossary = self._voice_profile_glossary()
                parts.append(f"{vp_intro}\n\n{glossary}" if glossary else vp_intro)

            if not parts:
                return None

            # Only label the context as marker guidance when the marker
            # instruction is actually present; a glossary-only context
            # uses its own heading.
            heading = (
                "## Voice Transcription Markers" if teach_marker else "## Voice Profile"
            )
            return {"context": heading + "\n\n" + "\n\n".join(parts)}
        except Exception:
            logger.debug("pre_llm_call hook failed", exc_info=True)
            return None
