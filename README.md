# hermes-inworld

Hermes Agent voice plugin — Inworld TTS-2 / TTS-2 Flash with expressive steering on TTS-2, and Inworld Realtime STT with Voice Profile analysis.

## Features

### Text-to-Speech (TTS-2 / TTS-2 Flash)
- Steering tags: `[say excitedly]`, `[very slow]`, `[whisper]`, `[laugh]`, emphasis via `CAPITALISATION`, non-verbals, and combinations like `[say sadly with deliberate pauses in a low voice and hushed style]`. Instruction tags and capitalization emphasis apply to `inworld-tts-2` only; `inworld-tts-2-flash` ignores instruction tags (emphasis is unverified on Flash) but still supports non-verbal sounds.
- Two models: `inworld-tts-2` (flagship, full steering) and `inworld-tts-2-flash` (lowest latency and cost; instruction tags are ignored, non-verbals still work).
- Adaptive first-turn context: full steering catalog for `inworld-tts-2`, non-verbals plus formatting for `inworld-tts-2-flash`, and no TTS context for deprecated/unknown model IDs (one-time warning per model).
- Choose a voice via `tts.inworld-tts.voice` in config (defaults to "Sarah").
- `{polish}`, `{japanese}`, ... language prefix to override auto-detected language.
- ffmpeg-aware: `voice_compatible` flag (Opus/OGG output) for voice-messaging platforms when `ffmpeg` is on `PATH`.
- Configurable `delivery_mode` (STABLE / BALANCED / CREATIVE), HTTP timeout, and a retry policy with exponential backoff for 5xx and 429 responses.

### Speech-to-Text (STT)
- Optional prefix marker prepended to transcribed messages with a system prompt so the agent replies with voice as well.
- Voice Profile analysis (emotion, pitch, vocal style) when `voice_profile: true` is set in config.
- Configurable Voice Profile skip keys (defaults to `age`, `gender`, `accent`).
- Configurable custom vocabulary prompts (e.g. project codenames).
- Auto-converts unsupported audio (e.g. WEBM from the desktop app) to OGG_OPUS via `ffmpeg`.
- Configurable HTTP, ffmpeg, and ffprobe timeouts, plus a retry policy with exponential backoff for 5xx and 429 responses and network errors. The pre-existing 500 → AUTO_DETECT encoding fallback is preserved.

### Cross-cutting
- `pre_llm_call` hooks inject model-adapted TTS guidance and STT voice-marker context on the first turn only — no per-turn overhead.

## Install

Drop this directory into `~/.hermes/plugins/hermes-inworld/`, then enable:

```bash
hermes plugins enable hermes-inworld
```

## Configure

### 1. API key (required)

In `~/.hermes/.env`:

```
INWORLD_API_KEY=your_key_here
```

### 2. Pick the providers in `~/.hermes/config.yaml`

```yaml
tts:
  provider: inworld-tts
  inworld-tts:
    voice: "Sarah"            # optional; defaults to "Sarah" when unset
    model: "inworld-tts-2"    # optional; inworld-tts-2 | inworld-tts-2-flash
    delivery_mode: "CREATIVE" # optional; STABLE | BALANCED | CREATIVE.
                              # Defaults to CREATIVE (matches the previous
                              # hardcoded payload value).
    timeout: 60               # optional; HTTP request timeout (seconds).
                              # Defaults to 60.
    retry:                    # optional; per-provider retry policy.
      max_retries: 3          # additional attempts after the first failure.
      initial_backoff: 1.0    # seconds; doubled by backoff_multiplier per
      backoff_multiplier: 2.0 # retry, capped at max_backoff.
      max_backoff: 30.0
      retry_on_5xx: true      # retry on server errors (500-599).
      retry_on_429: true      # retry on rate limit (honors Retry-After).
      retry_on_network_error: true # retry on URLError / DNS / etc.

stt:
  provider: inworld-stt
  inworld-stt:
    voice_prefix: "🎤"          # default: "🎤"; set "" to disable
    voice_profile: true         # default: false; append [emotion -> calm:0.85 | ...] suffix
    voice_profile_skip_keys:    # optional; override the keys stripped from
      - age                    # the transcript suffix. Defaults to
      - gender                 # ["age", "gender", "accent"].
      - accent
    voice_profile_min_confidence: 0.2  # optional; per-label confidence floor
                                     # (0.0-1.0). Labels below this
                                     # threshold are dropped from the
                                     # transcript suffix. Defaults to 0.2.
    prompts:                   # optional custom vocabulary
      - "Hermes"
    timeouts:                  # optional; per-stage timeouts (seconds).
      http: 120                # Synchronous STT request timeout.
      ffmpeg: 120              # Unsupported-format conversion (OGG_OPUS).
      ffprobe: 10              # Audio probe for sample rate / channels.
    retry:                     # optional; per-provider retry policy.
      max_retries: 3
      initial_backoff: 1.0
      backoff_multiplier: 2.0
      max_backoff: 30.0
      retry_on_5xx: true
      retry_on_429: true
      retry_on_network_error: true
```

The 500 → AUTO_DETECT encoding fallback in STT is preserved by default
(the first server-side 500 is followed by a single retry with
`audioEncoding: AUTO_DETECT`, with the Voice Profile suffix suppressed
on that path). Set `retry.max_retries: 0` to disable retries entirely.

### 3. TTS model selection

Set the model alongside the voice under `tts.inworld-tts` in
`~/.hermes/config.yaml`. The Hermes TTS picker writes only `tts.provider`,
so add the `model` key manually.

- Use the exact lowercase IDs shown above. Context selection is
  case-insensitive, but the configured value is sent to the API unchanged.
- `tts.inworld-tts.model` takes precedence over a top-level `tts.model`
  value forwarded by Hermes.
- When the model is unset, `inworld-tts-2` is used.
- `inworld-tts-1`, `inworld-tts-1-max`, `inworld-tts-1.5-max`, and
  `inworld-tts-1.5-mini` are deprecated and no longer offered in the catalog.
  A configured legacy ID still synthesizes, but gets no TTS hook context and
  logs one warning per model.
- Changing the model mid-session affects the next first turn; an
  already-injected context for the current session is not retracted.
