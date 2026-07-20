# hermes-inworld

Hermes Agent voice plugin — Inworld TTS-2 with expressive steering and Inworld Realtime STT with Voice Profile analysis.

## Features

### Text-to-Speech (TTS-2)
- Steering tags: `[say excitedly]`, `[very slow]`, `[whisper]`, `[laugh]`, emphasis via `CAPITALISATION`, non-verbals, and combinations like `[say sadly with deliberate pauses in a low voice and hushed style]`.
- Three models: `inworld-tts-2` (flagship, full steering), `inworld-tts-1.5-max`, `inworld-tts-1.5-mini`.
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
- `pre_llm_call` hooks inject TTS steering rules and STT voice-marker context on the first turn only — no per-turn overhead.

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
