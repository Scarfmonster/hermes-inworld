# hermes-inworld

Hermes Agent voice plugin — Inworld TTS-2 / TTS-2 Flash with manual or opt-in automatic steering on TTS-2, and Inworld Realtime STT with Voice Profile analysis.

## Features

### Text-to-Speech (TTS-2 / TTS-2 Flash)
- Manual steering tags: `[say excitedly]`, `[very slow]`, `[whisper]`, `[laugh]`,
  emphasis via `CAPITALISATION`, non-verbals, and combinations like
  `[say sadly with deliberate pauses in a low voice and hushed style]`.
  Instruction tags and capitalization emphasis apply to `inworld-tts-2` only;
  `inworld-tts-2-flash` ignores instruction tags (emphasis is unverified on
  Flash) but still supports non-verbal sounds.
- Two models: `inworld-tts-2` (flagship, full steering) and
  `inworld-tts-2-flash` (lowest latency and cost; instruction tags are ignored,
  non-verbals still work).
- Optional automatic steering rewrite: `tts.inworld-tts.steering_rewrite: true`
  runs the TTS input through a hidden auxiliary LLM pass
  (`auxiliary.tts_audio_tags`) to add tags before synthesis. Default `false`:
  text is synthesized as written.
- Rewrite prompts: `inworld-tts-2` gets the full prompt (instruction tags,
  non-verbals, emphasis, formatting); `inworld-tts-2-flash` gets the reduced
  prompt (non-verbals and formatting; no instruction tags or emphasis).
  Deprecated/unknown model IDs skip the rewrite and log one warning per model.
- Choose a voice via `tts.inworld-tts.voice` in config (defaults to "Sarah").
- Optional speaking rate (`speed`) forwarded by Hermes as
  `audioConfig.speakingRate`.
- Language override: prefix text with a language name or ISO code in braces,
  e.g. `{polish}` or `{pl}`. Supported: Arabic, Chinese, Dutch, English, French,
  German, Hebrew, Hindi, Italian, Japanese, Korean, Polish, Portuguese, Russian,
  Spanish.
- Audio formats: mp3, wav, ogg_opus, flac, and linear16. Opus/OGG is available
  for voice-messaging platforms when `ffmpeg` is on `PATH` (`voice_compatible`).
- Configurable `delivery_mode` (STABLE / BALANCED / CREATIVE), HTTP timeout, and
  a retry policy with exponential backoff for 5xx, 429, and network errors.

### Speech-to-Text (STT)
- Voice marker prefix (default `🎤`, disable with `""`) on transcriptions, plus
  a first-turn `pre_llm_call` prompt so the agent replies with voice as well.
- Voice Profile analysis (emotion, pitch, vocal style) when `voice_profile: true`;
  requires a Voice Profile-capable model (the default `inworld/inworld-stt-1`).
- Configurable Voice Profile skip keys (defaults to `age`, `gender`, `accent`).
- Configurable custom vocabulary prompts (e.g. project codenames).
- STT models: `inworld/inworld-stt-1` (default; Voice Profile support) and
  `groq/whisper-large-v3` (broad language coverage). The provider accepts the
  model forwarded by Hermes.
- Optional language hint forwarded by Hermes; unset or `auto` lets the API
  auto-detect.
- Auto-converts unsupported audio (e.g. WEBM from the desktop app) to OGG_OPUS
  via `ffmpeg`.
- Configurable HTTP, ffmpeg, and ffprobe timeouts, plus a retry policy with
  exponential backoff for 5xx and 429 responses and network errors. The
  pre-existing 500 → AUTO_DETECT encoding fallback is preserved.

### Cross-cutting
- When `inworld-stt` is the configured STT provider, the `pre_llm_call` hook
  injects voice-marker / Voice Profile context on the first turn only — no
  per-turn overhead. The former TTS prompt-injection hook has been removed;
  use `steering_rewrite` instead.

## Install

Drop this directory into `~/.hermes/plugins/hermes-inworld/`, then enable:

```bash
hermes plugins enable hermes-inworld
```

The plugin requires Python 3.13+. Config support uses `ruamel-yaml` (declared in
`pyproject.toml`); without it the providers fall back to config-less defaults.
`ffmpeg` is optional but enables Opus/OGG voice bubbles and STT conversion;
`ffprobe` improves STT audio probing.

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
    steering_rewrite: false   # optional; true = LLM-rewrite the TTS text
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

Provider config blocks are cached per provider instance. Restart Hermes/gateway
after changing `tts.inworld-tts` or `stt.inworld-stt` settings; the TTS `model`
key is re-read on each synthesis, so model changes apply without a restart.

The 500 → AUTO_DETECT encoding fallback in STT is preserved by default
(the first server-side 500 is followed by a single retry with
`audioEncoding: AUTO_DETECT`, with the Voice Profile suffix suppressed
on that path). Set `retry.max_retries: 0` to disable retries entirely.

### 3. TTS steering rewrite (optional)

`steering_rewrite` is off by default: normal text goes to TTS with no steering
tags and no TTS prompt injection.

Set `tts.inworld-tts.steering_rewrite: true` to have the TTS input rewritten
by the auxiliary LLM task `tts_audio_tags` before synthesis. The visible chat
text is unchanged; only the text sent to Inworld is rewritten.

- `inworld-tts-2`: full steering tags, non-verbals, emphasis, and formatting.
- `inworld-tts-2-flash`: reduced prompt (non-verbals and formatting; no
  instruction tags or emphasis).
- Deprecated/unknown model IDs: rewrite is skipped, the original text is
  synthesized, and one warning per model is logged.

If the auxiliary call fails or returns empty text, the original text is
synthesized.

Route the auxiliary task to a specific model if needed:

```yaml
auxiliary:
  tts_audio_tags:
    provider: auto
    model: ""
```

When enabled, the rewrite adds one auxiliary LLM call per synthesis; route it
to a fast/cheap model if latency or cost matters.

Because the provider config block is cached, changing `steering_rewrite`
requires a Hermes/gateway restart. When rewrite is enabled, the model is
resolved and checked on every synthesis, so a model change applies immediately.

The legacy first-turn TTS prompt injection has been removed. Set
`steering_rewrite: true` to restore automatic steering; there is no legacy
prompt-insertion mode.

Current limitation: there is no length guard, so a very long rewrite may exceed
Inworld's 2,000 UTF-16-unit synchronous request limit.

### 4. TTS model selection

Set the model alongside the voice under `tts.inworld-tts` in
`~/.hermes/config.yaml`. The Hermes TTS picker writes only `tts.provider`,
so add the `model` key manually.

- Use the exact lowercase IDs shown above. Model matching for the rewrite is
  case-insensitive, but the configured value is sent to the API unchanged.
- `tts.inworld-tts.model` takes precedence over a top-level `tts.model`
  value forwarded by Hermes.
- When the model is unset, `inworld-tts-2` is used.
- `inworld-tts-1`, `inworld-tts-1-max`, `inworld-tts-1.5-max`, and
  `inworld-tts-1.5-mini` are deprecated and no longer offered in the catalog.
  A configured legacy ID still synthesizes, but skips the steering rewrite and
  logs one warning per model when rewrite is enabled.
- Changing the model applies to the next synthesis. When rewrite is enabled,
  the prompt is selected per synthesis; there is no first-turn context to
  update.
