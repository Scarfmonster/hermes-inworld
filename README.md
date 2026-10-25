# hermes-inworld

Inworld voice plugin for Hermes Agent. Adds Inworld TTS-2 text-to-speech and
Inworld Realtime speech-to-text, with optional automatic steering tags and
Voice Profile annotations.

## Install

Requires Python 3.13+ and `ruamel-yaml`. `ffmpeg` is optional: it enables
Opus/OGG voice output and conversion of unsupported STT audio. `ffprobe`
improves audio probing for STT.

Copy this directory to `~/.hermes/plugins/hermes-inworld/` and enable it:

```bash
hermes plugins enable hermes-inworld
```

Restart Hermes after enabling.

## Configuration

Set your API key in `~/.hermes/.env` or the environment:

```dotenv
INWORLD_API_KEY=your_key_here
```

Add the providers to `~/.hermes/config.yaml`:

```yaml
tts:
  provider: inworld-tts
  inworld-tts:
    voice: "Sarah"            # default voice
    model: "inworld-tts-2"    # or inworld-tts-2-flash
    steering_rewrite: false   # add steering tags with an auxiliary LLM
    delivery_mode: "CREATIVE" # STABLE, BALANCED, or CREATIVE
    timeout: 60               # seconds
    retry:
      max_retries: 3
      initial_backoff: 1.0
      backoff_multiplier: 2.0
      max_backoff: 30.0

stt:
  provider: inworld-stt
  inworld-stt:
    voice_prefix: "🎤"        # set "" to disable
    voice_profile: false      # append voice characteristics to transcripts
    voice_profile_skip_keys: ["age", "gender", "accent"]
    voice_profile_min_confidence: 0.2
    prompts: ["Hermes"]       # optional vocabulary
    timeouts:
      http: 120
      ffmpeg: 120
      ffprobe: 10
```

The STT `retry` block accepts the same keys as TTS.

## Text-to-speech

`inworld-tts-2` is the default model and understands instruction tags:

```text
[say excitedly] Great news! [very slow] Let me explain. [reset] That's everything.
```

Non-verbal tags such as `[laugh]`, `[sigh]`, and `[clear throat]` work on both
models. `inworld-tts-2-flash` is cheaper and faster but ignores instruction
tags.

With `steering_rewrite: true`, each reply is passed through the
`auxiliary.tts_audio_tags` task, which adds tags automatically. The chat text
stays unchanged. Route the task to a different model with:

```yaml
auxiliary:
  tts_audio_tags:
    provider: auto
    model: ""
    timeout: 180                # seconds; raise if the rewrite call is slow
    reasoning_effort: "medium"  # optional; enables thinking for the rewrite
```

- `reasoning_effort` is applied by Hermes to the hidden rewrite call; the
  plugin never overrides it. Unset or `""` (the default) sends no reasoning
  setting, so the provider or model default applies — non-thinking on models
  that require an explicit enable. `none` disables thinking; enabled levels
  are `minimal`, `low`, `medium`, `high`, `xhigh`, `max`, and `ultra`, and
  routes clamp levels they do not support.
- Thinking can improve tag placement but adds latency and token cost on top of
  the auxiliary call every synthesis already makes. Raise `timeout` (default
  30 seconds) if the model needs longer; a timed-out rewrite falls back to the
  original text. A thinking response that returns the script only in a
  separate reasoning field is also treated as empty, so the original text is
  spoken.
- `tts_audio_tags` is shared with Hermes's Gemini and xAI audio-tag rewrites,
  so this setting affects those paths too.
- For provider-specific reasoning payloads, set
  `auxiliary.tts_audio_tags.extra_body.reasoning`; it takes precedence over
  `reasoning_effort`. Per-task reasoning requires a Hermes version that
  supports it; older versions ignore the key.

Language can be forced with a prefix: `{polish} Witaj!` or `{pl} Witaj!`.
Supported languages: arabic, chinese, dutch, english, french, german, hebrew,
hindi, italian, japanese, korean, polish, portuguese, russian, and spanish.

Voice output defaults to MP3. Other formats: `wav`, `ogg_opus`, `flac`, and
`linear16`; Opus requires `ffmpeg`. `speed` maps to Inworld's speaking rate
(0.5-1.5).

## Speech-to-text

Two models are supported: `inworld/inworld-stt-1` (default) and
`groq/whisper-large-v3`.

`voice_prefix` (default `🎤`) is prepended to every transcript. With
`voice_profile: true`, detected characteristics are appended:

```text
[emotion -> calm:0.85, sad:0.10 | vocalStyle -> normal:0.92]
```

Labels below `voice_profile_min_confidence` are dropped, as are keys listed in
`voice_profile_skip_keys` and the `unclear` label. `prompts` adds names or
terms that should be recognized more reliably.

Audio formats the API does not accept are converted to OGG_OPUS with
`ffmpeg`. If `ffmpeg` is missing, the original file is sent as-is.

## Retries and timeouts

Both providers retry 5xx, 429, and network errors with exponential backoff and
respect `Retry-After` on 429. Defaults are 3 retries, starting at 1 second,
doubling up to 30 seconds. `retry_on_5xx`, `retry_on_429`, and
`retry_on_network_error` are all enabled. TTS times out after 60 seconds; STT
uses 120 seconds for HTTP and ffmpeg, and 10 seconds for ffprobe. Set
`retry.max_retries: 0` to disable retries.

## Troubleshooting

- No audio and API errors: check that `INWORLD_API_KEY` is set.
- Opus/OGG output or WEBM transcription fails: install `ffmpeg`.
- Config changes have no effect: restart Hermes.
- Very long TTS replies fail: keep text under 2,000 UTF-16 code units.
