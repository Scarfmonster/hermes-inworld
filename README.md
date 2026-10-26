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
    streaming_extra_body: {}  # native reasoning-disable fields for streamed rewrites
    delivery_mode: "CREATIVE" # STABLE, BALANCED, or CREATIVE
    timeout: 60               # seconds
    retry:                    # TTS HTTP requests and the steering rewrite
      max_retries: 3
      initial_backoff: 1.0
      backoff_multiplier: 2.0
      max_backoff: 30.0

stt:
  provider: inworld-stt
  inworld-stt:
    voice_prefix: "🎤"        # set "" to disable
    voice_reply:              # native voice surfaces own reply playback
      platforms: ["desktop", "tui", "cli"]
      sources: ["voice_mode"]
      suppress_voice_prefix: true
      suppress_reply_prompt: true
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

- `reasoning_effort` is applied by Hermes to the hidden sync rewrite call; the
  plugin never overrides it. Unset or `""` (the default) sends no reasoning
  setting, so the provider or model default applies — non-thinking on models
  that require an explicit enable. `none` disables thinking; enabled levels
  are `minimal`, `low`, `medium`, `high`, `xhigh`, `max`, and `ultra`, and
  routes clamp levels they do not support.
- Streamed rewrites request thinking off. Explicit vendor reasoning keys in
  `extra_body` remain user-authoritative, and Hermes may lift the disable to
  its reasoning floor on routes that cannot disable reasoning.
- If a custom/OpenAI-compatible route (for example LiteLLM) ignores Hermes'
  `reasoning_effort: "none"` disable, add the provider's native off switch to
  `tts.inworld-tts.streaming_extra_body`. It is merged only into streamed
  rewrites, so the sync rewrite keeps following `reasoning_effort`:
  ```yaml
  tts:
    inworld-tts:
      streaming_extra_body:
        thinking:
          type: disabled   # DeepSeek V4 through LiteLLM
  ```
  Use the key the route actually understands; unknown keys can make the
  rewrite fail, in which case the original text is spoken.
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
- `streaming_extra_body` is passed to Hermes as `extra_body` on the streaming
  rewrite only. It does not affect sync synthesis or non-streaming calls.

Language can be forced with a prefix: `{polish} Witaj!` or `{pl} Witaj!`.
Supported languages: arabic, chinese, dutch, english, french, german, hebrew,
hindi, italian, japanese, korean, polish, portuguese, russian, and spanish.

Voice output defaults to MP3. Other formats: `wav`, `ogg_opus`, `flac`, and
`linear16`; Opus requires `ffmpeg`. `speed` maps to Inworld's speaking rate
(0.5-1.5).

Hermes's streaming voice surfaces use the plugin automatically when
`inworld-tts` is the provider, playing 24 kHz PCM16LE sentence by sentence
before the reply finishes. Each chunk is a separate request capped at 4,000
UTF-16 code units, so steering is limited to one short delivery direction per
chunk with no `[reset]`; flash remains direction-free. `tts.streaming.provider`
can pin a different streamer; under `auto`, built-in streamers are tried first.

## Speech-to-text

Two models are supported: `inworld/inworld-stt-1` (default) and
`groq/whisper-large-v3`.

`voice_prefix` (default `🎤`) is prepended to transcripts, except on native
voice surfaces, where the marker is omitted by default. On gateway platforms
the first-turn guidance asks the model to reply with both visible text and
the same content spoken through `text_to_speech`, so the written and spoken
versions stay aligned. With
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

The TTS HTTP call, the optional TTS steering rewrite, and STT HTTP calls retry
5xx, 429, and network errors with the same exponential backoff and
`Retry-After` handling. Defaults are 3 retries, starting at 1 second, doubling
up to 30 seconds. `retry_on_5xx`, `retry_on_429`, and
`retry_on_network_error` are all enabled. TTS times out after 60 seconds; STT
uses 120 seconds for HTTP and ffmpeg, and 10 seconds for ffprobe. Set
`retry.max_retries: 0` to disable retries. Streaming retries only the initial
HTTP open; once audio has been yielded, a mid-stream failure raises without a
retry.

The steering rewrite is an auxiliary LLM call, so it is also subject to
Hermes's `auxiliary.transient_retries` and provider fallback.
`tts.inworld-tts.retry.max_retries: 0` removes only the plugin's additional
rewrite attempts; Hermes still retries transient auxiliary failures. A failing
rewrite can therefore consume several auxiliary timeouts before the original
text is spoken. To bound that latency, lower `max_retries`, lower
`auxiliary.tts_audio_tags.timeout`, or disable `steering_rewrite`.
`auxiliary.transient_retries: 0` also disables Hermes-side transient retries,
but that setting is global to all auxiliary tasks. The plugin does not retry
non-429 4xx/auth errors or empty rewrite responses.

## Troubleshooting

- No audio and API errors: check that `INWORLD_API_KEY` is set.
- Opus/OGG output or WEBM transcription fails: install `ffmpeg`.
- Config changes have no effect: restart Hermes.
- Very long TTS replies fail: keep text under 2,000 UTF-16 code units for sync
  synthesis. Streaming handles up to 4,000 UTF-16 code units per sentence
  chunk; oversized chunks are truncated.
- Streaming audio is missing or falls back to whole-file synthesis: confirm
  Hermes supports plugin PCM streaming, that `tts.provider` is `inworld-tts`,
  and that `tts.streaming.provider` is not pinned to a different streamer.
