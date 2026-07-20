"""Hermes Agent voice plugin — Inworld TTS-2 + STT with Voice Profile.

Registers both :class:`.InworldTTSProvider` and :class:`.InworldSTTProvider`
via the Hermes plugin entrypoint::

    register(ctx) -> ctx.register_tts_provider(...); ctx.register_transcription_provider(...)
"""

from .stt import InworldSTTProvider
from .tts import InworldTTSProvider


def register(ctx):
    tts = InworldTTSProvider()
    stt = InworldSTTProvider()

    ctx.register_tts_provider(tts)
    ctx.register_transcription_provider(stt)

    tts.register_hooks(ctx)
    stt.register_hooks(ctx)
