"""First-turn TTS hook context prompts for the Inworld TTS provider.

The full steering prompt is intentionally preserved byte-for-byte; the
Flash variant reuses the model-neutral sections and omits the steering
catalog because Flash ignores instruction tags.
"""

__all__ = ("TTS_FLASH_HOOK_CONTEXT", "TTS_HOOK_CONTEXT")

_TTS_HOOK_HEADER = "## How to Talk When Using TTS (`text_to_speech`)\n\n"

_TTS_HOOK_STEERING_INTRO = (
    "When your response gets spoken aloud, drop a bracketed "
    "steering tag at the start of the sentence it applies to.\n"
    "Natural language only. No commas or capitals inside the brackets.\n\n"
)

_TTS_HOOK_FLASH_INTRO = (
    "The selected model is Realtime TTS-2 Flash. Instruction tags like "
    "[say excitedly] are ignored; non-verbal sounds still work.\n\n"
)

_TTS_HOOK_INSTRUCTION_TAGS = (
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
)

_TTS_HOOK_COMBINING = (
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
)

_TTS_HOOK_NON_VERBALS = (
    "### Non-verbals (inline anywhere)\n"
    "[laugh] [sigh] [breathe] [clear throat] [cough] [yawn]\n\n"
)

_TTS_HOOK_EMPHASIS = (
    "### Emphasis via capitalization\n"
    'Full word: "I told you NOT to do that"\n'
    'Syllable: "AbsoLUTEly"\n'
    "Use sparingly or it loses impact.\n\n"
)

_TTS_HOOK_RULES = (
    "### Rules\n"
    "- One steering tag per passage. Change it only when the "
    "delivery shifts.\n"
    "- Match the tag to the content. Don't [say sadly] on "
    "celebratory lines.\n"
    "- Avoid opposites in the same tag — [whisper] and [very loud] "
    "together produce unpredictable results.\n"
    "- Non-verbals can go inline wherever natural.\n\n"
)

_TTS_HOOK_FORMATTING = (
    "### Formatting for speech\n"
    '- Write numbers as words: "twenty-three" not "23"\n'
    '- Write dates as spoken: "march fifteenth"\n'
    "- Use contractions: don't, can't, I'm, we're\n"
    "- No markdown, no bullet points, no emoji in speech\n"
    "- Include filler words where natural: uh, um, well, like, you know\n"
    "- Vary sentence length for natural rhythm\n\n"
)

_TTS_HOOK_GOAL_STEERING = (
    "### The goal\n"
    "Don't force it. Use steering when it fits the moment — "
    "hype when something's exciting, quiet when something matters, "
    "[laugh] when something's funny. If it's a simple reply, "
    "just talk. The tags are there to make the spoken delivery sound natural, "
    "not like a robot reading a script.\n"
)

_TTS_HOOK_GOAL_FLASH = (
    "### The goal\n"
    "Don't force it. Use non-verbal sounds when they fit the moment — "
    "[laugh] when something's funny, [sigh] when something's heavy. "
    "If it's a simple reply, just talk. The sounds are there to make the "
    "spoken delivery sound natural, not like a robot reading a script.\n\n"
)

_TTS_HOOK_SCOPE_STEERING = (
    "### When to use\n"
    "Use this steering in the input to text_to_speech tool only. "
    "Don't use it in the normal text output to the user.\n\n"
)

_TTS_HOOK_SCOPE_FLASH = (
    "### When to use\n"
    "Use this guidance in the input to text_to_speech tool only. "
    "Don't use it in the normal text output to the user.\n\n"
)

_TTS_HOOK_LANGUAGE = (
    "### Language override\n"
    "To speak in a language, put the language "
    "name in curly braces at the very start of the text. "
    "Supported: {polish}, {spanish}, {french}, {german}, {italian}, "
    "{portuguese}, {dutch}, {russian}, {japanese}, {chinese}, "
    "{korean}, {arabic}, {hebrew}, {hindi}. "
    'Example: "{polish} Witaj, świecie!" will speak Polish.\n'
)

# Each section constant supplies its own trailing separators (some end with
# ``\n``, others with ``\n\n``). Do not strip or reflow them; the rendered
# contexts are byte-sensitive.
TTS_HOOK_CONTEXT = (
    _TTS_HOOK_HEADER
    + _TTS_HOOK_STEERING_INTRO
    + _TTS_HOOK_INSTRUCTION_TAGS
    + _TTS_HOOK_COMBINING
    + _TTS_HOOK_NON_VERBALS
    + _TTS_HOOK_EMPHASIS
    + _TTS_HOOK_RULES
    + _TTS_HOOK_FORMATTING
    + _TTS_HOOK_GOAL_STEERING
    + _TTS_HOOK_SCOPE_STEERING
    + _TTS_HOOK_LANGUAGE
)

TTS_FLASH_HOOK_CONTEXT = (
    _TTS_HOOK_HEADER
    + _TTS_HOOK_FLASH_INTRO
    + _TTS_HOOK_NON_VERBALS
    + _TTS_HOOK_FORMATTING
    + _TTS_HOOK_GOAL_FLASH
    + _TTS_HOOK_SCOPE_FLASH
    + _TTS_HOOK_LANGUAGE
)
