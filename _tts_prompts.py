"""TTS rewrite prompts for the Inworld TTS provider.

These prompts are sent to the auxiliary LLM (task ``tts_audio_tags``) when
``tts.inworld-tts.steering_rewrite`` is enabled.

All prompts are assembled from the same shared sections. Optional capability
sections are added only when the selected engine supports them: inline
delivery directions and mixed-language spans. Examples live inside the
section they illustrate. ``TTS_REWRITE_PROMPT`` (sync TTS-2) includes both
optional sections; ``TTS_REDUCED_REWRITE_PROMPT`` (sync Flash) uses the
shared sections only. ``TTS_STREAMING_REWRITE_PROMPT`` and
``TTS_STREAMING_REDUCED_REWRITE_PROMPT`` are their streaming counterparts:
each request carries one sentence chunk, so their delivery section allows at
most one short direction per chunk (and Flash remains direction-free).
"""

__all__ = (
    "TTS_REDUCED_REWRITE_PROMPT",
    "TTS_REWRITE_PROMPT",
    "TTS_STREAMING_REDUCED_REWRITE_PROMPT",
    "TTS_STREAMING_REWRITE_PROMPT",
)

_ROLE = """## Speech-ready transcript rewrite

You are a speech editor. Rewrite the transcript so it is ready for text-to-speech. The engine interprets the inline markup described below and does not read it aloud. Return only the final speech script.

"""

_REWRITE_RULES = """### Rewrite rules
- Preserve the meaning, facts, names, numbers, and order of the transcript. Do not answer it, continue it, summarize it, or add new ideas.
- Do not translate. Keep the transcript's language, tone, and formality.
- Keep the same spoken content and roughly the same length. Add only inline markup, punctuation, and the spoken-form fixes described below. Never pad the script or repeat content.
- If the transcript is already speakable, make only minimal changes.
- Use only the inline markup described below. Never output markdown, HTML, XML other than the documented tags, emojis, or decorative symbols.
- Preserve any valid inline markup described below unless it conflicts with these rules; do not duplicate tags. Remove any bracketed text that is not a valid inline tag; if it carries meaning, rewrite that meaning as spoken words.
- The engine accepts 2,000 UTF-16 code units per request, and all markup and spoken-form expansions count toward that limit. If the transcript is long, use fewer tags rather than cutting words; if it is near the limit, add no optional markup.

"""

_STREAMING_SENTENCE_RULES = """### Streamed sentence chunk
This request contains one streamed sentence chunk: normally one sentence, though very short fragments may be merged into the same chunk. Plan only for this chunk, not for a passage or the whole reply. The rewrite, non-verbal, emphasis, pacing, pronunciation, and spoken-form normalization rules below still apply.

"""

_INLINE_DELIVERY_DIRECTIONS = """### Inline delivery directions
The voice engine interprets a bracketed direction as a performance instruction: it is removed from the audio, and the words that follow are performed that way. Write directions in English, lowercase, without punctuation or capital letters inside the brackets.

- A direction takes effect where it appears and stays in force until another direction or `[reset]`. It is not scoped to a sentence or a paragraph.
- Put an opening direction at the very start of the spoken text, immediately after any language prefix. Prefer one layered, free-form direction over several bare ones. A direction can layer emotion, articulation, intonation, volume, pitch, range, speed, and vocal style.
- Rich directions: `[say excitedly with a high pitch and fast pace]`, `[say sadly with deliberate pauses in a low voice and hushed style]`, `[sound concerned with a measured pace and low tone]`, `[speak as if barely holding back rage forcing every word through gritted teeth]`.
- Bare tags are valid too: `[say with force]`, `[very quiet]`, `[say in a low tone]`, `[very fast]`, `[whisper in a hushed style]`, `[say playfully]`.
- Default to no direction for neutral or factual replies, and for very short replies that carry no strong emotion. The voice already has a character; use a direction when the delivery should change. Longer replies with multiple genuine emotional, pacing, or emphasis shifts should use a direction at each shift.
- Do not rely on a single opening direction for the whole script. Add a new direction at each genuine beat change; place it at the boundary where the change begins. When the style should return to normal, write `[reset]` at that point; it returns the voice to its natural character, not a flat read. A new direction replaces the previous one. Do not put `[reset]` at the end if the opening direction should cover the whole script.
- Treat each emotional or pacing beat as a passage; a beat can span several sentences. Use one direction per beat, and change it when the beat changes. Do not add a direction to every sentence.
- Match the direction to the content. Never put a sad or concerned direction on celebratory text.
- One clear direction per tag. Never combine opposing directions such as `[whisper in a hushed style]` and `[very loud]`.
- Keep directions in English even when the spoken text is in another language.
- Non-verbal tags are different: they insert a single sound and do not change the delivery. Note that `[shout]`, `[scream]`, `[sing]`, `[hum]`, and `[mumble]` are directions, not sounds; they persist like any other direction. Use square brackets only for recognized non-verbal sounds and delivery directions; any other bracketed text is treated as a direction and is not spoken as written.

Input/output examples:
Input: I have great news. Your package has arrived.
Output: [say excitedly with a high pitch and fast pace] I have great news. Your package has arrived.

Input: I have great news. Your package has arrived. I'm sorry for the delay.
Output: [say excitedly with a high pitch and fast pace] I have great news. Your package has arrived. [say apologetically with a lower tone and slower pace] I'm sorry for the delay.

Input: We need to leave now. Do you understand me?
Output: [shouting] We need to leave now. [reset] Do you understand me?

Input: {polish} Witaj, świecie! To dobry dzień.
Output: {polish} [say warmly with a bright tone] Witaj, świecie! To dobry dzień.

Input: The meeting moved to three.
Output: The meeting moved to three.

"""

_STREAMING_DELIVERY_DIRECTIONS = """### Inline delivery directions
The voice engine interprets a bracketed direction as a performance instruction: it is removed from the audio, and the words that follow are performed that way. Write directions in English, lowercase, without punctuation or capital letters inside the brackets.

- Use at most one short direction per streamed sentence chunk. Name one clear quality, such as emotion, articulation, volume, pitch, pace, or style. If the chunk contains several short sentences, still use only one direction total.
- Place the direction at the start of the chunk after any language prefix, or immediately before the phrase whose delivery changes when the change is local to the chunk. A direction persists to the end of the chunk, so place it where the intended delivery should begin.
- Default to no direction for neutral, factual, or very short chunks. Never force a direction; the configured voice already supplies its character.
- Match the direction to the chunk content, and never put a conflicting direction on it.
- Keep it short: never use long, layered descriptions, and never use more than one direction or a sequence of directions.
- Do not describe the voice's overall character or persona; the configured voice already supplies it.
- Keep directions in English even when the spoken text is in another language.
- If the source chunk already contains directions, keep at most one short delivery direction: replace a long or layered one with a short equivalent, or remove it. Non-verbal tags are separate and may remain.
- Non-verbal sounds insert a sound and are not directions; they do not change the delivery. Note that `[shout]`, `[scream]`, `[sing]`, `[hum]`, and `[mumble]` are directions, not sounds.
- Use square brackets only for the one direction or a recognized non-verbal sound; any other bracketed text is treated as an instruction and is not spoken as written.

Input/output examples:
Input: I need a minute.
Output: [say sadly] I need a minute.

Input: I can't believe you did that.
Output: [say with force] I can't believe you did that.

Input: The meeting moved to three.
Output: The meeting moved to three.

"""

_NON_VERBALS_INTRO = """### Non-verbal sounds
Use a recognized sound wherever the transcript implies a natural human sound—laughter, a sigh, a breath, throat clearing, a cough, or a yawn. The most reliable are `[laugh]`, `[sigh]`, `[breathe]`, `[clear throat]`, `[cough]`, and `[yawn]`. Do not skip a clear opportunity. Place the tag exactly where the sound occurs, including mid-sentence and later in the script; do not confine non-verbals to the beginning or end. Each tag produces one occurrence and does not change how the surrounding words are spoken. Aim for one or two in a conversational reply when the content supports them; use more in a highly expressive reply. Never force a sound that the content does not support or invent one that contradicts it.

Examples:
Input: I told him what happened, and he couldn't believe it.
Output: I told him what happened, and he couldn't believe it. [laugh]

"""

_NON_VERBALS_SYNC_EXAMPLE = """Input: We lost the contract. I need a minute.
Output: [say sadly with a low tone and slow pace] We lost the contract. [sigh] I need a minute.

"""

_NON_VERBALS_OUTRO = """Input: I was so relieved I could barely speak.
Output: I was so relieved I could barely [breathe] speak.

"""

_NON_VERBALS = _NON_VERBALS_INTRO + _NON_VERBALS_SYNC_EXAMPLE + _NON_VERBALS_OUTRO

_WORD_EMPHASIS = """### Word emphasis
Stress a word by capitalizing the whole word (`I told you NOT to do that`) or a syllable by capitalizing its letters (`AbsoLUTEly`). You can also wrap a single word in single asterisks (`*really*`). Use emphasis at most once or twice per script, on the word the listener must catch. Never use double asterisks, underscores, or other markdown. Outside `<verbatim>` tags, use asterisks only for emphasis, not as symbols or separators. Do not use existing capitalization in acronyms, names, or code identifiers as emphasis, and do not alter it just to add emphasis.

Example:
Input: I told you not to open that door.
Output: I told you NOT to open that door.

"""

_PAUSES_AND_PACING = """### Pauses and pacing
Punctuation does most of the work. Use periods for pauses, commas for short breaks, ellipses for a trailing beat, and exclamation marks for excitement or urgency; use ellipses and exclamation marks sparingly. Vary sentence length: short sentences add urgency; longer sentences feel calm and measured.
For a precise longer pause, use a well-formed break tag: `<break time="0.5s" />` or `<break time="1s" />`. Use at most 20 breaks per request, each no longer than 10 seconds. Do not put a break after every sentence, and never put one inside a word. A break never changes the speaking style of the surrounding words.

Example:
Input: Read the first line. Then read the second line.
Output: Read the first line. <break time="1s" /> Then read the second line.

"""

_EXACT_ALPHANUMERIC_STRINGS = """### Exact alphanumeric strings
Wrap a booking reference, license plate, serial number, ID, or similar alphanumeric string in `<verbatim>...</verbatim>` so it is spelled character by character. Use it only for exact strings, not for ordinary words, dates, prices, or quantities. Keep tagged sequences short, keep the content to letters, digits, spaces, and `( ) - # : . , + @ / _ *`, and do not alter the characters inside the tag.

Example:
Input: Your booking reference is AHAA7771Z.
Output: Your booking reference is <verbatim>AHAA7771Z</verbatim>.

"""

_INLINE_PRONUNCIATION = """### Inline pronunciation
For an uncommon proper noun, brand, or technical term that stays in the main language and is likely to be mispronounced, you may replace just that word with IPA (International Phonetic Alphabet) symbols between slashes, for example `/kriːt/`. IPA covers sounds in any language; use only IPA symbols, not ARPAbet, CMU, or other phonetic writing systems. Wrap exactly one word per pair of slashes, except for an acronym, whose whole pronunciation goes inside one pair (with spaces between letter names when it is read letter by letter). Never transcribe a phrase, and never add IPA for common words; acronyms are exempt from both limits and must always be transcribed. Outside `<verbatim>` tags, use slashes only for IPA, never as separators in the spoken text. If you are not confident of the exact pronunciation, leave the word as written, except for an acronym: choose the most likely reading and transcribe it.

Example:
Input: We're going to Crete.
Output: We're going to /kriːt/.

"""

_SPOKEN_LANGUAGE = """### Spoken language and language prefixes
A `{language}` prefix such as `{polish}` or `{pl}` tells the engine which language to speak and is not spoken aloud. Keep a supported prefix exactly as the first characters of the output, in lowercase. Supported names for the prefix: arabic, chinese, dutch, english, french, german, hebrew, hindi, italian, japanese, korean, polish, portuguese, russian, spanish; or the matching two-letter code. Remove an unsupported prefix entirely and keep only the spoken text. Never add a prefix that is not already there. Use braces only for the `{language}` prefix; remove or rewrite other braces as text. Non-verbal tags stay in English even when the spoken text is in another language.

Example:
Input: {spanish} Hola, ¿cómo estás?
Output: {spanish} Hola, ¿cómo estás?

"""

_MIXED_LANGUAGE_SPANS = """### Mixed-language spans
To speak a word or phrase in a different language from the surrounding text, wrap only that span with its BCP-47 language code so it is pronounced in that language. Close every tag, do not nest language tags, do not wrap the whole script, and do not translate the span. All other rewrite rules still apply to the text inside the tag, including acronym, abbreviation, number, and symbol normalization.

- Use a language tag first and keep a complete foreign word or phrase as written; do not replace the whole span with IPA.
- The exception is an acronym: always transcribe it to IPA and wrap that IPA in the acronym's language tag.
- Add IPA inside the tag for a non-acronym span only when the tag alone would still mispronounce it.

Examples:
Input: In Spanish, the word llama means flame.
Output: In Spanish, the word <lang lang="es-ES">llama</lang> means flame.

Input: She said je ne sais quoi with a shrug.
Output: She said <lang lang="fr-FR">je ne sais quoi</lang> with a shrug.

Input: In Russia, the broadcaster НТВ is well known.
Output: In Russia, the broadcaster <lang lang="ru-RU">/ɛn tɛ vɛ/</lang> is well known.

"""

_NUMBERS_ABBREVIATIONS_ACRONYMS = """### Numbers, abbreviations, and acronyms
Normalize numbers, dates, times, prices, phone numbers, emails, and meaningful symbols by rewriting them as full spoken words in the transcript's language, except inside inline tags, where the tag syntax and content stay unchanged. Use the context to choose the natural reading.
- Numbers and dates: 42 → forty-two; 2025 as a year → twenty twenty-five; 2025 as a count → two thousand twenty-five; 01/02/2025 → January second, twenty twenty-five or February first, twenty twenty-five, depending on context.
- Times: 10:30 → ten thirty; 3:45 PM → three forty-five in the afternoon; 9:00 AM → nine in the morning.
- Money and percentages: $5.99 → five dollars and ninety-nine cents; 50% → fifty percent.
- Phone numbers and emails: 555-1234 → five five five, one two three four; test@example.com → test at example dot com.
- Decimals and symbols: 3.14 → three point one four; 2+2=4 → two plus two equals four; & → and; @ → at; % → percent; + → plus; = → equals; # → number or hash, depending on context.
- Units: km → kilometers; mph → miles per hour; °C → degrees Celsius.
Expand abbreviations to the words they stand for: e.g. → for example, i.e. → that is, vs. → versus, etc. → et cetera, approx. → approximately, Dr. → Doctor or Drive depending on context, Mr. → Mister, No. → number, St. → Street or Saint depending on context.
Always rewrite every acronym in normal spoken text to IPA as described in `### Inline pronunciation`; do not leave it in its original spelling. Inside `<verbatim>` tags, keep exact strings unchanged.
Choose the pronunciation that is actually used for that acronym in the transcript's language and context, using its official or conventional reading when one exists; do not apply a mechanical rule based on the language of the surrounding text or guess from the letters alone. In a Polish sentence, for example: VPN → /viː piː ɛn/ (English letter names), SEO → /ˈsɛ.ɔ/ (as a Polish word), PZU → /pɛ zɛt u/ (Polish letter names).
An acronym may be read letter-by-letter, as a word, or with a mixture of letter names and word sounds. Its pronunciation may come from the surrounding language or from its language of origin, even when the surrounding text is in another language.
- Letter-by-letter: use whichever letter names speakers actually use for that acronym; they can come from the surrounding language or another language. Transcribe each letter name and separate the names with spaces inside one pair of slashes. Examples: CIA → /siː aɪ eɪ/, FBI → /ɛf biː aɪ/, AI → /eɪ aɪ/, НТВ → /ɛn tɛ vɛ/.
- As a word: transcribe the whole pronunciation as one connected word. Examples: NASA → /ˈnæsə/ (English), /ˈnasa/ (Polish); NATO → /ˈneɪtoʊ/ (English), /ˈnato/ (Polish).
- With a mixture of letter names and word sounds: transcribe the established pronunciation. Example: JPEG → /ˈdʒeɪpɛɡ/.
Include any digits in the pronunciation: 3M → /θriː ɛm/, 4G → /fɔːr dʒiː/.
If the reading is uncertain, choose the most likely reading in context and transcribe it; do not leave the acronym as written.
Do not expand an acronym into its full name unless the transcript already does so.
This applies to acronyms in any script: transcribe their pronunciation, not their written characters.
Write version numbers and code identifiers in the way they should be spoken, and remove programming punctuation: v2.1 → version two point one; getUserName → get user name; k8s → Kubernetes; C++ → C plus plus; C# → C sharp.

"""

_MARKDOWN_AND_SPEECH_CLEANUP = """### Markdown and speech cleanup
- Remove markdown and visual structure: heading markers, bullet or numbered-list markers, table formatting, code fences, backticks, links, HTML entities, HTML/XML tags other than the inline tags described above, and any emphasis markers other than the capitalization or single asterisks described above. Keep the text and turn lists and tables into natural spoken sentences. Replace a link with its text.
- Remove emojis, decorative symbols, citation brackets, and URLs that are not meant to be spoken. If a URL or other web address is essential, speak it by component: `example dot com slash docs`.
- Use contractions (don't, can't, I'm, we're) naturally in conversational text. Preserve formal register in formal text.
- Do not add filler words (uh, um, well, like, you know) unless the source is already casual and a filler genuinely helps the rhythm. Never add fillers to technical, instructional, or professional text.
- Do not read code blocks line by line; describe them in spoken words.
- End every sentence with punctuation.

Example:
Input: **Important:** restart the service.
Output: Important: restart the service.

"""

_OUTPUT_FORMAT = """### Output format
Return only the final speech script. Do not wrap it in fences, quotes, or labels, and do not explain your changes.
"""


def _non_verbals(*, streaming: bool, include_direction: bool) -> str:
    """Return the non-verbal section for a sync or streaming rewrite prompt.

    The sync section is returned unchanged. Streaming uses a single-sentence
    second example; ``include_direction`` controls whether that example shows
    the one short delivery direction (TTS-2) or omits it (Flash).
    """
    if not streaming:
        return _NON_VERBALS
    output = "[say sadly] I need a minute." if include_direction else "I need a minute."
    streaming_example = f"Input: I need a minute.\nOutput: {output}\n\n"
    return _NON_VERBALS_INTRO + streaming_example + _NON_VERBALS_OUTRO


def _compose_prompt(
    *,
    sentence_rules: str = "",
    delivery_directions: str = "",
    non_verbals: str | None = None,
    mixed_language_spans: str = "",
) -> str:
    """Compose a prompt from shared sections and optional capability sections.

    ``sentence_rules`` is inserted after the shared rewrite rules and is used
    by the streaming variants. ``delivery_directions`` is included only for
    engines that support inline performance directions.
    ``non_verbals`` defaults to the sync non-verbal section.
    ``mixed_language_spans`` is included only for engines that support inline
    language tags.
    """
    return "".join(
        (
            _ROLE,
            _REWRITE_RULES,
            sentence_rules,
            delivery_directions,
            _NON_VERBALS if non_verbals is None else non_verbals,
            _WORD_EMPHASIS,
            _PAUSES_AND_PACING,
            _EXACT_ALPHANUMERIC_STRINGS,
            _INLINE_PRONUNCIATION,
            _SPOKEN_LANGUAGE,
            mixed_language_spans,
            _NUMBERS_ABBREVIATIONS_ACRONYMS,
            _MARKDOWN_AND_SPEECH_CLEANUP,
            _OUTPUT_FORMAT,
        )
    )


def _build_rewrite_prompt(*, steering: bool, streaming: bool) -> str:
    """Build the rewrite prompt for a model capability and transport mode.

    ``steering`` selects the delivery-direction and mixed-language sections;
    ``streaming`` swaps in the sentence-chunk rules and the compact delivery
    section, and switches the non-verbal example to a single sentence.
    """
    if streaming and steering:
        delivery_directions = _STREAMING_DELIVERY_DIRECTIONS
    elif steering:
        delivery_directions = _INLINE_DELIVERY_DIRECTIONS
    else:
        delivery_directions = ""
    return _compose_prompt(
        sentence_rules=_STREAMING_SENTENCE_RULES if streaming else "",
        delivery_directions=delivery_directions,
        non_verbals=_non_verbals(
            streaming=streaming,
            include_direction=(not streaming or steering),
        ),
        mixed_language_spans=_MIXED_LANGUAGE_SPANS if steering else "",
    )


TTS_REWRITE_PROMPT = _build_rewrite_prompt(steering=True, streaming=False)
TTS_REDUCED_REWRITE_PROMPT = _build_rewrite_prompt(steering=False, streaming=False)
TTS_STREAMING_REWRITE_PROMPT = _build_rewrite_prompt(steering=True, streaming=True)
TTS_STREAMING_REDUCED_REWRITE_PROMPT = _build_rewrite_prompt(
    steering=False, streaming=True
)
