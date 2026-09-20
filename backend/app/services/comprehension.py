"""Target comprehension: from what the user said to what the rover looks for.

This is the seam for Aleesha's comprehension work. A provider registered with
``register_comprehension_provider`` is tried first; until one exists (or when
it fails) a small rule-based fallback does the job.

The fallback is deliberately modest and honest about it:

* It only *parses* the sentence -- strips lead-ins ("can you find my ..."),
  separates the target phrase from landmark phrases ("... beside my
  backpack"), and maps the target phrase to a COCO class for local screening.
* It never rewrites the description: OMNI still receives the user's words.
* Transcription does not resolve references. "The other one", "that one" or
  "it" name nothing the rover can look for, so they come back with
  ``needs_clarification=True`` instead of a guess.
* Landmarks are context only. They are never used to gate or steer movement.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, replace
from typing import Callable

from app.services.yolo_detector import COCO_TARGET_MAP

logger = logging.getLogger(__name__)

RULE_BASED_RESOLVER = "rule_based_fallback"

CLARIFICATION_QUESTION = "Which item do you mean? Tell me what it is, e.g. 'the blue water bottle'."
EMPTY_REQUEST_QUESTION = "What should I look for? Tell me what it is, e.g. 'the blue water bottle'."


@dataclass(frozen=True)
class ComprehensionContext:
    previous_target_text: str | None = None
    rejected_descriptions: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResolvedTarget:
    raw_text: str                 # what the user typed/said
    target_text: str              # description handed to OMNI (fallback: raw_text trimmed)
    category: str | None          # COCO class for YOLO screening, or None -> OMNI sampling
    landmarks: tuple[str, ...]    # "beside my backpack" -> ("backpack",); NEVER used for gating
    needs_clarification: bool
    clarification_question: str | None
    resolver: str                 # "rule_based_fallback" | provider name


ComprehensionProvider = Callable[[str, ComprehensionContext], "ResolvedTarget | None"]

_provider_lock = threading.Lock()
_provider: tuple[str, ComprehensionProvider] | None = None


def register_comprehension_provider(name: str, fn: ComprehensionProvider) -> None:
    """Install the provider tried before the rule-based fallback (one at a
    time; registering again replaces it). ``fn(raw_text, context)`` returns a
    ResolvedTarget, or None to decline; raising also falls back."""
    global _provider
    if not name or not callable(fn):
        raise ValueError("a comprehension provider needs a name and a callable")
    with _provider_lock:
        _provider = (name, fn)


def clear_comprehension_provider() -> None:
    global _provider
    with _provider_lock:
        _provider = None


def _ask_provider(raw_text: str, context: ComprehensionContext) -> ResolvedTarget | None:
    with _provider_lock:
        provider = _provider
    if provider is None:
        return None
    name, fn = provider
    try:
        result = fn(raw_text, context)
    except Exception:  # noqa: BLE001 - a broken provider must not block a search
        logger.exception("comprehension provider %r failed; using the rule-based fallback", name)
        return None
    if result is None:
        logger.info("comprehension provider %r declined; using the rule-based fallback", name)
        return None
    if not isinstance(result, ResolvedTarget):
        logger.warning(
            "comprehension provider %r returned %s, not a ResolvedTarget; using the rule-based fallback",
            name,
            type(result).__name__,
        )
        return None
    # Stamp the registered name so callers can always tell who resolved it.
    category = result.category
    if category is not None:
        normalized = category.strip().lower() if isinstance(category, str) else None
        if normalized not in set(COCO_TARGET_MAP.values()):
            logger.warning("comprehension provider %r returned unsupported category %r", name, category)
            normalized = None
        category = normalized
    return replace(result, resolver=name, category=category)


def resolve_target(raw_text: str, context: ComprehensionContext | None = None) -> ResolvedTarget:
    raw_text = raw_text or ""
    context = context or ComprehensionContext()
    return _ask_provider(raw_text, context) or _resolve_rule_based(raw_text, context)


# --- rule-based fallback -------------------------------------------------------

# Spoken/typed wrappers around the actual target, stripped from the front
# (repeatedly, so "hey patch can you please find ..." unwinds fully). Longest
# first so "can you find" wins over "can you".
_LEAD_INS = sorted(
    [
        "please", "hey", "hi", "hello", "ok", "okay", "hey patch", "hi patch", "hello patch",
        "ok patch", "okay patch",
        "can you", "could you", "would you", "will you", "can u", "can you please", "could you please",
        "i need you to", "i want you to", "i would like you to", "i'd like you to",
        "help me", "help me to", "go", "go and", "try to", "just",
        "find", "find me", "can you find", "could you find", "help me find", "go find",
        "look for", "look around for", "search for", "locate", "spot",
        "where is", "where are", "where's", "wheres", "where did i put", "where did i leave",
        "where have i put", "where have i left",
        "i'm looking for", "im looking for", "i am looking for", "we're looking for", "we are looking for",
        "i'm searching for", "im searching for", "i am searching for", "looking for", "searching for",
        "get", "get me", "go get", "bring", "bring me", "fetch", "fetch me", "grab", "grab me",
        "i lost", "i've lost", "i have lost", "i can't find", "i cant find", "i cannot find",
        "i need", "i want", "i need to find", "i want to find",
        "have you seen", "do you see", "can you see", "show me",
    ],
    key=len,
    reverse=True,
)
_LEAD_IN_RE = re.compile(r"^(?:" + "|".join(re.escape(p) for p in _LEAD_INS) + r")\b\s*")
_WAKE_WORD_RE = re.compile(r"^patch\s*[,:!.]\s*")
_TRAILERS_RE = re.compile(r"\s+(?:please|for me|thanks|thank you|right now|now)$")

# Articles / possessives / demonstratives dropped from the front of a phrase.
_DETERMINERS_RE = re.compile(
    r"^(?:my|the|a|an|our|your|his|her|their|this|that|these|those|some|any|one of)\b\s*"
)

# Where the target phrase ends and a landmark phrase begins. "with" is NOT
# here on purpose: "bottle with a red cap" describes the bottle itself.
_PREPOSITIONS = sorted(
    [
        "beside", "next to", "near", "near to", "nearby", "by", "on", "on top of", "under", "underneath",
        "beneath", "below", "above", "behind", "in front of", "inside", "inside of", "in", "at",
        "between", "in between", "left of", "right of", "to the left of", "to the right of",
        "on the left of", "on the right of", "close to", "against", "alongside",
        "around", "from", "off", "off of", "within", "over", "across", "across from",
        "opposite", "opposite to", "toward", "towards", "into", "onto", "among", "amongst",
    ],
    key=len,
    reverse=True,
)
_PREPOSITION_RE = re.compile(r"\b(?:" + "|".join(re.escape(p) for p in _PREPOSITIONS) + r")\b")

# What follows a relation but is not a place: "the bottle from before", "my
# keys over there". Never recorded as a landmark.
_NOT_A_PLACE_WORDS = frozenset(
    {
        "here", "there", "somewhere", "anywhere", "everywhere", "nowhere",
        "before", "earlier", "previously", "yesterday", "today", "tonight", "now", "then", "again",
        "ago", "last", "time", "night", "week", "morning", "afternoon", "evening",
    }
)

# The screening category is read from the HEAD NOUN only. Any of these after
# the first word ends the head-noun segment, because what follows describes or
# locates the target and may name a different object ("case for my phone",
# "keys, they're with my phone", "keys my laptop is next to"):
#   * every landmark preposition above;
#   * post-modifiers and relative/subordinate clause words;
#   * a pronoun, which starts a new clause ("... i think", "... they're");
#   * a determiner/possessive, which starts a second noun phrase.
# Cutting too early only costs the local accelerator (category None -> OMNI is
# asked on every stop); cutting too late screens for the wrong object.
_HEAD_NOUN_ENDERS = sorted(
    [
        *_PREPOSITIONS,
        "with", "without", "of", "for", "to", "like",
        "that", "that's", "thats", "which", "who", "whose", "whom", "where", "when", "while",
        "because", "since", "but", "though", "although", "if", "so", "maybe", "probably", "perhaps",
        "i", "i'm", "im", "i've", "ive", "i'd", "i'll", "we", "we're", "we've", "you", "he", "she",
        "it", "it's", "its", "they", "they're", "theyre", "they've", "them", "there's", "theres",
        "my", "the", "a", "an", "our", "your", "his", "her", "their", "this", "these", "those",
        "some", "any",
    ],
    key=len,
    reverse=True,
)
_HEAD_NOUN_END_RE = re.compile(r"\b(?:" + "|".join(re.escape(p) for p in _HEAD_NOUN_ENDERS) + r")(?![\w'])")
# Two targets ("my keys and phone") do not make one reliable screening class.
_CONJUNCTION_RE = re.compile(r"\b(?:and|or|plus)\b")

# Unresolved references. A phrase is one when it contains a word that points
# at something without naming it ...
_POINTING_WORDS = frozenset(
    {
        "it", "this", "that", "these", "those", "them", "they", "one", "ones", "other", "others",
        "another", "same", "different", "else", "thing", "things", "stuff", "item", "items",
        "object", "objects", "something", "anything",
    }
)
# ... and nothing else but words that cannot name an object on their own:
# articles, fillers, positions, and the plain colour/size adjectives people
# attach to a pronoun ("the red one"). A lone "orange" is still a fruit -- it
# contains no pointing word -- but "the orange one" is a reference.
_NON_NAMING_WORDS = frozenset(
    {
        "the", "a", "an", "my", "our", "your", "his", "her", "their", "some", "any",
        "again", "instead", "too", "also", "not", "no", "nope", "please", "just",
        "first", "second", "third", "last", "next", "previous",
        "here", "there", "over", "left", "right", "middle", "up", "down",
        "red", "blue", "green", "yellow", "black", "white", "orange", "purple", "pink", "brown",
        "grey", "gray", "silver", "gold", "big", "small", "little", "large", "tiny", "new", "old",
    }
)

_CURLY_APOSTROPHES = str.maketrans({chr(0x2018): "'", chr(0x2019): "'"})


def _normalize(raw_text: str) -> str:
    text = raw_text.strip().lower().translate(_CURLY_APOSTROPHES)
    text = _WAKE_WORD_RE.sub("", text)
    # Punctuation and hyphens become spaces. \w keeps non-Latin letters, so a
    # request in another language is passed through to OMNI, not treated as empty.
    text = re.sub(r"[^\w' ]+|_", " ", text)
    return " ".join(text.split())


def _strip_lead_ins(text: str) -> str:
    previous = None
    while text != previous:
        previous = text
        text = _LEAD_IN_RE.sub("", text)
        text = _TRAILERS_RE.sub("", text)
    return text.strip()


def _strip_determiners(phrase: str) -> str:
    previous = None
    while phrase != previous:
        previous = phrase
        phrase = _DETERMINERS_RE.sub("", phrase)
    return phrase.strip()


def _is_reference_only(phrase: str) -> bool:
    words = phrase.split()
    return any(word in _POINTING_WORDS for word in words) and all(
        word in _POINTING_WORDS or word in _NON_NAMING_WORDS for word in words
    )


@dataclass(frozen=True)
class _Parse:
    request: str                  # normalised text with lead-ins removed
    head: str                     # text before the first spatial preposition, determiners intact
    target_phrase: str            # head without leading articles/possessives
    landmarks: tuple[str, ...]


def _parse(raw_text: str) -> _Parse:
    request = _strip_lead_ins(_normalize(raw_text))
    segments = _PREPOSITION_RE.split(request)
    head = segments[0].strip()

    landmarks: list[str] = []
    for segment in segments[1:]:
        for part in re.split(r"\band\b", segment):
            landmark = _strip_determiners(part.strip())
            # "next to it" names nothing; keep only landmarks that do.
            if landmark and not _is_reference_only(landmark) and not all(w in _NOT_A_PLACE_WORDS for w in landmark.split()) and landmark not in landmarks:
                landmarks.append(landmark)
    return _Parse(request, head, _strip_determiners(head), tuple(landmarks))


def parse_target_phrase(raw_text: str) -> tuple[str, tuple[str, ...]]:
    """Split a request into (target phrase, landmark phrases), both normalised
    to lower case with lead-ins and leading articles/possessives removed.

    "Find my bottle with a red cap beside my backpack" ->
    ("bottle with a red cap", ("backpack",)). The target phrase may be empty.
    """
    parsed = _parse(raw_text)
    return parsed.target_phrase, parsed.landmarks


def category_for_phrase(phrase: str) -> str | None:
    """COCO class named by ``phrase``: word-boundary match of the
    COCO_TARGET_MAP keywords (simple plurals allowed), longest keyword first.
    None when nothing matches or when two different classes do -- ambiguity
    goes to OMNI sampling rather than screening for the wrong thing."""
    text = _strip_determiners(_normalize(phrase))
    if _CONJUNCTION_RE.search(text):
        return None
    ender = _HEAD_NOUN_END_RE.search(text)
    if ender:
        text = text[:ender.start()]
    claimed: list[tuple[int, int]] = []
    classes: set[str] = set()
    for keyword in sorted(COCO_TARGET_MAP, key=len, reverse=True):
        for match in re.finditer(rf"\b{re.escape(keyword)}(?:s|es)?\b", text):
            start, end = match.span()
            # "cell phone" already claimed these characters; don't count "phone" again.
            if any(start >= s and end <= e for s, e in claimed):
                continue
            claimed.append((start, end))
            classes.add(COCO_TARGET_MAP[keyword])
    return classes.pop() if len(classes) == 1 else None


def _result(
    raw_text: str,
    *,
    category: str | None = None,
    landmarks: tuple[str, ...] = (),
    question: str | None = None,
) -> ResolvedTarget:
    return ResolvedTarget(
        raw_text=raw_text,
        target_text=raw_text.strip(),  # the fallback never rewrites what OMNI is told to look for
        category=category,
        landmarks=landmarks,
        needs_clarification=question is not None,
        clarification_question=question,
        resolver=RULE_BASED_RESOLVER,
    )


def _resolve_rule_based(raw_text: str, context: ComprehensionContext) -> ResolvedTarget:
    # `context` is there for providers. The fallback does not use it to guess:
    # knowing the previous target does not tell us which object "the other
    # one" is, and a wrong guess would send the rover after the wrong thing.
    parsed = _parse(raw_text)
    if not parsed.request:
        return _result(raw_text, question=EMPTY_REQUEST_QUESTION)  # empty, or nothing but "can you find"

    if _is_reference_only(parsed.request) or _is_reference_only(parsed.head):
        # "the other one", "it", "not that one", "the one next to it" ...
        return _result(raw_text, question=CLARIFICATION_QUESTION)

    if not parsed.target_phrase:
        if not parsed.landmarks:
            return _result(raw_text, question=CLARIFICATION_QUESTION)  # "next to it": nothing named at all
        # Unusual word order ("next to the sofa is my bottle"): we cannot tell
        # target from landmark, so claim neither and let OMNI read the sentence.
        return _result(raw_text)

    return _result(raw_text, category=category_for_phrase(parsed.target_phrase), landmarks=parsed.landmarks)
