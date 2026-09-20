"""The comprehension adapter: rule-based fallback, provider seam, YOLO delegation."""

import pytest

from app.services import comprehension
from app.services.comprehension import (
    RULE_BASED_RESOLVER,
    ComprehensionContext,
    ResolvedTarget,
    category_for_phrase,
    clear_comprehension_provider,
    parse_target_phrase,
    register_comprehension_provider,
    resolve_target,
)
from app.services.yolo_detector import COCO_TARGET_MAP, match_coco_class


@pytest.fixture(autouse=True)
def _no_leftover_provider():
    clear_comprehension_provider()
    yield
    clear_comprehension_provider()


def provided(raw_text: str, **overrides) -> ResolvedTarget:
    fields = dict(
        raw_text=raw_text,
        target_text="the blue water bottle from before",
        category="bottle",
        landmarks=(),
        needs_clarification=False,
        clarification_question=None,
        resolver="",
    )
    fields.update(overrides)
    return ResolvedTarget(**fields)


# --- target vs landmark ------------------------------------------------------


def test_target_is_separated_from_the_landmark():
    result = resolve_target("Find my bottle beside my backpack")
    assert result.category == "bottle"
    assert result.landmarks == ("backpack",)
    assert result.needs_clarification is False
    assert result.clarification_question is None
    assert result.resolver == RULE_BASED_RESOLVER


def test_swapping_the_roles_swaps_the_category():
    result = resolve_target("my backpack next to the bottle")
    assert result.category == "backpack"
    assert result.landmarks == ("bottle",)


def test_unmapped_target_falls_back_to_omni_sampling():
    result = resolve_target("my keys")
    assert result.category is None
    assert result.landmarks == ()
    assert result.needs_clarification is False


def test_a_coco_word_that_is_only_a_landmark_never_becomes_the_category():
    result = resolve_target("my keys on top of the laptop")
    assert result.category is None
    assert result.landmarks == ("laptop",)


def test_fallback_hands_omni_the_users_own_words():
    result = resolve_target("  Find my bottle beside my backpack  ")
    assert result.raw_text == "  Find my bottle beside my backpack  "
    assert result.target_text == "Find my bottle beside my backpack"


@pytest.mark.parametrize(
    "text, category, landmarks",
    [
        ("Hey Patch, can you please find my cell phone on the desk next to the laptop?", "cell phone", ("desk", "laptop")),
        ("Patch, where's my iPhone?", "cell phone", ()),
        ("I’m looking for my water-bottle", "bottle", ()),
        ("i can't find my suitcase", "suitcase", ()),
        ("bring me the remote please", "remote", ()),
        ("where are my books", "book", ()),
        ("my purse in the kitchen", "handbag", ("kitchen",)),
        ("the bottle between the laptop and the backpack", "bottle", ("laptop", "backpack")),
        ("the bottle to the left of the fridge", "bottle", ("fridge",)),
        ("look for the book underneath the big blue chair", "book", ("big blue chair",)),
        ("bottle in front of the tv behind the sofa", "bottle", ("tv", "sofa")),
        ("that bottle", "bottle", ()),
        ("my bottle next to it", "bottle", ()),  # a pronoun is not a landmark
    ],
)
def test_lead_ins_prepositions_and_landmarks(text, category, landmarks):
    result = resolve_target(text)
    assert (result.category, result.landmarks) == (category, landmarks)
    assert result.needs_clarification is False


# --- the category comes from the head noun only (review finding C10) -------------
# A COCO word that belongs to a SECOND object must never become the screening
# class: it gates OMNI on the wrong thing, picks the wrong proximity profile and
# lets YOLO boxes of that other object steer the approach.

LEAKING_PHRASES = [
    "find my keys around my laptop",
    "my wallet from my backpack",
    "find my keys, they're with my phone",
    "find my charger for my laptop",
    "find the lid of my bottle",
    "find the case for my phone",
    "find my keys across from the fridge",
    "find my keys opposite the fridge",
    "find my keys, I think they fell off the book",
    "find my wallet within the backpack",
]


@pytest.mark.parametrize("text", LEAKING_PHRASES)
def test_a_second_object_in_the_request_never_becomes_the_category(text):
    result = resolve_target(text)
    assert result.category is None
    assert match_coco_class(text) is None
    assert result.needs_clarification is False
    assert result.target_text == text  # OMNI still gets the user's own words


@pytest.mark.parametrize(
    "text",
    [
        "find my glasses, I left them with my laptop",
        "find my keys they're with my phone",             # speech: no punctuation
        "find my keys theyre next to my phone",           # speech: no apostrophe either
        "find my keys it's on the book",
        "my keys, my laptop is next to them",             # a second noun phrase, no preposition before it
        "find my keys attached to my backpack",
        "find my keys that were in my backpack",
        "the charger which goes with the laptop",
        "my wallet, I put it in the suitcase",
        "find the keys my phone was lying on",
        "find my keys and my phone",                      # two targets, one class: not a reliable category
        "find my keys and phone",
        "my headphones over the laptop",
        "my keys off the fridge",
        "my wallet towards the fridge",
        "my keys into the backpack",
        "my keys among the books",
        "my glasses without the bottle",
    ],
)
def test_other_relations_and_clauses_do_not_leak_a_category_either(text):
    assert resolve_target(text).category is None


@pytest.mark.parametrize(
    "phrase, category",
    [
        ("keys around my laptop", None),
        ("case for my phone", None),
        ("lid of my bottle", None),
        ("my keys near my laptop", None),      # a leading determiner does not hide the cut
        ("charger that fits my laptop", None),
        ("bottle with a red cap", "bottle"),
        ("bottle of water", "bottle"),
        ("remote for the tv", "remote"),
        ("the book that i borrowed", "book"),
        ("backpack without a zip", "backpack"),
        ("blue water bottle", "bottle"),
    ],
)
def test_category_for_phrase_only_reads_the_head_noun(phrase, category):
    assert category_for_phrase(phrase) == category


@pytest.mark.parametrize(
    "text, category, landmarks",
    [
        ("Find my bottle beside my backpack", "bottle", ("backpack",)),
        ("the blue water bottle from before", "bottle", ()),   # "before" is not a place
        ("my bottle over there", "bottle", ()),                 # neither is "there"
        ("find my keys around my laptop", None, ("laptop",)),
        ("my wallet from my backpack", None, ("backpack",)),
        ("find my keys across from the fridge", None, ("fridge",)),
        ("find my keys opposite the fridge", None, ("fridge",)),
        ("find my wallet within the backpack", None, ("backpack",)),
        ("find my keys, I think they fell off the book", None, ("book",)),
        ("the jacket over the suitcase", None, ("suitcase",)),
        ("my bottle above the fridge", "bottle", ("fridge",)),
        ("my bottle below the laptop", "bottle", ("laptop",)),
        ("the bottle from the fridge", "bottle", ("fridge",)),
    ],
)
def test_more_spatial_relations_become_landmarks(text, category, landmarks):
    result = resolve_target(text)
    assert (result.category, result.landmarks) == (category, landmarks)
    assert result.needs_clarification is False


def test_the_head_noun_cut_never_touches_what_omni_is_told():
    for text in ["find the case for my phone", "bottle with a red cap", "the blue water bottle from before"]:
        result = resolve_target(text)
        assert result.target_text == text
        assert result.raw_text == text
    # "with"/"for"/"of" stay attributes of the target phrase; only the category ignores them.
    assert parse_target_phrase("find the case for my phone") == ("case for my phone", ())
    assert parse_target_phrase("find the lid of my bottle") == ("lid of my bottle", ())


def test_a_reference_followed_by_a_new_relation_still_needs_clarification():
    for text in ["the one from before", "that one over there", "the other one around the corner"]:
        result = resolve_target(text)
        assert result.needs_clarification is True
        assert result.category is None


# --- "with" is an attribute, not a landmark ------------------------------------


def test_with_keeps_the_attribute_in_the_target_phrase():
    assert parse_target_phrase("bottle with a red cap") == ("bottle with a red cap", ())
    result = resolve_target("bottle with a red cap")
    assert result.category == "bottle"
    assert result.landmarks == ()
    assert result.target_text == "bottle with a red cap"


def test_with_attribute_and_a_real_landmark_together():
    assert parse_target_phrase("Find my bottle with a red cap beside my backpack") == (
        "bottle with a red cap",
        ("backpack",),
    )
    result = resolve_target("Find my bottle with a red cap beside my backpack")
    assert (result.category, result.landmarks) == ("bottle", ("backpack",))


# --- category matching ---------------------------------------------------------


@pytest.mark.parametrize(
    "phrase, category",
    [
        ("phone", "cell phone"),
        ("cell phone", "cell phone"),   # longest keyword wins; still one class
        ("phones", "cell phone"),
        ("phone case", "cell phone"),   # loose, and accepted as such
        ("headphones", None),           # word boundary: no "phone" inside "headphones"
        ("notebook", None),
        ("handbag", "handbag"),         # not "bag" -> backpack
        ("bag", "backpack"),
        ("fridge", "refrigerator"),
        ("laptop bag", None),           # two different classes -> let OMNI decide
        ("bottle and book", None),
        ("keys", None),
        ("", None),
    ],
)
def test_category_for_phrase(phrase, category):
    assert category_for_phrase(phrase) == category


def test_every_mapped_keyword_resolves_to_its_own_class():
    for keyword, coco_class in COCO_TARGET_MAP.items():
        assert resolve_target(f"find my {keyword}").category == coco_class


# --- unresolved references -------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "the other one",
        "that one",
        "it",
        "this",
        "another one",
        "the same one",
        "not that one",
        "the one next to it",
        "Find the other one please",
        "No, not that one!",
        "the red one",
        "bring me that thing",
        "next to it",
    ],
)
def test_pure_references_need_clarification(text):
    result = resolve_target(text)
    assert result.needs_clarification is True
    assert result.category is None
    assert result.landmarks == ()
    assert result.clarification_question
    assert "Which item do you mean" in result.clarification_question
    assert result.resolver == RULE_BASED_RESOLVER
    assert result.target_text == text.strip()


def test_context_does_not_make_the_fallback_guess():
    context = ComprehensionContext(previous_target_text="my bottle", rejected_descriptions=("a green bottle",))
    result = resolve_target("the other one", context)
    assert result.needs_clarification is True
    assert result.category is None  # NOT "bottle": knowing the last target does not identify "the other one"


@pytest.mark.parametrize("text", ["", "   ", "???", "can you find", None])
def test_empty_requests_ask_what_to_look_for(text):
    result = resolve_target(text)
    assert result.needs_clarification is True
    assert result.category is None
    assert "What should I look for" in result.clarification_question


@pytest.mark.parametrize(
    "text, category",
    [
        ("the other bottle", "bottle"),   # vague, but it names something OMNI can look for
        ("find my orange", None),         # a colour word alone is a perfectly good noun
        ("xbox one", None),
        ("我的水瓶", None),  # not English: pass it through, do not call it empty
        ("next to the sofa is my bottle", None),
    ],
)
def test_named_things_are_not_mistaken_for_references(text, category):
    result = resolve_target(text)
    assert result.needs_clarification is False
    assert result.category == category


def test_unusual_word_order_claims_neither_target_nor_landmark():
    result = resolve_target("next to the sofa is my bottle")
    assert (result.category, result.landmarks) == (None, ())
    assert result.target_text == "next to the sofa is my bottle"


# --- provider seam ---------------------------------------------------------------


def test_registered_provider_is_tried_first_and_gets_the_context():
    seen = []

    def provider(raw_text, context):
        seen.append((raw_text, context))
        return provided(raw_text, landmarks=("sink",))

    context = ComprehensionContext(previous_target_text="my bottle", rejected_descriptions=("a green bottle",))
    register_comprehension_provider("omni_comprehension", provider)
    result = resolve_target("the other one", context)

    assert seen == [("the other one", context)]
    assert result.needs_clarification is False  # the provider could resolve what the rules cannot
    assert result.category == "bottle"
    assert result.target_text == "the blue water bottle from before"
    assert result.landmarks == ("sink",)
    assert result.resolver == "omni_comprehension"


def test_provider_gets_a_default_context_when_none_is_passed():
    seen = []
    register_comprehension_provider("p", lambda raw, ctx: seen.append(ctx) or None)
    resolve_target("my bottle")
    assert seen == [ComprehensionContext()]


def test_provider_failure_falls_back_to_the_rules(caplog):
    def provider(raw_text, context):
        raise RuntimeError("model offline")

    register_comprehension_provider("flaky", provider)
    with caplog.at_level("ERROR", logger=comprehension.logger.name):
        result = resolve_target("Find my bottle beside my backpack")

    assert (result.category, result.landmarks) == ("bottle", ("backpack",))
    assert result.resolver == RULE_BASED_RESOLVER
    assert any("flaky" in record.getMessage() for record in caplog.records)  # the failure is logged, not hidden


@pytest.mark.parametrize("bad_return", [None, "bottle", {"category": "bottle"}])
def test_provider_declining_or_returning_junk_falls_back(bad_return):
    register_comprehension_provider("shy", lambda raw, ctx: bad_return)
    result = resolve_target("the other one")
    assert result.resolver == RULE_BASED_RESOLVER
    assert result.needs_clarification is True


def test_registering_again_replaces_and_clearing_restores_the_fallback():
    register_comprehension_provider("first", lambda raw, ctx: provided(raw, category="book"))
    register_comprehension_provider("second", lambda raw, ctx: provided(raw, category="laptop"))
    result = resolve_target("my keys")
    assert (result.category, result.resolver) == ("laptop", "second")

    clear_comprehension_provider()
    result = resolve_target("my keys")
    assert (result.category, result.resolver) == (None, RULE_BASED_RESOLVER)


# A category the local detector never emits would make screening come back
# empty on every frame (a permanent miss), so it is dropped, loudly (finding L05).


@pytest.mark.parametrize("bogus", ["phone", "water bottle", "keys", "bottles", "", "   ", 7, ["bottle"]])
def test_a_provider_category_that_is_not_a_coco_class_is_dropped_with_a_warning(bogus, caplog):
    register_comprehension_provider("sloppy", lambda raw, ctx: provided(raw, category=bogus))
    with caplog.at_level("WARNING", logger=comprehension.logger.name):
        result = resolve_target("the other one")

    assert result.category is None  # -> OMNI sampling on every stop, never gating on a class YOLO cannot see
    assert result.resolver == "sloppy"  # everything else the provider said is kept
    assert result.target_text == "the blue water bottle from before"
    assert result.needs_clarification is False
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "sloppy" in warnings[0].getMessage() and repr(bogus) in warnings[0].getMessage()


def test_every_coco_class_value_is_accepted_from_a_provider(caplog):
    with caplog.at_level("WARNING", logger=comprehension.logger.name):
        for coco_class in sorted(set(COCO_TARGET_MAP.values())):
            register_comprehension_provider("tidy", lambda raw, ctx, c=coco_class: provided(raw, category=c))
            assert resolve_target("the other one").category == coco_class
        # Case and stray whitespace are not worth losing the accelerator over.
        register_comprehension_provider("tidy", lambda raw, ctx: provided(raw, category="  Cell Phone "))
        assert resolve_target("the other one").category == "cell phone"
        # None is the documented way to ask for OMNI sampling: no warning.
        register_comprehension_provider("tidy", lambda raw, ctx: provided(raw, category=None))
        assert resolve_target("the other one").category is None
    assert [r for r in caplog.records if r.levelname == "WARNING"] == []


def test_provider_registration_is_validated():
    with pytest.raises(ValueError):
        register_comprehension_provider("", lambda raw, ctx: None)
    with pytest.raises(ValueError):
        register_comprehension_provider("name", "not callable")  # type: ignore[arg-type]


# --- yolo_detector.match_coco_class delegates here --------------------------------


def test_match_coco_class_uses_the_target_phrase_not_a_substring_scan():
    assert match_coco_class("Find my bottle beside my backpack") == "bottle"  # was "backpack"
    assert match_coco_class("my backpack next to the bottle") == "backpack"
    assert match_coco_class("my headphones") is None                          # was "cell phone"
    assert match_coco_class("phone") == "cell phone"
    assert match_coco_class("my keys") is None
    assert match_coco_class("the other one") is None


def test_match_coco_class_follows_a_registered_provider():
    register_comprehension_provider("always_laptop", lambda raw, ctx: provided(raw, category="laptop"))
    assert match_coco_class("my keys") == "laptop"
    clear_comprehension_provider()
    assert match_coco_class("my keys") is None
