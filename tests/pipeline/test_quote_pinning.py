"""Quote pinning: lexical categories, protected names, the budget."""

from types import SimpleNamespace

from mtc.core.config import Settings
from mtc.pipeline.quote_pinning import (
    PIN_CATEGORIES,
    PROTECTED_NAME_CATEGORY,
    extract_pinned_quotes,
    pin_category,
)


def _m(speaker, content, role="user"):
    return {"role": role, "speaker": speaker, "content": content}


def _no_names():
    return SimpleNamespace(PROTECTED_NAMES="")


def test_commitment_vulnerability_and_promise_categories():
    assert set(PIN_CATEGORIES) == {"promise", "commitment", "vulnerability"}
    messages = [
        _m("Alice", "The weather was mild. I promise to call on Sunday."),
        _m("Testagent", "I will keep the list of open questions.", "assistant"),
        _m("Alice", "Honestly I'm worried about the exam next week."),
        _m("Operator", "The meeting moved to the second floor."),
        _m("Alice", "Let's look at the budget again tomorrow."),
    ]
    quotes = extract_pinned_quotes(messages, settings=_no_names())
    assert quotes == [
        'Alice: "I promise to call on Sunday."',
        'Testagent: "I will keep the list of open questions."',
        'Alice: "Honestly I\'m worried about the exam next week."',
        'Alice: "Let\'s look at the budget again tomorrow."',
    ]
    assert pin_category("I promise.") == "promise"
    assert pin_category("We'll finish the draft.") == "commitment"
    assert pin_category("I am afraid of the dark.") == "vulnerability"
    assert pin_category("It hurts when I think about it.") == "vulnerability"
    assert pin_category("The report is due.") is None
    # the typographic apostrophe matches as well
    assert pin_category("I" + chr(0x2019) + "ll be there.") == "commitment"


def test_protected_names_empty_by_default(monkeypatch):
    monkeypatch.delenv("MTC_PROTECTED_NAMES", raising=False)
    assert Settings(_env_file=None).PROTECTED_NAMES in ("", [])
    messages = [_m("Operator", "Alice visited the library on Tuesday.")]
    assert extract_pinned_quotes(messages, settings=Settings(_env_file=None)) == []
    configured = SimpleNamespace(PROTECTED_NAMES="Alice, Guest")
    assert extract_pinned_quotes(messages, settings=configured) == [
        'Operator: "Alice visited the library on Tuesday."'
    ]


def test_protected_name_whole_word():
    names = ["Alice"]
    messages = [
        _m("Operator", "Aliceville has a new bridge."),
        _m("Operator", "The postcard came from alice herself."),
        _m("Operator", "Visit (Alice) soon."),
    ]
    quotes = extract_pinned_quotes(messages, protected_names=names)
    assert quotes == [
        'Operator: "The postcard came from alice herself."',
        'Operator: "Visit (Alice) soon."',
    ]
    assert pin_category("Aliceville is quiet.", ()) is None
    # an explicit empty list overrides the configured names
    configured = SimpleNamespace(PROTECTED_NAMES="Alice")
    assert (
        extract_pinned_quotes(messages[1:], protected_names=[], settings=configured)
        == []
    )
    assert PROTECTED_NAME_CATEGORY == "protected_name"


def test_budget_respected():
    messages = [
        _m("Alice", f"I promise item number {index} gets done.") for index in range(50)
    ]
    quotes = extract_pinned_quotes(messages, max_chars=200, settings=_no_names())
    assert quotes
    assert len("\n".join(quotes)) <= 200
    assert quotes[0] == 'Alice: "I promise item number 0 gets done."'
    assert extract_pinned_quotes(messages, max_chars=0, settings=_no_names()) == []
    # duplicates are quoted once
    same = [_m("Alice", "I promise to call.")] * 3
    assert extract_pinned_quotes(same, settings=_no_names()) == [
        'Alice: "I promise to call."'
    ]


def test_no_valence_guessing():
    messages = [
        _m("Alice", "What a wonderful, joyful, lovely afternoon that was!"),
        _m("Alice", "That was a terrible, awful, miserable afternoon."),
        _m("Alice", "I'm worried the train is late."),
    ]
    quotes = extract_pinned_quotes(messages, settings=_no_names())
    # sentiment alone pins nothing; the quote is the sentence as written,
    # with no label, mood or score attached
    assert quotes == ['Alice: "I\'m worried the train is late."']
    for word in ("positive", "negative", "valence", "mood", "sentiment"):
        assert all(word not in quote.lower() for quote in quotes)


def test_clock_notes_never_quoted():
    note = {
        "role": "user",
        "speaker": "clock",
        "content": "I promise this is a clock note.",
        "is_clock_note": True,
    }
    assert extract_pinned_quotes([note], settings=_no_names()) == []
