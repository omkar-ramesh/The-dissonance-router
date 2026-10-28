"""Hope: zero-token review of a failure, a fix that is checked, and rules that stick."""
import pytest

from src.agent import hope, signals
from src.db.database import init_db

COMPLAINT = "ugh, that's a wall of text, I'm not reading all that"
NEGATION = "Don't keep it short this time, I need every step of backprop"


@pytest.fixture(autouse=True)
def clean_rules():
    signals.reset_rules()
    yield
    signals.reset_rules()


def test_review_checks_the_fix_without_changing_the_reader():
    r = hope.review("MISSED_SIGNAL", COMPLAINT)
    assert r["verified"] and r["tokens"] == 0 and not r["applied"]
    assert signals.SignalExtractor().extract(COMPLAINT) == {}      # only a scratch copy was tried


def test_negation_fix_reverses_the_reading():
    assert signals.SignalExtractor().extract(NEGATION)["length"] == "concise"     # the known bug
    assert hope.review("MISREAD_NEGATION", NEGATION)["verified"]


def test_apply_keeps_the_rule_and_it_survives_a_restart(tmp_path):
    db = str(tmp_path / "t.db")
    init_db(db)
    assert hope.apply(db, "MISSED_SIGNAL", "casey") == 1
    assert signals.SignalExtractor().extract(COMPLAINT) == {"length": "concise"}
    assert hope.apply(db, "MISSED_SIGNAL") == 0                                      # no duplicates
    signals.reset_rules()
    assert hope.load_rules(db) == 1
    assert signals.SignalExtractor().extract(COMPLAINT) == {"length": "concise"}


@pytest.mark.parametrize("message", ["wow, that is so wordy", "too much text, I'm not reading any of it"])
def test_the_fix_generalises_to_other_wordings(message):
    hope.review("MISSED_SIGNAL", message)
    signals.add_rule("length", "concise", hope.PLAYBOOK["MISSED_SIGNAL"]["rules"][0][2])
    assert signals.SignalExtractor().extract(message).get("length") == "concise"


def test_rules_do_not_break_normal_phrasings():
    for dim, value, pattern in hope.PLAYBOOK["MISREAD_NEGATION"]["rules"]:
        signals.add_rule(dim, value, pattern)
    ex = signals.SignalExtractor()
    assert ex.extract("keep it short please")["length"] == "concise"
    assert ex.extract("don't keep it short")["length"] == "detailed"


def test_unverified_link_is_removed_and_manual_codes_have_no_fix():
    link = "https://arxiv.org/abs/2410.99999"
    r = hope.review("UNVERIFIED_SOURCE", answer=f"See {link}", unverified=[link])
    assert r["verified"] and link not in r["fixed_answer"]
    assert hope.review("UNDER_ROUTED")["kind"] == "manual"
    assert hope.apply(":memory:", "UNDER_ROUTED") == 0
