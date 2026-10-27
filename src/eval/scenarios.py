"""Scripted conversations, each with the outcome a good router should produce.

Four kinds, mirroring how the agent is judged:
  success      - the agent should get every graded turn right.
  failure      - known limitations of the keyword reader. Each lists the reason codes we
                 expect it to hit, so the report shows *how* it fails instead of hiding it.
  heldout      - written after the signal readers were tuned, in phrasings nothing was tested
                 on, and not edited after the first run. Graded like success; failures stay.
  false_claim  - a model that makes things up; the checker has to catch it.
"""
from dataclasses import dataclass, field
from typing import Optional

# Reason codes. Every finding in the report carries one of these.
MISSED_SIGNAL = "MISSED_SIGNAL"          # a real habit never reached the observed profile
FALSE_SIGNAL = "FALSE_SIGNAL"            # a habit was learned that the user does not have
UNDER_ROUTED = "UNDER_ROUTED"            # a smaller tier than the situation needed
OVER_ROUTED = "OVER_ROUTED"              # a bigger tier than the situation needed (wasted spend)
IGNORED_FORMAT = "IGNORED_FORMAT"        # the model ignored the answer format it was given
MISSING_SUGGESTION = "MISSING_SUGGESTION"
BACKEND_FAILED = "BACKEND_FAILED"        # the model call itself failed
FALSE_CLAIM = "FALSE_CLAIM"              # one of the agent's own explanations contradicts the database
UNVERIFIED_SOURCE = "UNVERIFIED_SOURCE"  # an answer cites a link or source nothing backs up

CODES = {
    MISSED_SIGNAL: "A real habit never reached the observed profile",
    FALSE_SIGNAL: "Learned a habit the user does not have",
    UNDER_ROUTED: "Used a smaller model than the situation needed",
    OVER_ROUTED: "Used a bigger model than the situation needed (wasted spend)",
    IGNORED_FORMAT: "The model ignored the answer format it was given",
    MISSING_SUGGESTION: "Did not offer to update the preference",
    BACKEND_FAILED: "The model call failed",
    FALSE_CLAIM: "The agent's explanation contradicts its own records",
    UNVERIFIED_SOURCE: "An answer cites a link or source that nothing backs up",
}


@dataclass
class Turn:
    say: str
    ideal: Optional[str] = None                 # tier a good router uses here; None = not graded
    observed_after: dict = field(default_factory=dict)   # observed profile values expected after this turn
    expect_format: bool = False                 # an Ultra answer must open with a TL;DR line
    expect_suggestion: Optional[bool] = None    # whether "update your preference?" should be offered
    action: Optional[tuple] = None              # ("accept", dimension) done before this turn


@dataclass
class Scenario:
    id: str
    title: str
    kind: str                                   # success | failure | false_claim
    why: str                                    # what this case demonstrates, in one sentence
    turns: list
    stated: dict = field(default_factory=dict)
    expect_codes: frozenset = frozenset()       # reason codes this scenario is expected to produce
    model: str = "real"                         # "real" = whatever backend runs; "lying" = invents sources


def _neutral(n):
    return [Turn(f"Explain topic number {i} to me") for i in range(n)]


SCENARIOS = [
    # ---- success ---------------------------------------------------------------
    Scenario("aligned_user", "Says short, asks for short", "success",
             "No dissonance: everything stays on the cheapest tier.",
             stated={"length": "concise"},
             turns=[Turn("Explain recursion, keep it short", ideal="nano"),
                    Turn("What is a hash map? Be brief.", ideal="nano", observed_after={"length": "concise"}),
                    Turn("Summarize how DNS works, shorter please", ideal="nano")]),

    Scenario("classic_contradiction", "Says short, keeps asking for detail", "success",
             "The core case: a real contradiction escalates to Ultra, which reconciles both sides.",
             stated={"length": "concise"},
             turns=[Turn("Explain attention, in more detail please", ideal="nano"),
                    Turn("Can you elaborate on the softmax?", ideal="ultra", expect_format=True,
                         observed_after={"length": "detailed"}),
                    Turn("Go deeper on multi-head attention", ideal="ultra", expect_format=True,
                         expect_suggestion=True)]),

    Scenario("emerging_preference", "Never said anything, but keeps asking for detail", "success",
             "A habit nobody declared gets a middle tier and an offer to save it.",
             turns=[Turn("Elaborate on tokenizers", ideal="nano"),
                    Turn("Go deeper on BPE", ideal="super", expect_suggestion=True,
                         observed_after={"length": "detailed"})]),

    Scenario("mixed_topics", "A normal chat: unrelated questions, no style requests", "success",
             "Topic changes are not preference changes: nothing is learned and nothing escalates.",
             turns=[Turn("What's the capital of France?", ideal="nano"),
                    Turn("Explain how a TCP handshake works", ideal="nano"),
                    Turn("Write a haiku about autumn rain", ideal="nano"),
                    Turn("Why does ice float on water?", ideal="nano",
                         observed_after={"length": "balanced", "tone": "balanced", "structure": "balanced"}),
                    Turn("Thanks!", ideal="nano")]),

    Scenario("quick_messages", "Thanks and hi in the middle of a contradiction", "success",
             "Quick messages stay cheap and do not count as more evidence.",
             stated={"length": "concise"},
             turns=[Turn("Explain transformers in more detail", ideal="nano"),
                    Turn("Elaborate on the encoder", ideal="ultra", expect_format=True),
                    Turn("thanks!", ideal="nano"),
                    Turn("hi", ideal="nano"),
                    Turn("Why does layer norm help?", ideal="ultra", expect_format=True)]),

    Scenario("one_off_request", "One-off instruction", "success",
             "'Less formal' is obeyed this once and not learned as a habit.",
             stated={"tone": "formal"},
             turns=[Turn("Explain recursion, make it less formal", ideal="nano", observed_after={"tone": "balanced"}),
                    Turn("What is a closure?", ideal="nano")]),

    Scenario("accepting_resolves", "Accepts the suggestion", "success",
             "Once the user adopts the habit, the contradiction is closed and the cost drops back.",
             stated={"length": "concise"},
             turns=[Turn("Explain RNNs in more detail", ideal="nano"),
                    Turn("Elaborate on the vanishing gradient", ideal="ultra", expect_format=True,
                         expect_suggestion=True),
                    Turn("Explain GRUs", ideal="nano", action=("accept", "length"))]),

    Scenario("habit_fades", "A habit that stops showing up", "success",
             "Old evidence fades: after enough ordinary messages the agent stops acting on it.",
             turns=[Turn("Elaborate on tokenizers", ideal="nano"),
                    Turn("Go deeper on BPE", ideal="super")] + _neutral(11)
                   + [Turn("Explain the last topic to me", ideal="nano", observed_after={"length": "balanced"})]),

    # ---- failure: known limitations -----------------------------------------------
    Scenario("implicit_complaint", "Wants more detail but never says the words", "failure",
             "The detector only reads explicit phrasings, so 'I'm still lost' teaches it nothing.",
             stated={"length": "concise"},
             turns=[Turn("Explain gradient descent"),
                    Turn("I still don't get it"),
                    Turn("Still confused honestly"),
                    Turn("Okay I'm lost, what does the learning rate do?", ideal="ultra",
                         observed_after={"length": "detailed"})],
             expect_codes=frozenset({MISSED_SIGNAL, UNDER_ROUTED})),

    Scenario("other_language", "Asks for detail in Spanish", "failure",
             "The keyword detector is English-only.",
             stated={"length": "concise"},
             turns=[Turn("Explica la atención, con más detalle por favor"),
                    Turn("Profundiza en el softmax, explícalo más a fondo", ideal="ultra",
                         observed_after={"length": "detailed"})],
             expect_codes=frozenset({MISSED_SIGNAL, UNDER_ROUTED})),

    # Expectation corrected 2026-10-28: "don't make it shorter ... in full" asks for detail, so
    # "detailed" is right (it was "learn nothing"). Found when the Nano reader got it right.
    Scenario("negation", "'Don't make it shorter'", "failure",
             "A keyword match can't read negation, so it learns the opposite of what was meant.",
             stated={"length": "detailed"},
             turns=[Turn("Don't make it shorter, explain RNNs in full"),
                    Turn("Again, don't make it shorter. What is an LSTM?", ideal="nano",
                         observed_after={"length": "detailed"})],
             expect_codes=frozenset({FALSE_SIGNAL, OVER_ROUTED})),

    Scenario("quoted_phrase", "Quoting someone else's request", "failure",
             "Words inside quotes are not the user's own request, but the detector can't tell.",
             stated={"length": "detailed"},
             turns=[Turn('My manager keeps telling me to "keep it short". Explain Kafka partitions.'),
                    Turn('He said "make it shorter" again, but I like depth. What is a consumer group?',
                         ideal="nano", observed_after={"length": "balanced"})],
             expect_codes=frozenset({FALSE_SIGNAL, OVER_ROUTED})),

    # ---- held-out: written 2026-10-28 after tuning, not edited after the first run ----
    Scenario("ho_wall_of_text", "Says detailed, complains it's too much", "heldout",
             "A complaint about length in everyday words, not a keyword.",
             stated={"length": "detailed"},
             turns=[Turn("That was a wall of text. What's a CPU cache?"),
                    Turn("Again way too much to read. What is RAM?", ideal="ultra",
                         observed_after={"length": "concise"})]),

    Scenario("ho_spell_it_out", "Says concise, wants every step", "heldout",
             "Asking for depth without the word 'detail'.",
             stated={"length": "concise"},
             turns=[Turn("Walk me through how HTTPS works, every step"),
                    Turn("Spell out how certificates are checked, leave nothing out", ideal="ultra",
                         observed_after={"length": "detailed"})]),

    Scenario("ho_keep_it_chill", "Says formal, talks casual", "heldout",
             "A tone request in slang.",
             stated={"tone": "formal"},
             turns=[Turn("explain closures but keep it chill pls"),
                    Turn("keep it chill again: what is hoisting", ideal="ultra",
                         observed_after={"tone": "casual"})]),

    Scenario("ho_receipts", "Says no sources, asks where claims come from", "heldout",
             "Wanting sources, phrased two different ways.",
             stated={"sourcing": "minimal"},
             turns=[Turn("Explain how vaccines train the immune system, and show me where that comes from"),
                    Turn("What causes inflation? Back it with references", ideal="ultra",
                         observed_after={"sourcing": "full"})]),

    Scenario("ho_have_another_go", "Keeps sending drafts back", "heldout",
             "Rejecting answers without saying 'try again'.",
             turns=[Turn("Explain recursion with an example"),
                    Turn("Hmm, that misses the point, explain it differently"),
                    Turn("Still not right, have another go at it", ideal="super",
                         observed_after={"revision": "rejects"})]),

    Scenario("ho_small_talk", "Small talk around a real question", "heldout",
             "Chit-chat is not a preference.",
             turns=[Turn("haha nice", ideal="nano"),
                    Turn("ok what's the difference between a list and a tuple?", ideal="nano"),
                    Turn("cool thanks", ideal="nano",
                         observed_after={"length": "balanced", "tone": "balanced", "revision": "balanced"})]),

    Scenario("ho_topic_words", "Style words that are really topics", "heldout",
             "'Detailed balance' and 'formal grammar' are subjects, not requests.",
             stated={"length": "concise"},
             turns=[Turn("Explain the detailed balance condition in Markov chains", ideal="nano"),
                    Turn("What is a formal grammar?", ideal="nano",
                         observed_after={"length": "balanced", "tone": "balanced"})]),

    # ---- false claims -------------------------------------------------------------
    Scenario("invented_source", "A model that invents a citation", "false_claim",
             "The model attaches a link nothing backs up. The check has to flag it, not pass it along.",
             turns=[Turn("Who proposed the transformer architecture?", ideal="nano")],
             expect_codes=frozenset({UNVERIFIED_SOURCE}), model="lying"),
]

BY_ID = {s.id: s for s in SCENARIOS}
