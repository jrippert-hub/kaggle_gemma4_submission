"""
baselines.py — Baseline classifiers for comparison against Caspar.

PerPromptClassifier  — scores each turn independently using keyword heuristics
                       and simple negative-sentiment word lists. No ML model,
                       no cross-turn context. Represents naive per-message filtering.

TruncatingClassifier — same scorer, but explicitly truncates to the first 512
                       tokens before scoring, simulating production classifiers
                       that can't handle long context and miss late-appearing harms.

Both baselines are intentionally simple. The goal is to demonstrate what
rule-based and window-limited classifiers can and cannot detect, vs Caspar's
full arc-level reasoning with Gemma 4.
"""

import re
from typing import List, Optional

MAX_TOKENS = 512          # truncating classifier's hard cap (~375 words)
FLAG_THRESHOLD = 0.55     # score threshold to flag a turn


# ---------------------------------------------------------------------------
# Keyword lexicons
# ---------------------------------------------------------------------------

# High-weight crisis / hopelessness language
_CRISIS_PATTERNS = [
    r"\bsuicid\w*\b",
    r"\bkill\s+myself\b",
    r"\bend\s+(it|my\s+life|everything)\b",
    r"\bwant\s+to\s+die\b",
    r"\bbetter\s+off\s+(dead|without\s+me)\b",
    r"\bno\s+(point|reason|purpose)\s+(anymore|left|in\s+living)\b",
    r"\bcan'?t\s+(go\s+on|continue|do\s+this\s+anymore)\b",
    r"\bgive\s+up\b",
    r"\bgiving\s+up\b",
    r"\bself.harm\b",
    r"\bhurt\s+myself\b",
]

# Medium-weight distress signals
_DISTRESS_PATTERNS = [
    r"\bhopeless\b",
    r"\bworthless\b",
    r"\buseless\b",
    r"\bburden\b",
    r"\bnobody\s+(cares?|loves?\s+me)\b",
    r"\bdon'?t\s+(care|matter)\s+anymore\b",
    r"\balone\b",
    r"\bempty\b",
    r"\bnumb\b",
    r"\bexhausted\b",
    r"\bdesperate\b",
    r"\btrapped\b",
    r"\bpointless\b",
    r"\bmiserable\b",
    r"\bfall\s+apart\b",
    r"\bfalling\s+apart\b",
    r"\bcan'?t\s+(sleep|eat|function|cope|concentrate)\b",
    r"\bstopped\s+(caring|trying|bothering)\b",
    r"\bgiven\s+up\b",
]

# Low-weight withdrawal / negative framing signals
_WITHDRAWAL_PATTERNS = [
    r"\bdon'?t\s+see\s+the\s+point\b",
    r"\bwhat'?s\s+the\s+point\b",
    r"\bpull\s+(away|back)\b",
    r"\bpulling\s+(away|back)\b",
    r"\bwithdr\w+\b",
    r"\bisolat\w+\b",
    r"\bdisconnect\w*\b",
    r"\bstop\s+texting\b",
    r"\bno\s+one\s+(understands?|gets?\s+it)\b",
    r"\bnever\s+(gets?\s+better|improves?)\b",
    r"\balways\s+(like\s+this|been\s+this\s+way)\b",
    r"\bcouldn'?t\b",
    r"\bwouldn'?t\b",
    r"\bdidn'?t\b.*\bever\b",
]

_CRISIS_RE     = re.compile("|".join(_CRISIS_PATTERNS),     re.IGNORECASE)
_DISTRESS_RE   = re.compile("|".join(_DISTRESS_PATTERNS),   re.IGNORECASE)
_WITHDRAWAL_RE = re.compile("|".join(_WITHDRAWAL_PATTERNS), re.IGNORECASE)


def _keyword_score(text: str) -> float:
    """
    Score text 0.0–1.0 using weighted keyword matching.
    Crisis language: 0.60 per match (capped at 0.90)
    Distress language: 0.20 per match (capped at 0.60)
    Withdrawal language: 0.08 per match (capped at 0.30)
    Scores are additive then clamped to [0, 1].
    """
    crisis_hits     = len(_CRISIS_RE.findall(text))
    distress_hits   = len(_DISTRESS_RE.findall(text))
    withdrawal_hits = len(_WITHDRAWAL_RE.findall(text))

    score = (
        min(crisis_hits     * 0.60, 0.90)
        + min(distress_hits   * 0.20, 0.60)
        + min(withdrawal_hits * 0.08, 0.30)
    )
    return min(score, 1.0)


def _truncate_to_tokens(text: str, max_tokens: int = MAX_TOKENS) -> str:
    """Approximate token truncation: ~0.75 words per token."""
    words = text.split()
    return " ".join(words[: int(max_tokens * 0.75)])


# ---------------------------------------------------------------------------
# PerPromptClassifier
# ---------------------------------------------------------------------------

class PerPromptClassifier:
    """
    Scores each user turn independently with no cross-turn context.
    For long documents, chunks the text and returns the max chunk score —
    flagging if ANY chunk exceeds the threshold.

    Represents naive per-message keyword filtering.
    No memory, no arc reasoning, no context window advantage.
    """

    def score(self, text: str) -> float:
        return _keyword_score(text)

    def flag(self, text: str) -> bool:
        return self.score(text) >= FLAG_THRESHOLD

    def first_flag_turn(self, turns: List[dict]) -> Optional[int]:
        """Return 1-indexed user turn number of first flag, or None."""
        user_turn = 0
        for t in turns:
            if t["role"] != "user":
                continue
            user_turn += 1
            if self.flag(t["content"]):
                return user_turn
        return None

    def score_document(self, text: str, chunk_words: int = 300) -> float:
        """Score a long document by chunking and returning the max chunk score."""
        words = text.split()
        chunks = [
            " ".join(words[i: i + chunk_words])
            for i in range(0, len(words), chunk_words)
        ]
        if not chunks:
            return 0.0
        return max(self.score(chunk) for chunk in chunks)

    def flag_document(self, text: str) -> bool:
        return self.score_document(text) >= FLAG_THRESHOLD


# ---------------------------------------------------------------------------
# TruncatingClassifier
# ---------------------------------------------------------------------------

class TruncatingClassifier:
    """
    Truncates input to the first 512 tokens before scoring.

    Simulates production classifiers with a fixed context window that only
    ever see the beginning of a long document. For LC Escalating examples
    where distress content appears after word ~4000, this classifier reads
    only benign preamble and returns a near-zero score — the core failure
    mode Caspar is designed to address.
    """

    def score(self, text: str) -> float:
        truncated = _truncate_to_tokens(text, MAX_TOKENS)
        return _keyword_score(truncated)

    def flag(self, text: str) -> bool:
        return self.score(text) >= FLAG_THRESHOLD

    def first_flag_turn(self, turns: List[dict]) -> Optional[int]:
        """Return 1-indexed user turn number of first flag, or None."""
        user_turn = 0
        for t in turns:
            if t["role"] != "user":
                continue
            user_turn += 1
            if self.flag(t["content"]):
                return user_turn
        return None

    def score_document(self, text: str) -> float:
        """Score only the first 512 tokens — ignores everything after."""
        return self.score(text)

    def flag_document(self, text: str) -> bool:
        return self.score_document(text) >= FLAG_THRESHOLD
