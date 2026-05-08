"""
baselines.py — Baseline classifiers for comparison against Caspar.

PerPromptClassifier  — scores each turn independently (no context).
TruncatingClassifier — explicitly truncates to 512 tokens before scoring,
                       simulating what production classifiers do under long context.
"""

from typing import List, Optional

NEGATIVE_THRESHOLD = 0.7
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MAX_TOKENS = 512


def _load_pipeline():
    from transformers import pipeline
    return pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        truncation=True,
        max_length=MAX_TOKENS,
    )


def _negative_score(result: dict) -> float:
    label = result["label"].lower()
    score = result["score"]
    if "negative" in label:
        return score
    if "neutral" in label:
        return 0.15
    return 0.0


class PerPromptClassifier:
    """
    Scores text turn by turn with no cross-turn context.
    For long-context documents, chunks the text and scores each chunk —
    flagging if ANY chunk exceeds the threshold.
    """

    def __init__(self):
        self._pipe = _load_pipeline()

    def score(self, text: str) -> float:
        result = self._pipe(text)[0]
        return _negative_score(result)

    def flag(self, text: str) -> bool:
        return self.score(text) >= NEGATIVE_THRESHOLD

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

    def score_document(self, text: str, chunk_words: int = 200) -> float:
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
        return self.score_document(text) >= NEGATIVE_THRESHOLD


class TruncatingClassifier:
    """
    Explicitly truncates input to the first 512 tokens before scoring.
    This simulates production classifiers that can't handle long context —
    they only ever see the beginning of a long document, missing late harms.
    """

    def __init__(self):
        self._pipe = _load_pipeline()
        try:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        except Exception:
            self._tokenizer = None

    def _truncate(self, text: str) -> str:
        if self._tokenizer is not None:
            tokens = self._tokenizer.encode(text, truncation=True, max_length=MAX_TOKENS)
            return self._tokenizer.decode(tokens, skip_special_tokens=True)
        # Fallback: rough word-level truncation (~0.75 words per token)
        words = text.split()
        return " ".join(words[: int(MAX_TOKENS * 0.75)])

    def score(self, text: str) -> float:
        truncated = self._truncate(text)
        result = self._pipe(truncated)[0]
        return _negative_score(result)

    def flag(self, text: str) -> bool:
        return self.score(text) >= NEGATIVE_THRESHOLD

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
        return self.score_document(text) >= NEGATIVE_THRESHOLD
