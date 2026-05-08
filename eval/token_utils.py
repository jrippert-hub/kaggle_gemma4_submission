"""
token_utils.py — Lightweight token counting for eval context sizing.

Uses tiktoken (cl100k_base) if available, otherwise falls back to a
word-count approximation (~1.3 tokens per word for English prose).
"""

from typing import List

_encoder = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        try:
            import tiktoken
            _encoder = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            _encoder = "fallback"
    return _encoder


def count_tokens(text: str) -> int:
    enc = _get_encoder()
    if enc == "fallback":
        return int(len(text.split()) * 1.3)
    return len(enc.encode(text))


def count_turns_tokens(turns: List[dict]) -> int:
    """Total tokens across all turns in a conversation."""
    return sum(count_tokens(t.get("content", "")) for t in turns)


def context_size_label(n_tokens: int) -> str:
    """Human-readable label: e.g. '12.4k tokens'."""
    if n_tokens >= 1000:
        return f"{n_tokens / 1000:.1f}k"
    return str(n_tokens)
