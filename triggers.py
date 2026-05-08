import logging
import os
import re
from typing import List, Optional, Tuple

import numpy as np

from database import get_all_turns, get_user_turn_count

logger = logging.getLogger(__name__)

# Loaded once at startup via load_sentiment_model()
_sentiment_pipeline = None

FIRST_PERSON = re.compile(r"\b(i|me|my|myself|mine)\b", re.IGNORECASE)
PAST_MODALS = re.compile(
    r"\b(couldn't|wouldn't|didn't|shouldn't|can't|won't|don't|wasn't|weren't|hadn't|haven't|isn't)\b",
    re.IGNORECASE,
)

# Composite score weights — must sum to 1.0
W_INTERVAL = 0.15
W_VALENCE = 0.45
W_LINGUISTIC = 0.40

# TEST_MODE=true lowers thresholds so the shadow agent fires after just a few messages
_TEST_MODE = os.getenv("TEST_MODE", "false").lower() == "true"

COMPOSITE_THRESHOLD = 0.05 if _TEST_MODE else 0.4
VALENCE_DRIFT_THRESHOLD = 0.05 if _TEST_MODE else 0.3
INTERVAL_TURNS = 3 if _TEST_MODE else 15


def load_sentiment_model() -> None:
    global _sentiment_pipeline
    try:
        from transformers import pipeline as hf_pipeline
        _sentiment_pipeline = hf_pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            truncation=True,
            max_length=512,
        )
        logger.info("[triggers] sentiment model loaded")
    except Exception:
        logger.warning("[triggers] sentiment model failed to load — valence trigger disabled")


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------

def _negative_score(text: str) -> float:
    """Map sentiment pipeline output to a 0-1 negativity score."""
    if _sentiment_pipeline is None:
        return 0.0
    result = _sentiment_pipeline(text)[0]
    label = result["label"].lower()
    score = result["score"]
    if "negative" in label:
        return score
    if "neutral" in label:
        return 0.15
    return 0.0


def _valence_drift(user_turns: List[str]) -> Tuple[bool, float]:
    """Compare negativity of first-5 vs last-5 user turns."""
    if len(user_turns) < 10:
        return False, 0.0
    baseline = float(np.mean([_negative_score(t) for t in user_turns[:5]]))
    recent = float(np.mean([_negative_score(t) for t in user_turns[-5:]]))
    drift = recent - baseline
    return drift > VALENCE_DRIFT_THRESHOLD, max(0.0, drift)


def _linguistic_flags(user_turns: List[str]) -> float:
    """Return 0-1 score from three linguistic risk signals."""
    if len(user_turns) < 3:
        return 0.0

    sub_scores = []

    # 1. First-person pronoun rate increase (early vs late half)
    if len(user_turns) >= 6:
        mid = len(user_turns) // 2
        early, late = user_turns[:mid], user_turns[mid:]
        rate = lambda turns: np.mean(  # noqa: E731
            [len(FIRST_PERSON.findall(t)) / max(len(t.split()), 1) for t in turns]
        )
        delta = float(rate(late) - rate(early))
        sub_scores.append(min(1.0, max(0.0, delta / 0.10)))

    # 2. Past-tense modal density in last 5 turns
    recent_5 = user_turns[-5:]
    modal_count = sum(len(PAST_MODALS.findall(t)) for t in recent_5)
    sub_scores.append(min(1.0, modal_count / 5.0))

    # 3. Sentence length shortening over last 10 turns
    recent_10 = user_turns[-10:]
    if len(recent_10) >= 4:
        lengths = [len(t.split()) for t in recent_10]
        mid = len(lengths) // 2
        first_half_avg = float(np.mean(lengths[:mid])) or 1.0
        second_half_avg = float(np.mean(lengths[mid:]))
        shortening = (first_half_avg - second_half_avg) / first_half_avg
        sub_scores.append(min(1.0, max(0.0, shortening)))

    return float(np.mean(sub_scores)) if sub_scores else 0.0


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def evaluate_triggers(session_id: str) -> float:
    all_turns = await get_all_turns(session_id)
    user_turn_count = sum(1 for t in all_turns if t["role"] == "user")
    user_texts = [t["content"] for t in all_turns if t["role"] == "user"]

    # Signal 1 — interval
    interval = user_turn_count > 0 and user_turn_count % INTERVAL_TURNS == 0

    # Signal 2 — valence drift
    valence_triggered, drift = _valence_drift(user_texts)

    # Signal 3 — linguistic flags
    ling_score = _linguistic_flags(user_texts)

    composite = (
        W_INTERVAL * float(interval)
        + W_VALENCE * float(valence_triggered)
        + W_LINGUISTIC * ling_score
    )

    logger.info(
        "[triggers] session=%s turns=%d interval=%s valence=%s(drift=%.2f) linguistic=%.2f composite=%.2f",
        session_id, user_turn_count, interval, valence_triggered, drift, ling_score, composite,
    )

    return composite
