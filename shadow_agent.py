import json
import logging
import re
from typing import List, Optional

import httpx

from database import (
    get_all_turns,
    get_anchor_turns,
    get_safety_state,
    save_anchor_turns,
    upsert_safety_state,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a safety analysis agent embedded in a conversational AI system.

Your task is to review the conversation arc below and reason about whether there is any trajectory toward psychological distress, crisis, or policy-relevant risk.

IMPORTANT GUIDELINES:
- Reason at the arc level — patterns across the whole conversation, not individual messages.
- Most conversations are benign. If you see no concerning arc, set activated to false and risk_score below 0.3.
- Only flag escalating or high_risk trajectories when there is clear multi-turn evidence.
- Consider: thematic drift, emotional tone shifts, increasing self-reference, linguistic markers of distress, hopelessness language, withdrawal signals.
- anchor_turns should list turn_numbers of specific turns that are most evidentially significant. Leave empty if none.

You MUST respond with ONLY a valid JSON object — no markdown, no code fences, no commentary:

{
  "turn_count": <int — total user turns in session>,
  "baseline": "<tone and topic of opening turns>",
  "observed_themes": ["<theme>"],
  "trajectory": "<stable | mild_concern | escalating | high_risk>",
  "risk_score": <float 0.0–1.0>,
  "trend_direction": "<stable | escalating | deescalating>",
  "confidence": <float 0.0–1.0>,
  "anchor_turns": [<turn_number int>, ...],
  "last_reasoning": "<chain-of-thought summary — what you observed and why you scored it this way>",
  "recommended_action": "<none | tone_shift | resource_injection>"
}"""


def _build_prompt(safety_state: dict, anchor_turns: List[dict], recent_turns: List[dict]) -> str:
    parts = []

    if safety_state.get("turn_count", 0) > 0:
        parts.append(f"[SAFETY STATE]\n{json.dumps(safety_state, indent=2)}")
    else:
        parts.append("[SAFETY STATE]\nNo prior activations — this is the first analysis.")

    if anchor_turns:
        flagged = "\n".join(f"Turn {a['turn_number']}: {a['content']}" for a in anchor_turns)
        parts.append(f"[FLAGGED MOMENTS]\n{flagged}")
    else:
        parts.append("[FLAGGED MOMENTS]\nNone recorded.")

    turns_text = "\n".join(
        f"[{t['role'].upper()} | turn {t['turn_number']}]: {t['content']}"
        for t in recent_turns
    )
    parts.append(f"[RECENT TURNS]\n{turns_text}")

    return "\n\n".join(parts)


def _parse_response(raw: str) -> Optional[dict]:
    # Strip markdown fences if the model wraps output despite instructions
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Last resort: grab the first {...} block
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


async def run_shadow_agent(session_id: str, ollama_base_url: str, model: str) -> None:
    logger.info("[shadow] activating — session=%s", session_id)

    safety_state = await get_safety_state(session_id)
    all_turns = await get_all_turns(session_id)
    anchor_turns = await get_anchor_turns(session_id)
    recent_turns = all_turns[-20:]
    user_turn_count = sum(1 for t in all_turns if t["role"] == "user")

    prompt = _build_prompt(safety_state, anchor_turns, recent_turns)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{ollama_base_url}/api/chat",
                json={"model": model, "messages": messages, "stream": False},
            )
            resp.raise_for_status()
            raw = resp.json()["message"]["content"]
    except Exception:
        logger.exception("[shadow] Ollama call failed — session=%s", session_id)
        return

    new_state = _parse_response(raw)
    if new_state is None:
        logger.error("[shadow] JSON parse failed — session=%s raw=%s", session_id, raw[:300])
        return

    new_state["turn_count"] = user_turn_count

    await upsert_safety_state(session_id, new_state)

    new_anchors = new_state.get("anchor_turns", [])
    if new_anchors:
        await save_anchor_turns(session_id, new_anchors, all_turns)

    logger.info(
        "[shadow] complete — session=%s trajectory=%s risk=%.2f action=%s anchors=%s",
        session_id,
        new_state.get("trajectory", "?"),
        float(new_state.get("risk_score", 0.0)),
        new_state.get("recommended_action", "?"),
        new_anchors,
    )
