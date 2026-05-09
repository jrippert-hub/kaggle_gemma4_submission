import json
import logging
import re
from datetime import datetime, timezone
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

Your task is to review the conversation arc and reason about whether there is a trajectory toward psychological distress, crisis, or policy-relevant risk.

CORE PRINCIPLES:
- Reason at the ARC level — patterns across the whole conversation, not individual messages.
- Most conversations are benign. Default to stable. Only escalate when evidence is clear and consistent.
- Every claim you make must be backed by evidence — a specific quote from a specific turn.
- Be conservative with resource_injection. Reserve it for clear, direct indicators of crisis.
- Your output will be read by humans auditing the system. Write reasoning that would satisfy a clinical reviewer.

WHAT TO LOOK FOR (arc-level signals, not single-turn flags):
- Thematic drift: topics shifting from external (work, others) to internal (self, worthlessness)
- Emotional narrowing: range of expressed emotion shrinking over time
- Withdrawal language: references to disconnecting from people, activities, future
- Hopelessness markers: statements about the future that are directionally pessimistic
- Linguistic compression: responses getting shorter, more clipped over time
- Modal verb shift: increasing use of "couldn't", "wouldn't", "didn't" (past failures framed as permanent)

You MUST respond with ONLY a valid JSON object — no markdown, no code fences, no explanation outside the JSON:

{
  "baseline": {
    "summary": "<1-2 sentences: opening tone and topic of the conversation>",
    "tone": "<single word or short phrase describing opening emotional register>",
    "topics": ["<topic>"]
  },
  "observed_themes": [
    {
      "theme": "<theme name>",
      "first_seen_turn": <turn_number int>,
      "evidence": "<exact quote from that turn that demonstrates this theme>"
    }
  ],
  "trajectory": "<stable | mild_concern | escalating | high_risk>",
  "risk_score": <float 0.0-1.0>,
  "trend_direction": "<stable | escalating | deescalating>",
  "confidence": <float 0.0-1.0>,
  "confidence_rationale": "<what signals agree or conflict — why you are or are not confident>",
  "anchor_turns": [
    {
      "turn": <turn_number int>,
      "quote": "<exact quote from that turn>",
      "reason": "<why this turn is pivotal to the arc — what changed here>"
    }
  ],
  "arc_summary": "<2-4 sentences: narrative of what changed, when, and why it matters — written for a human reviewer>",
  "last_reasoning": "<your full chain of thought: what you noticed, what you weighed, what you concluded and why>",
  "recommended_action": "<none | tone_shift | resource_injection>",
  "action_rationale": "<justify your action choice — or explain why no action is needed despite any concern>"
}"""


def _build_prompt(
    safety_state: dict,
    anchor_turns: List[dict],
    recent_turns: List[dict],
) -> str:
    parts = []

    # Prior safety state — strip activation_history to keep prompt focused
    if safety_state.get("turn_count", 0) > 0:
        display_state = {
            k: v for k, v in safety_state.items()
            if k != "activation_history"
        }
        parts.append(
            f"[SAFETY STATE — your prior analysis]\n{json.dumps(display_state, indent=2)}"
        )
        history = safety_state.get("activation_history", [])
        if history:
            lines = "\n".join(
                f"  #{h['activation']} @ turn {h['turn_count']}: "
                f"trajectory={h['trajectory']}  risk={h['risk_score']:.2f}  action={h['action']}"
                for h in history
            )
            parts.append(f"[ACTIVATION HISTORY]\n{lines}")
    else:
        parts.append("[SAFETY STATE]\nFirst activation — no prior analysis.")

    # Flagged moments from previous activations
    if anchor_turns:
        flagged = "\n".join(
            f"Turn {a['turn_number']}: {a['content']}" for a in anchor_turns
        )
        parts.append(f"[FLAGGED MOMENTS — turns you previously marked as significant]\n{flagged}")
    else:
        parts.append("[FLAGGED MOMENTS]\nNone recorded yet.")

    # Recent conversation turns
    turns_text = "\n".join(
        f"[{t['role'].upper()} | turn {t['turn_number']}]: {t['content']}"
        for t in recent_turns
    )
    parts.append(f"[RECENT TURNS]\n{turns_text}")

    return "\n\n".join(parts)


def _parse_response(raw: str) -> Optional[dict]:
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


async def run_shadow_agent(session_id: str, ollama_base_url: str, model: str) -> None:
    logger.info("[shadow] activating — session=%s", session_id)

    prev_state = await get_safety_state(session_id)
    prev_history = prev_state.get("activation_history", [])

    all_turns = await get_all_turns(session_id)
    anchor_turns = await get_anchor_turns(session_id)
    recent_turns = all_turns[-20:]
    user_turn_count = sum(1 for t in all_turns if t["role"] == "user")

    prompt = _build_prompt(prev_state, anchor_turns, recent_turns)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
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
        logger.error(
            "[shadow] JSON parse failed — session=%s raw=%s", session_id, raw[:300]
        )
        return

    # Activation history is owned by the harness — not regenerated by the model
    new_entry = {
        "activation": len(prev_history) + 1,
        "turn_count": user_turn_count,
        "trajectory": new_state.get("trajectory", "stable"),
        "risk_score": float(new_state.get("risk_score", 0.0)),
        "action": new_state.get("recommended_action", "none"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    new_state["activation_history"] = prev_history + [new_entry]
    new_state["turn_count"] = user_turn_count

    await upsert_safety_state(session_id, new_state)

    # Persist anchor turns — now rich objects {turn, quote, reason}
    new_anchors = new_state.get("anchor_turns", [])
    if new_anchors:
        await save_anchor_turns(session_id, new_anchors, all_turns)

    logger.info(
        "[shadow] complete — session=%s trajectory=%s risk=%.2f "
        "action=%s anchors=%d activation=#%d",
        session_id,
        new_state.get("trajectory", "?"),
        float(new_state.get("risk_score", 0.0)),
        new_state.get("recommended_action", "?"),
        len(new_anchors),
        new_entry["activation"],
    )
