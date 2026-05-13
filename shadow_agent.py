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

SYSTEM_PROMPT = """You are a safety analysis agent. Analyse the conversation below and output ONLY a JSON object with exactly these fields. No prose, no markdown, no code fences, no disclaimer, no preamble — raw JSON only. Start your response with { and end with }.

OUTPUT FORMAT (copy these exact field names):
{
  "trajectory": "stable",
  "risk_score": 0.0,
  "trend_direction": "stable",
  "confidence": 0.5,
  "recommended_action": "none",
  "arc_summary": "one sentence summary",
  "last_reasoning": "your reasoning here",
  "action_rationale": "why you chose this action",
  "confidence_rationale": "why you are this confident",
  "baseline": {"summary": "", "tone": "neutral", "topics": []},
  "observed_themes": [],
  "anchor_turns": []
}

FIELD RULES:
- trajectory: must be exactly one of: stable | mild_concern | escalating | high_risk
- risk_score: float between 0.0 and 1.0
- trend_direction: must be exactly one of: stable | escalating | deescalating
- recommended_action: must be exactly one of: none | tone_shift | resource_injection
- observed_themes: list of {theme, first_seen_turn, evidence} — quote directly from the conversation
- anchor_turns: list of {turn, quote, reason} — pivotal moments only

SCORING GUIDE:
- 0.0–0.2: benign, everyday topics, no distress signals
- 0.3–0.5: mild concern — stress or sadness present but manageable
- 0.6–0.8: escalating — hopelessness, withdrawal, or persistent negative themes
- 0.8–1.0: high risk — suicidal ideation, self-harm, crisis language

Be conservative. Default to stable unless you see clear, consistent arc-level evidence of distress."""


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
    """Extract and parse the first valid JSON object from raw model output.
    Uses brace-matching so disclaimers or preamble before the { are ignored."""
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()

    # Fast path: entire output is valid JSON
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Brace-match: find the first { and walk to its matching }
    start = cleaned.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i, ch in enumerate(cleaned[start:], start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(cleaned[start : i + 1])
                except json.JSONDecodeError:
                    break

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
