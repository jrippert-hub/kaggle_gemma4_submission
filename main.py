import asyncio
import logging
import os
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import List, Optional

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from database import (
    get_next_turn_number,
    get_recent_turns,
    get_safety_state,
    init_db,
    save_turn,
    upsert_safety_state,
)
from shadow_agent import run_shadow_agent
from triggers import COMPOSITE_THRESHOLD, INTERVAL_TURNS, evaluate_triggers, load_sentiment_model

_TEST_MODE = os.getenv("TEST_MODE", "false").lower() == "true"
# EVAL_MODE disables background auto-triggers so force-evaluate has exclusive
# Ollama access during eval runs. Set with EVAL_MODE=true alongside TEST_MODE.
_EVAL_MODE = os.getenv("EVAL_MODE", "false").lower() == "true"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "gemma4:e4b")

BASE_SYSTEM_PROMPT = "You are a helpful, thoughtful assistant."
TONE_SHIFT_ADDENDUM = (
    " Note: this user may be experiencing distress. Prioritise warm, grounded responses."
)
RESOURCE_INJECTION_ADDENDUM = (
    " If it feels appropriate, you may gently mention that support is available —"
    " such as speaking with a trusted person or contacting a helpline."
)

# ---------------------------------------------------------------------------
# Logging — console + rotating file
# ---------------------------------------------------------------------------

Path("logs").mkdir(exist_ok=True)

_file_handler = RotatingFileHandler(
    "logs/app.log", maxBytes=10 * 1024 * 1024, backupCount=3
)
_file_handler.setFormatter(
    logging.Formatter("%(asctime)s %(levelname)s %(name)s — %(message)s")
)
logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.StreamHandler(), _file_handler],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shadow agent queue
# ---------------------------------------------------------------------------

_shadow_queue: asyncio.Queue = asyncio.Queue()


async def _shadow_worker() -> None:
    while True:
        session_id = await _shadow_queue.get()
        try:
            await run_shadow_agent(session_id, OLLAMA_BASE_URL, OLLAMA_CHAT_MODEL)
        except Exception:
            logger.exception("[shadow_worker] unhandled error — session=%s", session_id)
        finally:
            _shadow_queue.task_done()


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    load_sentiment_model()
    worker = asyncio.create_task(_shadow_worker())
    if _TEST_MODE:
        logger.warning(
            "TEST_MODE enabled — interval=%d turns, composite threshold=%.2f",
            INTERVAL_TURNS, COMPOSITE_THRESHOLD,
        )
    if _EVAL_MODE:
        logger.warning("EVAL_MODE enabled — background triggers disabled, use force-evaluate")
    logger.info("App started — model=%s ollama=%s", OLLAMA_CHAT_MODEL, OLLAMA_BASE_URL)
    yield
    worker.cancel()
    try:
        await worker
    except asyncio.CancelledError:
        pass


app = FastAPI(title="Gemma 4 Safety Chat", version="2.0.0", lifespan=lifespan)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_system_prompt(recommended_action: str) -> str:
    prompt = BASE_SYSTEM_PROMPT
    if recommended_action == "tone_shift":
        prompt += TONE_SHIFT_ADDENDUM
    elif recommended_action == "resource_injection":
        prompt += TONE_SHIFT_ADDENDUM + RESOURCE_INJECTION_ADDENDUM
    return prompt


async def _ollama_chat(messages: List[dict]) -> str:
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={"model": OLLAMA_CHAT_MODEL, "messages": messages, "stream": False},
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]


# ---------------------------------------------------------------------------
# Background task
# ---------------------------------------------------------------------------

async def _run_trigger_evaluation(session_id: str) -> None:
    score = await evaluate_triggers(session_id)
    if score > COMPOSITE_THRESHOLD:
        logger.info(
            "[main] composite=%.2f > threshold=%.2f — enqueuing shadow agent for session=%s",
            score, COMPOSITE_THRESHOLD, session_id,
        )
        await _shadow_queue.put(session_id)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    turn_count: int
    recommended_action: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, background_tasks: BackgroundTasks) -> ChatResponse:
    # Load current safety state to inject into system prompt
    safety_state = await get_safety_state(req.session_id)
    recommended_action = safety_state.get("recommended_action", "none")
    system_prompt = _build_system_prompt(recommended_action)

    # Load last 20 turns for context
    history = await get_recent_turns(req.session_id, limit=20)

    # Save user turn
    turn_number = await get_next_turn_number(req.session_id)
    await save_turn(req.session_id, turn_number, "user", req.message)

    # Build Ollama messages
    messages = [{"role": "system", "content": system_prompt}]
    messages += [{"role": t["role"], "content": t["content"]} for t in history]
    messages.append({"role": "user", "content": req.message})

    # Call Ollama
    try:
        reply = await _ollama_chat(messages)
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=502, detail=f"Ollama error: {exc.response.text}")
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=f"Ollama unreachable: {exc}")

    # Save assistant turn
    await save_turn(req.session_id, turn_number + 1, "assistant", reply)

    # Count user turns for response
    user_turn_count = sum(1 for t in history if t["role"] == "user") + 1

    # Fire trigger evaluation after response is sent
    # Skip in EVAL_MODE — force-evaluate handles shadow agent calls exclusively
    if not _EVAL_MODE:
        background_tasks.add_task(_run_trigger_evaluation, req.session_id)

    return ChatResponse(
        session_id=req.session_id,
        reply=reply,
        turn_count=user_turn_count,
        recommended_action=recommended_action,
    )


@app.get("/risk-state/{session_id}")
async def risk_state(session_id: str) -> dict:
    state = await get_safety_state(session_id)
    if state.get("turn_count", 0) == 0:
        raise HTTPException(status_code=404, detail="No safety state recorded for this session yet")
    return state


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model": OLLAMA_CHAT_MODEL, "test_mode": _TEST_MODE}


@app.post("/admin/force-evaluate/{session_id}")
async def force_evaluate(session_id: str) -> dict:
    """
    Directly invoke the shadow agent, bypassing trigger thresholds.
    Used by the eval harness for long-context documents where a single
    message would not normally cross the trigger threshold.
    """
    await run_shadow_agent(session_id, OLLAMA_BASE_URL, OLLAMA_CHAT_MODEL)
    return await get_safety_state(session_id)


@app.get("/", response_class=HTMLResponse)
async def chat_ui() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Gemma 4 Safety Chat</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #0f1117; color: #e0e0e0; display: flex; height: 100vh; }
  #sidebar { width: 300px; background: #1a1d27; border-right: 1px solid #2e3047; display: flex; flex-direction: column; padding: 16px; gap: 12px; overflow-y: auto; }
  #sidebar h2 { font-size: 13px; text-transform: uppercase; letter-spacing: .08em; color: #888; }
  #session-id { background: #252836; border: 1px solid #3a3d52; border-radius: 6px; padding: 8px 10px; color: #e0e0e0; font-size: 13px; width: 100%; }
  #risk-panel { background: #252836; border-radius: 8px; padding: 12px; font-size: 12px; line-height: 1.8; }
  .risk-row { display: flex; justify-content: space-between; }
  .risk-label { color: #888; }
  .risk-val { font-weight: 600; }
  .badge { padding: 2px 8px; border-radius: 99px; font-size: 11px; font-weight: 700; }
  .stable      { background: #1a3a2a; color: #4caf82; }
  .mild_concern{ background: #3a3010; color: #f0b429; }
  .escalating  { background: #3a2010; color: #f07829; }
  .high_risk   { background: #3a1010; color: #f04040; }
  .none        { background: #1a2a3a; color: #4a9eff; }
  .tone_shift  { background: #3a2a10; color: #f0c040; }
  .resource_injection { background: #3a1010; color: #ff6060; }
  #main { flex: 1; display: flex; flex-direction: column; }
  #messages { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 12px; }
  .msg { max-width: 75%; padding: 10px 14px; border-radius: 12px; font-size: 14px; line-height: 1.5; white-space: pre-wrap; word-break: break-word; }
  .user { align-self: flex-end; background: #3a4fd4; color: #fff; border-bottom-right-radius: 3px; }
  .assistant { align-self: flex-start; background: #252836; color: #e0e0e0; border-bottom-left-radius: 3px; }
  .thinking { opacity: .5; font-style: italic; }
  #input-bar { padding: 16px; border-top: 1px solid #2e3047; display: flex; gap: 10px; }
  #msg-input { flex: 1; background: #252836; border: 1px solid #3a3d52; border-radius: 8px; padding: 10px 14px; color: #e0e0e0; font-size: 14px; resize: none; height: 44px; }
  #msg-input:focus { outline: none; border-color: #5a6eff; }
  #send-btn { background: #3a4fd4; color: #fff; border: none; border-radius: 8px; padding: 0 20px; font-size: 14px; cursor: pointer; font-weight: 600; }
  #send-btn:disabled { opacity: .4; cursor: not-allowed; }
  #turn-count { font-size: 11px; color: #555; text-align: right; margin-top: 4px; }
</style>
</head>
<body>
<div id="sidebar">
  <h2>Session</h2>
  <input id="session-id" type="text" placeholder="session ID" value="test-session-1" />
  <div id="turn-count">turns: 0</div>

  <h2 style="margin-top:8px">Risk State</h2>
  <div id="risk-panel">
    <div class="risk-row"><span class="risk-label">trajectory</span><span class="risk-val" id="r-trajectory">—</span></div>
    <div class="risk-row"><span class="risk-label">risk score</span><span class="risk-val" id="r-score">—</span></div>
    <div class="risk-row"><span class="risk-label">trend</span><span class="risk-val" id="r-trend">—</span></div>
    <div class="risk-row"><span class="risk-label">confidence</span><span class="risk-val" id="r-confidence">—</span></div>
    <div class="risk-row"><span class="risk-label">action</span><span class="risk-val" id="r-action">—</span></div>
    <div style="margin-top:8px;color:#888;font-size:11px" id="r-reasoning">No activations yet.</div>
  </div>

  <h2 style="margin-top:8px">Themes</h2>
  <div id="r-themes" style="font-size:12px;color:#aaa;line-height:1.8">—</div>
</div>

<div id="main">
  <div id="messages"></div>
  <div id="input-bar">
    <textarea id="msg-input" placeholder="Type a message… (Enter to send, Shift+Enter for newline)"></textarea>
    <button id="send-btn">Send</button>
  </div>
</div>

<script>
const messagesEl = document.getElementById('messages');
const inputEl    = document.getElementById('msg-input');
const sendBtn    = document.getElementById('send-btn');
const sessionEl  = document.getElementById('session-id');
const turnEl     = document.getElementById('turn-count');

function addMsg(role, text) {
  const d = document.createElement('div');
  d.className = 'msg ' + role;
  d.textContent = text;
  messagesEl.appendChild(d);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return d;
}

function badge(val, cls) {
  return `<span class="badge ${cls}">${val}</span>`;
}

async function pollRisk(sessionId) {
  try {
    const r = await fetch(`/risk-state/${encodeURIComponent(sessionId)}`);
    if (!r.ok) return;
    const s = await r.json();
    document.getElementById('r-trajectory').innerHTML  = badge(s.trajectory, s.trajectory);
    document.getElementById('r-score').textContent     = s.risk_score.toFixed(2);
    document.getElementById('r-trend').textContent     = s.trend_direction;
    document.getElementById('r-confidence').textContent= s.confidence.toFixed(2);
    document.getElementById('r-action').innerHTML      = badge(s.recommended_action, s.recommended_action);
    document.getElementById('r-reasoning').textContent = s.last_reasoning || '—';
    document.getElementById('r-themes').innerHTML      = (s.observed_themes || []).map(t => `• ${t}`).join('<br>') || '—';
  } catch(_) {}
}

async function send() {
  const text = inputEl.value.trim();
  const sessionId = sessionEl.value.trim() || 'default';
  if (!text) return;

  inputEl.value = '';
  sendBtn.disabled = true;
  addMsg('user', text);
  const thinking = addMsg('assistant', '…');
  thinking.classList.add('thinking');

  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId, message: text }),
    });
    const data = await res.json();
    thinking.classList.remove('thinking');
    thinking.textContent = res.ok ? data.reply : (data.detail || 'Error');
    if (res.ok) {
      turnEl.textContent = `turns: ${data.turn_count}`;
      // Poll risk state — shadow agent may fire in background, check a few times
      pollRisk(sessionId);
      setTimeout(() => pollRisk(sessionId), 3000);
      setTimeout(() => pollRisk(sessionId), 8000);
    }
  } catch(e) {
    thinking.textContent = 'Network error: ' + e.message;
  }
  sendBtn.disabled = false;
  inputEl.focus();
}

sendBtn.addEventListener('click', send);
inputEl.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});
</script>
</body>
</html>"""
