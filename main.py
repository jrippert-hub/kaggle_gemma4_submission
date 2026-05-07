import asyncio
import logging
import os
from contextlib import asynccontextmanager

import httpx
import numpy as np
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

from database import get_session, init_db, update_safety_state, upsert_session

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "gemma4:9b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "gemma4:9b")

# Trigger: fire every N user turns
TURN_COUNT_TRIGGER: int = int(os.getenv("TURN_COUNT_TRIGGER", "10"))
# Trigger: fire when cosine similarity to session centroid drops below this
EMBEDDING_ANOMALY_THRESHOLD: float = float(os.getenv("EMBEDDING_ANOMALY_THRESHOLD", "0.70"))
# Minimum past turns before anomaly trigger is active
ANOMALY_MIN_HISTORY: int = 3

shadow_queue: asyncio.Queue[str] = asyncio.Queue()


# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------

async def _ollama_chat(messages: list[dict]) -> str:
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={"model": OLLAMA_CHAT_MODEL, "messages": messages, "stream": False},
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]


async def _get_embedding(text: str) -> list[float]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": OLLAMA_EMBED_MODEL, "prompt": text},
        )
        resp.raise_for_status()
        return resp.json()["embedding"]


# ---------------------------------------------------------------------------
# Trigger evaluation
# ---------------------------------------------------------------------------

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0.0:
        return 1.0
    return float(np.dot(va, vb) / denom)


async def evaluate_triggers(
    session_id: str,
    turn_count: int,
    message_embedding: list[float],
) -> bool:
    session = await get_session(session_id)
    if session is None:
        return False

    fired = False

    # Trigger 1 — periodic turn-count checkpoint
    if turn_count > 0 and turn_count % TURN_COUNT_TRIGGER == 0:
        logger.info("[triggers] turn-count fired: session=%s turn=%d", session_id, turn_count)
        fired = True

    # Trigger 2 — embedding anomaly vs. running session centroid
    past_embeddings = [
        t["embedding"] for t in session["turns"] if "embedding" in t
    ]
    if len(past_embeddings) >= ANOMALY_MIN_HISTORY:
        centroid = np.mean(past_embeddings, axis=0).tolist()
        sim = _cosine_similarity(message_embedding, centroid)
        if sim < EMBEDDING_ANOMALY_THRESHOLD:
            logger.info(
                "[triggers] embedding anomaly fired: session=%s sim=%.3f threshold=%.3f",
                session_id, sim, EMBEDDING_ANOMALY_THRESHOLD,
            )
            fired = True

    return fired


# ---------------------------------------------------------------------------
# Shadow agent
# ---------------------------------------------------------------------------

async def run_shadow_agent(session_id: str) -> None:
    session = await get_session(session_id)
    if session is None:
        logger.warning("[shadow] session not found: %s", session_id)
        return

    turn_count = len([t for t in session["turns"] if t["role"] == "user"])
    logger.info("[shadow] running for session=%s at user_turn=%d", session_id, turn_count)

    await update_safety_state(session_id, "shadow_active", turn_count)

    # ------------------------------------------------------------------
    # Insert policy / review logic here:
    # e.g. call a separate Ollama model, emit an alert, annotate the DB,
    # update safety_state to "flagged" / "cleared", etc.
    # ------------------------------------------------------------------

    logger.info("[shadow] completed for session=%s", session_id)


async def _shadow_worker() -> None:
    while True:
        session_id = await shadow_queue.get()
        try:
            await run_shadow_agent(session_id)
        except Exception:
            logger.exception("[shadow] unhandled error for session=%s", session_id)
        finally:
            shadow_queue.task_done()


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    worker = asyncio.create_task(_shadow_worker())
    yield
    worker.cancel()
    try:
        await worker
    except asyncio.CancelledError:
        pass


app = FastAPI(title="Gemma 4 Chat API", version="1.0.0", lifespan=lifespan)


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
    safety_state: str


class SessionState(BaseModel):
    session_id: str
    turn_count: int
    safety_state: str
    last_activation_turn: int


# ---------------------------------------------------------------------------
# Background task wired to /chat
# ---------------------------------------------------------------------------

async def _post_chat_tasks(
    session_id: str,
    turn_count: int,
    message_embedding: list[float],
) -> None:
    triggered = await evaluate_triggers(session_id, turn_count, message_embedding)
    if triggered:
        await shadow_queue.put(session_id)
        logger.info("[chat] shadow agent enqueued for session=%s", session_id)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, background_tasks: BackgroundTasks) -> ChatResponse:
    # Load or initialise session
    session = await get_session(req.session_id) or {
        "session_id": req.session_id,
        "turns": [],
        "safety_state": "nominal",
        "last_activation_turn": 0,
    }

    turns: list[dict] = session["turns"]

    # Embed the incoming message (fail-safe: zero vector on error)
    try:
        embedding = await _get_embedding(req.message)
    except Exception:
        logger.warning("[chat] embedding failed, using zero vector")
        embedding = [0.0] * 2048

    # Build Ollama message history (omit stored embeddings from payload)
    messages = [{"role": t["role"], "content": t["content"]} for t in turns]
    messages.append({"role": "user", "content": req.message})

    # Call Ollama
    try:
        reply = await _ollama_chat(messages)
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=502, detail=f"Ollama error: {exc.response.text}")
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=f"Ollama unreachable: {exc}")

    # Persist turns; embed stored on user turn for future anomaly detection
    turns.append({"role": "user", "content": req.message, "embedding": embedding})
    turns.append({"role": "assistant", "content": reply})
    turn_count = sum(1 for t in turns if t["role"] == "user")

    await upsert_session(
        req.session_id, turns, session["safety_state"], session["last_activation_turn"]
    )

    # Schedule evaluate_triggers + conditional shadow-agent enqueue after response
    background_tasks.add_task(
        _post_chat_tasks, req.session_id, turn_count, embedding
    )

    return ChatResponse(
        session_id=req.session_id,
        reply=reply,
        turn_count=turn_count,
        safety_state=session["safety_state"],
    )


@app.get("/session/{session_id}", response_model=SessionState)
async def get_session_state(session_id: str) -> SessionState:
    session = await get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    turn_count = sum(1 for t in session["turns"] if t["role"] == "user")
    return SessionState(
        session_id=session_id,
        turn_count=turn_count,
        safety_state=session["safety_state"],
        last_activation_turn=session["last_activation_turn"],
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
