import json
from datetime import datetime, timezone
from typing import List, Optional

import aiosqlite

DB_PATH = "sessions.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT    NOT NULL,
    turn_number  INTEGER NOT NULL,
    role         TEXT    NOT NULL,
    content      TEXT    NOT NULL,
    timestamp    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS anchor_turns (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT    NOT NULL,
    turn_number  INTEGER NOT NULL,
    content      TEXT    NOT NULL,
    UNIQUE(session_id, turn_number)
);

CREATE TABLE IF NOT EXISTS safety_state (
    session_id   TEXT PRIMARY KEY,
    state_json   TEXT NOT NULL
);
"""

EMPTY_SAFETY_STATE = {
    "turn_count": 0,
    "baseline": "",
    "observed_themes": [],
    "trajectory": "stable",
    "risk_score": 0.0,
    "trend_direction": "stable",
    "confidence": 0.0,
    "anchor_turns": [],
    "last_reasoning": "",
    "recommended_action": "none",
}


async def init_db() -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript(SCHEMA)
        await db.commit()


# ---------------------------------------------------------------------------
# turns
# ---------------------------------------------------------------------------

async def get_next_turn_number(session_id: str) -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT COALESCE(MAX(turn_number), 0) + 1 FROM turns WHERE session_id = ?",
            (session_id,),
        ) as cur:
            row = await cur.fetchone()
    return row[0] if row else 1


async def save_turn(session_id: str, turn_number: int, role: str, content: str) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO turns (session_id, turn_number, role, content, timestamp) VALUES (?,?,?,?,?)",
            (session_id, turn_number, role, content, ts),
        )
        await db.commit()


async def get_recent_turns(session_id: str, limit: int = 20) -> List[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """
            SELECT turn_number, role, content, timestamp FROM turns
            WHERE session_id = ? ORDER BY turn_number DESC LIMIT ?
            """,
            (session_id, limit),
        ) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in reversed(rows)]


async def get_all_turns(session_id: str) -> List[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT turn_number, role, content, timestamp FROM turns WHERE session_id = ? ORDER BY turn_number",
            (session_id,),
        ) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def get_user_turn_count(session_id: str) -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT COUNT(*) FROM turns WHERE session_id = ? AND role = 'user'",
            (session_id,),
        ) as cur:
            row = await cur.fetchone()
    return row[0] if row else 0


# ---------------------------------------------------------------------------
# anchor_turns
# ---------------------------------------------------------------------------

async def save_anchor_turns(session_id: str, turn_numbers: List[int], all_turns: List[dict]) -> None:
    turn_map = {t["turn_number"]: t["content"] for t in all_turns}
    async with aiosqlite.connect(DB_PATH) as db:
        for tn in turn_numbers:
            await db.execute(
                "INSERT OR IGNORE INTO anchor_turns (session_id, turn_number, content) VALUES (?,?,?)",
                (session_id, tn, turn_map.get(tn, "")),
            )
        await db.commit()


async def get_anchor_turns(session_id: str) -> List[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT turn_number, content FROM anchor_turns WHERE session_id = ? ORDER BY turn_number",
            (session_id,),
        ) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# safety_state
# ---------------------------------------------------------------------------

async def get_safety_state(session_id: str) -> dict:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT state_json FROM safety_state WHERE session_id = ?",
            (session_id,),
        ) as cur:
            row = await cur.fetchone()
    if row is None:
        return dict(EMPTY_SAFETY_STATE)
    return json.loads(row[0])


async def upsert_safety_state(session_id: str, state: dict) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO safety_state (session_id, state_json) VALUES (?,?)
            ON CONFLICT(session_id) DO UPDATE SET state_json = excluded.state_json
            """,
            (session_id, json.dumps(state)),
        )
        await db.commit()
