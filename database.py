import json

import aiosqlite

DB_PATH = "sessions.db"

CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS sessions (
        session_id          TEXT    PRIMARY KEY,
        turns               TEXT    NOT NULL DEFAULT '[]',
        safety_state        TEXT    NOT NULL DEFAULT 'nominal',
        last_activation_turn INTEGER NOT NULL DEFAULT 0
    )
"""


async def init_db() -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(CREATE_TABLE)
        await db.commit()


async def get_session(session_id: str) -> dict | None:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ) as cursor:
            row = await cursor.fetchone()
    if row is None:
        return None
    return {
        "session_id": row["session_id"],
        "turns": json.loads(row["turns"]),
        "safety_state": row["safety_state"],
        "last_activation_turn": row["last_activation_turn"],
    }


async def upsert_session(
    session_id: str,
    turns: list,
    safety_state: str,
    last_activation_turn: int,
) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO sessions (session_id, turns, safety_state, last_activation_turn)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                turns                = excluded.turns,
                safety_state         = excluded.safety_state,
                last_activation_turn = excluded.last_activation_turn
            """,
            (session_id, json.dumps(turns), safety_state, last_activation_turn),
        )
        await db.commit()


async def update_safety_state(
    session_id: str, safety_state: str, last_activation_turn: int
) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            UPDATE sessions
            SET safety_state = ?, last_activation_turn = ?
            WHERE session_id = ?
            """,
            (safety_state, last_activation_turn, session_id),
        )
        await db.commit()
