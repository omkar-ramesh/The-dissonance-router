"""Past conversations: archived compressed, kept 10 days unless the user keeps them.

"New chat" archives the current conversation to a gzip'd JSON file (one per conversation,
in a folder per user) and starts a fresh thread. The profile is not touched: a new chat is
not a new person.

An archived conversation is deleted automatically after RETENTION_DAYS unless the user
pressed Keep. Deletion removes the file AND the message text in the database (routing
numbers such as tier and score stay, for statistics), so "deleted" means deleted.
Reminders are computed from dates: no model call, no tokens.
"""
from __future__ import annotations    # the method named `list` would otherwise shadow list[...] hints

import gzip
import hashlib
import json
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

from src.db.database import get_connection

REMIND_DAYS = 3


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class MemoryStore:
    def __init__(self, db_path: str, folder, retention_days: float = 10, now: Callable[[], datetime] = _utc_now):
        self.db_path, self.folder, self.retention, self.now = db_path, Path(folder), retention_days, now

    # ------------------------------------------------------------ archive
    def boundary(self, user_id: str) -> int:
        """Id of the last archived interaction: the current conversation starts after it."""
        with closing(get_connection(self.db_path)) as conn:
            row = conn.execute("SELECT MAX(last_id) AS b FROM conversations WHERE user_id = ?", (user_id,)).fetchone()
        return row["b"] or 0

    def archive(self, user_id: str) -> Optional[dict]:
        """Compress the current conversation to a file and start a new one. None if it was empty."""
        start = self.boundary(user_id)
        with closing(get_connection(self.db_path)) as conn:
            rows = [dict(r) for r in conn.execute(
                "SELECT id, user_message, agent_response, model_used, conflict_score, routing_reason, timestamp "
                "FROM interactions WHERE user_id = ? AND id > ? ORDER BY id", (user_id, start)).fetchall()]
        if not rows:
            return None
        now = self.now()
        title = next((r["user_message"] for r in rows if r["user_message"].strip()), "Conversation")[:80]
        with closing(get_connection(self.db_path)) as conn:
            with conn:
                cid = conn.execute(
                    "INSERT INTO conversations (user_id, title, first_id, last_id, turns, bytes, file, archived_at, expires_at) "
                    "VALUES (?, ?, ?, ?, ?, 0, '', ?, ?)",
                    (user_id, title, rows[0]["id"], rows[-1]["id"], len(rows), now.isoformat(),
                     (now + timedelta(days=self.retention)).isoformat())).lastrowid
                path = self._path(user_id, cid)
                path.parent.mkdir(parents=True, exist_ok=True)
                payload = {"id": cid, "title": title, "archived_at": now.isoformat(), "turns": rows}
                with gzip.open(path, "wt", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False)
                conn.execute("UPDATE conversations SET file = ?, bytes = ? WHERE id = ?",
                             (path.name, path.stat().st_size, cid))
        return self.get(user_id, cid)

    # ------------------------------------------------------------ read
    def list(self, user_id: str) -> list[dict]:
        with closing(get_connection(self.db_path)) as conn:
            rows = conn.execute("SELECT * FROM conversations WHERE user_id = ? ORDER BY id DESC", (user_id,)).fetchall()
        return [self._item(r) for r in rows]

    def get(self, user_id: str, cid: int) -> Optional[dict]:
        row = self._row(user_id, cid)
        return self._item(row) if row else None

    def read(self, user_id: str, cid: int) -> dict:
        """The archived transcript (decompressed)."""
        row = self._row(user_id, cid)
        if not row:
            raise LookupError(f"No conversation {cid} for this user")
        with gzip.open(self._path(user_id, cid), "rt", encoding="utf-8") as f:
            return json.load(f)

    def file(self, user_id: str, cid: int) -> Path:
        if not self._row(user_id, cid):
            raise LookupError(f"No conversation {cid} for this user")
        return self._path(user_id, cid)

    def reminders(self, user_id: str) -> list[dict]:
        """Unkept conversations that will be deleted within REMIND_DAYS."""
        return [i for i in self.list(user_id) if not i["kept"] and i["days_left"] <= REMIND_DAYS]

    # ------------------------------------------------------------ change
    def keep(self, user_id: str, cid: int, kept: bool = True) -> dict:
        """Keep forever, or undo that (the 10 days start again from now)."""
        if not self._row(user_id, cid):
            raise LookupError(f"No conversation {cid} for this user")
        expires = (self.now() + timedelta(days=self.retention)).isoformat()
        with closing(get_connection(self.db_path)) as conn:
            with conn:
                if kept:
                    conn.execute("UPDATE conversations SET saved = 1 WHERE id = ?", (cid,))
                else:
                    conn.execute("UPDATE conversations SET saved = 0, expires_at = ? WHERE id = ?", (expires, cid))
        return self.get(user_id, cid)

    def delete(self, user_id: str, cid: int) -> None:
        row = self._row(user_id, cid)
        if not row:
            raise LookupError(f"No conversation {cid} for this user")
        self._erase(row)

    def sweep(self) -> int:
        """Delete every unkept conversation past its date (all users). Returns how many."""
        now = self.now().isoformat()
        with closing(get_connection(self.db_path)) as conn:
            due = conn.execute("SELECT * FROM conversations WHERE saved = 0 AND expires_at <= ?", (now,)).fetchall()
        for row in due:
            self._erase(row)
        return len(due)

    # ------------------------------------------------------------ internals
    def _erase(self, row) -> None:
        self._path(row["user_id"], row["id"]).unlink(missing_ok=True)
        with closing(get_connection(self.db_path)) as conn:
            with conn:
                # the words go; tier, score and reason stay, for statistics
                conn.execute("UPDATE interactions SET user_message = '', agent_response = '' "
                             "WHERE user_id = ? AND id BETWEEN ? AND ?", (row["user_id"], row["first_id"], row["last_id"]))
                # quotes the note bot kept from this conversation go too
                conn.execute("UPDATE events SET detail = '' WHERE user_id = ? AND interaction_id BETWEEN ? AND ?",
                             (row["user_id"], row["first_id"], row["last_id"]))
                conn.execute("DELETE FROM conversations WHERE id = ?", (row["id"],))

    def _row(self, user_id: str, cid: int):
        with closing(get_connection(self.db_path)) as conn:
            return conn.execute("SELECT * FROM conversations WHERE id = ? AND user_id = ?", (cid, user_id)).fetchone()

    def _path(self, user_id: str, cid: int) -> Path:
        # Folder named by a hash of the id: user ids may contain dots, so never trust them in a path.
        folder = hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:16]
        return self.folder / folder / f"conversation-{int(cid)}.json.gz"

    def _item(self, row) -> dict:
        expires = datetime.fromisoformat(row["expires_at"])
        left = (expires - self.now()).total_seconds() / 86400
        return {"id": row["id"], "title": row["title"], "turns": row["turns"], "bytes": row["bytes"],
                "archived_at": row["archived_at"], "kept": bool(row["saved"]),
                "expires_at": None if row["saved"] else row["expires_at"],
                "days_left": None if row["saved"] else max(0.0, round(left, 2))}
