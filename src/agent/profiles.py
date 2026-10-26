"""Dual style profiles: what the user says (stated) vs. what they do (observed)."""
from contextlib import closing

from src.db.database import get_connection

DIMENSIONS = ("tone", "length", "sourcing", "structure", "revision")

# Each dimension is a spectrum: pole, "balanced", opposite pole.
VALID_VALUES = {
    "tone": ("formal", "balanced", "casual"),
    "length": ("concise", "balanced", "detailed"),
    "sourcing": ("minimal", "balanced", "full"),
    "structure": ("bullet", "balanced", "paragraph"),
    "revision": ("rejects", "balanced", "accepts"),
}

DEFAULT_VALUE = "balanced"
PROFILE_KINDS = ("stated", "observed")


def validate(dimension: str, value: str) -> None:
    """Raise ValueError if the dimension or value is not part of the profile spec."""
    if dimension not in VALID_VALUES:
        raise ValueError(f"Unknown dimension {dimension!r}; expected one of {DIMENSIONS}")
    if value not in VALID_VALUES[dimension]:
        raise ValueError(
            f"Invalid value {value!r} for {dimension}; expected one of {VALID_VALUES[dimension]}"
        )


class ProfileManager:
    """Reads and writes both profiles from the `profiles` table (one row per user)."""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def get_stated_profile(self, user_id: str) -> dict:
        """Return the user's stated preferences. Unknown users get all-"balanced"."""
        return self._extract(self._fetch_row(user_id), "stated")

    def get_observed_profile(self, user_id: str) -> dict:
        """Return the user's observed behavior. Unknown users get all-"balanced"."""
        return self._extract(self._fetch_row(user_id), "observed")

    def get_all_profiles(self, user_id: str) -> tuple[dict, dict]:
        """Return (stated, observed) from a single read."""
        row = self._fetch_row(user_id)
        return self._extract(row, "stated"), self._extract(row, "observed")

    def update_stated(self, user_id: str, dimension: str, value: str) -> bool:
        """Set one stated preference, creating the user's row if needed."""
        return self._update(user_id, "stated", dimension, value)

    def update_observed(self, user_id: str, dimension: str, value: str) -> bool:
        """Set one observed preference, creating the user's row if needed."""
        return self._update(user_id, "observed", dimension, value)

    def _fetch_row(self, user_id: str):
        with closing(get_connection(self.db_path)) as conn:
            return conn.execute("SELECT * FROM profiles WHERE user_id = ?", (user_id,)).fetchone()

    @staticmethod
    def _extract(row, kind: str) -> dict:
        if row is None:
            return {dim: DEFAULT_VALUE for dim in DIMENSIONS}
        return {dim: row[f"{kind}_{dim}"] for dim in DIMENSIONS}

    def _update(self, user_id: str, kind: str, dimension: str, value: str) -> bool:
        validate(dimension, value)
        # Safe to interpolate: kind and dimension are both whitelisted above.
        column = f"{kind}_{dimension}"
        with closing(get_connection(self.db_path)) as conn:
            with conn:
                conn.execute(
                    f"INSERT INTO profiles (user_id, {column}) VALUES (?, ?) "
                    f"ON CONFLICT(user_id) DO UPDATE SET {column} = excluded.{column}, "
                    "updated_at = CURRENT_TIMESTAMP",
                    (user_id, value),
                )
        return True
