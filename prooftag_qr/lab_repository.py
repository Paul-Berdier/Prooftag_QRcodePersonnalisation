from __future__ import annotations

import threading
from datetime import UTC, datetime

from sqlalchemy import insert, select, update
from sqlalchemy.engine import Engine

from .db import lab_campaigns, lab_ratings, lab_trials


class LabRepository:
    """Persistence boundary for campaigns, trials and human ratings."""

    def __init__(self, engine: Engine):
        self.engine = engine
        self._lock = threading.RLock()

    def create_campaign(self, values: dict) -> dict:
        with self._lock, self.engine.begin() as connection:
            connection.execute(insert(lab_campaigns).values(**values))
        return self.get_campaign(values["id"])

    def get_campaign(self, campaign_id: str) -> dict | None:
        with self.engine.connect() as connection:
            row = (
                connection.execute(
                    select(lab_campaigns).where(lab_campaigns.c.id == campaign_id)
                )
                .mappings()
                .first()
            )
        return dict(row) if row else None

    def list_campaigns(self, limit: int = 100) -> list[dict]:
        with self.engine.connect() as connection:
            rows = (
                connection.execute(
                    select(lab_campaigns)
                    .order_by(lab_campaigns.c.created_at.desc())
                    .limit(limit)
                )
                .mappings()
                .all()
            )
        return [dict(row) for row in rows]

    def update_campaign(self, campaign_id: str, **values) -> dict:
        values["updated_at"] = datetime.now(UTC)
        with self._lock, self.engine.begin() as connection:
            connection.execute(
                update(lab_campaigns)
                .where(lab_campaigns.c.id == campaign_id)
                .values(**values)
            )
        campaign = self.get_campaign(campaign_id)
        if campaign is None:
            raise KeyError(campaign_id)
        return campaign

    def mark_running_interrupted(self) -> None:
        now = datetime.now(UTC)
        with self._lock, self.engine.begin() as connection:
            connection.execute(
                update(lab_trials)
                .where(lab_trials.c.status.in_(["queued", "running"]))
                .values(
                    status="error",
                    completed_at=now,
                    error="API restarted before this trial completed",
                )
            )
            connection.execute(
                update(lab_campaigns)
                .where(lab_campaigns.c.status.in_(["queued", "running"]))
                .values(
                    status="interrupted",
                    updated_at=now,
                    error="API restarted; payload is intentionally not persisted",
                )
            )

    def create_trial(self, values: dict) -> dict:
        with self._lock, self.engine.begin() as connection:
            connection.execute(insert(lab_trials).values(**values))
        trial = self.get_trial(values["id"])
        if trial is None:
            raise KeyError(values["id"])
        return trial

    def get_trial(self, trial_id: str) -> dict | None:
        with self.engine.connect() as connection:
            row = (
                connection.execute(select(lab_trials).where(lab_trials.c.id == trial_id))
                .mappings()
                .first()
            )
        return dict(row) if row else None

    def list_trials(self, campaign_id: str) -> list[dict]:
        with self.engine.connect() as connection:
            rows = (
                connection.execute(
                    select(lab_trials)
                    .where(lab_trials.c.campaign_id == campaign_id)
                    .order_by(
                        lab_trials.c.prompt_id,
                        lab_trials.c.seed,
                        lab_trials.c.method_id,
                    )
                )
                .mappings()
                .all()
            )
        return [dict(row) for row in rows]

    def update_trial(self, trial_id: str, **values) -> dict:
        with self._lock, self.engine.begin() as connection:
            connection.execute(
                update(lab_trials).where(lab_trials.c.id == trial_id).values(**values)
            )
        trial = self.get_trial(trial_id)
        if trial is None:
            raise KeyError(trial_id)
        return trial

    def save_rating(self, trial_id: str, values: dict) -> dict:
        now = datetime.now(UTC)
        with self._lock, self.engine.begin() as connection:
            existing = (
                connection.execute(
                    select(lab_ratings.c.id).where(lab_ratings.c.trial_id == trial_id)
                )
                .mappings()
                .first()
            )
            if existing:
                connection.execute(
                    update(lab_ratings)
                    .where(lab_ratings.c.id == existing["id"])
                    .values(**values, updated_at=now)
                )
                rating_id = existing["id"]
            else:
                result = connection.execute(
                    insert(lab_ratings).values(
                        trial_id=trial_id,
                        created_at=now,
                        updated_at=now,
                        **values,
                    )
                )
                rating_id = result.inserted_primary_key[0]
        rating = self.get_rating_by_id(rating_id)
        if rating is None:
            raise KeyError(rating_id)
        return rating

    def get_rating(self, trial_id: str) -> dict | None:
        with self.engine.connect() as connection:
            row = (
                connection.execute(
                    select(lab_ratings).where(lab_ratings.c.trial_id == trial_id)
                )
                .mappings()
                .first()
            )
        return dict(row) if row else None

    def get_rating_by_id(self, rating_id: int) -> dict | None:
        with self.engine.connect() as connection:
            row = (
                connection.execute(
                    select(lab_ratings).where(lab_ratings.c.id == rating_id)
                )
                .mappings()
                .first()
            )
        return dict(row) if row else None
