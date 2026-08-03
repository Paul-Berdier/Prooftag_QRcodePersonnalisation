"""Persist E017 provenance and repeated phone-scan calibration data."""

import sqlalchemy as sa
from alembic import op

revision = "0005_e017_phone_calibration"
down_revision = "0004_human_verdicts"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    run_columns = {column["name"] for column in sa.inspect(bind).get_columns("runs")}
    if "provenance" not in run_columns:
        op.add_column(
            "runs",
            sa.Column(
                "provenance",
                sa.JSON(),
                nullable=False,
                server_default=sa.text("'{}'"),
            ),
        )
    rating_columns = {
        column["name"] for column in sa.inspect(bind).get_columns("lab_ratings")
    }
    additions = [
        ("human_scan_attempts", sa.Integer(), "0"),
        ("human_scan_successes", sa.Integer(), "0"),
        ("human_scan_device", sa.String(length=200), "''"),
    ]
    for name, column_type, default in additions:
        if name not in rating_columns:
            op.add_column(
                "lab_ratings",
                sa.Column(
                    name,
                    column_type,
                    nullable=False,
                    server_default=sa.text(default),
                ),
            )


def downgrade() -> None:
    bind = op.get_bind()
    rating_columns = {
        column["name"] for column in sa.inspect(bind).get_columns("lab_ratings")
    }
    for name in (
        "human_scan_device",
        "human_scan_successes",
        "human_scan_attempts",
    ):
        if name in rating_columns:
            op.drop_column("lab_ratings", name)
    run_columns = {column["name"] for column in sa.inspect(bind).get_columns("runs")}
    if "provenance" in run_columns:
        op.drop_column("runs", "provenance")
