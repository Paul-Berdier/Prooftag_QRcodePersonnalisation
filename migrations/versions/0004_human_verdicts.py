"""Add simple human aesthetic and scan verdicts to laboratory ratings."""

import sqlalchemy as sa
from alembic import op

revision = "0004_human_verdicts"
down_revision = "0003_research_variant_selection"
branch_labels = None
depends_on = None


def upgrade() -> None:
    existing = {
        column["name"]
        for column in sa.inspect(op.get_bind()).get_columns("lab_ratings")
    }
    if "aesthetic_ok" not in existing:
        op.add_column("lab_ratings", sa.Column("aesthetic_ok", sa.Boolean()))
    if "human_scan_result" not in existing:
        op.add_column(
            "lab_ratings",
            sa.Column(
                "human_scan_result",
                sa.String(length=20),
                nullable=False,
                server_default="not_tested",
            ),
        )


def downgrade() -> None:
    existing = {
        column["name"]
        for column in sa.inspect(op.get_bind()).get_columns("lab_ratings")
    }
    if "human_scan_result" in existing:
        op.drop_column("lab_ratings", "human_scan_result")
    if "aesthetic_ok" in existing:
        op.drop_column("lab_ratings", "aesthetic_ok")
