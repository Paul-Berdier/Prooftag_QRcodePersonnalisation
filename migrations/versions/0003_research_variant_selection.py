"""Record the actual research candidate selected by a laboratory trial."""

import sqlalchemy as sa
from alembic import op

revision = "0003_research_variant_selection"
down_revision = "0002_lab_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Migration 0001 uses the current SQLAlchemy metadata with ``create_all``.
    # A fresh database can therefore already contain future columns while an
    # existing production database does not. Keep this migration safe for both.
    existing = {column["name"] for column in sa.inspect(op.get_bind()).get_columns("runs")}
    if "selected_variant" not in existing:
        op.add_column("runs", sa.Column("selected_variant", sa.String(length=100), nullable=True))
    if "selection_mode" not in existing:
        op.add_column(
            "runs",
            sa.Column(
                "selection_mode",
                sa.String(length=20),
                nullable=False,
                server_default="delivery",
            ),
        )
    if "stage1_reused" not in existing:
        op.add_column(
            "runs",
            sa.Column("stage1_reused", sa.Boolean(), nullable=False, server_default=sa.false()),
        )
    if "stage1_source_run_id" not in existing:
        op.add_column(
            "runs",
            sa.Column("stage1_source_run_id", sa.String(length=36), nullable=True),
        )


def downgrade() -> None:
    existing = {column["name"] for column in sa.inspect(op.get_bind()).get_columns("runs")}
    for column_name in (
        "stage1_source_run_id",
        "stage1_reused",
        "selection_mode",
        "selected_variant",
    ):
        if column_name in existing:
            op.drop_column("runs", column_name)
