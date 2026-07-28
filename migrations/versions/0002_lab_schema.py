"""Add the persistent web laboratory schema."""

from alembic import op

from prooftag_qr.db import lab_campaigns, lab_ratings, lab_trials

revision = "0002_lab_schema"
down_revision = "0001_initial_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    lab_campaigns.create(bind=bind, checkfirst=True)
    lab_trials.create(bind=bind, checkfirst=True)
    lab_ratings.create(bind=bind, checkfirst=True)


def downgrade() -> None:
    bind = op.get_bind()
    lab_ratings.drop(bind=bind, checkfirst=True)
    lab_trials.drop(bind=bind, checkfirst=True)
    lab_campaigns.drop(bind=bind, checkfirst=True)
