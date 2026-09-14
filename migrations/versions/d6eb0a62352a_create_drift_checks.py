"""create drift_checks

Revision ID: d6eb0a62352a
Revises: 882d88f1dfda
Create Date: 2026-09-14 12:48:10.221304

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "d6eb0a62352a"
down_revision: str | Sequence[str] | None = "882d88f1dfda"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

STATUSES = ("ok", "alert", "warming_up", "insufficient")


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "drift_checks",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column(
            "checked_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("model_name", sa.Text(), nullable=False),
        # The observed window: the most recent 30 days of live, resolved calls.
        sa.Column("window_start", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("window_end", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("observed_n", sa.Integer(), nullable=False),
        sa.Column("observed_accuracy", sa.Double(), nullable=True),
        sa.Column("majority_baseline", sa.Double(), nullable=True),
        # The reference: every live, resolved call before the window. Drift is
        # observed-minus-reference, so the reference is the live log itself
        # rather than a walk-forward number that selection has inflated.
        sa.Column("reference_n", sa.Integer(), nullable=False),
        sa.Column("reference_accuracy", sa.Double(), nullable=True),
        sa.Column("z_score", sa.Double(), nullable=True),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        # PSI per feature against the active version's training distribution.
        # Always computed, only surfaced when status is 'alert': it answers
        # "did the world move or did the model" and is not an alert source.
        sa.Column("feature_psi", postgresql.JSONB(), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_drift_checks"),
        sa.CheckConstraint(
            "status IN ('ok', 'alert', 'warming_up', 'insufficient')",
            name="ck_drift_checks_status",
        ),
        sa.CheckConstraint("window_end > window_start", name="ck_drift_checks_window"),
    )
    op.create_index(
        "ix_drift_checks_model_checked_at",
        "drift_checks",
        ["model_name", sa.text("checked_at DESC")],
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_drift_checks_model_checked_at", table_name="drift_checks")
    op.drop_table("drift_checks")
