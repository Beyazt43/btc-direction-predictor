"""create predictions

Revision ID: 5327a947822d
Revises: 0f19e86ba092
Create Date: 2026-08-20 15:25:44.118902

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "5327a947822d"
down_revision: str | Sequence[str] | None = "0f19e86ba092"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "predictions",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("model_name", sa.Text(), nullable=False),
        sa.Column("model_version", sa.Text(), nullable=False),
        # The hour being predicted, not the hour the prediction was made from.
        sa.Column("target_open_time", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column(
            "predicted_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("predicted_direction", sa.SmallInteger(), nullable=False),
        sa.Column("predicted_proba", sa.Double(), nullable=True),
        # Resolution columns: null until the target hour closes and is ingested.
        sa.Column("actual_direction", sa.SmallInteger(), nullable=True),
        # Realized return is stored alongside the label so accuracy can be sliced
        # by move magnitude post-hoc, instead of baking a dead zone into the label.
        sa.Column("actual_log_return", sa.Double(), nullable=True),
        sa.Column("resolved_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_predictions"),
        # One prediction per model version per target hour; makes re-running the
        # prediction job for an already-predicted hour a no-op rather than a dupe.
        sa.UniqueConstraint(
            "model_name",
            "model_version",
            "target_open_time",
            name="uq_predictions_model_version_target",
        ),
    )
    # Drift monitoring reads rolling windows per model across all versions, which
    # the unique constraint above cannot serve efficiently (model_version sits
    # between the two columns actually being filtered on).
    op.create_index(
        "ix_predictions_model_name_target_open_time",
        "predictions",
        ["model_name", "target_open_time"],
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_predictions_model_name_target_open_time", table_name="predictions")
    op.drop_table("predictions")
