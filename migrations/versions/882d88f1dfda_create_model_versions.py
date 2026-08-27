"""create model_versions

Revision ID: 882d88f1dfda
Revises: 5327a947822d
Create Date: 2026-08-21 16:02:11.443027

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "882d88f1dfda"
down_revision: str | Sequence[str] | None = "5327a947822d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "model_versions",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("model_name", sa.Text(), nullable=False),
        sa.Column("model_version", sa.Text(), nullable=False),
        sa.Column(
            "trained_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        # Training window bounds: without these a version cannot be reproduced,
        # and "was this model allowed to see hour T?" is unanswerable.
        sa.Column("train_start", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("train_end", sa.TIMESTAMP(timezone=True), nullable=False),
        # Validity window. Answers "which version was live at hour T" even for
        # hours where no prediction was logged, which is exactly the downtime
        # case the back-generation rule needs.
        sa.Column("activated_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("retired_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("artifact_path", sa.Text(), nullable=False),
        sa.Column("hyperparameters", postgresql.JSONB(), nullable=True),
        # The feature list the model was trained against. A serving-time
        # mismatch is train/serve skew, and this is what makes it detectable.
        sa.Column("feature_names", postgresql.JSONB(), nullable=True),
        # Reference walk-forward metrics, stored per version so drift can be
        # measured as observed-minus-reference rather than against a constant.
        sa.Column("reference_metrics", postgresql.JSONB(), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_model_versions"),
        sa.UniqueConstraint("model_name", "model_version", name="uq_model_versions_name_version"),
        sa.CheckConstraint("train_end > train_start", name="ck_model_versions_train_window"),
        sa.CheckConstraint(
            "retired_at IS NULL OR activated_at IS NOT NULL",
            name="ck_model_versions_retired_implies_activated",
        ),
    )

    # At most one live version per model. A second active row would make
    # "which version was live" ambiguous precisely when it matters.
    op.create_index(
        "uq_model_versions_one_active_per_model",
        "model_versions",
        ["model_name"],
        unique=True,
        postgresql_where=sa.text("activated_at IS NOT NULL AND retired_at IS NULL"),
    )

    # Every logged prediction must point at a registered version. Without this
    # a prediction could name a version whose validity window is unknown, which
    # is the failure the registry exists to prevent.
    op.create_foreign_key(
        "fk_predictions_model_version",
        "predictions",
        "model_versions",
        ["model_name", "model_version"],
        ["model_name", "model_version"],
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint("fk_predictions_model_version", "predictions", type_="foreignkey")
    op.drop_index("uq_model_versions_one_active_per_model", table_name="model_versions")
    op.drop_table("model_versions")
