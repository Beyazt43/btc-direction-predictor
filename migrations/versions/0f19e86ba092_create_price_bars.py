"""create price_bars

Revision ID: 0f19e86ba092
Revises:
Create Date: 2026-08-20 15:25:37.912684

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0f19e86ba092"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "price_bars",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.Text(), server_default=sa.text("'BTCUSDT'"), nullable=False),
        # open_time and close_time are kept separately so we can always be precise
        # about what information was actually available at prediction time.
        sa.Column("open_time", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("close_time", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("open", sa.Numeric(18, 8), nullable=False),
        sa.Column("high", sa.Numeric(18, 8), nullable=False),
        sa.Column("low", sa.Numeric(18, 8), nullable=False),
        sa.Column("close", sa.Numeric(18, 8), nullable=False),
        sa.Column("volume", sa.Numeric(24, 8), nullable=False),
        sa.Column("quote_volume", sa.Numeric(24, 8), nullable=True),
        sa.Column("num_trades", sa.Integer(), nullable=True),
        # Distinct from open_time: a leakage tripwire / audit trail. A row whose
        # ingested_at postdates when it should have been available is a bug.
        sa.Column(
            "ingested_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("source", sa.Text(), server_default=sa.text("'binance'"), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_price_bars"),
        # Makes ingestion idempotent: the scheduler double-firing or a backfill
        # overlapping existing rows is safe via ON CONFLICT.
        sa.UniqueConstraint("symbol", "open_time", name="uq_price_bars_symbol_open_time"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("price_bars")
