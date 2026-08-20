"""Core table definitions.

The migrations remain the source of truth for DDL; these definitions mirror them
so queries can be built against real column objects, and so Alembic can diff the
two (`alembic check`) and catch drift between schema and code.
"""

import sqlalchemy as sa

metadata = sa.MetaData()

price_bars = sa.Table(
    "price_bars",
    metadata,
    sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
    sa.Column("symbol", sa.Text(), server_default=sa.text("'BTCUSDT'"), nullable=False),
    sa.Column("open_time", sa.TIMESTAMP(timezone=True), nullable=False),
    sa.Column("close_time", sa.TIMESTAMP(timezone=True), nullable=False),
    sa.Column("open", sa.Numeric(18, 8), nullable=False),
    sa.Column("high", sa.Numeric(18, 8), nullable=False),
    sa.Column("low", sa.Numeric(18, 8), nullable=False),
    sa.Column("close", sa.Numeric(18, 8), nullable=False),
    sa.Column("volume", sa.Numeric(24, 8), nullable=False),
    sa.Column("quote_volume", sa.Numeric(24, 8), nullable=True),
    sa.Column("num_trades", sa.Integer(), nullable=True),
    sa.Column(
        "ingested_at",
        sa.TIMESTAMP(timezone=True),
        server_default=sa.text("now()"),
        nullable=False,
    ),
    sa.Column("source", sa.Text(), server_default=sa.text("'binance'"), nullable=False),
    sa.PrimaryKeyConstraint("id", name="pk_price_bars"),
    sa.UniqueConstraint("symbol", "open_time", name="uq_price_bars_symbol_open_time"),
)

predictions = sa.Table(
    "predictions",
    metadata,
    sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
    sa.Column("model_name", sa.Text(), nullable=False),
    sa.Column("model_version", sa.Text(), nullable=False),
    sa.Column("target_open_time", sa.TIMESTAMP(timezone=True), nullable=False),
    sa.Column(
        "predicted_at",
        sa.TIMESTAMP(timezone=True),
        server_default=sa.text("now()"),
        nullable=False,
    ),
    sa.Column("predicted_direction", sa.SmallInteger(), nullable=False),
    sa.Column("predicted_proba", sa.Double(), nullable=True),
    sa.Column("actual_direction", sa.SmallInteger(), nullable=True),
    sa.Column("actual_log_return", sa.Double(), nullable=True),
    sa.Column("resolved_at", sa.TIMESTAMP(timezone=True), nullable=True),
    sa.PrimaryKeyConstraint("id", name="pk_predictions"),
    sa.UniqueConstraint(
        "model_name",
        "model_version",
        "target_open_time",
        name="uq_predictions_model_version_target",
    ),
    sa.Index("ix_predictions_model_name_target_open_time", "model_name", "target_open_time"),
)
