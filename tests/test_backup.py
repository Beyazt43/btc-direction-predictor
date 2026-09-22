import gzip
import json
from datetime import UTC, datetime
from decimal import Decimal

import pytest
import sqlalchemy as sa

from btcpred.backup.service import (
    MANIFEST,
    TABLES,
    BackupError,
    Manifest,
    _decode,
    _encode,
    list_backups,
    prune,
    verify_backup,
)
from btcpred.db.tables import drift_checks, model_versions, predictions, price_bars

# --- encoding round trip: the failure the first restore actually hit -----------


def test_every_column_type_survives_encode_decode():
    """CSV has one type and the driver does no coercion, so each column type has
    to come back as itself. Handing a BIGINT the string "1" fails at bind time."""
    cases = [
        (price_bars.c.id, 42),
        (price_bars.c.symbol, "BTCUSDT"),
        (price_bars.c.open_time, datetime(2026, 9, 22, 13, tzinfo=UTC)),
        (price_bars.c.close, Decimal("86024.01000000")),
        (price_bars.c.num_trades, 136686),
        (predictions.c.predicted_direction, 1),
        (predictions.c.predicted_proba, 0.4547),
        (predictions.c.actual_log_return, -0.000901),
        (model_versions.c.hyperparameters, {"max_depth": 3, "n_estimators": 144}),
        (model_versions.c.feature_names, ["ret_lag_1", "close_pos"]),
        (drift_checks.c.z_score, -2.4),
        (drift_checks.c.status, "alert"),
    ]
    for column, value in cases:
        restored = _decode(str(_encode(value)), column)
        assert restored == value, f"{column.name}: {value!r} -> {restored!r}"
        assert type(restored) is type(value), f"{column.name}: type changed"


def test_null_round_trips_and_is_not_an_empty_string():
    """CSV cannot tell NULL from '', so NULL is written as \\N."""
    assert _encode(None) == r"\N"
    for column in (predictions.c.actual_direction, model_versions.c.hyperparameters):
        assert _decode(r"\N", column) is None
    assert _decode("", predictions.c.model_name) == ""


def test_float_columns_do_not_become_decimal():
    """Double inherits Float inherits Numeric; the wrong branch order silently
    turns probabilities into Decimal, which asyncpg then rejects."""
    assert isinstance(_decode("0.5", predictions.c.predicted_proba), float)
    assert isinstance(_decode("86024.01", price_bars.c.close), Decimal)


def test_tables_are_ordered_so_foreign_keys_resolve():
    names = [t.name for t in TABLES]
    assert names.index("model_versions") < names.index("predictions")


# --- manifest and verification -------------------------------------------------


def make_backup(root, name="btcpred-20260922T120000Z", rows=None, revision="d6eb0a62352a"):
    rows = rows or {t.name: 2 for t in TABLES}
    path = root / name
    path.mkdir(parents=True)
    digests = {}
    for table, count in rows.items():
        target = path / f"{table}.csv.gz"
        with gzip.open(target, "wt", encoding="utf-8", newline="") as fh:
            fh.write("id\n" + "".join(f"{i}\n" for i in range(count)))
        from btcpred.backup.service import _digest

        digests[target.name] = _digest(target)
    manifest = Manifest(
        created_at="2026-09-22T12:00:00+00:00",
        alembic_revision=revision,
        rows=rows,
        sha256=digests,
    )
    (path / MANIFEST).write_text(manifest.to_json(), encoding="utf-8")
    return path


def test_verify_accepts_an_intact_backup(tmp_path):
    path = make_backup(tmp_path)
    assert verify_backup(path).alembic_revision == "d6eb0a62352a"


def test_verify_detects_a_corrupted_file(tmp_path):
    path = make_backup(tmp_path)
    target = path / "predictions.csv.gz"
    # Same row count, different bytes: only the checksum can catch this.
    with gzip.open(target, "wt", encoding="utf-8", newline="") as fh:
        fh.write("id\n7\n8\n")
    with pytest.raises(BackupError, match="checksum"):
        verify_backup(path)


def test_verify_detects_a_missing_file(tmp_path):
    path = make_backup(tmp_path)
    (path / "predictions.csv.gz").unlink()
    with pytest.raises(BackupError, match="missing"):
        verify_backup(path)


def test_verify_detects_a_truncated_file(tmp_path):
    """Checksums alone would catch this, but row counts say what went wrong."""
    path = make_backup(tmp_path, rows={t.name: 5 for t in TABLES})
    manifest = json.loads((path / MANIFEST).read_text())
    manifest["rows"]["predictions"] = 9
    (path / MANIFEST).write_text(json.dumps(manifest))
    with pytest.raises(BackupError, match="rows"):
        verify_backup(path)


def test_verify_reports_a_missing_manifest(tmp_path):
    path = make_backup(tmp_path)
    (path / MANIFEST).unlink()
    with pytest.raises(BackupError, match="manifest"):
        verify_backup(path)


# --- listing and retention -----------------------------------------------------


def test_backups_list_newest_first(tmp_path):
    for day in (20, 22, 21):
        name = f"btcpred-202609{day}T120000Z"
        make_backup(tmp_path, name)
    assert [b.name for b in list_backups(tmp_path)] == [
        "btcpred-20260922T120000Z",
        "btcpred-20260921T120000Z",
        "btcpred-20260920T120000Z",
    ]


def test_listing_skips_an_unreadable_backup_instead_of_failing(tmp_path):
    make_backup(tmp_path, "btcpred-20260922T120000Z")
    broken = tmp_path / "btcpred-20260921T120000Z"
    broken.mkdir()
    (broken / MANIFEST).write_text("not json")
    assert [b.name for b in list_backups(tmp_path)] == ["btcpred-20260922T120000Z"]


def test_listing_an_absent_directory_is_empty_not_an_error(tmp_path):
    assert list_backups(tmp_path / "nope") == []


def test_prune_keeps_the_newest(tmp_path):
    for day in (18, 19, 20, 21, 22):
        make_backup(tmp_path, f"btcpred-202609{day}T120000Z")
    removed = prune(tmp_path, keep=2)

    assert sorted(removed) == [
        "btcpred-20260918T120000Z",
        "btcpred-20260919T120000Z",
        "btcpred-20260920T120000Z",
    ]
    assert [b.name for b in list_backups(tmp_path)] == [
        "btcpred-20260922T120000Z",
        "btcpred-20260921T120000Z",
    ]


def test_prune_refuses_to_delete_everything(tmp_path):
    make_backup(tmp_path)
    with pytest.raises(ValueError, match="at least 1"):
        prune(tmp_path, keep=0)


def test_prune_is_a_no_op_when_under_the_limit(tmp_path):
    make_backup(tmp_path)
    assert prune(tmp_path, keep=14) == []
    assert len(list_backups(tmp_path)) == 1


def test_manifest_json_round_trips(tmp_path):
    path = make_backup(tmp_path)
    assert Manifest.from_path(path) == verify_backup(path)


def test_every_backed_up_table_is_a_real_table():
    for table in TABLES:
        assert isinstance(table, sa.Table)
