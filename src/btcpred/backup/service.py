"""Backup and restore of the irreplaceable tables.

`price_bars` can be re-downloaded from Binance and the schema lives in Alembic,
in git. What cannot be rebuilt is `predictions`: every row was committed before
its outcome was knowable, and nothing in this system back-fills. `model_versions`
and `drift_checks` are the context that makes those rows interpretable.

So a backup here is *data plus the revision it belongs to*, not a schema dump. A
restore is `alembic upgrade head` followed by a load, which means the backup
cannot drift out of step with the migrations the way a captured DDL snapshot can.
That also avoids needing a `pg_dump` binary matched to the server version inside
the application image.

Every table is written as gzipped CSV with a manifest recording the Alembic
revision, per-table row counts and a SHA-256 of each file, so a backup can be
verified without restoring it.
"""

import csv
import gzip
import hashlib
import io
import json
import logging
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.asyncio import AsyncConnection

from btcpred.db.tables import drift_checks, metadata, model_versions, predictions, price_bars

logger = logging.getLogger(__name__)

# Written and restored in this order: model_versions before predictions, because
# predictions carries a foreign key to it.
TABLES: tuple[sa.Table, ...] = (price_bars, model_versions, predictions, drift_checks)

MANIFEST = "manifest.json"
STAMP = "%Y%m%dT%H%M%SZ"


class BackupError(RuntimeError):
    """A backup is missing, malformed, or fails verification."""


@dataclass(frozen=True, slots=True)
class Manifest:
    created_at: str
    alembic_revision: str | None
    rows: dict[str, int]
    sha256: dict[str, str]
    format_version: int = 1

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @staticmethod
    def from_path(path: Path) -> "Manifest":
        try:
            raw = json.loads((path / MANIFEST).read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise BackupError(f"{path} has no {MANIFEST}") from exc
        return Manifest(**raw)


@dataclass(frozen=True, slots=True)
class BackupInfo:
    path: Path
    manifest: Manifest

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def total_rows(self) -> int:
        return sum(self.manifest.rows.values())

    @property
    def size_bytes(self) -> int:
        return sum(f.stat().st_size for f in self.path.glob("*") if f.is_file())


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


async def _revision(conn: AsyncConnection) -> str | None:
    """The Alembic revision this data belongs to.

    Restoring into a database at a different revision is the one way this
    backup format can go wrong, so the revision travels with the data.
    """
    try:
        return await conn.scalar(sa.text("SELECT version_num FROM alembic_version"))
    except Exception:
        return None


async def _dump_table(conn: AsyncConnection, table: sa.Table, dest: Path) -> int:
    rows = (await conn.execute(sa.select(table))).mappings().all()
    columns = [c.name for c in table.columns]

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({c: _encode(row[c]) for c in columns})

    with gzip.open(dest, "wt", encoding="utf-8", newline="") as fh:
        fh.write(buffer.getvalue())
    return len(rows)


def _encode(value: Any) -> Any:
    """CSV cannot distinguish NULL from empty string, so NULL is written as \\N.

    That is Postgres' own convention for text-format COPY, which keeps the files
    readable by psql if this module ever goes away.
    """
    if value is None:
        return r"\N"
    if isinstance(value, dict | list):
        return json.dumps(value, sort_keys=True)
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _decode(value: str, column: sa.Column) -> Any:
    """Turn a CSV field back into the type its column expects.

    CSV has exactly one type and the driver does no coercion: handing a BIGINT
    column the string "1" fails at bind time. So the column type drives the
    conversion rather than being guessed from the text.
    """
    if value == r"\N":
        return None
    kind = column.type
    if isinstance(kind, postgresql.JSONB | sa.JSON):
        return json.loads(value)
    # Order matters: Double inherits Float inherits Numeric, so the float branch
    # has to come first or NUMERIC would swallow it.
    if isinstance(kind, sa.Float):
        return float(value)
    if isinstance(kind, sa.Numeric):
        return Decimal(value)
    if isinstance(kind, sa.Integer):
        return int(value)
    if isinstance(kind, sa.DateTime):
        return datetime.fromisoformat(value)
    if isinstance(kind, sa.Boolean):
        return value.strip().lower() in {"true", "t", "1"}
    return value


async def create_backup(
    conn: AsyncConnection, root: Path, *, now: datetime | None = None
) -> BackupInfo:
    """Write a new backup directory under `root`.

    Assembled in a `.partial` directory and renamed on success, so an
    interrupted run cannot leave something that looks like a usable backup.
    """
    now = now or datetime.now(UTC)
    final = root / f"btcpred-{now.strftime(STAMP)}"
    staging = final.with_suffix(".partial")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    rows: dict[str, int] = {}
    digests: dict[str, str] = {}
    for table in TABLES:
        target = staging / f"{table.name}.csv.gz"
        rows[table.name] = await _dump_table(conn, table, target)
        digests[target.name] = _digest(target)

    manifest = Manifest(
        created_at=now.isoformat(),
        alembic_revision=await _revision(conn),
        rows=rows,
        sha256=digests,
    )
    (staging / MANIFEST).write_text(manifest.to_json(), encoding="utf-8")

    if final.exists():
        shutil.rmtree(final)
    staging.rename(final)
    logger.info("backup %s written (%d rows)", final.name, sum(rows.values()))
    return BackupInfo(final, manifest)


def list_backups(root: Path) -> list[BackupInfo]:
    """Every readable backup under `root`, newest first."""
    if not root.exists():
        return []
    found: list[BackupInfo] = []
    for path in sorted(root.glob("btcpred-*"), reverse=True):
        if not path.is_dir():
            continue
        try:
            found.append(BackupInfo(path, Manifest.from_path(path)))
        except (BackupError, TypeError, json.JSONDecodeError):
            logger.warning("skipping unreadable backup %s", path.name)
    return found


def verify_backup(path: Path) -> Manifest:
    """Re-hash every file and compare against the manifest.

    A backup nobody has verified is a hope, not a backup.
    """
    manifest = Manifest.from_path(path)
    for name, expected in manifest.sha256.items():
        target = path / name
        if not target.exists():
            raise BackupError(f"{path.name}: {name} is missing")
        actual = _digest(target)
        if actual != expected:
            raise BackupError(f"{path.name}: {name} checksum mismatch")
    for table_name, count in manifest.rows.items():
        with gzip.open(path / f"{table_name}.csv.gz", "rt", encoding="utf-8", newline="") as fh:
            actual = sum(1 for _ in csv.DictReader(fh))
        if actual != count:
            raise BackupError(f"{path.name}: {table_name} has {actual} rows, manifest says {count}")
    return manifest


def prune(root: Path, keep: int) -> list[str]:
    """Delete all but the newest `keep` backups. Returns what was removed."""
    if keep < 1:
        raise ValueError("keep must be at least 1")
    removed = []
    for info in list_backups(root)[keep:]:
        shutil.rmtree(info.path)
        removed.append(info.name)
    if removed:
        logger.info("pruned %d old backup(s): %s", len(removed), ", ".join(removed))
    return removed


async def restore_backup(conn: AsyncConnection, path: Path) -> dict[str, int]:
    """Replace the contents of every backed-up table from `path`.

    Destructive by nature: the tables are truncated first. The caller is
    responsible for confirming intent — the CLI requires an explicit flag.

    The identity sequences are reset afterwards. Without that, the next insert
    would reuse an id the restored rows already hold, which surfaces much later
    as a primary key violation rather than at restore time.
    """
    manifest = verify_backup(path)

    current = await _revision(conn)
    if manifest.alembic_revision and current and manifest.alembic_revision != current:
        raise BackupError(
            f"backup is at revision {manifest.alembic_revision}, database is at {current}; "
            "bring them into line before restoring"
        )

    restored: dict[str, int] = {}
    # Truncate in reverse dependency order, load in forward order.
    await conn.execute(
        sa.text(f"TRUNCATE {', '.join(t.name for t in reversed(TABLES))} RESTART IDENTITY CASCADE")
    )

    for table in TABLES:
        source = path / f"{table.name}.csv.gz"
        with gzip.open(source, "rt", encoding="utf-8", newline="") as fh:
            rows = [
                {k: _decode(v, table.c[k]) for k, v in row.items()} for row in csv.DictReader(fh)
            ]
        if rows:
            await conn.execute(sa.insert(table), rows)
        restored[table.name] = len(rows)

        sequence = await conn.scalar(
            sa.text(f"SELECT pg_get_serial_sequence('{table.name}', 'id')")
        )
        if sequence and rows:
            await conn.execute(
                sa.text(f"SELECT setval('{sequence}', (SELECT max(id) FROM {table.name}))")
            )

    logger.info("restored %d rows from %s", sum(restored.values()), path.name)
    return restored


def table_names() -> list[str]:
    return [t.name for t in TABLES]


__all__ = [
    "TABLES",
    "BackupError",
    "BackupInfo",
    "Manifest",
    "create_backup",
    "list_backups",
    "metadata",
    "prune",
    "restore_backup",
    "table_names",
    "verify_backup",
]
