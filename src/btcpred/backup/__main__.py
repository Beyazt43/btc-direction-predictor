"""Backup and restore from the command line.

python -m btcpred.backup create              # write a new backup, then prune
python -m btcpred.backup list                # what exists, with sizes
python -m btcpred.backup verify <name|path>  # re-hash and count, without restoring
python -m btcpred.backup restore <name|path> --yes-destroy-current-data
"""

import argparse
import asyncio
import logging
from pathlib import Path

from btcpred.backup.service import (
    BackupError,
    create_backup,
    list_backups,
    prune,
    restore_backup,
    verify_backup,
)
from btcpred.config import get_settings
from btcpred.db.session import get_engine


def _resolve(root: Path, name: str) -> Path:
    """Accept either a backup name or a full path."""
    candidate = Path(name)
    if candidate.is_dir():
        return candidate
    candidate = root / name
    if candidate.is_dir():
        return candidate
    raise BackupError(f"no backup named {name!r} under {root}")


async def cmd_create(args: argparse.Namespace) -> int:
    settings = get_settings()
    root = Path(settings.backup_dir)
    root.mkdir(parents=True, exist_ok=True)

    async with get_engine().begin() as conn:
        info = await create_backup(conn, root)

    verify_backup(info.path)
    removed = prune(root, args.keep or settings.backup_keep)

    print(f"{info.name}  {info.total_rows} rows  {info.size_bytes / 1024:.0f} KB  verified")
    for table, count in sorted(info.manifest.rows.items()):
        print(f"  {table:16} {count:8d}")
    if removed:
        print(f"pruned: {', '.join(removed)}")
    return 0


async def cmd_list(args: argparse.Namespace) -> int:
    root = Path(get_settings().backup_dir)
    found = list_backups(root)
    if not found:
        print(f"no backups under {root}")
        return 0
    for info in found:
        print(
            f"{info.name}  {info.total_rows:7d} rows  {info.size_bytes / 1024:6.0f} KB  "
            f"rev={info.manifest.alembic_revision}"
        )
    return 0


async def cmd_verify(args: argparse.Namespace) -> int:
    root = Path(get_settings().backup_dir)
    path = _resolve(root, args.name)
    manifest = verify_backup(path)
    print(f"{path.name}: OK ({sum(manifest.rows.values())} rows, rev={manifest.alembic_revision})")
    return 0


async def cmd_restore(args: argparse.Namespace) -> int:
    root = Path(get_settings().backup_dir)
    path = _resolve(root, args.name)

    if not args.yes_destroy_current_data:
        manifest = verify_backup(path)
        print(f"{path.name} verified: {sum(manifest.rows.values())} rows.")
        print("Restoring TRUNCATES every backed-up table and replaces it with this snapshot.")
        print("Re-run with --yes-destroy-current-data to proceed.")
        return 1

    async with get_engine().begin() as conn:
        restored = await restore_backup(conn, path)
    for table, count in sorted(restored.items()):
        print(f"  {table:16} {count:8d}")
    print("restored; run the scheduler to resume")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Back up and restore the prediction record.")
    sub = parser.add_subparsers(dest="command", required=True)

    create = sub.add_parser("create", help="write a new backup and prune old ones")
    create.add_argument("--keep", type=int, default=None, help="override BACKUP_KEEP")
    create.set_defaults(func=cmd_create)

    listing = sub.add_parser("list", help="show existing backups")
    listing.set_defaults(func=cmd_list)

    verify = sub.add_parser("verify", help="re-hash and re-count a backup without restoring")
    verify.add_argument("name")
    verify.set_defaults(func=cmd_verify)

    restore = sub.add_parser("restore", help="replace current data with a backup")
    restore.add_argument("name")
    restore.add_argument(
        "--yes-destroy-current-data",
        action="store_true",
        help="required: restoring truncates the backed-up tables first",
    )
    restore.set_defaults(func=cmd_restore)

    args = parser.parse_args()
    logging.basicConfig(
        level=get_settings().log_level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    async def run() -> int:
        try:
            return await args.func(args)
        except BackupError as exc:
            print(f"error: {exc}")
            return 1
        finally:
            await get_engine().dispose()

    raise SystemExit(asyncio.run(run()))


if __name__ == "__main__":
    main()
