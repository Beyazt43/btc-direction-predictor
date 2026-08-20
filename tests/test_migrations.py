from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _script_directory() -> ScriptDirectory:
    return ScriptDirectory.from_config(Config(str(PROJECT_ROOT / "alembic.ini")))


def test_single_head():
    """A second head means two migrations share a parent and will not both apply."""
    assert len(_script_directory().get_heads()) == 1


def test_every_revision_is_reversible():
    script = _script_directory()
    for revision in script.walk_revisions():
        source = Path(revision.path).read_text()
        assert "def downgrade()" in source, f"{revision.revision} has no downgrade"
        body = source.split("def downgrade()", 1)[1]
        assert "pass" not in body, f"{revision.revision} has an empty downgrade"
