"""Publish complete local artifacts without truncating the previous version."""
from contextlib import contextmanager
from pathlib import Path
import json
import os
import tempfile


@contextmanager
def atomic_output(path: Path, *, binary: bool = False):
    path = Path(path)
    options = {} if binary else {"encoding": "utf-8", "newline": ""}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb" if binary else "w", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp",
                                         delete=False, **options) as stream:
            temporary = Path(stream.name)
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def save_json(path: Path, value: dict) -> None:
    with atomic_output(path) as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
