from __future__ import annotations

from typing import TYPE_CHECKING

import os, json, glob as std_glob, logging
from io import TextIOWrapper
from zipfile import ZipFile
from pathlib import PurePosixPath

if TYPE_CHECKING:
    from typing import Any, IO, Union
    from collections.abc import Iterator

__all__ = [
    "open_file",
    "load_json",
    "save_json",
    "read_lines",
    "glob",
    "setup_logging"
]

def open_file(path: str, mode: str = "r", root: Union[str, ZipFile, None] = None) -> IO:
    # Single absolute or relative path
    if root is None:
        f = open(path, mode)
    # Relative path to file system root
    elif isinstance(root, str):
        f = open(os.path.join(root, path), mode)
    # Path within ZIP archive
    elif isinstance(root, ZipFile):
        is_text_mode = "b" not in mode
        zip_mode = mode.replace("b", "")

        f = root.open(path, zip_mode)
        if is_text_mode:
            f = TextIOWrapper(f)
    
    return f

def load_json(path: str, root: Union[str, ZipFile, None] = None, **kwargs: Any) -> Any:
    with open_file(path, root=root) as f:
        return json.load(f, **kwargs)

def save_json(obj: Any, path: str, root: Union[str, ZipFile, None] = None, **kwargs: Any):
    with open_file(path, "w", root=root) as f:
        json.dump(obj, f, **kwargs)

def read_lines(path: str, root: Union[str, ZipFile, None] = None) -> Iterator[str]:
    with open_file(path, root=root) as f:
        for line in f:
            line = line.strip()
            if line:
                yield line

def glob(pattern: str, root: Union[str, ZipFile, None] = None) -> list[str]:
    # Single absolute or relative path
    if root is None:
        return std_glob.glob(pattern)
    # Relative path to file system root
    elif isinstance(root, str):
        return std_glob.glob(os.path.join(root, pattern))
    # Path within ZIP archive
    elif isinstance(root, ZipFile):
        return [
            path for path in root.namelist() \
            if PurePosixPath(path).match(pattern)
        ]

def setup_logging(level: Union[str, int] = logging.INFO):
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=level
    )
