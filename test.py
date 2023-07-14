#!/usr/bin/env python3

import pathlib
import subprocess

_SCRIPT_DIR = pathlib.Path(__file__).parent
_TEST_SUITE_DIR = _SCRIPT_DIR / "craftinginterpreters" / "test"


def main() -> None:
    _init_submodules()


def _init_submodules() -> None:
    if not _TEST_SUITE_DIR.exists():
        _info("submodule 'craftinginterpreters' not initialized")
        _info("initializing submodules now")
        subprocess.run(
            ["git", "submodule", "update", "--init"],
            cwd=_SCRIPT_DIR,
        )
    _info(f"{str(_TEST_SUITE_DIR)} found, assuming submodules are initialized")


def _info(*args, **kwargs) -> None:
    print("[test.py]", *args, **kwargs)


if __name__ == "__main__":
    main()
