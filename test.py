#!/usr/bin/env python3

import argparse
import pathlib
import subprocess

_SCRIPT_DIR = pathlib.Path(__file__).parent
_TEST_SUITE_DIR = _SCRIPT_DIR / "craftinginterpreters" / "test"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        help="Test suite directory if not using the default set.",
        default=None,
    )
    parser.add_argument(
        "--init-submodules",
        dest="allow_submodule_init",
        help="Initialize submodules if they are not already.",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    try:
        suite: pathlib.Path
        if args.suite:
            if args.allow_submodule_init:
                _error(
                    "argument '--init-submodules' not allowed when"
                    "'--suite' is provided"
                )
                raise TestSetupError()

            _info("using user-provided test suite path")
            suite = pathlib.Path(args.suite)
        else:
            _init_submodules(allow_submodule_init=args.allow_submodule_init)
            suite = _TEST_SUITE_DIR

        _info(f"using test suite directory: {str(suite)}")
    except TestSetupError:
        exit(1)  # Assume relevant information is logged before raising.


def _init_submodules(allow_submodule_init: bool = False) -> None:
    if not _TEST_SUITE_DIR.exists():
        _info("submodule 'craftinginterpreters' not initialized")
        if not allow_submodule_init:
            _error("unable to run tests when submodules are not initialized")
            _error("rerun with '--init-submodules' to do so automatically")
            raise TestSetupError()
        _info("initializing submodules now")
        subprocess.run(
            ["git", "submodule", "update", "--init"],
            cwd=_SCRIPT_DIR,
        )
    _info(f"{str(_TEST_SUITE_DIR)} found, assuming submodules are initialized")


class TestSetupError(RuntimeError):
    pass


def _info(*args, **kwargs) -> None:
    print("[test.py]", *args, **kwargs)


def _error(*args, **kwargs) -> None:
    print("[ERROR|test.py]", *args, **kwargs)


if __name__ == "__main__":
    main()
