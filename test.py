#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import enum
import pathlib
import reprlib
import subprocess
import sys

_SCRIPT_DIR = pathlib.Path(__file__).parent
_TEST_SUITE_DIR = _SCRIPT_DIR / "craftinginterpreters" / "test"


class SuiteType(enum.Enum):
    ALL = 1
    JAVA = 2
    C = 3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "LOX_PATH",
        help="Path to Lox interpreter under test.",
    )
    parser.add_argument(
        "-s",
        "--suite",
        help="Test suite to run.",
        choices=[st.name.lower() for st in SuiteType],
        required=True,
    )
    parser.add_argument(
        "-r",
        "--suite-root",
        help="Test suite root directory if not using the default.",
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
        suite_root: pathlib.Path
        if args.suite_root:
            if args.allow_submodule_init:
                _error(
                    "argument '--init-submodules' not allowed when"
                    "'--suite' is provided"
                )
                raise TestSetupError()

            _info("using user-provided test suite path")
            suite_root = pathlib.Path(args.suite_root)
        else:
            _init_submodules(allow_submodule_init=args.allow_submodule_init)
            suite_root = _TEST_SUITE_DIR

        _info(f"using test suite directory: {str(suite_root)}")

        tests = [Test.from_file(path) for path in suite_root.glob("**/*.lox")]
    except TestSetupError:
        exit(1)  # Assume relevant information is logged before raising.

    # placeholder for now...
    import pprint

    pprint.pprint(tests)


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


@dataclasses.dataclass(frozen=True)
class Test:
    name: str
    contents: str

    def __repr__(self) -> str:
        return (
            f"Test(name={self.name}, "
            f"contents={reprlib.repr(self.contents)})"
        )

    def __str__(self) -> str:
        return self.name

    @classmethod
    def from_file(cls, path: pathlib.Path) -> Test:
        return cls(name=path.stem, contents=path.read_text())


class TestSetupError(RuntimeError):
    pass


def _info(*args, **kwargs) -> None:
    print("[test.py]", *args, **kwargs)


def _error(*args, **kwargs) -> None:
    print("[ERROR|test.py]", *args, file=sys.stderr, **kwargs)


if __name__ == "__main__":
    main()
