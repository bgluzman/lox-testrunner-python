#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import enum
import pathlib
import re
import reprlib
import subprocess
import sys

_SCRIPT_DIR = pathlib.Path(__file__).parent
_TEST_SUITE_DIR = _SCRIPT_DIR / "craftinginterpreters" / "test"

_EXPECTED_OUTPUT_PATTERN = re.compile(r"// expect: ?(.*)")
_EXPECTED_ERROR_PATTERN = re.compile(r"// (Error.*)")
_EXPECTED_LINE_PATTERN = re.compile(r"// \[((java|c) )?line (\d+)\] (Error.*)")
_EXPECTED_RUNTIME_ERROR_PATTERN = re.compile(r"// expect runtime error: (.+)")
_EXPECTE_NONTEST_PATTERN = re.compile(r"// nontest")


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
            _info("using default test suite path")
            _init_submodules(allow_submodule_init=args.allow_submodule_init)
            suite_root = _TEST_SUITE_DIR

        _info(f"using test suite directory: {str(suite_root)}")

        tests = [Test.from_file(path) for path in suite_root.glob("**/*.lox")]
    except TestSetupError:
        exit(1)  # Assume relevant information is logged before raising.

    # placeholder for now...
    import pprint

    test_expects = {
        t.name: {
            "output": [
                ExpectedOutput.try_from_line(i + 1, l)
                for i, l in enumerate(t.contents.split("\n"))
                if _EXPECTED_OUTPUT_PATTERN.search(l)
            ],
            "compileErrors": [
                ExpectedCompileError.try_from_line(i + 1, l)
                for i, l in enumerate(t.contents.split("\n"))
                if _EXPECTED_ERROR_PATTERN.search(l)
                or _EXPECTED_LINE_PATTERN.search(l)
            ],
            "runtimeErrors": [
                ExpectedRuntimeError.try_from_line(i + 1, l)
                for i, l in enumerate(t.contents.split("\n"))
                if _EXPECTED_RUNTIME_ERROR_PATTERN.search(l)
            ],
        }
        for t in tests
    }
    pprint.pprint(test_expects)


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
class ExpectedOutput:
    line_num: int
    output: str

    @classmethod
    def try_from_line(cls, line_num: int, line: str) -> ExpectedOutput | None:
        mat = _EXPECTED_OUTPUT_PATTERN.search(line)
        if mat:
            return cls(line_num=line_num, output=mat.group(1))
        return None


@dataclasses.dataclass(frozen=True)
class ExpectedCompileError:
    line_num: int
    error: str
    exit_code: int = 65
    suite_type: SuiteType | None = None

    @classmethod
    def try_from_line(
        cls,
        line_num: int,
        line: str,
    ) -> ExpectedCompileError | None:
        if mat := _EXPECTED_ERROR_PATTERN.search(line):
            return cls(line_num=line_num, error=mat.group(1))
        if mat := _EXPECTED_LINE_PATTERN.search(line):
            suite_type = None
            if language := mat.group(2):
                suite_type = SuiteType[language.upper()]
            return cls(
                line_num=line_num,
                error=mat.group(1),
                suite_type=suite_type,
            )
        return None


@dataclasses.dataclass(frozen=True)
class ExpectedRuntimeError:
    line_num: int
    error: str
    exit_code: int = 70

    @classmethod
    def try_from_line(
        cls,
        line_num: int,
        line: str,
    ) -> ExpectedRuntimeError | None:
        mat = _EXPECTED_RUNTIME_ERROR_PATTERN.search(line)
        if mat:
            return cls(line_num=line_num, error=mat.group(1))
        return None


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
