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
_SYNTAX_ERROR_PATTERN = re.compile(r"\[.*line (\d+)\] (Error.+)")
_STACK_TRACE_PATTERN = re.compile(r"\[line (\d+)\]")
_NONTEST_PATTERN = re.compile(r"// nontest")


class Language(enum.Enum):
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
        "-l",
        "--language",
        help="Language test suite to run.",
        choices=[lang.name.lower() for lang in Language],
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

        # TODO (bgluzman): filter loading for only selected suites...
        test_defs = [
            TestDefinition.from_file(path)
            for path in suite_root.glob("**/*.lox")
        ]
    except TestSetupError:
        exit(1)  # Assume relevant information is logged before raising.

    # TODO (bgluzman): support for running actual suites...
    for td in test_defs:
        if str(td.path).endswith("super/missing_arguments.lox"):
            print(Test(td, args.LOX_PATH).run())
            continue


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
    suite_type: Language | None = None

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
                suite_type = Language[language.upper()]
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
class TestDefinition:
    name: str
    path: pathlib.Path
    contents: str

    expected_outputs: list[ExpectedOutput]
    expected_compile_errors: list[ExpectedCompileError]
    expected_runtime_error: ExpectedRuntimeError | None

    def __repr__(self) -> str:
        return (
            f"TestDefinition(name={self.name}, path={self.path}, "
            f"contents={reprlib.repr(self.contents)}, "
            f"expected_outputs={self.expected_outputs}, "
            f"expected_compile_errors={self.expected_compile_errors}, "
            f"expected_runtime_errors={self.expected_runtime_error})"
        )

    def __str__(self) -> str:
        return self.name

    @property
    def expected_exit_code(self) -> int:
        if self.expected_compile_errors:
            return self.expected_compile_errors[0].exit_code
        elif self.expected_runtime_error:
            return self.expected_runtime_error.exit_code
        else:
            return 0

    @classmethod
    def from_file(cls, path: pathlib.Path) -> TestDefinition:
        name = path.stem
        contents = path.read_text()
        expected_outputs: list[ExpectedOutput] = []
        expected_compile_errors: list[ExpectedCompileError] = []
        expected_runtime_error: ExpectedRuntimeError | None = None
        for line_num, line in enumerate(contents.split("\n"), 1):
            if eo := ExpectedOutput.try_from_line(line_num, line):
                expected_outputs.append(eo)
                continue
            if ece := ExpectedCompileError.try_from_line(line_num, line):
                expected_compile_errors.append(ece)
                continue
            if ecr := ExpectedRuntimeError.try_from_line(line_num, line):
                expected_runtime_error = ecr
                continue

        # Unlikely to ever happen, but check just in case.
        if expected_compile_errors:
            exit_codes = [ecr.exit_code for ecr in expected_compile_errors]
            if not all(exit_codes[0] == ec for ec in exit_codes):
                raise TestSetupError("errors with differing exit codes")

        return cls(
            name=name,
            path=path,
            contents=contents,
            expected_outputs=expected_outputs,
            expected_compile_errors=expected_compile_errors,
            expected_runtime_error=expected_runtime_error,
        )


class TestSetupError(RuntimeError):
    pass


@dataclasses.dataclass
class Test:
    definition: TestDefinition
    executable: pathlib.Path

    @dataclasses.dataclass(frozen=True)
    class Result:
        failures: list[str]

        @property
        def is_success(self) -> bool:
            return not bool(self.failures)

        @property
        def is_fail(self) -> bool:
            return bool(self.failures)

    def run(self) -> Result:
        process = subprocess.run(
            [self.executable, self.definition.path],
            capture_output=True,
        )
        output = process.stdout.decode("utf-8")
        output_lines = [ol for ol in output.split("\n") if ol]
        error = process.stderr.decode("utf-8")
        error_lines = [el for el in error.split("\n") if el]
        exit_code = process.returncode

        failures: list[str] = []
        print(f"{output_lines=} {error_lines=}")
        if self.definition.expected_runtime_error:
            failures += self._validateRuntimeErrors(error_lines)
        else:
            failures += self._validateCompileErrors(error_lines)
        failures += self._validateExitCode(
            exit_code,
            error_lines,
        )
        failures += self._validateOutput(
            output_lines,
            error_lines,
        )

        return Test.Result(failures=failures)

    def _validateExitCode(
        self,
        actual: int,
        error_lines: list[str],
    ) -> list[str]:
        expected = self.definition.expected_exit_code
        if expected == actual:
            return []

        if len(error_lines) > 10:
            error_lines = error_lines[:10] + ["(truncated...)"]
        return [
            f"Expected return code {expected} and got {actual}. Stderr:",
            *error_lines,
        ]

    def _validateOutput(
        self,
        output_lines: list[str],
        error_lines: list[str],
    ) -> list[str]:
        failures: list[str] = []
        for index in range(0, len(output_lines)):
            line = output_lines[index]
            if index >= len(self.definition.expected_outputs):
                failures += [f"Got output '{line}' when none was expected"]
                continue

            expected = self.definition.expected_outputs[index]
            if expected.output != line:
                failures += [
                    f"Expected output '{expected.output}' on "
                    f"line {expected.line_num} and got {line}."
                ]

        index = len(output_lines)
        while index < len(self.definition.expected_outputs):
            expected = self.definition.expected_outputs[index]
            failures += [
                f"Missing expected output '{expected.output}' on "
                f"line {expected.line_num}."
            ]
            index += 1

        return failures

    def _validateRuntimeErrors(
        self,
        error_lines: list[str],
    ) -> list[str]:
        expected = self.definition.expected_runtime_error
        assert expected, "Should not be None in this context."

        if not error_lines:
            return [f"Expected runtime error '{expected.error}' and got none"]

        if error_lines[0] != expected.error:
            return [
                f"Expected runtime error '{expected.error}' and got:",
                error_lines[0],
            ]

        stack_match = None
        stack_lines = error_lines[1:]
        for line in stack_lines:
            stack_match = _STACK_TRACE_PATTERN.search(line)
            if stack_match:
                break
        else:
            # No matching line found...
            return ["Expected stack trace and got:", *stack_lines]

        stack_line = int(stack_match.group(1))
        if stack_line != expected.line_num:
            return [
                f"Expected runtime error on line {expected.line_num} "
                f"but was on line {stack_line}"
            ]

        return []

    def _validateCompileErrors(
        self,
        error_lines: list[str],
    ) -> list[str]:
        return []  # TODO (bgluzman)


def _info(*args, **kwargs) -> None:
    print("[test.py]", *args, **kwargs)


def _error(*args, **kwargs) -> None:
    print("[ERROR|test.py]", *args, file=sys.stderr, **kwargs)


if __name__ == "__main__":
    main()
