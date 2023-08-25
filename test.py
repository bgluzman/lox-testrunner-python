#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import enum
import pathlib
import re
import subprocess
import sys

_SCRIPT_DIR = pathlib.Path(__file__).parent
_TESTS_ROOT_DIR = _SCRIPT_DIR / "craftinginterpreters" / "test"

_EXPECTED_OUTPUT_PATTERN = re.compile(r"// expect: ?(.*)")
_EXPECTED_ERROR_PATTERN = re.compile(r"// (Error.*)")
_EXPECTED_LINE_PATTERN = re.compile(r"// \[((java|c) )?line (\d+)\] (Error.*)")
_EXPECTED_RUNTIME_ERROR_PATTERN = re.compile(r"// expect runtime error: (.+)")
_SYNTAX_ERROR_PATTERN = re.compile(r"\[.*line (\d+)\] (Error.+)")
_STACK_TRACE_PATTERN = re.compile(r"\[line (\d+)\]")
_NONTEST_PATTERN = re.compile(r"// nontest")

# TODO (bgluzman): populate this for real
_BUILTIN_SUITE_SELECTIONS = {"all": {"development"}}


PassCount = int
FailCount = int
ExpectationCount = int


class Language(enum.Enum):
    JAVA = 1
    C = 2


def main() -> None:
    # TODO (bgluzman): add '--arguments' for interpreter args?
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--interpreter",
        help="Path to interpreter.",
        required=True,
    )
    parser.add_argument(
        "SUITE",
        help="Test suite name.",
    )
    parser.add_argument(
        "--tests-root",
        help="Root test file directory (if not using default).",
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
        tests_root: pathlib.Path
        if args.tests_root:
            if args.allow_submodule_init:
                _error(
                    "argument '--init-submodules' not allowed when"
                    "'--suite' is provided"
                )
                raise TestSetupError()

            _info("using user-provided test suite path")
            tests_root = pathlib.Path(args.tests_root)
        else:
            _info("using default test suite path")
            _init_submodules(allow_submodule_init=args.allow_submodule_init)
            tests_root = _TESTS_ROOT_DIR

        _info(f"using test suite directory: {str(tests_root)}")

        tests = {
            str(path.relative_to(tests_root)): Test.from_file(path)
            for path in tests_root.glob("**/*.lox")
        }
    except TestSetupError:
        exit(1)  # Assumes relevant information is logged before raising.

    # TODO (bgluzman): populate remaining suites...
    suites = _builtin_suites(args.interpreter)
    if args.SUITE == "all":
        run_suites(suites, _BUILTIN_SUITE_SELECTIONS["all"], tests)
    else:
        _error(f"unknown suite '{args.suite}'")
        exit(1)


def run_suites(
    suites: dict[str, Suite],
    selections: set[str],
    tests: dict[str, Test],
) -> None:
    any_failures: bool = False
    selected: list[Suite] = [suites[sel] for sel in selections]
    for suite in selected:
        print(f"=== {suite.name} ===")
        passed, failed, expectations = run_suite(suite, tests)
        if not failed:
            print(
                f"All {_green(passed)} tests passed "
                f"({expectations} expectations)."
            )
        else:
            any_failures = True
            print(
                f"{_green(passed)} tests passed. "
                f"{_red(failed)} tests failed."
            )
    if any_failures:
        exit(1)


def run_suite(
    suite: Suite,
    tests: dict[str, Test],
) -> tuple[PassCount, FailCount, ExpectationCount]:
    passed, failed, skipped, expectations = 0, 0, 0, 0
    for test_name, state in suite.tests.items():
        if state == "skip":
            skipped += 1
            continue

        # TODO (bgluzman): count expectations...
        test = tests[test_name]
        result = test.run(suite)
        if result.is_success:
            passed += 1
        else:
            assert result.is_fail, "Test cannot both fail and succeed."
            print(f"{_red('FAIL')} {test_name}")
            print("")
            for failure in result.failures:
                print(f"    {_yellow(failure)}")
            print("")
            failed += 1

    return passed, failed, expectations


class TestSetupError(RuntimeError):
    pass


def _init_submodules(allow_submodule_init: bool = False) -> None:
    if not _TESTS_ROOT_DIR.exists():
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
    _info(f"{str(_TESTS_ROOT_DIR)} found, assuming submodules are initialized")


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
    language: Language | None = None

    @classmethod
    def try_from_line(
        cls,
        line_num: int,
        line: str,
    ) -> ExpectedCompileError | None:
        if mat := _EXPECTED_ERROR_PATTERN.search(line):
            return cls(
                line_num=line_num,
                error=f"[line {line_num}] {mat.group(1)}",
            )
        if mat := _EXPECTED_LINE_PATTERN.search(line):
            language = None
            if language_match := mat.group(2):
                language = Language[language_match.upper()]
            return cls(
                line_num=line_num,
                error=f"[line {mat.group(3)}] {mat.group(4)}",
                language=language,
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
class Suite:
    name: str
    language: Language
    executable: pathlib.Path
    tests: dict[str, str]  # path str -> "pass" | "skip"


@dataclasses.dataclass(frozen=True)
class Test:
    name: str
    path: pathlib.Path

    expected_outputs: list[ExpectedOutput]
    expected_compile_errors: list[ExpectedCompileError]
    expected_runtime_error: ExpectedRuntimeError | None

    def __repr__(self) -> str:
        return (
            f"Test(name={self.name}, path={self.path}, "
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
    def from_file(
        cls,
        path: pathlib.Path,
    ) -> Test:
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
            expected_outputs=expected_outputs,
            expected_compile_errors=expected_compile_errors,
            expected_runtime_error=expected_runtime_error,
        )

    @dataclasses.dataclass(frozen=True)
    class Result:
        failures: list[str]

        @property
        def is_success(self) -> bool:
            return not bool(self.failures)

        @property
        def is_fail(self) -> bool:
            return bool(self.failures)

    def run(self, suite: Suite) -> Result:
        process = subprocess.run(
            [suite.executable, self.path],
            capture_output=True,
        )
        output = process.stdout.decode("utf-8")
        output_lines = [ol for ol in output.split("\n") if ol]
        error = process.stderr.decode("utf-8")
        error_lines = [el for el in error.split("\n") if el]
        exit_code = process.returncode

        failures: list[str] = []
        if self.expected_runtime_error:
            failures += self._validate_runtime_errors(error_lines)
        else:
            failures += self._validate_compile_errors(
                suite.language,
                error_lines,
            )
        failures += self._validate_exit_code(
            exit_code,
            error_lines,
        )
        failures += self._validate_output(
            output_lines,
            error_lines,
        )

        return Test.Result(failures=failures)

    def _validate_exit_code(
        self,
        actual: int,
        error_lines: list[str],
    ) -> list[str]:
        expected = self.expected_exit_code
        if expected == actual:
            return []

        if len(error_lines) > 10:
            error_lines = error_lines[:10] + ["(truncated...)"]
        return [
            f"Expected return code {expected} and got {actual}. Stderr:",
            *error_lines,
        ]

    def _validate_output(
        self,
        output_lines: list[str],
        error_lines: list[str],
    ) -> list[str]:
        failures: list[str] = []
        for index in range(0, len(output_lines)):
            line = output_lines[index]
            if index >= len(self.expected_outputs):
                failures += [f"Got output '{line}' when none was expected"]
                continue

            expected = self.expected_outputs[index]
            if expected.output != line:
                failures += [
                    f"Expected output '{expected.output}' on "
                    f"line {expected.line_num} and got {line}."
                ]

        index = len(output_lines)
        while index < len(self.expected_outputs):
            expected = self.expected_outputs[index]
            failures += [
                f"Missing expected output '{expected.output}' on "
                f"line {expected.line_num}."
            ]
            index += 1

        return failures

    def _validate_runtime_errors(
        self,
        error_lines: list[str],
    ) -> list[str]:
        expected = self.expected_runtime_error
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

    def _validate_compile_errors(
        self,
        suite_language: Language,
        error_lines: list[str],
    ) -> list[str]:
        expected_errors: set[str] = {
            ece.error
            for ece in self.expected_compile_errors
            if not ece.language or ece.language == suite_language
        }

        failures: list[str] = []
        found_errors: set[str] = set()
        unexpected_count: int = 0
        for line in error_lines:
            syntax_match = _SYNTAX_ERROR_PATTERN.search(line)
            if syntax_match:
                error = (
                    f"[line {syntax_match.group(1)}] "
                    f"{syntax_match.group(2)}"
                )
                if error in expected_errors:
                    found_errors.add(error)
                else:
                    if unexpected_count < 10:
                        failures += ["Unexpected output on stderr:", line]
                    unexpected_count += 1

        if unexpected_count > 10:
            failures += [f"(truncated {unexpected_count - 10} more...)"]

        for error in expected_errors - found_errors:
            failures += [f"Missing expected error: {error}"]

        return failures


def _builtin_suites(executable: pathlib.Path) -> dict[str, Suite]:
    return {
        "development": Suite(
            "development",
            Language.C,
            executable,
            {"class/local_inherit_self.lox": "pass"},
        )
    }


# from blender build scripts:
# https://svn.blender.org/
#   svnroot/bf-blender/trunk/blender/build_files/scons/tools/bcolors.py
class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def _color(color: str, message: str) -> str:
    return f"{color}{message}{bcolors.ENDC}"


def _green(message, **kwargs) -> str:
    return _color(bcolors.OKGREEN, str(message))


def _yellow(message, **kwargs) -> str:
    return _color(bcolors.WARNING, str(message))


def _red(message, **kwargs) -> str:
    return _color(bcolors.FAIL, str(message))


def _info(*args, **kwargs) -> None:
    print("[test.py]", *args, **kwargs)


def _error(*args, **kwargs) -> None:
    print("[ERROR|test.py]", *args, file=sys.stderr, **kwargs)


if __name__ == "__main__":
    main()
