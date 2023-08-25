# lox-testrunner-python
Python-based Test Runner for Lox Language Test Suite (from "Crafting Interpreters")

This is a port of [`test.dart`](https://github.com/munificent/craftinginterpreters/blob/01e6f5b8f3e5dfa65674c2f9cf4700d73ab41cf8/tool/bin/test.dart) from the official [`craftinginterpreters` repo](https://github.com/munificent/craftinginterpreters). This is intended to make it easier to run the [test suite defined there](https://github.com/munificent/craftinginterpreters/tree/master/test) for those who find it easier to depend on Python rather than [Dart](https://dart.dev/).

This does **not** (currently) maintain compatibility with `test.dart`. In particular, an interpreter must be specified via the `-i` option. A few options present in `test.dart` are not supported. Please submit an issue if you would benefit from seeing these features added.

Note the above means the 'all' test suite is broken at the moment. Since the interpreter must be specified on invocation, Java and C suites cannot run successfully and simultaneously.