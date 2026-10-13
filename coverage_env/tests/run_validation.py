"""Run the independent Coverage Environment validation suite."""

from __future__ import annotations

import unittest
from pathlib import Path


class ValidationResult(unittest.TextTestResult):
    """Print an explicit PASS or FAIL line for every individual test."""

    def addSuccess(self, test):
        super().addSuccess(test)
        self.stream.writeln(f"PASS {test.id()}")

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.stream.writeln(f"FAIL {test.id()}")

    def addError(self, test, err):
        super().addError(test, err)
        self.stream.writeln(f"FAIL {test.id()} (error)")


class ValidationRunner(unittest.TextTestRunner):
    resultclass = ValidationResult


def main() -> int:
    """Discover and run all independent transition-validation files."""
    validation_dir = Path(__file__).with_name("validation")
    suite = unittest.defaultTestLoader.discover(
        start_dir=str(validation_dir), pattern="test_*.py",
        top_level_dir=str(Path(__file__).parents[1]),
    )
    result = ValidationRunner(verbosity=0).run(suite)
    status = "PASS" if result.wasSuccessful() else "FAIL"
    print(
        f"\nVALIDATION {status}: {result.testsRun - len(result.failures) - len(result.errors)}/"
        f"{result.testsRun} passed; {len(result.failures)} failed; "
        f"{len(result.errors)} errors."
    )
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
