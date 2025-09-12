#!/usr/bin/env python3
"""
Test runner script for GraphJudge Streamlit Pipeline.

This script provides a unified interface for running different types of tests
with proper configuration and reporting.

Usage:
    python run_tests.py                    # Run all unit tests
    python run_tests.py --integration      # Include integration tests
    python run_tests.py --performance      # Include performance tests
    python run_tests.py --coverage         # Generate coverage report
    python run_tests.py --all             # Run all tests including slow ones
"""

import subprocess
import sys
import argparse
from pathlib import Path


def run_command(cmd: list, description: str = None) -> int:
    """Run a command and return the exit code."""
    if description:
        print(f"\n{'=' * 60}")
        print(f"Running: {description}")
        print(f"{'=' * 60}")
    
    print(f"Running: {' '.join(cmd)}")
    print("-" * 40)
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    
    if result.returncode == 0:
        print(f"SUCCESS: {description or 'Command'} completed successfully")
    else:
        print(f"FAILED: {description or 'Command'} failed with exit code {result.returncode}")
    
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Run GraphJudge Streamlit Pipeline tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py                    # Basic unit tests
  python run_tests.py --integration      # Unit + integration tests
  python run_tests.py --coverage         # Tests with coverage report
  python run_tests.py --all --performance  # All tests including performance
  python run_tests.py --quick            # Quick smoke tests only
        """
    )
    
    # Test selection options
    parser.add_argument(
        "--unit", action="store_true", default=True,
        help="Run unit tests (default: True)"
    )
    parser.add_argument(
        "--integration", action="store_true", 
        help="Include integration tests"
    )
    parser.add_argument(
        "--performance", action="store_true",
        help="Include performance tests"
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Run only smoke tests"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all tests including slow ones"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run only quick tests (no slow tests)"
    )
    
    # Reporting options
    parser.add_argument(
        "--coverage", action="store_true",
        help="Generate test coverage report"
    )
    parser.add_argument(
        "--html-coverage", action="store_true",
        help="Generate HTML coverage report"
    )
    parser.add_argument(
        "--junit", action="store_true",
        help="Generate JUnit XML report"
    )
    
    # Test execution options
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose test output"
    )
    parser.add_argument(
        "--parallel", "-n", type=int, metavar="N",
        help="Run tests in parallel (N processes)"
    )
    parser.add_argument(
        "--fail-fast", "-x", action="store_true", 
        help="Stop on first failure"
    )
    parser.add_argument(
        "--pdb", action="store_true",
        help="Drop into debugger on failures"
    )
    
    # Test filtering
    parser.add_argument(
        "--pattern", "-k", type=str,
        help="Run tests matching pattern"
    )
    parser.add_argument(
        "--module", "-m", type=str,
        help="Run specific test module"
    )
    
    args = parser.parse_args()
    
    # Build pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Test selection
    test_paths = ["tests/"]
    
    if args.module:
        test_paths = [f"tests/{args.module}"]
    elif args.smoke:
        cmd.extend(["-m", "smoke"])
    elif args.quick:
        cmd.extend(["-m", "not slow and not performance"])
    elif args.performance:
        cmd.extend(["--performance"])
    elif args.integration:
        cmd.extend(["-m", "unit or integration"])
    
    # Add test paths
    cmd.extend(test_paths)
    
    # Test execution options
    if args.verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    
    if args.fail_fast:
        cmd.append("-x")
    
    if args.pdb:
        cmd.append("--pdb")
    
    if args.parallel:
        cmd.extend(["-n", str(args.parallel)])
    
    if args.pattern:
        cmd.extend(["-k", args.pattern])
    
    # Coverage options
    if args.coverage or args.html_coverage:
        cmd.extend([
            "--cov=core",
            "--cov=utils",
            "--cov-report=term-missing"
        ])
        
        if args.html_coverage:
            cmd.append("--cov-report=html:htmlcov")
    
    # Reporting options
    if args.junit:
        cmd.append("--junit-xml=test-results.xml")
    
    # Additional pytest options from pytest.ini will be automatically loaded
    
    print("GraphJudge Streamlit Pipeline Test Runner")
    print("=" * 60)
    print(f"Python: {sys.version}")
    print(f"Working directory: {Path.cwd()}")
    
    if args.all:
        cmd.extend(["--slow", "--e2e"])
        description = "Running ALL tests (unit, integration, performance, slow)"
    elif args.integration:
        description = "Running unit and integration tests"
    elif args.performance:
        description = "Running performance tests"
    elif args.smoke:
        description = "Running smoke tests"
    elif args.quick:
        description = "Running quick tests only"
    else:
        description = "Running unit tests"
    
    # Run the tests
    exit_code = run_command(cmd, description)
    
    # Generate additional reports if requested
    if args.html_coverage and exit_code == 0:
        print(f"\nCoverage report generated: {Path.cwd() / 'htmlcov' / 'index.html'}")
    
    if args.junit and exit_code == 0:
        print(f"\nJUnit report generated: {Path.cwd() / 'test-results.xml'}")
    
    # Final summary
    print(f"\n{'=' * 60}")
    if exit_code == 0:
        print("SUCCESS: All tests completed successfully!")
    else:
        print("FAILED: Some tests failed. Check the output above for details.")
        print("\nDebugging tips:")
        print("- Use --verbose for more detailed output")
        print("- Use --fail-fast to stop on first failure")
        print("- Use --pdb to debug failures interactively")
        print("- Use --pattern to run specific tests")
    print(f"{'=' * 60}")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())