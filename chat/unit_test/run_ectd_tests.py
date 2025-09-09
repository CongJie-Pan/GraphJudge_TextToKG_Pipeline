"""
ECTD Improvements Test Runner

This script runs comprehensive unit tests for all ECTD improvements and generates
detailed JSON reports for analysis and CI/CD integration.

Tests Covered:
- Entity Cleaner Tool (tools/clean_entities.py)
- Type Annotator Tool (tools/type_annotator.py)
- Alignment Checker Tool (tools/alignment_checker.py)
- Enhanced KIMI Entity Extraction (run_entity.py improvements)

Output:
- Individual JSON reports for each test module
- Combined test report with overall statistics
- Coverage analysis if pytest-cov is available
- Performance metrics and execution times

Usage:
    python run_ectd_tests.py [--verbose] [--coverage] [--output-dir DIR]
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any


class ECTDTestRunner:
    """
    Comprehensive test runner for ECTD improvements with detailed reporting.
    
    This class orchestrates the execution of all ECTD-related unit tests,
    generates individual and combined reports, and provides performance
    analysis for the testing process.
    """
    
    def __init__(self, output_dir: str = "test_reports", verbose: bool = False):
        """
        Initialize the test runner.
        
        Args:
            output_dir (str): Directory to store test reports
            verbose (bool): Enable verbose output
        """
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        self.test_results = {}
        self.start_time = None
        self.end_time = None
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Test modules to run
        self.test_modules = [
            {
                "name": "clean_entities",
                "file": "test_tools_clean_entities.py",
                "description": "Entity Cleaner Tool Tests",
                "report": "test_clean_entities_report.json"
            },
            {
                "name": "type_annotator", 
                "file": "test_tools_type_annotator.py",
                "description": "Type Annotator Tool Tests",
                "report": "test_type_annotator_report.json"
            },
            {
                "name": "alignment_checker",
                "file": "test_tools_alignment_checker.py", 
                "description": "Alignment Checker Tool Tests",
                "report": "test_alignment_checker_report.json"
            },
            {
                "name": "kimi_entity_improvements",
                "file": "test_run_entity.py",
                "description": "Enhanced KIMI Entity Extraction Tests", 
                "report": "test_kimi_entity_report.json"
            }
        ]
    
    def print_banner(self):
        """Print a formatted banner for the test execution."""
        print("=" * 80)
        print("🧪 ECTD IMPROVEMENTS COMPREHENSIVE TEST SUITE")
        print("=" * 80)
        print(f"📅 Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📂 Output Directory: {self.output_dir.absolute()}")
        print(f"🎯 Test Modules: {len(self.test_modules)}")
        print("=" * 80)
    
    def check_dependencies(self) -> bool:
        """
        Check if required dependencies are available.
        
        Returns:
            bool: True if all dependencies are available
        """
        print("🔍 Checking test dependencies...")
        
        try:
            import pytest
            print(f"✓ pytest version: {pytest.__version__}")
        except ImportError:
            print("✗ pytest not found. Install with: pip install pytest")
            return False
        
        try:
            import pytest_json_report
            print("✓ pytest-json-report available")
        except ImportError:
            print("⚠️  pytest-json-report not found. Installing...")
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "pytest-json-report"], 
                             check=True, capture_output=True)
                print("✓ pytest-json-report installed successfully")
            except subprocess.CalledProcessError:
                print("✗ Failed to install pytest-json-report")
                return False
        
        try:
            import pytest_cov
            print("✓ pytest-cov available for coverage analysis")
        except ImportError:
            print("⚠️  pytest-cov not found. Coverage analysis will be skipped.")
        
        print("✅ Dependency check completed\n")
        return True
    
    def run_single_test(self, test_module: Dict[str, str]) -> Dict[str, Any]:
        """
        Run a single test module and collect results.
        
        Args:
            test_module (Dict[str, str]): Test module configuration
            
        Returns:
            Dict[str, Any]: Test execution results
        """
        print(f"🧪 Running {test_module['description']}...")
        start_time = time.time()
        
        # Prepare pytest command
        report_path = self.output_dir / test_module['report']
        cmd = [
            sys.executable, "-m", "pytest",
            test_module['file'],
            "--json-report",
            f"--json-report-file={report_path}",
            "--tb=short"
        ]
        
        if self.verbose:
            cmd.append("-v")
        else:
            cmd.append("-q")
        
        # Execute test
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Parse test results
            if report_path.exists():
                with open(report_path, 'r', encoding='utf-8') as f:
                    json_report = json.load(f)
                
                test_result = {
                    "module": test_module['name'],
                    "description": test_module['description'],
                    "status": "PASSED" if result.returncode == 0 else "FAILED",
                    "return_code": result.returncode,
                    "execution_time": execution_time,
                    "tests_collected": json_report.get('summary', {}).get('collected', 0),
                    "tests_passed": json_report.get('summary', {}).get('passed', 0),
                    "tests_failed": json_report.get('summary', {}).get('failed', 0),
                    "tests_errors": json_report.get('summary', {}).get('error', 0),
                    "tests_skipped": json_report.get('summary', {}).get('skipped', 0),
                    "coverage": None,  # Will be added if coverage is available
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "report_path": str(report_path)
                }
            else:
                test_result = {
                    "module": test_module['name'],
                    "description": test_module['description'],
                    "status": "ERROR",
                    "return_code": result.returncode,
                    "execution_time": execution_time,
                    "error": "No JSON report generated",
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "report_path": None
                }
            
            # Print summary
            if test_result["status"] == "PASSED":
                print(f"✅ {test_module['name']}: {test_result['tests_passed']} passed, "
                     f"{test_result['tests_failed']} failed ({execution_time:.2f}s)")
            else:
                print(f"❌ {test_module['name']}: FAILED ({execution_time:.2f}s)")
                if self.verbose:
                    print(f"   Error: {result.stderr}")
            
            return test_result
            
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            
            error_result = {
                "module": test_module['name'],
                "description": test_module['description'],
                "status": "ERROR",
                "execution_time": execution_time,
                "error": str(e),
                "report_path": None
            }
            
            print(f"💥 {test_module['name']}: ERROR - {e}")
            return error_result
    
    def run_coverage_analysis(self) -> Dict[str, Any]:
        """
        Run coverage analysis for the tools directory.
        
        Returns:
            Dict[str, Any]: Coverage analysis results
        """
        print("📊 Running coverage analysis...")
        
        try:
            # Check if tools directory exists
            tools_dir = Path(__file__).parent.parent / "tools"
            if not tools_dir.exists():
                print("⚠️  Tools directory not found, skipping coverage analysis")
                return {"status": "skipped", "reason": "tools directory not found"}
            
            # Run coverage analysis
            coverage_report_path = self.output_dir / "coverage_report.json"
            cmd = [
                sys.executable, "-m", "pytest",
                "--cov=tools",
                "--cov-report=json:" + str(coverage_report_path),
                "--cov-report=term",
                "test_tools_*.py"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)
            
            if coverage_report_path.exists():
                with open(coverage_report_path, 'r', encoding='utf-8') as f:
                    coverage_data = json.load(f)
                
                print(f"✅ Coverage analysis completed: {coverage_data.get('totals', {}).get('percent_covered', 0):.1f}%")
                
                return {
                    "status": "completed",
                    "total_coverage": coverage_data.get('totals', {}).get('percent_covered', 0),
                    "lines_covered": coverage_data.get('totals', {}).get('covered_lines', 0),
                    "lines_missing": coverage_data.get('totals', {}).get('missing_lines', 0),
                    "report_path": str(coverage_report_path),
                    "stdout": result.stdout
                }
            else:
                print("⚠️  Coverage report not generated")
                return {"status": "failed", "reason": "no report generated"}
                
        except Exception as e:
            print(f"❌ Coverage analysis failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def generate_combined_report(self) -> None:
        """Generate a combined test report with overall statistics."""
        print("📋 Generating combined test report...")
        
        # Calculate overall statistics
        total_tests = sum(r.get('tests_collected', 0) for r in self.test_results.values())
        total_passed = sum(r.get('tests_passed', 0) for r in self.test_results.values())
        total_failed = sum(r.get('tests_failed', 0) for r in self.test_results.values())
        total_errors = sum(r.get('tests_errors', 0) for r in self.test_results.values())
        total_skipped = sum(r.get('tests_skipped', 0) for r in self.test_results.values())
        
        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        total_execution_time = self.end_time - self.start_time if self.end_time and self.start_time else 0
        
        # Create combined report
        combined_report = {
            "ectd_test_suite": {
                "execution_timestamp": datetime.now().isoformat(),
                "total_execution_time": total_execution_time,
                "python_version": sys.version,
                "working_directory": str(Path.cwd()),
                "output_directory": str(self.output_dir.absolute())
            },
            "overall_statistics": {
                "modules_tested": len(self.test_modules),
                "modules_passed": len([r for r in self.test_results.values() if r.get('status') == 'PASSED']),
                "modules_failed": len([r for r in self.test_results.values() if r.get('status') == 'FAILED']),
                "modules_error": len([r for r in self.test_results.values() if r.get('status') == 'ERROR']),
                "total_tests": total_tests,
                "total_passed": total_passed,
                "total_failed": total_failed,
                "total_errors": total_errors,
                "total_skipped": total_skipped,
                "success_rate": success_rate
            },
            "module_results": self.test_results,
            "recommendations": self._generate_recommendations()
        }
        
        # Save combined report
        combined_report_path = self.output_dir / "ectd_combined_test_report.json"
        with open(combined_report_path, 'w', encoding='utf-8') as f:
            json.dump(combined_report, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Combined report saved: {combined_report_path}")
    
    def _generate_recommendations(self) -> List[str]:
        """
        Generate recommendations based on test results.
        
        Returns:
            List[str]: List of recommendations
        """
        recommendations = []
        
        # Check overall success rate
        total_tests = sum(r.get('tests_collected', 0) for r in self.test_results.values())
        total_passed = sum(r.get('tests_passed', 0) for r in self.test_results.values())
        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        if success_rate == 100:
            recommendations.append("🎉 Excellent! All tests passed. ECTD improvements are working correctly.")
        elif success_rate >= 90:
            recommendations.append("✅ Good test coverage with high success rate. Address any failing tests.")
        elif success_rate >= 70:
            recommendations.append("⚠️ Moderate success rate. Review failing tests and improve code quality.")
        else:
            recommendations.append("🚨 Low success rate. Significant issues need to be addressed.")
        
        # Check for specific module issues
        failed_modules = [name for name, result in self.test_results.items() 
                         if result.get('status') != 'PASSED']
        
        if failed_modules:
            recommendations.append(f"🔧 Failed modules requiring attention: {', '.join(failed_modules)}")
        
        # Check execution time
        total_time = sum(r.get('execution_time', 0) for r in self.test_results.values())
        if total_time > 300:  # 5 minutes
            recommendations.append("⏰ Tests are taking a long time. Consider optimizing test performance.")
        
        return recommendations
    
    def print_summary(self):
        """Print a formatted summary of test results."""
        print("\n" + "=" * 80)
        print("📊 ECTD TEST EXECUTION SUMMARY")
        print("=" * 80)
        
        total_tests = sum(r.get('tests_collected', 0) for r in self.test_results.values())
        total_passed = sum(r.get('tests_passed', 0) for r in self.test_results.values())
        total_failed = sum(r.get('tests_failed', 0) for r in self.test_results.values())
        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        print(f"🎯 Total Tests: {total_tests}")
        print(f"✅ Passed: {total_passed}")
        print(f"❌ Failed: {total_failed}")
        print(f"📈 Success Rate: {success_rate:.1f}%")
        print(f"⏱️  Total Time: {self.end_time - self.start_time:.2f}s")
        
        print("\n📋 Module Results:")
        for name, result in self.test_results.items():
            status_icon = "✅" if result.get('status') == 'PASSED' else "❌"
            print(f"  {status_icon} {result.get('description', name)}: "
                 f"{result.get('tests_passed', 0)}/{result.get('tests_collected', 0)} "
                 f"({result.get('execution_time', 0):.2f}s)")
        
        print(f"\n📂 Reports saved to: {self.output_dir.absolute()}")
        print("=" * 80)
    
    def run_all_tests(self, include_coverage: bool = False) -> Dict[str, Any]:
        """
        Run all ECTD tests and generate comprehensive reports.
        
        Args:
            include_coverage (bool): Whether to include coverage analysis
            
        Returns:
            Dict[str, Any]: Overall test execution results
        """
        self.start_time = time.time()
        
        # Print banner
        self.print_banner()
        
        # Check dependencies
        if not self.check_dependencies():
            return {"status": "error", "message": "Dependency check failed"}
        
        # Run individual test modules
        print("🏃 Running individual test modules...\n")
        for test_module in self.test_modules:
            result = self.run_single_test(test_module)
            self.test_results[test_module['name']] = result
        
        # Run coverage analysis if requested
        coverage_result = None
        if include_coverage:
            coverage_result = self.run_coverage_analysis()
        
        self.end_time = time.time()
        
        # Generate combined report
        self.generate_combined_report()
        
        # Print summary
        self.print_summary()
        
        return {
            "status": "completed",
            "test_results": self.test_results,
            "coverage": coverage_result,
            "execution_time": self.end_time - self.start_time,
            "output_directory": str(self.output_dir.absolute())
        }


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive tests for ECTD improvements",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_ectd_tests.py
  python run_ectd_tests.py --verbose --coverage
  python run_ectd_tests.py --output-dir custom_reports
        """
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose test output'
    )
    
    parser.add_argument(
        '--coverage', '-c',
        action='store_true',
        help='Include coverage analysis'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        default='test_reports',
        help='Directory to store test reports (default: test_reports)'
    )
    
    args = parser.parse_args()
    
    # Initialize and run test suite
    runner = ECTDTestRunner(output_dir=args.output_dir, verbose=args.verbose)
    results = runner.run_all_tests(include_coverage=args.coverage)
    
    # Exit with appropriate code
    if results["status"] == "completed":
        failed_modules = len([r for r in results["test_results"].values() 
                            if r.get('status') != 'PASSED'])
        sys.exit(0 if failed_modules == 0 else 1)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
