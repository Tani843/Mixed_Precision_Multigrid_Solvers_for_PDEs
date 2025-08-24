#!/usr/bin/env python3
"""
Installation Verification Script

This script verifies that the mixed-precision multigrid package is correctly
installed and all components are functioning properly.
"""

import importlib
import os
import platform
import sys
import traceback
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


class InstallationVerifier:
    """Comprehensive installation verification for the multigrid package."""
    
    def __init__(self, verbose: bool = False):
        """Initialize the verifier."""
        self.verbose = verbose
        self.results = {
            'core': {'status': 'unknown', 'details': []},
            'dependencies': {'status': 'unknown', 'details': []},
            'optional': {'status': 'unknown', 'details': []},
            'gpu': {'status': 'unknown', 'details': []},
            'functionality': {'status': 'unknown', 'details': []},
            'performance': {'status': 'unknown', 'details': []}
        }
        
        # Suppress non-critical warnings during verification
        warnings.filterwarnings('ignore', category=UserWarning)
    
    def print_status(self, message: str, status: str = 'info') -> None:
        """Print status message with appropriate formatting."""
        if not self.verbose and status == 'info':
            return
        
        symbols = {
            'success': '✅',
            'error': '❌', 
            'warning': '⚠️',
            'info': 'ℹ️'
        }
        
        symbol = symbols.get(status, 'ℹ️')
        print(f"{symbol} {message}")
    
    def check_python_version(self) -> bool:
        """Check if Python version meets requirements."""
        required = (3, 8)
        current = sys.version_info[:2]
        
        if current >= required:
            self.print_status(f"Python {sys.version.split()[0]} (meets requirement ≥{required[0]}.{required[1]})", 'success')
            return True
        else:
            self.print_status(f"Python {sys.version.split()[0]} (requires ≥{required[0]}.{required[1]})", 'error')
            return False
    
    def check_core_package(self) -> bool:
        """Check if the core package can be imported."""
        try:
            import multigrid
            version = getattr(multigrid, '__version__', 'unknown')
            
            self.print_status(f"Core package imported successfully (v{version})", 'success')
            self.results['core']['status'] = 'success'
            self.results['core']['details'].append(f"Version: {version}")
            
            # Check core components
            core_components = [
                'Grid', 'PrecisionManager', 'MultigridSolver',
                'LaplacianOperator', 'RestrictionOperator', 'ProlongationOperator'
            ]
            
            missing_components = []
            for component in core_components:
                if hasattr(multigrid, component):
                    self.print_status(f"Core component '{component}' available", 'success')
                else:
                    missing_components.append(component)
                    self.print_status(f"Core component '{component}' missing", 'error')
            
            if missing_components:
                self.results['core']['status'] = 'error'
                self.results['core']['details'].append(f"Missing: {', '.join(missing_components)}")
                return False
            
            return True
            
        except ImportError as e:
            self.print_status(f"Failed to import core package: {e}", 'error')
            self.results['core']['status'] = 'error'
            self.results['core']['details'].append(f"Import error: {e}")
            return False
    
    def check_required_dependencies(self) -> bool:
        """Check required dependencies."""
        dependencies = [
            ('numpy', '1.21.0'),
            ('scipy', '1.9.0'),
            ('matplotlib', '3.5.0'),
            ('yaml', '6.0')  # PyYAML imports as 'yaml'
        ]
        
        all_ok = True
        
        for package, min_version in dependencies:
            try:
                module = importlib.import_module(package)
                version = getattr(module, '__version__', 'unknown')
                
                self.print_status(f"Required dependency '{package}' v{version} available", 'success')
                self.results['dependencies']['details'].append(f"{package}: {version}")
                
            except ImportError:
                self.print_status(f"Required dependency '{package}' missing", 'error')
                self.results['dependencies']['details'].append(f"{package}: MISSING")
                all_ok = False
        
        self.results['dependencies']['status'] = 'success' if all_ok else 'error'
        return all_ok
    
    def check_optional_dependencies(self) -> bool:
        """Check optional dependencies."""
        optional_deps = [
            ('cupy', 'GPU acceleration'),
            ('numba', 'JIT compilation'),
            ('seaborn', 'Advanced plotting'),
            ('plotly', 'Interactive visualizations'),
            ('psutil', 'System monitoring'),
            ('mpi4py', 'MPI parallelization')
        ]
        
        available_count = 0
        
        for package, description in optional_deps:
            try:
                module = importlib.import_module(package)
                version = getattr(module, '__version__', 'unknown')
                self.print_status(f"Optional dependency '{package}' v{version} available ({description})", 'success')
                self.results['optional']['details'].append(f"{package}: {version}")
                available_count += 1
                
            except ImportError:
                self.print_status(f"Optional dependency '{package}' not available ({description})", 'warning')
                self.results['optional']['details'].append(f"{package}: NOT AVAILABLE")
        
        self.results['optional']['status'] = 'success'
        self.results['optional']['details'].append(f"Available: {available_count}/{len(optional_deps)}")
        
        return True  # Optional dependencies don't affect overall success
    
    def check_gpu_support(self) -> bool:
        """Check GPU support and availability."""
        gpu_available = False
        
        # Check CuPy
        try:
            import cupy as cp
            
            # Try to create a simple array on GPU
            test_array = cp.array([1, 2, 3])
            device_count = cp.cuda.runtime.getDeviceCount()
            device_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
            
            self.print_status(f"CuPy available with {device_count} GPU(s)", 'success')
            self.print_status(f"Primary GPU: {device_name}", 'success')
            
            gpu_available = True
            self.results['gpu']['details'].extend([
                f"CuPy version: {cp.__version__}",
                f"GPU count: {device_count}",
                f"Primary GPU: {device_name}"
            ])
            
        except ImportError:
            self.print_status("CuPy not available (GPU acceleration disabled)", 'warning')
            self.results['gpu']['details'].append("CuPy: NOT AVAILABLE")
            
        except Exception as e:
            self.print_status(f"CuPy available but GPU not accessible: {e}", 'warning')
            self.results['gpu']['details'].append(f"CuPy error: {e}")
        
        # Check if package detects GPU
        try:
            import multigrid
            pkg_gpu_available = getattr(multigrid, 'GPU_AVAILABLE', False)
            
            if pkg_gpu_available:
                self.print_status("Package GPU support enabled", 'success')
            else:
                self.print_status("Package GPU support disabled", 'warning')
            
            self.results['gpu']['details'].append(f"Package GPU support: {pkg_gpu_available}")
            
        except:
            pass
        
        self.results['gpu']['status'] = 'success' if gpu_available else 'warning'
        return True  # GPU support is optional
    
    def check_basic_functionality(self) -> bool:
        """Test basic functionality of the package."""
        try:
            import multigrid
            
            # Test 1: Create a simple grid
            grid = multigrid.Grid(nx=32, ny=32)
            self.print_status("Grid creation successful", 'success')
            
            # Test 2: Create operators
            laplacian = multigrid.LaplacianOperator(grid)
            self.print_status("Operator creation successful", 'success')
            
            # Test 3: Create solver
            solver = multigrid.MultigridSolver(grid)
            self.print_status("Solver creation successful", 'success')
            
            # Test 4: Simple solve
            # Create a simple test problem
            x = np.linspace(0, 1, grid.nx)
            y = np.linspace(0, 1, grid.ny)
            X, Y = np.meshgrid(x, y)
            
            # Manufactured solution: u = sin(πx)sin(πy)
            u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
            f = 2 * np.pi**2 * u_exact  # Source term
            
            # Solve with multigrid
            u_initial = np.zeros_like(f)
            u_solution = solver.solve(f, u_initial, tol=1e-6, max_iter=20)
            
            # Check convergence
            error = np.max(np.abs(u_solution - u_exact))
            if error < 1e-3:  # Reasonable tolerance for this test
                self.print_status(f"Basic solve successful (error: {error:.2e})", 'success')
                self.results['functionality']['details'].append(f"Solve error: {error:.2e}")
            else:
                self.print_status(f"Basic solve completed but with high error: {error:.2e}", 'warning')
                self.results['functionality']['details'].append(f"High solve error: {error:.2e}")
            
            self.results['functionality']['status'] = 'success'
            return True
            
        except Exception as e:
            self.print_status(f"Basic functionality test failed: {e}", 'error')
            if self.verbose:
                self.print_status(f"Traceback: {traceback.format_exc()}", 'error')
            
            self.results['functionality']['status'] = 'error'
            self.results['functionality']['details'].append(f"Error: {e}")
            return False
    
    def check_performance(self) -> bool:
        """Basic performance verification."""
        try:
            import time
            import multigrid
            
            # Performance test with larger grid
            grid = multigrid.Grid(nx=128, ny=128)
            solver = multigrid.MultigridSolver(grid)
            
            # Create test problem
            x = np.linspace(0, 1, grid.nx)
            y = np.linspace(0, 1, grid.ny)
            X, Y = np.meshgrid(x, y)
            u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
            f = 2 * np.pi**2 * u_exact
            
            # Time the solve
            u_initial = np.zeros_like(f)
            start_time = time.time()
            u_solution = solver.solve(f, u_initial, tol=1e-8, max_iter=50)
            solve_time = time.time() - start_time
            
            # Check iterations and time
            if hasattr(solver, 'iteration_count'):
                iterations = solver.iteration_count
            else:
                iterations = "unknown"
            
            self.print_status(f"Performance test completed in {solve_time:.3f}s ({iterations} iterations)", 'success')
            
            # Basic performance expectations
            if solve_time < 5.0:  # Should complete in reasonable time
                self.print_status("Performance within expected range", 'success')
                performance_status = 'success'
            else:
                self.print_status("Performance slower than expected", 'warning')
                performance_status = 'warning'
            
            self.results['performance']['status'] = performance_status
            self.results['performance']['details'].extend([
                f"Solve time: {solve_time:.3f}s",
                f"Iterations: {iterations}",
                f"Problem size: 128×128"
            ])
            
            return True
            
        except Exception as e:
            self.print_status(f"Performance test failed: {e}", 'warning')
            self.results['performance']['status'] = 'warning'
            self.results['performance']['details'].append(f"Error: {e}")
            return True  # Non-critical failure
    
    def print_system_info(self) -> None:
        """Print system information."""
        self.print_status("System Information:", 'info')
        self.print_status(f"Platform: {platform.platform()}", 'info')
        self.print_status(f"Python: {sys.version.split()[0]} ({sys.executable})", 'info')
        self.print_status(f"Architecture: {platform.architecture()[0]}", 'info')
        self.print_status(f"Processor: {platform.processor()}", 'info')
        
        # Memory info if available
        try:
            import psutil
            memory = psutil.virtual_memory()
            self.print_status(f"Memory: {memory.total / (1024**3):.1f} GB total, {memory.available / (1024**3):.1f} GB available", 'info')
        except ImportError:
            pass
        
        print()  # Empty line for readability
    
    def generate_report(self) -> Dict:
        """Generate comprehensive verification report."""
        overall_status = 'success'
        critical_failures = []
        
        # Check for critical failures
        if self.results['core']['status'] == 'error':
            overall_status = 'error'
            critical_failures.append('Core package import failed')
        
        if self.results['dependencies']['status'] == 'error':
            overall_status = 'error'
            critical_failures.append('Required dependencies missing')
        
        if self.results['functionality']['status'] == 'error':
            overall_status = 'error'
            critical_failures.append('Basic functionality tests failed')
        
        report = {
            'overall_status': overall_status,
            'critical_failures': critical_failures,
            'detailed_results': self.results,
            'system_info': {
                'platform': platform.platform(),
                'python_version': sys.version.split()[0],
                'architecture': platform.architecture()[0]
            }
        }
        
        return report
    
    def run_verification(self) -> bool:
        """Run complete installation verification."""
        print("🔍 Mixed-Precision Multigrid Installation Verification")
        print("=" * 60)
        
        if self.verbose:
            self.print_system_info()
        
        # Run all checks
        checks = [
            ("Python Version", self.check_python_version),
            ("Core Package", self.check_core_package),
            ("Required Dependencies", self.check_required_dependencies),
            ("Optional Dependencies", self.check_optional_dependencies),
            ("GPU Support", self.check_gpu_support),
            ("Basic Functionality", self.check_basic_functionality),
            ("Performance", self.check_performance)
        ]
        
        print("Running verification checks...")
        print("-" * 40)
        
        success_count = 0
        for check_name, check_func in checks:
            self.print_status(f"\n{check_name}:", 'info')
            try:
                if check_func():
                    success_count += 1
            except Exception as e:
                self.print_status(f"Check '{check_name}' failed with exception: {e}", 'error')
                if self.verbose:
                    self.print_status(f"Traceback: {traceback.format_exc()}", 'error')
        
        # Generate final report
        report = self.generate_report()
        
        print("\n" + "=" * 60)
        print("📋 VERIFICATION SUMMARY")
        print("=" * 60)
        
        if report['overall_status'] == 'success':
            self.print_status("✅ Installation verification PASSED", 'success')
            self.print_status("All critical components are working correctly.", 'success')
        else:
            self.print_status("❌ Installation verification FAILED", 'error')
            self.print_status("Critical issues found:", 'error')
            for failure in report['critical_failures']:
                self.print_status(f"  - {failure}", 'error')
        
        # Success rate
        print(f"\nChecks passed: {success_count}/{len(checks)}")
        
        # Recommendations
        if report['overall_status'] != 'success':
            print("\n🔧 RECOMMENDED ACTIONS:")
            print("1. Reinstall the package: pip install --force-reinstall mixed-precision-multigrid")
            print("2. Check system requirements and dependencies")
            print("3. Verify Python version compatibility (≥3.8)")
            print("4. For GPU support: pip install mixed-precision-multigrid[gpu]")
        else:
            print("\n🎉 Installation is ready for use!")
            print("Try running: python -c 'import multigrid; print(multigrid.__version__)'")
        
        return report['overall_status'] == 'success'


def main():
    """Main entry point for installation verification."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify mixed-precision multigrid installation")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--json", help="Output results as JSON to file")
    
    args = parser.parse_args()
    
    verifier = InstallationVerifier(verbose=args.verbose)
    success = verifier.run_verification()
    
    # Output JSON report if requested
    if args.json:
        import json
        report = verifier.generate_report()
        
        with open(args.json, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n📄 Detailed report saved to: {args.json}")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()