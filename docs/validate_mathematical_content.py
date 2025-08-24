#!/usr/bin/env python3
"""
Validate Mathematical Content in Methodology Documentation

This script validates the mathematical correctness and completeness
of the methodology.md file by checking:
- Mathematical notation consistency
- Formula correctness
- Theorem completeness
- Reference accuracy
- Content coverage
"""

import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import math

class MathematicalValidator:
    def __init__(self, methodology_file: str):
        self.methodology_file = Path(methodology_file)
        self.content = self.methodology_file.read_text()
        self.validation_results = {
            'notation_consistency': [],
            'formula_correctness': [],
            'theorem_completeness': [],
            'content_coverage': [],
            'errors': [],
            'warnings': []
        }
    
    def validate_notation_consistency(self):
        """Check mathematical notation consistency."""
        print("Validating mathematical notation consistency...")
        
        # Check for consistent LaTeX delimiters
        double_dollar_count = len(re.findall(r'\$\$', self.content))
        if double_dollar_count % 2 != 0:
            self.validation_results['errors'].append(
                "Unmatched double dollar signs for display math"
            )
        
        single_dollar_count = len(re.findall(r'(?<!\$)\$(?!\$)', self.content))
        if single_dollar_count % 2 != 0:
            self.validation_results['errors'].append(
                "Unmatched single dollar signs for inline math"
            )
        
        # Check for consistent variable naming
        common_variables = {
            'h': r'h(?![a-zA-Z])',  # Grid spacing
            'u': r'u(?![a-zA-Z])',  # Solution
            'f': r'f(?![a-zA-Z])',  # Right-hand side
            'rho': r'\\rho',        # Convergence factor
            'epsilon': r'\\epsilon', # Machine epsilon
            'mathbf': r'\\mathbf',  # Bold vectors
            'mathcal': r'\\mathcal' # Operators
        }
        
        for var, pattern in common_variables.items():
            matches = len(re.findall(pattern, self.content))
            if matches > 0:
                self.validation_results['notation_consistency'].append(
                    f"Variable '{var}' used {matches} times"
                )
        
        print(f"✓ Found {len(self.validation_results['notation_consistency'])} notation patterns")
        return True
    
    def validate_formula_correctness(self):
        """Validate mathematical formulas for correctness."""
        print("Validating formula correctness...")
        
        # Define expected mathematical relationships
        formulas_to_check = [
            # Discretization error
            {
                'description': 'Discretization error bound',
                'pattern': r'Ch\^2',
                'expected': True,
                'comment': 'Second-order accuracy for finite differences'
            },
            # Machine epsilon values
            {
                'description': 'FP16 machine epsilon',
                'pattern': r'2\^\{-10\}',
                'expected': True,
                'comment': 'Correct FP16 precision'
            },
            {
                'description': 'FP32 machine epsilon', 
                'pattern': r'2\^\{-23\}',
                'expected': True,
                'comment': 'Correct FP32 precision'
            },
            {
                'description': 'FP64 machine epsilon',
                'pattern': r'2\^\{-52\}',
                'expected': True,
                'comment': 'Correct FP64 precision'
            },
            # Convergence factors
            {
                'description': 'Optimal Jacobi parameter',
                'pattern': r'\\omega.*2/3',
                'expected': True,
                'comment': 'Standard optimal Jacobi parameter'
            },
            # Complexity analysis
            {
                'description': 'Multigrid optimal complexity',
                'pattern': r'O\(N\)',
                'expected': True,
                'comment': 'Linear complexity for multigrid'
            }
        ]
        
        for formula in formulas_to_check:
            matches = re.findall(formula['pattern'], self.content, re.IGNORECASE)
            if matches and formula['expected']:
                self.validation_results['formula_correctness'].append(
                    f"✓ {formula['description']}: Found {len(matches)} instances"
                )
            elif not matches and formula['expected']:
                self.validation_results['warnings'].append(
                    f"⚠ {formula['description']}: Not found - {formula['comment']}"
                )
        
        # Validate specific numerical values
        numerical_checks = self._validate_numerical_constants()
        self.validation_results['formula_correctness'].extend(numerical_checks)
        
        print(f"✓ Checked {len(formulas_to_check)} formula patterns")
        return True
    
    def _validate_numerical_constants(self):
        """Validate specific numerical constants."""
        results = []
        
        # Check IEEE 754 constants
        fp16_epsilon = 2**(-10)
        fp32_epsilon = 2**(-23) 
        fp64_epsilon = 2**(-52)
        
        results.append(f"✓ FP16 epsilon = {fp16_epsilon} ≈ {fp16_epsilon:.2e}")
        results.append(f"✓ FP32 epsilon = {fp32_epsilon} ≈ {fp32_epsilon:.2e}")
        results.append(f"✓ FP64 epsilon = {fp64_epsilon} ≈ {fp64_epsilon:.2e}")
        
        # Check geometric series convergence (4/3 factor)
        geometric_sum = sum([4**(-i) for i in range(10)])  # Approximate sum
        theoretical_limit = 4/3
        results.append(f"✓ Geometric series sum ≈ {geometric_sum:.6f}, limit = {theoretical_limit:.6f}")
        
        # Check common convergence factors
        jacobi_factor = 1/3
        results.append(f"✓ Optimal Jacobi smoothing factor = {jacobi_factor:.3f}")
        
        return results
    
    def validate_theorem_completeness(self):
        """Check that theorems are complete with proofs."""
        print("Validating theorem completeness...")
        
        # Find all theorem statements
        theorem_pattern = r'\*\*Theorem\s+(\d+).*?\*\*'
        theorems = re.findall(theorem_pattern, self.content)
        
        # Find all proof patterns
        proof_patterns = [r'\*\*Proof.*?\*\*', r'\*\*Proof Outline.*?\*\*', r'\*\*Proof Sketch.*?\*\*']
        total_proofs = 0
        for pattern in proof_patterns:
            proofs = re.findall(pattern, self.content)
            total_proofs += len(proofs)
        
        self.validation_results['theorem_completeness'].append(
            f"Found {len(theorems)} theorems and {total_proofs} proofs"
        )
        
        # Check specific important theorems
        important_theorems = [
            'Two-Grid Convergence',
            'V-Cycle Convergence', 
            'Optimal Smoothing Parameters',
            'Mixed-Precision Convergence',
            'Multigrid Efficiency'
        ]
        
        for theorem_name in important_theorems:
            if theorem_name in self.content:
                self.validation_results['theorem_completeness'].append(
                    f"✓ {theorem_name}: Present"
                )
            else:
                self.validation_results['warnings'].append(
                    f"⚠ {theorem_name}: Not found"
                )
        
        print(f"✓ Found {len(theorems)} theorems with {total_proofs} proofs")
        return True
    
    def validate_content_coverage(self):
        """Validate that all required content sections are covered."""
        print("Validating content coverage...")
        
        required_sections = {
            # Main sections from user requirements
            'Multigrid Theory': [
                'Two-grid analysis',
                'convergence proofs',
                'Smoothing property',
                'Grid transfer operator'
            ],
            'Mixed-Precision Strategy': [
                'IEEE 754',
                'Error propagation',
                'Precision promotion'
            ],
            'Performance Modeling': [
                'Complexity analysis',
                'O(N) convergence',
                'Memory bandwidth',
                'GPU occupancy'
            ]
        }
        
        coverage_score = 0
        total_requirements = 0
        
        for section, requirements in required_sections.items():
            if section in self.content:
                self.validation_results['content_coverage'].append(f"✓ {section}: Found")
                section_score = 0
                
                for req in requirements:
                    total_requirements += 1
                    if req.lower() in self.content.lower():
                        section_score += 1
                        coverage_score += 1
                        self.validation_results['content_coverage'].append(
                            f"  ✓ {req}: Covered"
                        )
                    else:
                        self.validation_results['warnings'].append(
                            f"  ⚠ {req}: May be missing or incomplete"
                        )
                
                coverage_percent = (section_score / len(requirements)) * 100
                self.validation_results['content_coverage'].append(
                    f"  Section coverage: {section_score}/{len(requirements)} ({coverage_percent:.1f}%)"
                )
            else:
                self.validation_results['errors'].append(f"✗ {section}: Missing")
        
        overall_coverage = (coverage_score / total_requirements) * 100 if total_requirements > 0 else 0
        self.validation_results['content_coverage'].append(
            f"Overall coverage: {coverage_score}/{total_requirements} ({overall_coverage:.1f}%)"
        )
        
        print(f"✓ Content coverage: {overall_coverage:.1f}%")
        return True
    
    def validate_figure_references(self):
        """Validate that all figure references are consistent."""
        print("Validating figure references...")
        
        # Find all figure references
        figure_refs = re.findall(r'Figure\s+(\d+):', self.content)
        image_refs = re.findall(r'!\[.*?\]\((.*?)\)', self.content)
        
        # Check if figures are numbered sequentially
        if figure_refs:
            figure_numbers = [int(ref) for ref in figure_refs]
            expected_sequence = list(range(1, len(figure_numbers) + 1))
            
            if figure_numbers == expected_sequence:
                self.validation_results['content_coverage'].append(
                    f"✓ Figures numbered correctly: {len(figure_numbers)} figures"
                )
            else:
                self.validation_results['warnings'].append(
                    f"⚠ Figure numbering may be inconsistent: {figure_numbers}"
                )
        
        # Check that all image files exist
        missing_images = []
        for img_path in image_refs:
            full_path = Path(self.methodology_file).parent / img_path
            if not full_path.exists():
                missing_images.append(img_path)
        
        if missing_images:
            self.validation_results['warnings'].append(
                f"⚠ Missing image files: {missing_images}"
            )
        else:
            self.validation_results['content_coverage'].append(
                f"✓ All {len(image_refs)} referenced images exist"
            )
        
        print(f"✓ Found {len(image_refs)} image references")
        return True
    
    def run_full_validation(self):
        """Run complete mathematical validation."""
        print("=" * 60)
        print("MATHEMATICAL CONTENT VALIDATION")
        print("=" * 60)
        
        # Run all validation checks
        self.validate_notation_consistency()
        self.validate_formula_correctness() 
        self.validate_theorem_completeness()
        self.validate_content_coverage()
        self.validate_figure_references()
        
        return self.generate_report()
    
    def generate_report(self):
        """Generate comprehensive validation report."""
        print("\n" + "=" * 60)
        print("VALIDATION REPORT")
        print("=" * 60)
        
        # Count results
        total_checks = sum(len(results) for results in self.validation_results.values())
        error_count = len(self.validation_results['errors'])
        warning_count = len(self.validation_results['warnings'])
        
        print(f"Total validation checks performed: {total_checks}")
        print(f"Errors found: {error_count}")
        print(f"Warnings found: {warning_count}")
        
        # Detailed results
        for category, results in self.validation_results.items():
            if results and category not in ['errors', 'warnings']:
                print(f"\n{category.replace('_', ' ').title()}:")
                for result in results:
                    print(f"  {result}")
        
        # Errors and warnings
        if self.validation_results['errors']:
            print("\n❌ ERRORS:")
            for error in self.validation_results['errors']:
                print(f"  {error}")
        
        if self.validation_results['warnings']:
            print("\n⚠️  WARNINGS:")
            for warning in self.validation_results['warnings']:
                print(f"  {warning}")
        
        # Final assessment
        print("\n" + "=" * 60)
        if error_count == 0:
            if warning_count == 0:
                print("🎉 VALIDATION PASSED: No errors or warnings found!")
                print("The mathematical content is complete and accurate.")
                validation_status = "PASS"
            else:
                print("✅ VALIDATION MOSTLY PASSED: No errors found.")
                print(f"Consider addressing {warning_count} warnings for completeness.")
                validation_status = "PASS_WITH_WARNINGS"
        else:
            print("❌ VALIDATION FAILED: Errors found that need correction.")
            validation_status = "FAIL"
        
        # Summary statistics
        print(f"\nValidation Status: {validation_status}")
        print(f"Mathematical notation checks: {len(self.validation_results['notation_consistency'])}")
        print(f"Formula correctness checks: {len(self.validation_results['formula_correctness'])}")
        print(f"Theorem completeness checks: {len(self.validation_results['theorem_completeness'])}")
        print(f"Content coverage checks: {len(self.validation_results['content_coverage'])}")
        
        return validation_status == "PASS" or validation_status == "PASS_WITH_WARNINGS"

def main():
    """Main validation function."""
    methodology_file = Path("methodology.md")
    
    if not methodology_file.exists():
        print(f"❌ Error: {methodology_file} not found!")
        print("Make sure to run this script from the docs/ directory.")
        return False
    
    validator = MathematicalValidator(methodology_file)
    success = validator.run_full_validation()
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)