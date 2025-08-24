---
layout: default
title: Home
permalink: /
---

# Mixed-Precision Multigrid Solvers for PDEs

High-performance numerical methods combining multigrid techniques with mixed-precision arithmetic for efficient partial differential equation solving on modern hardware architectures.

## Project Overview

This project presents a comprehensive framework for solving partial differential equations using **mixed-precision multigrid methods**. By intelligently combining single and double precision arithmetic, we achieve significant performance improvements while maintaining computational accuracy.

### Key Features

- **High-Performance Computing**: Optimized for both CPU and GPU architectures
- **Mixed-Precision Arithmetic**: Intelligent precision switching for optimal performance/accuracy trade-offs  
- **Comprehensive Validation**: Method of Manufactured Solutions (MMS) framework for rigorous testing
- **Scalable Implementation**: Efficient algorithms with proven convergence properties
- **Production-Ready**: Extensive testing and benchmarking suite

## Quick Results

<div class="results-grid">
  <div class="result-card">
    <h3>GPU Acceleration</h3>
    <div class="metric">{{ site.data.benchmarks.cpu_gpu_speedup }}</div>
    <p>Maximum speedup over CPU implementation</p>
  </div>
  
  <div class="result-card">
    <h3>Mixed Precision</h3>
    <div class="metric">{{ site.data.benchmarks.mixed_precision_speedup }}</div>
    <p>Performance gain with mixed precision</p>
  </div>
  
  <div class="result-card">
    <h3>Memory Savings</h3>
    <div class="metric">{{ site.data.benchmarks.memory_savings }}</div>
    <p>Reduction in memory usage</p>
  </div>
  
  <div class="result-card">
    <h3>Convergence</h3>
    <div class="metric">{{ site.data.benchmarks.convergence_rate }}</div>
    <p>Optimal theoretical convergence rate</p>
  </div>
</div>

## Applications

Our framework successfully solves:

- **Poisson Equations**: Steady-state elliptic problems with various boundary conditions
- **Heat Equations**: Time-dependent parabolic problems with implicit time stepping
- **Complex Geometries**: Support for irregular domains and mixed boundary conditions
- **Multi-Scale Problems**: Efficient handling of problems with disparate length scales

## Performance Highlights

![Convergence Analysis]({{ '/assets/images/convergence_analysis.png' | relative_url }})
*Grid convergence study showing optimal O(h²) convergence rates across different problem types*

![CPU vs GPU Performance]({{ '/assets/images/performance_scaling.png' | relative_url }})  
*Performance comparison demonstrating significant GPU acceleration for large-scale problems*

## Mathematical Foundation

The framework is built on rigorous mathematical principles:

- **Multigrid Theory**: Proven convergence bounds and optimal complexity
- **Mixed-Precision Analysis**: Theoretical error bounds and stability analysis  
- **Numerical Validation**: Comprehensive testing with analytical solutions
- **Scalability Analysis**: Weak and strong scaling studies

## Getting Started

```python
from multigrid.applications import PoissonSolver2D, run_comprehensive_validation

# Create solver with mixed precision
solver = PoissonSolver2D(
    solver_type='gpu_multigrid',
    enable_mixed_precision=True,
    tolerance=1e-8
)

# Run comprehensive validation
results = run_comprehensive_validation(quick_mode=True)
print(f"Validation pass rate: {results.passed_tests/results.total_tests:.1%}")
```

## Navigation

<div class="nav-cards">
  <div class="nav-card">
    <h3><a href="{{ '/about/' | relative_url }}">About</a></h3>
    <p>Project motivation, objectives, and scope</p>
  </div>
  
  <div class="nav-card">
    <h3><a href="{{ '/methodology/' | relative_url }}">Methodology</a></h3>
    <p>Mathematical formulation and algorithmic details</p>
  </div>
  
  <div class="nav-card">
    <h3><a href="{{ '/results/' | relative_url }}">Results</a></h3>
    <p>Performance benchmarks and validation studies</p>
  </div>
  
  <div class="nav-card">
    <h3><a href="{{ '/conclusion/' | relative_url }}">Conclusion</a></h3>
    <p>Summary of findings and future directions</p>
  </div>
</div>

## Latest Updates

- ✅ **Comprehensive validation framework** with MMS testing
- ✅ **GPU acceleration** with CUDA implementation  
- ✅ **Mixed-precision analysis** with performance/accuracy trade-offs
- ✅ **Scalability studies** for large-scale problems
- ✅ **Interactive visualization** tools and analysis dashboard

---

<div class="project-info">
  <p><strong>Version:</strong> {{ site.project.version }} | 
     <strong>License:</strong> MIT | 
     <strong>Status:</strong> Production Ready</p>
  
  <p>
    <a href="{{ site.project.github_repo }}" class="btn btn-primary">View on GitHub</a>
    <a href="{{ site.project.documentation }}" class="btn btn-secondary">Documentation</a>
  </p>
</div>