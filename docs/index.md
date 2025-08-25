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

**GPU Acceleration:** {{ site.data.benchmarks.cpu_gpu_speedup }} average speedup  
**Mixed Precision Gain:** {{ site.data.benchmarks.mixed_precision_speedup }}  
**Memory Savings:** {{ site.data.benchmarks.memory_savings }}  
**Convergence:** {{ site.data.benchmarks.convergence_rate }}

## Applications

Our framework successfully solves:

- **Poisson Equations**: Steady-state elliptic problems with various boundary conditions
- **Heat Equations**: Time-dependent parabolic problems with implicit time stepping
- **Complex Geometries**: Support for irregular domains and mixed boundary conditions
- **Multi-Scale Problems**: Efficient handling of problems with disparate length scales

## Performance Highlights

![Precision Strategy]({{ '/assets/images/precision_strategy.png' | relative_url }})
*Mixed-precision strategy showing optimal performance/accuracy trade-offs*

![Performance Scaling]({{ '/assets/images/performance_scaling.svg' | relative_url }})
*Scaling analysis demonstrating GPU acceleration and multi-precision efficiency*

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

- [About]({{ '/about/' | relative_url }})
- [Methodology]({{ '/methodology/' | relative_url }})
- [Results]({{ '/results/' | relative_url }})
- [Conclusion]({{ '/conclusion/' | relative_url }})

## Latest Updates

- ✅ **Comprehensive validation framework** with MMS testing
- ✅ **GPU acceleration** with CUDA implementation  
- ✅ **Mixed-precision analysis** with performance/accuracy trade-offs
- ✅ **Scalability studies** for large-scale problems
- ✅ **Interactive visualization** tools and analysis dashboard

---

<div class="project-info">
  <p><strong>Version:</strong> 1.0.0 | 
     <strong>License:</strong> MIT | 
     <strong>Status:</strong> Production Ready</p>
  
  <p>
    <a href="https://github.com/tani843/Mixed_Precision_Multigrid_Solvers_for_PDEs" class="btn btn-primary">View on GitHub</a>
    <a href="{{ '/' | relative_url }}" class="btn btn-secondary">Documentation</a>
  </p>
</div>