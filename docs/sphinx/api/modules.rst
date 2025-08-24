API Reference
=============

This section contains the complete API reference for all modules in the Mixed-Precision Multigrid Solvers framework.

.. toctree::
   :maxdepth: 4
   :caption: API Documentation

   multigrid

Core Modules
------------

The core modules provide the fundamental building blocks for multigrid computations:

- :doc:`multigrid.core` - Grid management and precision control
- :doc:`multigrid.operators` - Discrete operators (Laplacian, transfer operators)  
- :doc:`multigrid.solvers` - Multigrid solvers and smoothers

Advanced Modules
----------------

Advanced modules provide specialized functionality for high-performance computing:

- :doc:`multigrid.gpu` - GPU acceleration with CUDA
- :doc:`multigrid.preconditioning` - Advanced preconditioning strategies
- :doc:`multigrid.optimization` - Performance optimization tools

Application Modules
-------------------

Application modules provide ready-to-use solvers for common problems:

- :doc:`multigrid.applications` - PDE solvers (Poisson, heat equation, etc.)
- :doc:`multigrid.benchmarks` - Performance benchmarking framework
- :doc:`multigrid.visualization` - Visualization and analysis tools

Utility Modules
---------------

Utility modules provide supporting functionality:

- :doc:`multigrid.utils` - Logging, performance monitoring, and utilities
- :doc:`multigrid.analysis` - Convergence analysis and validation tools

Quick Reference
---------------

Most Common Classes
~~~~~~~~~~~~~~~~~~~

.. currentmodule:: multigrid

.. autosummary::
   :toctree: generated/

   solvers.MixedPrecisionMultigrid
   applications.PoissonSolver
   applications.HeatEquationSolver
   gpu.GPUSolver
   visualization.SolutionVisualizer

Most Common Functions
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   solvers.solve_poisson
   applications.run_convergence_study
   benchmarks.run_performance_benchmark
   visualization.plot_solution_2d
   utils.setup_logging

Module Hierarchy
----------------

.. code-block:: text

   multigrid/
   ├── core/                 # Core functionality
   │   ├── grid.py          # Grid management
   │   └── precision.py     # Precision control
   ├── operators/           # Discrete operators
   │   ├── base.py          # Base operator classes
   │   ├── laplacian.py     # Laplacian operators
   │   └── transfer.py      # Restriction/prolongation
   ├── solvers/             # Multigrid solvers
   │   ├── base.py          # Base solver classes
   │   ├── multigrid.py     # Multigrid methods
   │   └── smoothers.py     # Smoothing operators
   ├── gpu/                 # GPU acceleration
   │   ├── cuda_kernels.py  # CUDA implementations
   │   ├── gpu_solver.py    # GPU-accelerated solvers
   │   └── memory_manager.py # GPU memory management
   ├── preconditioning/     # Preconditioning
   │   ├── base.py          # Base preconditioner classes
   │   └── multigrid_preconditioner.py
   ├── optimization/        # Performance optimization
   │   ├── cache_optimization.py
   │   └── memory_management.py
   ├── applications/        # Ready-to-use applications
   │   ├── poisson_solver.py
   │   ├── heat_solver.py
   │   └── validation.py
   ├── benchmarks/          # Benchmarking framework
   │   ├── performance_benchmark.py
   │   └── validation_suite.py
   ├── visualization/       # Visualization tools
   │   ├── solution_plots.py
   │   ├── convergence_plots.py
   │   └── performance_plots.py
   └── utils/              # Utilities
       ├── logging_utils.py
       └── performance.py