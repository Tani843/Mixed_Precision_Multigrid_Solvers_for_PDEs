"""Setup script for Mixed-Precision Multigrid Solvers."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text() if (this_directory / "README.md").exists() else ""

# Read requirements
def read_requirements(filename):
    """Read requirements from file, filtering out comments and optional dependencies."""
    requirements = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and 'optional' not in line:
                # Remove platform-specific markers for basic installation
                if ';' in line:
                    line = line.split(';')[0].strip()
                requirements.append(line)
    return requirements

setup(
    name="mixed-precision-multigrid",
    version="1.0.0",
    description="High-performance mixed-precision multigrid solvers for partial differential equations with GPU acceleration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Tanisha Gupta",
    author_email="tanisha.gupta@research.edu",
    url="https://github.com/tanishagupta/Mixed_Precision_Multigrid_Solvers_for_PDEs",
    project_urls={
        "Documentation": "https://mixed-precision-multigrid.readthedocs.io/",
        "Source Code": "https://github.com/tanishagupta/Mixed_Precision_Multigrid_Solvers_for_PDEs",
        "Bug Tracker": "https://github.com/tanishagupta/Mixed_Precision_Multigrid_Solvers_for_PDEs/issues",
        "Benchmarks": "https://tanishagupta.github.io/Mixed_Precision_Multigrid_Solvers_for_PDEs/results/",
    },
    
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0,<2.0",
        "scipy>=1.9.0", 
        "matplotlib>=3.5.0",
        "pyyaml>=6.0",
    ],
    
    extras_require={
        "gpu": [
            "cupy>=11.0.0",
            "numba>=0.58.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-benchmark>=4.0.0",
            "pytest-xdist>=3.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "isort>=5.12.0",
            "pre-commit>=3.0.0",
        ],
        "performance": [
            "psutil>=5.9.0",
            "memory-profiler>=0.61.0",
            "line-profiler>=4.0.0",
            "py-spy>=0.3.0",
        ],
        "visualization": [
            "seaborn>=0.12.0",
            "plotly>=5.17.0",
            "jupyter>=1.0.0",
            "notebook>=7.0.0",
            "ipywidgets>=8.0.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0", 
            "sphinx-autodoc-typehints>=1.24.0",
            "myst-parser>=2.0.0",
            "sphinx-copybutton>=0.5.0",
        ],
        "mpi": [
            "mpi4py>=3.1.4",
        ],
        "all": [
            "cupy>=11.0.0",
            "numba>=0.58.0",
            "seaborn>=0.12.0",
            "plotly>=5.17.0",
            "jupyter>=1.0.0",
            "notebook>=7.0.0",
            "psutil>=5.9.0",
            "memory-profiler>=0.61.0",
        ]
    },
    
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9", 
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: Implementation :: CPython",
        "Operating System :: OS Independent",
        "Environment :: GPU :: NVIDIA CUDA",
        "Natural Language :: English",
    ],
    
    keywords=[
        "multigrid", "pde", "numerical-methods", "finite-difference", "mixed-precision",
        "gpu", "cuda", "high-performance-computing", "scientific-computing", "linear-algebra",
        "partial-differential-equations", "iterative-methods", "computational-mathematics"
    ],
    
    entry_points={
        "console_scripts": [
            "multigrid-solve=multigrid.cli:main",
        ],
    },
    
    package_data={
        "multigrid": ["config/*.yaml", "config/*.json"],
    },
    
    include_package_data=True,
    zip_safe=False,
)