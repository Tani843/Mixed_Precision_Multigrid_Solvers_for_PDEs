# Installation Guide

This guide provides detailed installation instructions for the Mixed-Precision Multigrid Solvers package across different platforms and use cases.

## Quick Installation

For most users, the simplest installation method is via pip:

```bash
pip install mixed-precision-multigrid
```

## Platform-Specific Installation

### Linux (Ubuntu/Debian)

#### Prerequisites
```bash
# Update system packages
sudo apt update && sudo apt upgrade

# Install system dependencies
sudo apt install -y \
    build-essential \
    gcc g++ gfortran \
    libblas-dev liblapack-dev \
    libffi-dev libssl-dev \
    python3-dev python3-pip \
    git wget curl

# Install Python virtual environment (recommended)
sudo apt install python3-venv
```

#### Installation Steps
```bash
# Create and activate virtual environment
python3 -m venv multigrid-env
source multigrid-env/bin/activate

# Upgrade pip and install package
pip install --upgrade pip setuptools wheel
pip install mixed-precision-multigrid

# For GPU support (requires CUDA)
pip install mixed-precision-multigrid[gpu]
```

#### GPU Support on Linux
```bash
# Install NVIDIA drivers and CUDA toolkit
sudo apt install nvidia-driver-535  # or latest version
sudo apt install nvidia-cuda-toolkit

# Verify CUDA installation
nvcc --version
nvidia-smi

# Install GPU-enabled package
pip install mixed-precision-multigrid[gpu]
```

### macOS

#### Prerequisites using Homebrew
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install system dependencies
brew install gcc gfortran openblas lapack
brew install python@3.9
```

#### Installation Steps
```bash
# Create virtual environment
python3 -m venv multigrid-env
source multigrid-env/bin/activate

# Install package
pip install --upgrade pip setuptools wheel
pip install mixed-precision-multigrid

# For Apple Silicon Macs with Metal Performance Shaders support
pip install mixed-precision-multigrid[metal]
```

#### macOS M1/M2 Specific Notes
```bash
# For Apple Silicon, ensure you're using arm64 Python
python3 -c "import platform; print(platform.machine())"  # Should show 'arm64'

# Install with optimized BLAS for Apple Silicon
export OPENBLAS_NUM_THREADS=1
pip install mixed-precision-multigrid
```

### Windows

#### Prerequisites
1. Install **Python 3.8-3.11** from [python.org](https://python.org)
2. Install **Microsoft C++ Build Tools** or **Visual Studio 2019/2022**
3. Install **Git for Windows**

#### Using Command Prompt/PowerShell
```cmd
# Create virtual environment
python -m venv multigrid-env
multigrid-env\Scripts\activate

# Install package
python -m pip install --upgrade pip setuptools wheel
pip install mixed-precision-multigrid
```

#### GPU Support on Windows
```cmd
# Install CUDA toolkit from NVIDIA website
# Download and install CUDA 11.8 or 12.x

# Verify CUDA installation
nvcc --version

# Install GPU-enabled package
pip install mixed-precision-multigrid[gpu]
```

#### Using Windows Subsystem for Linux (WSL2)
```bash
# Follow Ubuntu installation instructions within WSL2
# This often provides better performance and compatibility

# Install WSL2 Ubuntu
wsl --install -d Ubuntu-20.04

# Then follow Linux installation steps
```

## Installation Options

### Basic Installation
```bash
pip install mixed-precision-multigrid
```

### With GPU Support
```bash
pip install mixed-precision-multigrid[gpu]
```

### Development Installation
```bash
pip install mixed-precision-multigrid[dev]
```

### Complete Installation (All Features)
```bash
pip install mixed-precision-multigrid[all]
```

### From Source
```bash
git clone https://github.com/tanishagupta/Mixed_Precision_Multigrid_Solvers_for_PDEs.git
cd Mixed_Precision_Multigrid_Solvers_for_PDEs
pip install -e ".[dev]"
```

## Docker Installation

### CPU-Only Container
```bash
docker pull tanishagupta/mixed-precision-multigrid:latest-cpu
docker run -it --rm tanishagupta/mixed-precision-multigrid:latest-cpu
```

### GPU-Enabled Container
```bash
# Requires NVIDIA Container Toolkit
docker pull tanishagupta/mixed-precision-multigrid:latest-gpu
docker run --gpus all -it --rm tanishagupta/mixed-precision-multigrid:latest-gpu
```

### Development Environment
```bash
# Clone repository and use docker-compose
git clone https://github.com/tanishagupta/Mixed_Precision_Multigrid_Solvers_for_PDEs.git
cd Mixed_Precision_Multigrid_Solvers_for_PDEs
docker-compose up multigrid-dev
# Access Jupyter at http://localhost:8888
```

## Conda Installation (Alternative)

```bash
# Create conda environment
conda create -n multigrid python=3.9
conda activate multigrid

# Install dependencies
conda install numpy scipy matplotlib numba
pip install mixed-precision-multigrid
```

## Verification

After installation, verify everything works correctly:

```python
import multigrid
print(multigrid.__version__)

# Test basic functionality
from multigrid.solvers import MixedPrecisionMultigrid
solver = MixedPrecisionMultigrid()
print("Installation successful!")
```

### GPU Verification
```python
# Test GPU availability
try:
    import cupy as cp
    print(f"GPU available: {cp.cuda.is_available()}")
    print(f"GPU count: {cp.cuda.runtime.getDeviceCount()}")
except ImportError:
    print("GPU support not installed")
```

## Performance Optimization

### Linux Performance Tuning
```bash
# Set optimal threading for NumPy/SciPy
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMBA_NUM_THREADS=4

# For NUMA systems
numactl --cpubind=0 --membind=0 python your_script.py
```

### Memory Optimization
```bash
# For large problems, increase available memory
export OMP_STACKSIZE=2G
ulimit -s unlimited
```

## Troubleshooting Common Issues

### Import Errors
```bash
# If you get import errors, try:
pip install --force-reinstall mixed-precision-multigrid

# Or install dependencies manually:
pip install numpy scipy matplotlib numba>=0.58.0
```

### Compilation Issues
```bash
# For compilation errors, ensure you have proper build tools
# Linux:
sudo apt install build-essential
# macOS:
xcode-select --install
# Windows: Install Visual Studio Build Tools
```

### GPU Issues
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Reinstall CuPy if needed
pip uninstall cupy-cuda11x
pip install cupy-cuda11x>=11.0.0
```

## Next Steps

After successful installation:
1. Check the [Quick Start Guide](examples/notebooks/01_Getting_Started.ipynb)
2. Read the [API Documentation](https://mixed-precision-multigrid.readthedocs.io/)
3. Run the [Tutorial Notebooks](examples/notebooks/)
4. See [Performance Benchmarks](benchmarks/README.md)

For additional help, see the [Troubleshooting Guide](TROUBLESHOOTING.md) or [open an issue](https://github.com/tanishagupta/Mixed_Precision_Multigrid_Solvers_for_PDEs/issues).