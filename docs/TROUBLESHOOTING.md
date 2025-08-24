# Troubleshooting Guide

This guide helps you resolve common issues encountered when installing and using the Mixed-Precision Multigrid Solvers package.

## Installation Issues

### 1. Import Errors

#### Problem: `ModuleNotFoundError: No module named 'multigrid'`

**Solution:**
```bash
# Check if package is properly installed
pip list | grep mixed-precision-multigrid

# Reinstall if missing
pip install mixed-precision-multigrid

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"
```

#### Problem: `ImportError: cannot import name 'MixedPrecisionMultigrid'`

**Solution:**
```bash
# Verify installation completeness
python -c "import multigrid; print(dir(multigrid))"

# Reinstall with dependencies
pip install --force-reinstall mixed-precision-multigrid[all]
```

### 2. Compilation Errors

#### Problem: `Microsoft Visual C++ 14.0 is required` (Windows)

**Solution:**
```bash
# Install Microsoft C++ Build Tools
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Or install via chocolatey
choco install visualstudio2022buildtools
```

#### Problem: `gcc: command not found` (Linux/macOS)

**Solution:**
```bash
# Linux (Ubuntu/Debian)
sudo apt install build-essential gcc g++ gfortran

# macOS
xcode-select --install
# Or via Homebrew
brew install gcc
```

#### Problem: `fatal error: 'Python.h' file not found`

**Solution:**
```bash
# Linux
sudo apt install python3-dev

# macOS
# Usually resolved by installing Xcode command line tools
xcode-select --install

# Windows
# Ensure Python was installed with development headers
# Reinstall Python from python.org with "Add to PATH" option
```

### 3. Dependency Issues

#### Problem: `numpy.distutils` deprecation warnings

**Solution:**
```bash
# Update to compatible NumPy version
pip install "numpy>=1.21.0,<1.25.0"

# Or use specific versions
pip install numpy==1.24.3
```

#### Problem: `BLAS/LAPACK` library issues

**Solution:**
```bash
# Linux
sudo apt install libblas-dev liblapack-dev

# macOS
brew install openblas lapack

# Verify BLAS installation
python -c "import numpy; numpy.show_config()"
```

## Runtime Issues

### 4. Performance Problems

#### Problem: Slow execution on multi-core systems

**Solution:**
```bash
# Check and set optimal threading
export OPENBLAS_NUM_THREADS=4  # Set to number of physical cores
export MKL_NUM_THREADS=4
export NUMBA_NUM_THREADS=4

# Check current settings
python -c "
import numpy as np
print('NumPy config:')
np.show_config()
print(f'Number of threads: {np.get_num_threads()}')
"
```

#### Problem: High memory usage

**Solution:**
```python
# Use memory profiling to identify issues
from memory_profiler import profile

@profile
def your_solver_function():
    # Your code here
    pass

# Monitor memory usage
import psutil
import os
process = psutil.Process(os.getpid())
print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB")

# Optimize grid sizes and precision settings
solver = MixedPrecisionMultigrid(
    max_grid_size=512,  # Reduce for less memory
    precision_strategy='adaptive'  # Use adaptive precision
)
```

### 5. GPU Issues

#### Problem: `CUDA out of memory` errors

**Solution:**
```python
# Check GPU memory
import cupy as cp
mempool = cp.get_default_memory_pool()
print(f"Used memory: {mempool.used_bytes() / 1024**2:.1f} MB")
print(f"Total memory: {mempool.total_bytes() / 1024**2:.1f} MB")

# Clear GPU memory
mempool.free_all_blocks()

# Use smaller batch sizes or grid sizes
solver = MixedPrecisionMultigrid(
    max_grid_size=256,  # Reduce grid size
    use_gpu=True,
    gpu_memory_fraction=0.8  # Use 80% of GPU memory
)
```

#### Problem: `CuPy is not correctly installed`

**Solution:**
```bash
# Check CUDA version
nvcc --version
nvidia-smi

# Install correct CuPy version
# For CUDA 11.x
pip uninstall cupy-cuda11x cupy
pip install cupy-cuda11x

# For CUDA 12.x
pip uninstall cupy-cuda12x cupy
pip install cupy-cuda12x

# Verify installation
python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"
```

#### Problem: GPU not detected

**Solution:**
```bash
# Check NVIDIA drivers
nvidia-smi

# Check CUDA installation
nvcc --version
ls /usr/local/cuda/lib64/

# Verify GPU accessibility
python -c "
import cupy as cp
print('CUDA available:', cp.cuda.is_available())
print('Device count:', cp.cuda.runtime.getDeviceCount())
if cp.cuda.is_available():
    print('Device name:', cp.cuda.runtime.getDeviceProperties(0)['name'])
"
```

### 6. Numerical Issues

#### Problem: Poor convergence or divergence

**Solution:**
```python
# Adjust solver parameters
solver = MixedPrecisionMultigrid(
    max_iterations=1000,         # Increase iterations
    tolerance=1e-8,             # Adjust tolerance
    damping_factor=0.8,         # Reduce damping
    precision_strategy='conservative'  # Use higher precision
)

# Check problem conditioning
import numpy as np
def check_condition_number(matrix):
    return np.linalg.cond(matrix)

# Use better initial guess
initial_guess = np.zeros_like(your_rhs)  # Instead of random
```

#### Problem: Precision loss in mixed-precision mode

**Solution:**
```python
# Use conservative precision strategy
solver = MixedPrecisionMultigrid(
    precision_strategy='conservative',
    switch_threshold=1e-6,  # Switch to double precision earlier
    min_precision='double'  # Ensure minimum double precision
)

# Monitor precision switching
solver.enable_precision_monitoring = True
result = solver.solve(problem)
print(f"Precision switches: {solver.precision_switches}")
```

### 7. Platform-Specific Issues

#### Problem: macOS Apple Silicon compatibility

**Solution:**
```bash
# Check architecture
python -c "import platform; print(platform.machine())"

# For Apple Silicon, ensure arm64 packages
pip install --upgrade --force-reinstall mixed-precision-multigrid

# Use Rosetta if needed (not recommended)
arch -x86_64 pip install mixed-precision-multigrid
```

#### Problem: Windows path length limitations

**Solution:**
```cmd
# Enable long path support (Windows 10/11)
# Run as Administrator
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
-Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force

# Or use short paths
cd C:\
git clone https://github.com/tanishagupta/Mixed_Precision_Multigrid_Solvers_for_PDEs.git mp-mg
cd mp-mg
```

## Docker Issues

### 8. Container Problems

#### Problem: Permission denied errors in Docker

**Solution:**
```bash
# Fix file permissions
sudo chown -R $USER:$USER /path/to/project

# Run Docker with user mapping
docker run -it --rm -u $(id -u):$(id -g) \
    -v $(pwd):/home/multigrid/workspace \
    mixed-precision-multigrid:dev
```

#### Problem: GPU not accessible in Docker

**Solution:**
```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Test GPU access
docker run --gpus all nvidia/cuda:11.8-base nvidia-smi
```

## Testing and Validation

### 9. Test Failures

#### Problem: Tests fail with numerical differences

**Solution:**
```bash
# Run tests with relaxed tolerance
python -m pytest tests/ --tb=short -v --strict-markers

# Check floating point precision
python -c "
import sys
print(f'Float info: {sys.float_info}')
print(f'Epsilon: {sys.float_info.epsilon}')
"

# Use appropriate test tolerances
pytest tests/ -k "not test_exact_precision"
```

#### Problem: Tests timeout

**Solution:**
```bash
# Run with increased timeout
python -m pytest tests/ --timeout=300

# Run specific test categories
python -m pytest tests/unit/ -v  # Fast unit tests only
python -m pytest tests/integration/ -v --maxfail=1  # Stop on first failure
```

## Performance Debugging

### 10. Profiling and Optimization

#### CPU Profiling
```python
import cProfile
import pstats

# Profile your code
profiler = cProfile.Profile()
profiler.enable()

# Your solver code here
result = solver.solve(problem)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

#### GPU Profiling
```bash
# Use NVIDIA Nsight Systems
nsys profile --trace=cuda,nvtx python your_script.py

# Or use built-in CUDA profiler
import cupy
with cupy.cuda.profile():
    result = solver.solve(problem)
```

## Getting Help

### 11. Reporting Issues

When reporting issues, please include:

```bash
# System information
python -c "
import sys
import platform
import numpy as np
print(f'Python: {sys.version}')
print(f'Platform: {platform.platform()}')
print(f'NumPy: {np.__version__}')
try:
    import cupy
    print(f'CuPy: {cupy.__version__}')
    print(f'CUDA: {cupy.cuda.runtime.runtimeGetVersion()}')
except ImportError:
    print('CuPy: Not installed')

import multigrid
print(f'Mixed-Precision Multigrid: {multigrid.__version__}')
"

# Package versions
pip freeze | grep -E "(numpy|scipy|cupy|numba|mixed-precision)"

# Hardware information
# Linux
lscpu | grep -E "(Model name|CPU\\(s\\)|Thread|Core)"
nvidia-smi  # If using GPU

# macOS  
sysctl -n machdep.cpu.brand_string
sysctl -n hw.ncpu

# Windows
systeminfo | findstr /C:"Processor"
```

### Contact and Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/tanishagupta/Mixed_Precision_Multigrid_Solvers_for_PDEs/issues)
- **Documentation**: [Read the full documentation](https://mixed-precision-multigrid.readthedocs.io/)
- **Discussions**: [Community discussions](https://github.com/tanishagupta/Mixed_Precision_Multigrid_Solvers_for_PDEs/discussions)

### Additional Resources

- [Installation Guide](INSTALLATION.md)
- [API Reference](https://mixed-precision-multigrid.readthedocs.io/en/latest/api/)
- [Tutorial Notebooks](../examples/notebooks/)
- [Performance Benchmarks](../benchmarks/README.md)