# Multi-stage Docker build for Mixed-Precision Multigrid Solvers
# Optimized for both CPU and GPU deployment

# Base stage with common dependencies
FROM python:3.9-slim as base

LABEL maintainer="Tanisha Gupta <tanisha.gupta@research.edu>"
LABEL description="Mixed-Precision Multigrid Solvers for PDEs"
LABEL version="1.0.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libblas-dev \
    liblapack-dev \
    libffi-dev \
    libssl-dev \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash multigrid
WORKDIR /home/multigrid

# CPU-only production image
FROM base as cpu-production

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Copy source code
COPY --chown=multigrid:multigrid src/ ./src/
COPY --chown=multigrid:multigrid setup.py pyproject.toml README.md LICENSE ./

# Install the package
RUN pip install -e .

# Switch to non-root user
USER multigrid

# CPU health check
HEALTHCHECK --interval=60s --timeout=15s --start-period=5s --retries=3 \
    CMD python -c "\
import sys; \
try: \
    import multigrid; \
    import numpy as np; \
    grid = multigrid.Grid(16, 16); \
    solver = multigrid.MultigridSolver(grid); \
    print(f'CPU Health Check: Multigrid v{multigrid.__version__} ready'); \
    print('System healthy'); \
except Exception as e: \
    print(f'Health check failed: {e}'); \
    sys.exit(1) \
"

# Default command
CMD ["python", "-c", "import multigrid; print('Mixed-Precision Multigrid CPU ready!')"]

# GPU-enabled image
FROM nvidia/cuda:11.8-devel-ubuntu20.04 as gpu-base

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libblas-dev \
    liblapack-dev \
    libffi-dev \
    libssl-dev \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -sf python3.9 /usr/bin/python3 && \
    ln -sf python3 /usr/bin/python

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Create non-root user
RUN useradd --create-home --shell /bin/bash multigrid
WORKDIR /home/multigrid

# GPU production image
FROM gpu-base as gpu-production

# Copy requirements and install Python dependencies
COPY requirements.txt .
COPY docker/requirements-gpu.txt .

RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install -r requirements.txt && \
    python3 -m pip install -r requirements-gpu.txt

# Copy source code
COPY --chown=multigrid:multigrid src/ ./src/
COPY --chown=multigrid:multigrid setup.py pyproject.toml README.md LICENSE ./

# Install the package with GPU support
RUN python3 -m pip install -e ".[gpu]"

# Switch to non-root user
USER multigrid

# GPU health check with comprehensive validation
HEALTHCHECK --interval=60s --timeout=30s --start-period=10s --retries=3 \
    CMD python3 -c "\
import sys; \
try: \
    import cupy as cp; \
    import multigrid; \
    device_count = cp.cuda.runtime.getDeviceCount(); \
    print(f'GPU Health Check: {device_count} devices accessible'); \
    test_array = cp.array([1,2,3]); \
    result = cp.sum(test_array); \
    assert result == 6; \
    print('GPU computation verified'); \
    print(f'Multigrid version: {multigrid.__version__}'); \
    print('System healthy'); \
except Exception as e: \
    print(f'Health check failed: {e}'); \
    sys.exit(1) \
"

# Default command
CMD ["python3", "-c", "import multigrid; import cupy; print('Mixed-Precision Multigrid GPU ready!')"]

# Development image with all tools
FROM base as development

# Install additional development tools
RUN apt-get update && apt-get install -y \
    vim \
    nano \
    htop \
    tmux \
    tree \
    && rm -rf /var/lib/apt/lists/*

# Copy all requirements
COPY requirements.txt .
COPY docker/requirements-dev.txt .

# Install all dependencies including development tools
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    pip install -r requirements-dev.txt

# Copy source code (with development tools)
COPY --chown=multigrid:multigrid . .

# Install in development mode
RUN pip install -e ".[dev,visualization,performance,docs]"

# Switch to non-root user
USER multigrid

# Expose Jupyter port
EXPOSE 8888

# Default development command
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# Multi-platform build arguments
ARG BUILDPLATFORM
ARG TARGETPLATFORM

# Final stage selection based on build args
FROM cpu-production as final

# Add build information
LABEL build.platform=$BUILDPLATFORM
LABEL target.platform=$TARGETPLATFORM
LABEL build.date=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
LABEL org.opencontainers.image.title="Mixed-Precision Multigrid Solvers"
LABEL org.opencontainers.image.description="High-performance PDE solvers with GPU acceleration"
LABEL org.opencontainers.image.source="https://github.com/tanishagupta/Mixed_Precision_Multigrid_Solvers_for_PDEs"
LABEL org.opencontainers.image.documentation="https://mixed-precision-multigrid.readthedocs.io/"
LABEL org.opencontainers.image.licenses="MIT"