# Docker Deployment Guide

This directory contains Docker configurations for the Mixed-Precision Multigrid Solvers package, providing containerized deployment options for development, testing, and production environments.

## 🚀 Quick Start

### Prerequisites

- Docker Engine 20.10+ with BuildKit support
- Docker Compose v3.8+
- For GPU support: NVIDIA Docker Runtime or Docker with GPU support
- At least 8GB RAM recommended (16GB for GPU workloads)

### Development Environment

Start the complete development environment:

```bash
# Start development environment with Jupyter Lab
docker-compose up multigrid-dev

# Access Jupyter Lab at http://localhost:8888
# Default token: multigrid-dev-token
```

### Production CPU Deployment

```bash
# Build and run CPU production image
docker-compose up multigrid-cpu
```

### GPU-Accelerated Deployment

```bash
# Build and run GPU-enabled image (requires GPU runtime)
docker-compose --profile gpu up multigrid-gpu
```

## 📦 Available Images

| Image Target | Description | Use Case | Size |
|-------------|-------------|----------|------|
| `cpu-production` | Optimized CPU-only runtime | Production deployment | ~2GB |
| `gpu-production` | CUDA-enabled with GPU support | GPU production workloads | ~8GB |
| `development` | Full development environment | Development, testing | ~4GB |

## 🏗️ Building Images

### Using the Build Script

The provided build script offers convenient building with proper tagging:

```bash
# Build all images
./docker/build.sh

# Build specific targets
./docker/build.sh cpu gpu dev

# Build and push to registry
./docker/build.sh --registry ghcr.io/username --push all

# Build for multiple platforms
./docker/build.sh --platform linux/amd64,linux/arm64 cpu

# Build without cache
./docker/build.sh --no-cache gpu
```

### Manual Building

```bash
# CPU production image
docker build --target cpu-production -t mixed-precision-multigrid:cpu .

# GPU production image  
docker build --target gpu-production -t mixed-precision-multigrid:gpu .

# Development image
docker build --target development -t mixed-precision-multigrid:dev .
```

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OMP_NUM_THREADS` | 4/8 | OpenMP thread count |
| `CUDA_VISIBLE_DEVICES` | 0 | GPU device selection |
| `JUPYTER_TOKEN` | multigrid-dev | Jupyter access token |
| `LOG_LEVEL` | INFO | Logging verbosity |
| `DEV_MODE` | false | Enable development features |

### Volume Mounts

| Host Path | Container Path | Purpose |
|-----------|---------------|---------|
| `./src` | `/home/multigrid/src` | Source code (development) |
| `./data` | `/home/multigrid/data` | Input data |
| `./results` | `/home/multigrid/results` | Output results |
| `./examples` | `/home/multigrid/examples` | Example notebooks |

## 🧪 Development Workflow

### Interactive Development

```bash
# Start development container
docker-compose up -d multigrid-dev

# Access container shell
docker-compose exec multigrid-dev bash

# Install package in development mode
docker-compose exec multigrid-dev pip install -e .

# Run tests
docker-compose exec multigrid-dev python -m pytest

# Start Jupyter Lab
docker-compose exec multigrid-dev jupyter lab --ip=0.0.0.0 --allow-root
```

### Code Hot-Reloading

The development container mounts your local source directory, enabling hot-reloading:

1. Edit code locally in your IDE
2. Changes are immediately available in the container
3. Restart Python kernel in Jupyter to pick up changes

### Testing

```bash
# Run complete test suite
docker-compose --profile test up multigrid-test

# Run specific test categories
docker-compose exec multigrid-dev python -m pytest tests/unit/
docker-compose exec multigrid-dev python -m pytest -m gpu tests/
```

## 📊 Benchmarking

### Running Benchmarks

```bash
# Run comprehensive benchmark suite
docker-compose --profile benchmark up multigrid-benchmark

# Results will be saved to ./results directory
```

### Custom Benchmarks

```bash
# Run custom benchmark
docker-compose exec multigrid-cpu python -m multigrid.benchmarks \
    --problem-sizes 256,512,1024 \
    --solvers Multigrid,Jacobi \
    --output-dir /home/multigrid/results
```

## 🔍 Monitoring and Observability

### Prometheus Metrics

```bash
# Start monitoring stack
docker-compose --profile monitoring up

# Access Grafana: http://localhost:3000 (admin/multigrid)
# Access Prometheus: http://localhost:9090
```

### Health Checks

All production containers include comprehensive health checks:

- **CPU containers**: Verify package import and basic functionality
- **GPU containers**: Test CUDA availability and GPU computation
- **Monitoring interval**: 60 seconds
- **Timeout**: 15-30 seconds
- **Retries**: 3 attempts

View health status:

```bash
docker-compose ps
docker inspect <container_name> | grep -A 10 Health
```

## 🗄️ Data Persistence

### Database Storage

Optional PostgreSQL database for storing benchmark results:

```bash
# Start with database
docker-compose --profile database up postgres multigrid-cpu

# Connection details:
# Host: localhost:5432
# Database: multigrid  
# User: multigrid
# Password: multigrid_password
```

Database includes:
- Benchmark results storage
- Performance profiles
- Validation results
- System metrics over time
- Experiment tracking

### Redis Caching

Optional Redis cache for performance optimization:

```bash
# Start with caching
docker-compose --profile cache up redis multigrid-cpu

# Access: localhost:6379
```

## 🔒 Security Considerations

### Non-Root User

All containers run as non-root user `multigrid` (UID: 1000) for security:

```dockerfile
USER multigrid
WORKDIR /home/multigrid
```

### Secrets Management

For production deployments, use Docker secrets or environment files:

```bash
# Using environment file
echo "JUPYTER_TOKEN=secure-random-token" > .env
docker-compose up
```

### Network Security

Containers use isolated network (`multigrid-network`) with controlled port exposure.

## 🐛 Troubleshooting

### Common Issues

**1. GPU not accessible in container**
```bash
# Verify GPU runtime
docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi

# Check GPU support in compose
docker-compose --profile gpu config
```

**2. Permission errors with mounted volumes**
```bash
# Fix ownership (Linux/macOS)
sudo chown -R 1000:1000 ./src ./results ./data
```

**3. Out of memory during build**
```bash
# Increase Docker memory limit
# Docker Desktop: Settings → Resources → Memory → 8GB+

# Use multi-stage build efficiently
docker build --target cpu-production .
```

**4. Slow container startup**
```bash
# Check health status
docker-compose ps
docker-compose logs multigrid-dev

# Increase timeout if needed
docker-compose up --timeout 300
```

### Performance Optimization

**CPU Performance:**
- Set `OMP_NUM_THREADS` to match available cores
- Use `--cpus` limit to control resource usage
- Monitor with `docker stats`

**GPU Performance:**
- Verify CUDA version compatibility
- Use `nvidia-smi` to monitor GPU utilization
- Check GPU memory with health checks

**Memory Optimization:**
- Use `.dockerignore` to reduce build context
- Enable BuildKit for better caching
- Set appropriate memory limits

### Debugging

```bash
# Access container logs
docker-compose logs -f multigrid-dev

# Debug container issues
docker-compose exec multigrid-dev bash
docker-compose run --rm multigrid-dev python -c "import multigrid; print('OK')"

# Check resource usage
docker stats
docker-compose top
```

## 📚 Advanced Usage

### Multi-Platform Builds

```bash
# Build for multiple architectures
docker buildx create --name multiplatform
docker buildx use multiplatform
docker buildx build --platform linux/amd64,linux/arm64 \
    --target cpu-production \
    --push -t myregistry/multigrid:cpu .
```

### Custom Configurations

Create `docker-compose.override.yml` for environment-specific settings:

```yaml
version: '3.8'
services:
  multigrid-dev:
    environment:
      - CUSTOM_CONFIG=value
    ports:
      - "9999:8888"  # Custom port mapping
```

### CI/CD Integration

See `.github/workflows/docker.yml` for automated building and deployment.

## 🆘 Support

- **Documentation**: See main README and docs/ directory
- **Issues**: GitHub Issues for bug reports and feature requests
- **Performance**: Use built-in profiling and monitoring tools
- **Security**: Report security issues privately

## 📋 Image Details

### Base Images

- **CPU**: `python:3.9-slim` (Debian-based, optimized for size)
- **GPU**: `nvidia/cuda:11.8-devel-ubuntu20.04` (CUDA 11.8 with development tools)
- **Development**: Extended with additional tools and dependencies

### Installed Packages

**All Images:**
- Python 3.9+ with scientific computing stack
- NumPy, SciPy, Matplotlib optimized with MKL/OpenBLAS
- Mixed-Precision Multigrid package and dependencies

**GPU Images:**
- CUDA 11.8 runtime and development libraries
- CuPy for GPU array operations
- Numba with CUDA support

**Development Images:**
- Jupyter Lab with extensions
- Testing frameworks (pytest, coverage)
- Code quality tools (black, flake8, mypy)
- Documentation tools (Sphinx)
- Performance profiling tools

### Build Args

| Argument | Default | Description |
|----------|---------|-------------|
| `PYTHON_VERSION` | 3.9 | Python version to use |
| `CUDA_VERSION` | 11.8 | CUDA version for GPU images |
| `BUILDKIT_INLINE_CACHE` | 1 | Enable build cache optimization |

---

For more information, see the main project documentation and the examples in the `examples/` directory.