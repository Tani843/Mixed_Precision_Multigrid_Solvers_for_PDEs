#!/bin/bash
# Build script for Mixed-Precision Multigrid Docker images
# Provides convenient build commands with proper tagging and multi-platform support

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
REGISTRY=""
TAG="latest"
PLATFORM="linux/amd64"
PUSH=false
CACHE=true
BUILD_ARGS=""
TARGET=""

# Function to print colored output
print_color() {
    printf "${1}${2}${NC}\n"
}

print_usage() {
    cat << EOF
Usage: $0 [OPTIONS] [TARGETS...]

Build Mixed-Precision Multigrid Docker images

TARGETS:
    cpu         Build CPU-only production image
    gpu         Build GPU-enabled production image  
    dev         Build development image with tools
    benchmark   Build benchmarking image
    all         Build all images (default)

OPTIONS:
    -r, --registry REGISTRY    Docker registry to use (e.g., ghcr.io/user)
    -t, --tag TAG             Tag for images (default: latest)
    -p, --platform PLATFORM   Target platform (default: linux/amd64)
    --push                    Push images to registry after build
    --no-cache                Disable Docker build cache
    --build-arg ARG=VALUE     Pass build arguments to Docker
    --help                    Show this help message

EXAMPLES:
    # Build all images locally
    $0

    # Build only CPU image
    $0 cpu

    # Build and push to registry
    $0 --registry ghcr.io/tanishagupta --push all

    # Build for multiple platforms
    $0 --platform linux/amd64,linux/arm64 cpu

    # Build with custom build arguments
    $0 --build-arg PYTHON_VERSION=3.11 dev

    # Build GPU image without cache
    $0 --no-cache gpu

EOF
}

# Parse command line arguments
TARGETS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        -t|--tag)
            TAG="$2"
            shift 2
            ;;
        -p|--platform)
            PLATFORM="$2"
            shift 2
            ;;
        --push)
            PUSH=true
            shift
            ;;
        --no-cache)
            CACHE=false
            shift
            ;;
        --build-arg)
            BUILD_ARGS="$BUILD_ARGS --build-arg $2"
            shift 2
            ;;
        --help)
            print_usage
            exit 0
            ;;
        cpu|gpu|dev|development|benchmark|all)
            TARGETS+=("$1")
            shift
            ;;
        *)
            print_color $RED "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Default to all targets if none specified
if [[ ${#TARGETS[@]} -eq 0 ]]; then
    TARGETS=("all")
fi

# Construct image name prefix
if [[ -n "$REGISTRY" ]]; then
    IMAGE_PREFIX="$REGISTRY/mixed-precision-multigrid"
else
    IMAGE_PREFIX="mixed-precision-multigrid"
fi

# Construct build arguments
DOCKER_BUILD_ARGS="$BUILD_ARGS"
DOCKER_BUILD_ARGS="$DOCKER_BUILD_ARGS --platform $PLATFORM"

if [[ "$CACHE" == "false" ]]; then
    DOCKER_BUILD_ARGS="$DOCKER_BUILD_ARGS --no-cache"
fi

# Add buildkit inline cache for faster rebuilds
DOCKER_BUILD_ARGS="$DOCKER_BUILD_ARGS --build-arg BUILDKIT_INLINE_CACHE=1"

# Function to build a specific target
build_target() {
    local target=$1
    local image_tag="${IMAGE_PREFIX}:${target}-${TAG}"
    local dockerfile_target=$target
    
    # Map target names to Dockerfile targets
    case $target in
        cpu)
            dockerfile_target="cpu-production"
            ;;
        gpu)
            dockerfile_target="gpu-production"
            ;;
        dev|development)
            dockerfile_target="development"
            image_tag="${IMAGE_PREFIX}:dev-${TAG}"
            ;;
        benchmark)
            dockerfile_target="cpu-production"  # Use CPU target for benchmarking
            image_tag="${IMAGE_PREFIX}:benchmark-${TAG}"
            ;;
    esac
    
    print_color $BLUE "🔨 Building $target image: $image_tag"
    print_color $YELLOW "   Target: $dockerfile_target"
    print_color $YELLOW "   Platform: $PLATFORM"
    
    # Build the image
    docker build \
        --target $dockerfile_target \
        --tag $image_tag \
        $DOCKER_BUILD_ARGS \
        .
    
    if [[ $? -eq 0 ]]; then
        print_color $GREEN "✅ Successfully built $image_tag"
        
        # Push if requested
        if [[ "$PUSH" == "true" ]]; then
            print_color $BLUE "🚀 Pushing $image_tag"
            docker push $image_tag
            if [[ $? -eq 0 ]]; then
                print_color $GREEN "✅ Successfully pushed $image_tag"
            else
                print_color $RED "❌ Failed to push $image_tag"
                return 1
            fi
        fi
    else
        print_color $RED "❌ Failed to build $image_tag"
        return 1
    fi
}

# Function to build all targets
build_all() {
    local targets=("cpu" "gpu" "dev")
    for target in "${targets[@]}"; do
        build_target $target
    done
}

# Function to check Docker buildx availability
check_buildx() {
    if docker buildx version >/dev/null 2>&1; then
        print_color $GREEN "✅ Docker Buildx available"
        export DOCKER_BUILDKIT=1
    else
        print_color $YELLOW "⚠️  Docker Buildx not available, using regular build"
    fi
}

# Function to check if GPU runtime is available
check_gpu_support() {
    if command -v nvidia-docker >/dev/null 2>&1 || docker info | grep -q nvidia; then
        print_color $GREEN "✅ GPU runtime available"
        return 0
    else
        print_color $YELLOW "⚠️  GPU runtime not detected, GPU image may not work correctly"
        return 1
    fi
}

# Pre-build checks
print_color $BLUE "🔍 Running pre-build checks..."

# Check Docker
if ! command -v docker >/dev/null 2>&1; then
    print_color $RED "❌ Docker not found. Please install Docker first."
    exit 1
fi

check_buildx

# Check GPU support if building GPU image
for target in "${TARGETS[@]}"; do
    if [[ "$target" == "gpu" || "$target" == "all" ]]; then
        check_gpu_support
        break
    fi
done

# Verify Dockerfile exists
if [[ ! -f "Dockerfile" ]]; then
    print_color $RED "❌ Dockerfile not found in current directory"
    exit 1
fi

# Print build information
print_color $BLUE "📋 Build Configuration:"
print_color $YELLOW "   Targets: ${TARGETS[*]}"
print_color $YELLOW "   Registry: ${REGISTRY:-"local"}"
print_color $YELLOW "   Tag: $TAG"
print_color $YELLOW "   Platform: $PLATFORM"
print_color $YELLOW "   Push: $PUSH"
print_color $YELLOW "   Cache: $CACHE"
echo

# Build targets
SUCCESS=true
for target in "${TARGETS[@]}"; do
    if [[ "$target" == "all" ]]; then
        build_all
    else
        build_target $target
    fi
    
    if [[ $? -ne 0 ]]; then
        SUCCESS=false
        break
    fi
done

# Final status
echo
if [[ "$SUCCESS" == "true" ]]; then
    print_color $GREEN "🎉 All builds completed successfully!"
    
    # Show built images
    print_color $BLUE "📦 Built images:"
    docker images | grep mixed-precision-multigrid | head -10
    
    if [[ "$PUSH" == "true" ]]; then
        print_color $GREEN "🚀 Images pushed to registry"
    else
        print_color $YELLOW "💡 To push images, use: docker push <image_name>"
    fi
    
    print_color $BLUE "🏃 Next steps:"
    print_color $YELLOW "   • Test images: docker run --rm ${IMAGE_PREFIX}:cpu-${TAG}"
    print_color $YELLOW "   • Start development: docker-compose up multigrid-dev"
    print_color $YELLOW "   • Run benchmarks: docker-compose --profile benchmark up"
    
else
    print_color $RED "❌ Some builds failed!"
    exit 1
fi