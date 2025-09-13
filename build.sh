#!/bin/bash

# FLUX.1-dev Kohya Training Worker Build Script
# Optimized for multi-stage Docker builds with proper error handling

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="flux-kohya-worker"
DOCKERFILE="Dockerfile"
BUILDKIT_ENABLED=${DOCKER_BUILDKIT:-1}

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check system requirements
check_requirements() {
    print_status "Checking system requirements..."

    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi

    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running. Please start Docker."
        exit 1
    fi

    # Check available disk space (need at least 50GB)
    AVAILABLE_SPACE=$(df / | tail -1 | awk '{print $4}')
    AVAILABLE_GB=$((AVAILABLE_SPACE / 1024 / 1024))

    if [ $AVAILABLE_GB -lt 50 ]; then
        print_warning "Available disk space: ${AVAILABLE_GB}GB"
        print_warning "Building FLUX.1-dev worker requires at least 50GB free space"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        print_success "Available disk space: ${AVAILABLE_GB}GB ✓"
    fi

    # Check if NVIDIA Docker is available (optional)
    if docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu20.04 nvidia-smi &> /dev/null; then
        print_success "NVIDIA Docker support detected ✓"
    else
        print_warning "NVIDIA Docker not detected. GPU acceleration may not work."
    fi
}

# Function to clean up Docker cache
cleanup_cache() {
    print_status "Cleaning up Docker cache..."
    docker system prune -f
    docker volume prune -f
    print_success "Docker cache cleaned"
}

# Function to build Docker image
build_image() {
    local build_args=""

    # Enable BuildKit for faster builds
    export DOCKER_BUILDKIT=$BUILDKIT_ENABLED

    print_status "Starting multi-stage Docker build..."
    print_status "This will take 20-30 minutes. Large model downloads will be cached."

    # Build with progress and timestamps
    if docker build \
        --progress=plain \
        --target runtime \
        -t $IMAGE_NAME \
        $build_args \
        . 2>&1; then

        print_success "Docker image built successfully!"
        print_status "Image size:"
        docker images $IMAGE_NAME

        return 0
    else
        print_error "Docker build failed!"
        print_status "Try the following troubleshooting steps:"
        echo "  1. Check internet connection for model downloads"
        echo "  2. Clear cache: docker system prune -a"
        echo "  3. Rebuild without cache: docker build --no-cache -t $IMAGE_NAME ."
        echo "  4. Check Docker logs above for specific error details"
        return 1
    fi
}

# Function to test the built image
test_image() {
    print_status "Testing built image..."

    # Test 1: Basic CUDA availability
    print_status "Testing CUDA availability..."
    if docker run --rm --gpus all $IMAGE_NAME python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'Current GPU: {torch.cuda.get_device_name(0)}')
else:
    print('CUDA not available - check GPU configuration')
"; then
        print_success "CUDA test passed ✓"
    else
        print_error "CUDA test failed"
        return 1
    fi

    # Test 2: Model files accessibility
    print_status "Testing model file accessibility..."
    if docker run --rm $IMAGE_NAME ls -la /workspace/models/; then
        print_success "Model files accessible ✓"
    else
        print_error "Model files not accessible"
        return 1
    fi

    # Test 3: Kohya repository
    print_status "Testing Kohya repository..."
    if docker run --rm $IMAGE_NAME ls -la /workspace/kohya/; then
        print_success "Kohya repository accessible ✓"
    else
        print_error "Kohya repository not accessible"
        return 1
    fi

    print_success "All tests passed! Image is ready for deployment."
}

# Function to show usage
show_usage() {
    echo "FLUX.1-dev Kohya Training Worker Build Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -c, --cleanup    Clean Docker cache before building"
    echo "  -t, --test       Test the built image after build"
    echo "  -h, --help       Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                    # Build image"
    echo "  $0 --cleanup --test   # Clean cache, build, and test"
    echo "  $0 --help             # Show this help"
}

# Main script
main() {
    local cleanup=false
    local test=false

    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -c|--cleanup)
                cleanup=true
                shift
                ;;
            -t|--test)
                test=true
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done

    print_status "FLUX.1-dev Kohya Training Worker Builder"
    print_status "========================================="

    # Check requirements
    check_requirements

    # Cleanup if requested
    if [ "$cleanup" = true ]; then
        cleanup_cache
    fi

    # Build the image
    if build_image; then
        # Test if requested
        if [ "$test" = true ]; then
            test_image
        fi

        print_success "Build completed successfully!"
        print_status "Next steps:"
        echo "  1. Push to registry: docker tag $IMAGE_NAME your-registry/$IMAGE_NAME && docker push your-registry/$IMAGE_NAME"
        echo "  2. Deploy to RunPod serverless"
        echo "  3. Test with sample training job"
    else
        print_error "Build failed!"
        exit 1
    fi
}

# Run main function
main "$@"