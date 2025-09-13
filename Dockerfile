# Multi-stage build for FLUX.1-dev Kohya training worker
# Stage 1: Builder stage for dependencies and model downloads
FROM nvidia/cuda:12.4.0-devel-ubuntu20.04 AS builder

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Set working directory
WORKDIR /workspace

# Install build dependencies with proper caching
RUN set -e && \
    echo "Updating package lists with retry..." && \
    for i in 1 2 3; do \
        if apt-get update; then \
            echo "Package list update successful"; \
            break; \
        else \
            echo "Attempt $i failed, retrying in 5 seconds..."; \
            sleep 5; \
        fi; \
    done && \
    echo "Installing build packages..." && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    wget \
    git \
    python3 \
    python3-dev \
    python3-venv \
    python3-pip \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-dev \
    pkg-config \
    && echo "Cleaning up..." && \
    apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install PyTorch with CUDA support
RUN pip install --no-cache-dir --upgrade pip
RUN set -e && \
    echo "Installing PyTorch with CUDA 12.4 support..." && \
    for i in 1 2 3; do \
        if pip install --no-cache-dir torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124; then \
            echo "PyTorch installation successful"; \
            break; \
        else \
            echo "Attempt $i failed, retrying in 10 seconds..."; \
            sleep 10; \
        fi; \
    done

# Install Python dependencies
COPY requirements.txt /workspace/
RUN set -e && \
    echo "Installing Python dependencies..." && \
    for i in 1 2 3; do \
        if pip install --no-cache-dir -r requirements.txt; then \
            echo "Python dependencies installation successful"; \
            break; \
        else \
            echo "Attempt $i failed, retrying in 10 seconds..."; \
            sleep 10; \
        fi; \
    done

# Download models in builder stage (cached)
RUN mkdir -p /workspace/models && \
    echo "Downloading FLUX.1-dev model..." && \
    for i in 1 2 3; do \
        if wget -q --show-progress -O /workspace/models/flux1-dev.safetensors \
            https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors; then \
            echo "FLUX.1-dev model downloaded successfully"; \
            break; \
        else \
            echo "Attempt $i failed, retrying in 10 seconds..."; \
            sleep 10; \
        fi; \
    done && \
    echo "Downloading AE model..." && \
    wget -q --show-progress -O /workspace/models/ae.safetensors \
        https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors && \
    echo "Downloading CLIP-L model..." && \
    wget -q --show-progress -O /workspace/models/clip_l.safetensors \
        https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors && \
    echo "Downloading T5XXL model..." && \
    wget -q --show-progress -O /workspace/models/t5xxl_fp16.safetensors \
        https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors

# Clone repositories
RUN echo "Cloning Kohya sd-scripts..." && \
    for i in 1 2 3; do \
        if git clone -b sd3 https://github.com/kohya-ss/sd-scripts.git /workspace/kohya; then \
            echo "Kohya repository cloned successfully"; \
            break; \
        else \
            echo "Attempt $i failed, retrying in 5 seconds..."; \
            sleep 5; \
        fi; \
    done && \
    echo "Cloning FluxGym..." && \
    git clone https://github.com/cocktailpeanut/fluxgym.git /workspace/fluxgym

# Stage 2: Runtime stage (minimal final image)
FROM nvidia/cuda:12.4.0-runtime-ubuntu20.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Set working directory
WORKDIR /workspace

# Install only runtime dependencies
RUN set -e && \
    echo "Updating package lists with retry..." && \
    for i in 1 2 3; do \
        if apt-get update; then \
            echo "Package list update successful"; \
            break; \
        else \
            echo "Attempt $i failed, retrying in 5 seconds..."; \
            sleep 5; \
        fi; \
    done && \
    echo "Installing runtime packages..." && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libgomp1 \
    libstdc++6 \
    && echo "Cleaning up..." && \
    apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy models from builder (cached layer)
COPY --from=builder /workspace/models /workspace/models

# Copy repositories from builder
COPY --from=builder /workspace/kohya /workspace/kohya
COPY --from=builder /workspace/fluxgym /workspace/fluxgym

# Set proper environment variables for CUDA and ML
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONHASHSEED=random

# NVIDIA Container Toolkit environment variables
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Copy application code
COPY handler.py /workspace/

# Create output directory
RUN mkdir -p /workspace/output

# Verify CUDA installation
RUN python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}')"

# Set proper permissions
RUN chmod +x /workspace/handler.py

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import torch; assert torch.cuda.is_available()" || exit 1

# Default command
CMD ["python3", "handler.py"]
