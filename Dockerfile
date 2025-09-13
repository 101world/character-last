FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for better Python performance
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONHASHSEED=random

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Upgrade pip
RUN pip3 install --upgrade pip

# Install PyTorch with CUDA support first (FLUX.1 requires PyTorch 2.6.0+)
RUN pip3 install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Install other Python dependencies
COPY requirements.txt /workspace/
RUN pip3 install -r requirements.txt

# Verify Python installation
RUN python3 --version && pip3 --version

# Verify CUDA and PyTorch
RUN python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Clone repositories
RUN git clone -b sd3 https://github.com/kohya-ss/sd-scripts.git /workspace/kohya
RUN git clone https://github.com/cocktailpeanut/fluxgym.git /workspace/fluxgym

# Download FLUX.1-dev required models
RUN mkdir -p /workspace/models
# Download FLUX.1-dev model
RUN wget -O /workspace/models/flux1-dev.safetensors https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors
# Download AE model
RUN wget -O /workspace/models/ae.safetensors https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors
# Download CLIP-L text encoder
RUN wget -O /workspace/models/clip_l.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors
# Download T5-XXL text encoder
RUN wget -O /workspace/models/t5xxl_fp16.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors

COPY handler.py /workspace/
CMD ["python3", "handler.py"]
