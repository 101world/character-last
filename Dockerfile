FROM nvidia/cuda:12.4.0-runtime-ubuntu20.04

WORKDIR /workspace

# Install system dependencies
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
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
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONHASHSEED=random
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Upgrade pip and install PyTorch
RUN pip3 install --upgrade pip
RUN pip3 install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Install Python dependencies
COPY requirements.txt /workspace/
RUN pip3 install -r requirements.txt

# Verify installation
RUN python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Clone repositories
RUN git clone -b sd3 https://github.com/kohya-ss/sd-scripts.git /workspace/kohya
RUN git clone https://github.com/cocktailpeanut/fluxgym.git /workspace/fluxgym

# Download FLUX models
RUN mkdir -p /workspace/models
RUN wget -O /workspace/models/flux1-dev.safetensors https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors
RUN wget -O /workspace/models/ae.safetensors https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors
RUN wget -O /workspace/models/clip_l.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors
RUN wget -O /workspace/models/t5xxl_fp16.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors

COPY handler.py /workspace/
CMD ["python3", "handler.py"]
