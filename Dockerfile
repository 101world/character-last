# Simple FLUX.1-dev Training Worker
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Set environment
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Clone Kohya scripts
RUN git clone https://github.com/kohya-ss/sd-scripts.git -b sd3 /workspace/kohya

# Install Kohya's dependencies
RUN pip install --no-cache-dir -r /workspace/kohya/requirements.txt

# Copy application
COPY handler.py .

# Create directories
RUN mkdir -p /workspace/models /workspace/output /workspace/data

# Simple health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"

CMD ["python3", "handler.py"]
