# Base image: PyTorch + CUDA 11.8
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Set environment
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# System dependencies
RUN apt-get update && apt-get install -y \
    git wget && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone Kohya scripts (FLUX training)
RUN git clone https://github.com/kohya-ss/sd-scripts.git -b sd3 /workspace/kohya
RUN pip install --no-cache-dir -r /workspace/kohya/requirements.txt

# Copy handler
COPY handler.py .

# Create required directories
RUN mkdir -p /workspace/models /workspace/output /workspace/data

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Start serverless handler
CMD ["python3", "handler.py"]
