# Simple FLUX.1-dev Training Worker
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN git clone https://github.com/kohya-ss/sd-scripts.git -b sd3 /workspace/kohya
RUN pip install --no-cache-dir -r /workspace/kohya/requirements.txt

COPY handler.py .

RUN mkdir -p /workspace/models /workspace/output /workspace/data

CMD ["python3", "handler.py"]
