# Use NVIDIA CUDA 12.9 base image with cuDNN 8 and Ubuntu 22.04
FROM nvidia/cuda:12.9.0-base-ubuntu22.04

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies and Python 3.10
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip git libgl1-mesa-glx libglib2.0-0\
    && python3.10 -m pip install --no-cache-dir --upgrade pip \
    && python3.10 -m pip install --no-cache-dir torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128 \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set working directory
WORKDIR /app/src

# Copy requirements.txt and install additional dependencies
COPY src/requirements.txt .
RUN python3.10 -m pip install --no-cache-dir -r requirements.txt

# Copy the src directory (code only)
COPY src/ .

# RUN PIPELINE:
CMD ["python3.10", "main.py"]


