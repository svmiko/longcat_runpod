FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace/LongCat-Video

# Copy all project files
COPY . .

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch (matching CUDA 12.4)
RUN pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Install flash-attn dependencies
RUN pip install ninja psutil packaging

# Install flash-attn (this takes a while to compile)
RUN pip install flash_attn==2.7.4.post1 --no-build-isolation

# Install main requirements
RUN pip install -r requirements.txt

# Install avatar requirements
RUN pip install -r requirements_avatar.txt

# Install huggingface CLI for model download
RUN pip install "huggingface_hub[cli]"

# Create output directories
RUN mkdir -p outputs_avatar_single audio_temp_file

# Download models
RUN huggingface-cli download meituan-longcat/LongCat-Video --local-dir ./weights/LongCat-Video
RUN huggingface-cli download meituan-longcat/LongCat-Video-Avatar --local-dir ./weights/LongCat-Video-Avatar

# Install RunPod
RUN pip install runpod

# Expose port for streamlit (optional)
EXPOSE 8501

# Default command - RunPod serverless handler
CMD ["python3", "-u", "rp_handler.py"]
