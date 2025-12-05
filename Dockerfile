FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .[inference] && \
    pip install --no-cache-dir runpod

# Copy the entire project
COPY . .

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Set all temp/cache directories to /tmp to avoid disk space issues
# /tmp typically has more space on RunPod instances
ENV HF_HOME=/tmp/huggingface_cache
ENV HUGGINGFACE_HUB_CACHE=/tmp/huggingface_cache
ENV TMPDIR=/tmp
ENV TMP=/tmp
ENV TEMP=/tmp

# Create directories
RUN mkdir -p /tmp/huggingface_cache /tmp/outputs

# Expose port (RunPod will handle this)
EXPOSE 8000

# Default command (RunPod will override with handler)
CMD ["python", "handler.py"]

