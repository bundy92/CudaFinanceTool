# CUDA Finance Tool Dockerfile
# Multi-stage build for optimal image size

# Base CUDA image
FROM nvidia/cuda:11.8-devel-ubuntu20.04 as cuda-base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    python3 \
    python3-pip \
    python3-dev \
    libpython3-dev \
    && rm -rf /var/lib/apt/lists/*

# Python stage
FROM cuda-base as python-stage

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Build stage
FROM cuda-base as build-stage

# Copy source code
COPY . /workspace
WORKDIR /workspace

# Build the application
RUN make clean && make all

# Runtime stage
FROM nvidia/cuda:11.8-runtime-ubuntu20.04 as runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies
COPY --from=python-stage /usr/local/lib/python3.8/dist-packages /usr/local/lib/python3.8/dist-packages
COPY --from=python-stage /usr/local/bin /usr/local/bin

# Copy built application
COPY --from=build-stage /workspace/bin /app/bin
COPY --from=build-stage /workspace/web_interface /app/web_interface
COPY --from=build-stage /workspace/requirements.txt /app/requirements.txt
COPY --from=build-stage /workspace/README.md /app/README.md

# Set working directory
WORKDIR /app

# Create non-root user
RUN useradd -m -u 1000 cuda_user && \
    chown -R cuda_user:cuda_user /app

USER cuda_user

# Expose ports
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

# Default command
CMD ["python3", "web_interface/app.py"] 