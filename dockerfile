# Multi-stage Dockerfile for MELD Emotion Recognition Project
# Supports both CPU and GPU environments

# Base image with CUDA support for GPU
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04 as base-gpu

# Prevent interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive

# Set Python version
ENV PYTHON_VERSION=3.9

# Install system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget \
    curl \
    git \
    build-essential \
    cmake \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libsndfile1 \
    ffmpeg \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-distutils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install pip
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python${PYTHON_VERSION} get-pip.py && \
    rm get-pip.py

# Set Python3.9 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1

# CPU-only base image
FROM ubuntu:20.04 as base-cpu

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.9

RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget \
    curl \
    git \
    build-essential \
    cmake \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libsndfile1 \
    ffmpeg \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-distutils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python${PYTHON_VERSION} get-pip.py && \
    rm get-pip.py

RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1

# Main build stage
ARG GPU=true
FROM base-${GPU:+gpu}${GPU:+gpu} as builder

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Upgrade pip and install wheel
RUN python -m pip install --upgrade pip wheel setuptools

# Install PyTorch with CUDA support if GPU is enabled
ARG GPU=true
RUN if [ "$GPU" = "true" ]; then \
        pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118; \
    else \
        pip install torch==2.0.1+cpu torchvision==0.15.2+cpu torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu; \
    fi

# Install TensorFlow
RUN if [ "$GPU" = "true" ]; then \
        pip install tensorflow[and-cuda]==2.13.0; \
    else \
        pip install tensorflow-cpu==2.13.0; \
    fi

# Install other requirements
RUN pip install \
    transformers==4.30.2 \
    numpy==1.24.3 \
    pandas==2.0.3 \
    scikit-learn==1.3.0 \
    matplotlib==3.7.2 \
    seaborn==0.12.2 \
    tqdm==4.65.0 \
    nltk==3.8.1 \
    spacy==3.6.0 \
    librosa==0.10.0 \
    opencv-python==4.8.0.74 \
    soundfile==0.12.1 \
    h5py==3.9.0 \
    pyarrow==12.0.1 \
    openpyxl==3.1.2 \
    pyyaml==6.0 \
    python-dotenv==1.0.0 \
    wget==3.2 \
    tabulate==0.9.0 \
    rich==13.4.2 \
    plotly==5.15.0 \
    wordcloud==1.9.2

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Final stage
FROM builder as final

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TF_CPP_MIN_LOG_LEVEL=2

# Create non-root user
RUN useradd -m -u 1000 meld && \
    chown -R meld:meld /app

# Copy project files
COPY --chown=meld:meld . /app

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/checkpoints /app/exported_models && \
    chown -R meld:meld /app

# Switch to non-root user
USER meld

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Expose ports for TensorBoard and model serving
EXPOSE 6006 8501

# Default command
CMD ["python", "main.py", "--help"]