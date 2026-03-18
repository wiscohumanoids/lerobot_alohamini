# AlohaMini Development Container
# Ubuntu 22.04 + Python 3.10 + LeRobot dependencies
# Allows safe exploration of codebase without physical hardware

FROM ubuntu:22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Set Python version
ENV PYTHON_VERSION=3.10

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Build essentials
    build-essential \
    cmake \
    ninja-build \
    git \
    wget \
    curl \
    \
    # Python
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-distutils \
    python3-pip \
    \
    # EGL / OpenGL (DEV headers required for egl-probe)
    libegl1-mesa-dev \
    mesa-utils \
    \
    # ZMQ (for networking)
    libzmq3-dev \
    \
    # OpenCV dependencies
    libopencv-dev \
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libx11-6 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    \
    # Image processing
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    \
    # Audio (for voice control)
    portaudio19-dev \
    libsndfile1 \
    \
    # RealSense camera support
    libusb-1.0-0-dev \
    libudev-dev \
    \
    # Serial communication (for motors)
    udev \
    \
    # Jupyter dependencies
    libffi-dev \
    libssl-dev \
    \
    # Utilities
    vim \
    nano \
    htop \
    tmux \
    screen \
    \
    # X server suport 
    libxkbcommon-x11-0 \
    libxcb-xinerama0 \
    libx11-6 \
    vulkan-tools \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python && \
    ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Install egl-probe explicitly (GitHub-only, not PyPI)
RUN pip install --no-cache-dir \
    "hf-egl-probe @ git+https://github.com/huggingface/egl_probe.git"

# Set working directory
WORKDIR /workspace/lerobot_alohamini

# Copy repository contents
COPY src/ /workspace/lerobot_alohamini/src/


# Install Python dependencies
# Install lerobot in editable mode and let pip resolve dependencies
# This is simpler than using the frozen requirements.txt which has version conflicts
RUN cd /workspace/lerobot_alohamini && \
    pip install --no-cache-dir -e ".[aloha, pusht, xarm, dora, umi, reachy2, so100, umi, libero]" && \
    pip install --no-deps git+https://github.com/huggingface/lerobot-libero.git@main

# Install Jupyter for interactive notebooks
RUN pip install --no-cache-dir \
    jupyter \
    jupyterlab \
    ipywidgets \
    notebook

# Install Feetech servo sdk for motor control, leader arms
RUN pip install feetech-servo-sdk \
    hebi-py \
    teleop \
    placo

# Expose ports
# 8888: Jupyter Notebook/Lab
# 5555: ZMQ command port
# 5556: ZMQ observation port
EXPOSE 8888 5555 5556

# Set environment variables
ENV PYTHONPATH=/workspace/lerobot_alohamini:$PYTHONPATH
#ENV MUJOCO_GL=egl

# Create directories for data and logs
RUN mkdir -p /workspace/data /workspace/logs /workspace/outputs

# Copy wiscohumanoid documentation to workspace for easy access
COPY wiscohumanoid/setup_docs/ /workspace/setup_docs/

# Set default command to bash
CMD ["/bin/bash"]