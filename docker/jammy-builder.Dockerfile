# Ubuntu 22.04 (Jammy) build environment for ARM64
# Matches glibc 2.35 on Jetson Orin devices
# Build with: docker build --network=host -t meter-sim-jammy-builder:1 -f docker/jammy-builder.Dockerfile .
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    pkg-config \
    libusb-1.0-0-dev \
    libssl-dev \
    libcfitsio-dev \
    libclang-dev \
    clang \
    libzmq3-dev \
    libfontconfig1-dev \
    libudev-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Set working directory
WORKDIR /build

# Default command runs cargo build
CMD ["cargo", "build", "--release"]
