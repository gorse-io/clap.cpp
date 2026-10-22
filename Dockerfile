# syntax=docker/dockerfile:1

ARG BACKEND=cpu

FROM ubuntu:24.04 AS build-cpu

FROM build-cpu AS build-vulkan
RUN apt-get update \
    && apt-get install -y --no-install-recommends libvulkan-dev glslc spirv-headers \
    && rm -rf /var/lib/apt/lists/*

FROM nvidia/cuda:12.9.1-devel-ubuntu24.04 AS build-cuda

FROM build-${BACKEND} AS build
ARG BACKEND
ARG TARGETARCH
ARG BUILD_JOBS
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake ca-certificates git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
RUN set -eu; \
    case "${TARGETARCH}" in amd64|arm64) ;; *) echo "Unsupported architecture: ${TARGETARCH}" >&2; exit 1 ;; esac; \
    case "${BACKEND}" in \
        cpu) variant=cpu; jobs="${BUILD_JOBS:-2}" ;; \
        vulkan) variant=vulkan; jobs="${BUILD_JOBS:-1}" ;; \
        cuda) variant=cuda12.9; jobs="${BUILD_JOBS:-1}" ;; \
        *) echo "Unsupported backend: ${BACKEND}" >&2; exit 1 ;; \
    esac; \
    cmake --preset "linux-${TARGETARCH}-${variant}"; \
    cmake --build build --config Release --target clap-cli clap-model --parallel "${jobs}"; \
    mkdir -p /out; \
    cp build/clap build/clap-model /out/

FROM ubuntu:24.04 AS runtime-cpu
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

FROM runtime-cpu AS runtime-vulkan
RUN apt-get update \
    && apt-get install -y --no-install-recommends libvulkan1 \
    && rm -rf /var/lib/apt/lists/*

# The CUDA preset links CUDA runtime and cuBLAS statically. Only the host's
# NVIDIA driver must be supplied by NVIDIA Container Toolkit at runtime.
FROM runtime-cpu AS runtime-cuda
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

FROM runtime-${BACKEND} AS runtime
COPY --from=build /out/ /usr/local/bin/
WORKDIR /data
ENTRYPOINT ["clap"]
CMD ["--help"]
