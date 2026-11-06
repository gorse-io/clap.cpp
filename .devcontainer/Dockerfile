FROM mcr.microsoft.com/devcontainers/cpp:2-debian-13

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
