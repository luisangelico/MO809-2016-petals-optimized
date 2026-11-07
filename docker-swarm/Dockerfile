# CPU-only Petals swarm-node image (multi-arch: linux/amd64, linux/arm64).
# Build from the repo root so `petals/` can be COPYed:
#   docker buildx build --platform <arch> -f docker-swarm/Dockerfile -t <tag> .
FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/*

# setuptools>=81 breaks the pinned hivemind commit's pkg_resources usage; grpcio-tools
# must stay <1.68 for the same commit to build. See petals-mac-install memory.
RUN pip install --no-cache-dir "setuptools<81" wheel "grpcio-tools>=1.33.2,<1.68"
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir --no-build-isolation \
    "git+https://github.com/learning-at-home/hivemind.git@213bff98a62accb91f254e2afdccbf1d69ebdea9"

COPY petals /app/petals
RUN pip install --no-cache-dir --no-build-isolation /app/petals

# Route optimizer (request-aware routing extension, imports installed petals)
COPY router /app/router

ENV HF_HUB_DISABLE_XET=1
ENV HF_HOME=/cache/huggingface
VOLUME /cache
WORKDIR /app
