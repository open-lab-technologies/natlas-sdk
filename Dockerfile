# pynatlas serve, CPU build.
#
#   docker build -t pynatlas .
#   docker run -p 8000:8000 -v pynatlas-data:/data -e NATLAS_API_KEY=change-me pynatlas
#
# Weights download to the /data volume on first start, not into the image.
# Add speech recognition with --build-arg EXTRAS=server,asr (adds about 1 GB)
# and pass HF_TOKEN, since the speech models are gated.

ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS build
ARG EXTRAS=server
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential cmake \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src
RUN pip wheel --no-cache-dir --wheel-dir /wheels ".[${EXTRAS}]"

FROM python:${PYTHON_VERSION}-slim
ARG EXTRAS=server
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libgomp1 \
 && rm -rf /var/lib/apt/lists/*
COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links /wheels "pynatlas[${EXTRAS}]" \
 && rm -rf /wheels \
 && useradd --create-home --uid 1000 natlas \
 && mkdir /data && chown natlas /data
USER natlas
ENV NATLAS_CACHE=/data \
    HF_HOME=/data/huggingface \
    PYTHONUNBUFFERED=1
VOLUME /data
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10m \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
# Skips the download when NATLAS_MODEL_PATH points at weights you mounted yourself.
ENTRYPOINT ["sh", "-c", "[ -n \"$NATLAS_MODEL_PATH\" ] || pynatlas download --quant \"${NATLAS_QUANT:-Q4_K_M}\" || exit 1; exec pynatlas serve --host 0.0.0.0 --port 8000 --quant \"${NATLAS_QUANT:-Q4_K_M}\" \"$@\"", "--"]
