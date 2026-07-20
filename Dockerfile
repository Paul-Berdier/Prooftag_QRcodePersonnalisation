FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    TRANSFORMERS_CACHE=/cache/huggingface

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libgl1 libglib2.0-0 libzbar0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY prooftag_qr ./prooftag_qr
RUN pip install --upgrade pip \
    && pip install '.[gpu]'

RUN useradd --create-home --uid 10001 app \
    && mkdir -p /data /cache \
    && chown -R app:app /app /data /cache
USER 10001

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl --fail http://127.0.0.1:8080/healthz || exit 1

CMD ["uvicorn", "prooftag_qr.api:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--proxy-headers"]

