FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ARG DIFFQRCODER_COMMIT=e24ea73ee2e13c7e6e87cb422e8b11784e70ae00

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/opt/torch-cache \
    PYTHONPATH=/opt/DiffQRCoder

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl git libgl1 libglib2.0-0 libzbar0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md alembic.ini ./
COPY prooftag_qr ./prooftag_qr
COPY migrations ./migrations
RUN pip install --upgrade pip \
    && pip install '.[gpu]' \
    && python -c "import lpips, torch, torchvision; from diffusers import ControlNetModel, DDIMScheduler; lpips.LPIPS(net='vgg', verbose=False); print('GPU stack and LPIPS weights OK:', torch.__version__, torchvision.__version__)"

RUN git clone https://github.com/jwliao1209/DiffQRCoder.git /opt/DiffQRCoder \
    && git -C /opt/DiffQRCoder checkout "$DIFFQRCODER_COMMIT" \
    && test "$(git -C /opt/DiffQRCoder rev-parse HEAD)" = "$DIFFQRCODER_COMMIT" \
    && rm -rf /opt/DiffQRCoder/.git \
    && python -c "from diffqrcoder import DiffQRCoderPipeline; print('DiffQRCoder revision OK:', '$DIFFQRCODER_COMMIT')"

RUN useradd --create-home --uid 10001 app \
    && mkdir -p /data /cache /opt/torch-cache \
    && chown -R app:app /app /data /cache /opt/torch-cache
USER 10001

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl --fail http://127.0.0.1:8080/healthz || exit 1

CMD ["uvicorn", "prooftag_qr.api:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--proxy-headers"]
