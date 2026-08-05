FROM rust:1.85-slim-bookworm AS qart-builder

ARG QART_COMMIT=6e0e00804a1994db7098432c19fadfc552071e30
RUN apt-get update \
    && apt-get install -y --no-install-recommends git pkg-config \
    && git clone https://github.com/andrewyur/qart.git /src/qart \
    && git -C /src/qart checkout "$QART_COMMIT" \
    && test "$(git -C /src/qart rev-parse HEAD)" = "$QART_COMMIT" \
    && cargo build --manifest-path /src/qart/Cargo.toml --release --locked

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ARG DIFFQRCODER_COMMIT=e24ea73ee2e13c7e6e87cb422e8b11784e70ae00
ARG QART_COMMIT=6e0e00804a1994db7098432c19fadfc552071e30
ARG WECHAT_QR_MODELS_COMMIT=3487ef7cde71d93c6a01bb0b84aa0f22c6128f6b

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/opt/torch-cache \
    PROOFTAG_QR_WECHAT_MODELS_DIR=/opt/wechat-qrcode \
    PYTHONPATH=/opt/DiffQRCoder

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl git libgl1 libglib2.0-0 libzbar0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=qart-builder /src/qart/target/release/qart /usr/local/bin/qart
COPY --from=qart-builder /src/qart/LICENSE /opt/licenses/qart-LICENSE

WORKDIR /app
COPY pyproject.toml README.md alembic.ini ./
COPY prooftag_qr ./prooftag_qr
COPY migrations ./migrations
RUN pip install --upgrade pip \
    && pip install '.[gpu,quality]' \
    && python -c "import lpips, torch, torchvision; from diffusers import ControlNetModel, DDIMScheduler; lpips.LPIPS(net='vgg', verbose=False); print('GPU stack and LPIPS weights OK:', torch.__version__, torchvision.__version__)" \
    && python -c "from pathlib import Path; import hpsv2; p=Path(hpsv2.__file__).parent/'src/open_clip/bpe_simple_vocab_16e6.txt.gz'; assert p.is_file(); import hpsv2.img_score as s; s.device='cpu'; print('HPSv2 source and CPU mode OK:', p)"

RUN git init /opt/wechat-qrcode \
    && git -C /opt/wechat-qrcode remote add origin \
        https://github.com/WeChatCV/opencv_3rdparty.git \
    && git -C /opt/wechat-qrcode fetch --depth 1 origin "$WECHAT_QR_MODELS_COMMIT" \
    && git -C /opt/wechat-qrcode checkout --detach FETCH_HEAD \
    && test "$(git -C /opt/wechat-qrcode rev-parse HEAD)" = "$WECHAT_QR_MODELS_COMMIT" \
    && rm -rf /opt/wechat-qrcode/.git \
    && python -c "from prooftag_qr.validation import WeChatQRCodeDecoder; WeChatQRCodeDecoder(); print('WeChat QR models OK:', '$WECHAT_QR_MODELS_COMMIT')"

RUN git clone https://github.com/jwliao1209/DiffQRCoder.git /opt/DiffQRCoder \
    && git -C /opt/DiffQRCoder checkout "$DIFFQRCODER_COMMIT" \
    && test "$(git -C /opt/DiffQRCoder rev-parse HEAD)" = "$DIFFQRCODER_COMMIT" \
    && rm -rf /opt/DiffQRCoder/.git \
    && python -c "from diffqrcoder import DiffQRCoderPipeline; print('DiffQRCoder revision OK:', '$DIFFQRCODER_COMMIT')"

RUN qart help >/dev/null \
    && echo "QArt revision OK: $QART_COMMIT"

RUN useradd --create-home --uid 10001 app \
    && mkdir -p /data /cache /opt/torch-cache \
    && chown -R app:app /app /data /cache /opt/torch-cache
USER 10001

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl --fail http://127.0.0.1:8080/healthz || exit 1

CMD ["uvicorn", "prooftag_qr.api:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--proxy-headers"]
