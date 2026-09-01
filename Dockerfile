FROM rust:1.85-slim-bookworm AS qart-builder

ARG QART_COMMIT=6e0e00804a1994db7098432c19fadfc552071e30
RUN apt-get update \
    && apt-get install -y --no-install-recommends git pkg-config \
    && git clone https://github.com/andrewyur/qart.git /src/qart \
    && git -C /src/qart checkout "$QART_COMMIT" \
    && test "$(git -C /src/qart rev-parse HEAD)" = "$QART_COMMIT" \
    && cargo build --manifest-path /src/qart/Cargo.toml --release --locked

FROM node:22.15.0-bookworm-slim AS qr-verify-builder

WORKDIR /opt/prooftag-qr-verify
COPY qr_verify_bridge/package.json qr_verify_bridge/package-lock.json ./
RUN npm ci --omit=dev --ignore-scripts=false --audit=false
COPY qr_verify_bridge/bridge.mjs ./bridge.mjs

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ARG DIFFQRCODER_COMMIT=e24ea73ee2e13c7e6e87cb422e8b11784e70ae00
ARG QART_COMMIT=6e0e00804a1994db7098432c19fadfc552071e30

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/opt/torch-cache \
    PROOFTAG_QR_QR_VERIFY_BRIDGE=/opt/prooftag-qr-verify/bridge.mjs \
    PROOFTAG_QR_NODE_EXECUTABLE=/usr/local/bin/node \
    PYTHONPATH=/opt/DiffQRCoder

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=qart-builder /src/qart/target/release/qart /usr/local/bin/qart
COPY --from=qart-builder /src/qart/LICENSE /opt/licenses/qart-LICENSE
COPY --from=qr-verify-builder /usr/local/bin/node /usr/local/bin/node
COPY --from=qr-verify-builder /opt/prooftag-qr-verify /opt/prooftag-qr-verify

WORKDIR /app
COPY pyproject.toml README.md alembic.ini ./
COPY prooftag_qr ./prooftag_qr
COPY migrations ./migrations
# E040's final-pipeline evidence must contain the archived Stage-1 raster.
COPY docs/e035-assets/e034-observed-stage1.png ./docs/e035-assets/e034-observed-stage1.png
RUN pip install --upgrade pip \
    && pip install '.[gpu,quality,advisor-runtime]' \
    && python -c "import hpsv2, joblib, lpips, sklearn, torch, torchvision; from diffusers import ControlNetModel, DDIMScheduler; lpips.LPIPS(net='vgg', verbose=False); print('GPU, LPIPS, advisor-runtime and quality stack OK:', torch.__version__, torchvision.__version__, joblib.__version__, sklearn.__version__)"

RUN git clone https://github.com/jwliao1209/DiffQRCoder.git /opt/DiffQRCoder \
    && git -C /opt/DiffQRCoder checkout "$DIFFQRCODER_COMMIT" \
    && test "$(git -C /opt/DiffQRCoder rev-parse HEAD)" = "$DIFFQRCODER_COMMIT" \
    && rm -rf /opt/DiffQRCoder/.git \
    && python -c "from diffqrcoder import DiffQRCoderPipeline; print('DiffQRCoder revision OK:', '$DIFFQRCODER_COMMIT')"

RUN qart help >/dev/null \
    && node -e "const p=require('/opt/prooftag-qr-verify/node_modules/qr-verify/package.json'); if(p.version!=='0.2.0') process.exit(1); console.log('qr-verify version OK:', p.version)" \
    && python -c "from prooftag_qr.qr import generate_diffqrcoder_qr; from prooftag_qr.validation import QRVerifyDecoder; p='https://ptag.io/t/build'; d=QRVerifyDecoder(); a=d.decode_presets(generate_diffqrcoder_qr(p).image); d.close(); assert len(a)==37 and all(x['text']==p for x in a); print('qr-verify WASM bridge OK: 37/37')" \
    && test -f /app/docs/e035-assets/e034-observed-stage1.png \
    && echo "QArt revision OK: $QART_COMMIT"

RUN useradd --create-home --uid 10001 app \
    && mkdir -p /data /cache /opt/torch-cache \
    && chown -R app:app /app /data /cache /opt/torch-cache
USER 10001

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl --fail http://127.0.0.1:8080/healthz || exit 1

CMD ["uvicorn", "prooftag_qr.api:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--proxy-headers"]
