FROM rust:1.85-slim-bookworm AS qart-builder

ARG QART_COMMIT=6e0e00804a1994db7098432c19fadfc552071e30
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl pkg-config \
    && curl --fail --location --retry 5 --retry-all-errors \
        "https://codeload.github.com/andrewyur/qart/tar.gz/${QART_COMMIT}" \
        -o /tmp/qart.tar.gz \
    && mkdir -p /src/qart \
    && tar -xzf /tmp/qart.tar.gz --strip-components=1 -C /src/qart \
    && rm -f /tmp/qart.tar.gz \
    && cargo build --manifest-path /src/qart/Cargo.toml --release --locked

FROM node:22.15.0-bookworm-slim AS qr-verify-builder

WORKDIR /opt/prooftag-qr-verify
COPY qr_verify_bridge/package.json qr_verify_bridge/package-lock.json ./
RUN npm ci --omit=dev --ignore-scripts=false --audit=false
COPY qr_verify_bridge/bridge.mjs ./bridge.mjs

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ARG DIFFQRCODER_COMMIT=e24ea73ee2e13c7e6e87cb422e8b11784e70ae00
ARG QART_COMMIT=6e0e00804a1994db7098432c19fadfc552071e30
ARG HPSV2_COMMIT=866735ecaae999fa714bd9edfa05aa2672669ee3

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/opt/torch-cache \
    PROOFTAG_QR_QR_VERIFY_BRIDGE=/opt/prooftag-qr-verify/bridge.mjs \
    PROOFTAG_QR_NODE_EXECUTABLE=/usr/local/bin/node \
    PYTHONPATH=/opt/DiffQRCoder

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl libgl1 libglib2.0-0 \
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

# HPSv2 is intentionally installed from the exact public source commit by archive,
# not `git clone`. This avoids Git smart-HTTP authentication failures during Docker
# builds while preserving the source package's BPE vocabulary file.
RUN pip install --upgrade pip \
    && curl --fail --location --retry 5 --retry-all-errors \
        "https://codeload.github.com/tgxs002/HPSv2/tar.gz/${HPSV2_COMMIT}" \
        -o /tmp/hpsv2.tar.gz \
    && pip install /tmp/hpsv2.tar.gz \
    && rm -f /tmp/hpsv2.tar.gz \
    && pip install '.[gpu,quality,advisor-runtime]' \
    && python -c "import hpsv2, inspect, pathlib, joblib, lpips, sklearn, torch, torchvision; import hpsv2.src.open_clip as oc; from diffusers import ControlNetModel, DDIMScheduler; bpe=pathlib.Path(inspect.getfile(oc)).resolve().parent/'bpe_simple_vocab_16e6.txt.gz'; assert bpe.is_file(), bpe; lpips.LPIPS(net='vgg', verbose=False); print('GPU, HPSv2 source+BPE, LPIPS, advisor-runtime and quality stack OK:', torch.__version__, torchvision.__version__, joblib.__version__, sklearn.__version__, bpe)"

# Same principle for DiffQRCoder: pin the exact commit but download its source archive
# instead of invoking Git's smart-HTTP protocol.
RUN curl --fail --location --retry 5 --retry-all-errors \
        "https://codeload.github.com/jwliao1209/DiffQRCoder/tar.gz/${DIFFQRCODER_COMMIT}" \
        -o /tmp/diffqrcoder.tar.gz \
    && mkdir -p /opt/DiffQRCoder \
    && tar -xzf /tmp/diffqrcoder.tar.gz --strip-components=1 -C /opt/DiffQRCoder \
    && rm -f /tmp/diffqrcoder.tar.gz \
    && python -c "from diffqrcoder import DiffQRCoderPipeline; print('DiffQRCoder revision archive OK:', '$DIFFQRCODER_COMMIT')"

RUN qart help >/dev/null \
    && node -e "const p=require('/opt/prooftag-qr-verify/node_modules/qr-verify/package.json'); if(p.version!=='0.2.0') process.exit(1); console.log('qr-verify version OK:', p.version)" \
    && python -c "from prooftag_qr.qr import generate_diffqrcoder_qr; from prooftag_qr.validation import QRVerifyDecoder; p='https://ptag.io/t/build'; d=QRVerifyDecoder(); a=d.decode_presets(generate_diffqrcoder_qr(p).image); d.close(); assert len(a)==37 and all(x['text']==p for x in a); print('qr-verify WASM bridge OK: 37/37')" \
    && test -f /app/docs/e035-assets/e034-observed-stage1.png \
    && echo "QArt revision archive OK: $QART_COMMIT"

RUN useradd --create-home --uid 10001 app \
    && mkdir -p /data /cache /opt/torch-cache \
    && chown -R app:app /app /data /cache /opt/torch-cache
USER 10001

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl --fail http://127.0.0.1:8080/healthz || exit 1

CMD ["uvicorn", "prooftag_qr.api:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--proxy-headers"]
