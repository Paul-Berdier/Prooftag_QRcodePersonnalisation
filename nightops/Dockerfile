ARG BASE_IMAGE
FROM ${BASE_IMAGE}
USER root
COPY requirements-media.txt /opt/qr-nightly/requirements-media.txt
RUN python -m pip install --no-cache-dir --no-deps -r /opt/qr-nightly/requirements-media.txt
COPY qrnight /opt/qr-nightly/qrnight
ENV PYTHONPATH=/opt/qr-nightly:/app:/opt/DiffQRCoder \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
    TOKENIZERS_PARALLELISM=false
RUN python -c "import sklearn,joblib,numpy,matplotlib,PIL,imageio,imageio_ffmpeg; print('night extras OK', sklearn.__version__, imageio_ffmpeg.get_ffmpeg_exe())"
USER 10001
WORKDIR /app
