# DXNN Converter Dockerfile
# x86 Ubuntu with ultralytics + DX-COM
# Build: docker compose up --build
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# ---- 1. System deps ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv \
        libgl1 libglib2.0-0 libgomp1 \
        libgl1-mesa-glx \
        ffmpeg \
        wget unzip \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# ---- 2. Python deps ----
# Install ultralytics with DEEPX export support.
# The dx_com compiler wheel comes from the DEEPX SDK index (not PyPI).
COPY requirements.txt .
RUN pip install --no-cache-dir \
        "ultralytics[export-deepx]" \
        --find-links https://sdk.deepx.ai/release/dxcom/v2.3.0/index.html \
    && pip install --no-cache-dir -r requirements.txt \
        --find-links https://sdk.deepx.ai/release/dxcom/v2.3.0/index.html

# ---- 3. App code ----
COPY app.py .

EXPOSE 8899

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8899"]
