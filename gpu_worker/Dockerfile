# PhysLens GPU worker: Meta's MapAnything (metric 3D from photos/video frames) on RunPod Serverless.
# Built by RunPod from this repo. Everything is installed at build time, so a worker only loads the model at startup.

# RunPod's own PyTorch image (torch 2.8, CUDA 12.8): cached on their machines, so only our layers are pulled.
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/models/hf \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 python3-venv \
    && rm -rf /var/lib/apt/lists/*

# A separate environment that still sees the image's PyTorch (--system-site-packages), so pip never has to
# modify packages owned by the OS Python (which fails on Ubuntu 24.04).
RUN python3 -m venv --system-site-packages /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# Pinned MapAnything commit (2026-08-07) so builds are reproducible. The constraints keep the image's torch build.
ARG MAPANYTHING_COMMIT=3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9
COPY constraints.txt /tmp/constraints.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -c /tmp/constraints.txt \
      "mapanything @ https://github.com/facebookresearch/map-anything/archive/${MAPANYTHING_COMMIT}.zip" runpod \
    && python -c "import torch, mapanything, runpod; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

# Bake the Apache-2.0 weights into the image.
RUN python -c "from mapanything.models import MapAnything; MapAnything.from_pretrained('facebook/map-anything-apache')"

COPY handler.py /handler.py
CMD ["python", "-u", "/handler.py"]
