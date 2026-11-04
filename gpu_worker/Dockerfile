# PhysLens GPU worker on RunPod Serverless: Meta's MapAnything (metric 3D, camera poses from many views) and
# Microsoft's MoGe-3 (sharp per-frame depth and normals at the image's own resolution).
# Built by RunPod from this repo. Everything is installed at build time, so a worker only loads the model at startup.

# RunPod's own PyTorch image (torch 2.8, CUDA 12.8): cached on their machines, so only our layers are pulled.
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/models/hf \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 python3-venv git \
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

# MoGe-3 (MIT code and weights), pinned. Installed without its web-demo extras (gradio); FlexGEMM is pure Triton
# unless FLEX_GEMM_BUILD_CUDA is set, so nothing is compiled here.
ARG MOGE_COMMIT=74fbce054ebed49800de42d0ad0e83495065719a
RUN pip install --no-cache-dir -c /tmp/constraints.txt click "trimesh>=4.11" scipy matplotlib huggingface-hub \
      "utils3d_moge @ git+https://github.com/EasternJournalist/utils3d-moge.git@62f09d58509485564e24d5d9f6aac9ee9ebc0c37" \
      "pipeline @ git+https://github.com/EasternJournalist/pipeline.git@1c511390d90226c00c101f34b84df26a0f8789b4" \
      "flex-gemm @ git+https://github.com/JeffreyXiang/FlexGEMM.git@b2fadb29d41846c7981ade6801ffc689fae119cf" \
    && (python -c "import cv2" || pip install --no-cache-dir opencv-python-headless) \
    && pip install --no-cache-dir --no-deps "moge @ https://github.com/microsoft/MoGe/archive/${MOGE_COMMIT}.zip" \
    && pip show moge flex_gemm utils3d_moge | grep -E "^(Name|Version)" \
    && python -c "import torch, mapanything, moge, importlib.util as u; assert u.find_spec('moge.model.v3'); print('moge installed, torch', torch.__version__)"
# (Importing moge.model.v3 here would fail: FlexGEMM asks the GPU for its name on import, and the build machine has
# no GPU. The worker imports it at first use, on the GPU.)

# Bake the MoGe-3 ViT-L weights (370M parameters, MIT) into the image.
RUN python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('Ruicheng/moge-3-vitl', 'model.pt'))"

COPY handler.py /handler.py
CMD ["python", "-u", "/handler.py"]
