"""Creates (or reuses) the PhysLens GPU worker on RunPod with only an API key: no image build, no GitHub link.

A network volume caches MapAnything and its weights; a template runs boot.sh on RunPod's PyTorch 2.8 / CUDA 12.8
image; a serverless endpoint with 0 minimum workers serves requests. Re-running is safe: existing resources with
the same names are reused. Writes RUNPOD_ENDPOINT_ID into backend/.env.

  cd backend && .venv/bin/python ../gpu_worker/runpod_deploy.py
"""
from __future__ import annotations

import base64
import os
import sys
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent / "backend"
sys.path.insert(0, str(BACKEND))
from physlens.config import load_env  # noqa: E402

load_env()
API = "https://rest.runpod.io/v1"
H = {"Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}"}

VOLUME = {"name": "physlens-cache", "size": 30, "dataCenterId": "EU-RO-1"}
# RunPod's own PyTorch image: kept cached on their machines, so workers start without a long download.
IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
# 24 GB cards.
GPUS = ["NVIDIA GeForce RTX 4090", "NVIDIA RTX A5000", "NVIDIA GeForce RTX 3090", "NVIDIA L4"]


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def find(kind: str, name: str):
    items = httpx.get(f"{API}/{kind}", headers=H, timeout=30).json()
    return next((i for i in items if i.get("name") == name), None)


def create(kind: str, body: dict) -> dict:
    r = httpx.post(f"{API}/{kind}", headers=H, json=body, timeout=60)
    if r.status_code >= 300:
        raise SystemExit(f"creating {kind} failed: HTTP {r.status_code} {r.text[:400]}")
    return r.json()


def save_endpoint_id(endpoint_id: str) -> None:
    env = BACKEND / ".env"
    lines = env.read_text().splitlines()
    lines = [f"RUNPOD_ENDPOINT_ID={endpoint_id}" if l.startswith("RUNPOD_ENDPOINT_ID=") else l for l in lines]
    if not any(l.startswith("RUNPOD_ENDPOINT_ID=") for l in lines):
        lines.append(f"RUNPOD_ENDPOINT_ID={endpoint_id}")
    env.write_text("\n".join(lines) + "\n")


def main() -> None:
    volume = find("networkvolumes", VOLUME["name"]) or create("networkvolumes", VOLUME)
    print("volume:  ", volume["id"], volume.get("dataCenterId"), f"{volume.get('size')} GB")

    template_body = {
        "name": "physlens-mapanything",
        "imageName": IMAGE,
        "isServerless": True,
        "containerDiskInGb": 20,
        "dockerStartCmd": ["bash", "-c", "printf '%s' \"$PHYSLENS_BOOT_B64\" | base64 -d > /boot.sh && bash /boot.sh"],
        "env": {
            "PHYSLENS_BOOT_B64": b64(HERE / "boot.sh"),
            "PHYSLENS_HANDLER_B64": b64(HERE / "handler.py"),
            "PHYSLENS_CONSTRAINTS": (HERE / "constraints.txt").read_text(),
        },
    }
    template = find("templates", template_body["name"])
    if template is None:
        template = create("templates", template_body)
    else:
        r = httpx.patch(f"{API}/templates/{template['id']}", headers=H, json={
            k: template_body[k] for k in ("imageName", "containerDiskInGb", "dockerStartCmd", "env")}, timeout=60)
        if r.status_code >= 300:
            raise SystemExit(f"updating template failed: HTTP {r.status_code} {r.text[:400]}")
    print("template:", template["id"])

    endpoint_settings = {"workersMin": 0, "workersMax": 1, "idleTimeout": 60, "minCudaVersion": "12.8"}
    endpoint = find("endpoints", "physlens-geometry")
    if endpoint is not None:
        r = httpx.patch(f"{API}/endpoints/{endpoint['id']}", headers=H, json=endpoint_settings, timeout=60)
        if r.status_code >= 300:
            raise SystemExit(f"updating endpoint failed: HTTP {r.status_code} {r.text[:400]}")
    endpoint = endpoint or create("endpoints", {
        "name": "physlens-geometry",
        "templateId": template["id"],
        "gpuTypeIds": GPUS,
        "workersMin": 0,
        "workersMax": 1,
        "idleTimeout": 60,
        "networkVolumeId": volume["id"],
        "dataCenterIds": [VOLUME["dataCenterId"]],
        "executionTimeoutMs": 600000,
        "flashboot": True,
        "minCudaVersion": "12.8",
    })
    print("endpoint:", endpoint["id"])
    save_endpoint_id(endpoint["id"])
    print("saved RUNPOD_ENDPOINT_ID to backend/.env")


if __name__ == "__main__":
    main()
