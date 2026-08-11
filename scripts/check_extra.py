#!/usr/bin/env python3
"""Import one installed optional dependency boundary without provider work."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import socket
import sys
from importlib import metadata
from pathlib import Path
from unittest import mock

EXTRA_IMPORTS: dict[str, tuple[str, ...]] = {
    "text": ("ragkit.adapters.textual",),
    "ocr": ("PIL", "pypdfium2", "pytesseract", "ragkit.adapters.ocr"),
    "layout": ("openpyxl", "pdfplumber", "pptx", "ragkit.adapters.layout"),
    "vision": ("PIL", "torch", "torchvision", "transformers", "ragkit.adapters.vision"),
    "media": ("faster_whisper", "scenedetect", "cv2", "ragkit.adapters.media"),
    "persistent": ("chromadb", "ragkit.adapters.chroma_store"),
    "hosted": ("openai", "ragkit.adapters.hosted"),
    "http": ("uvicorn", "ragkit.delivery.http", "ragkit.delivery.server"),
    "reranking": ("torch", "transformers", "ragkit.adapters.cross_encoder_reranker"),
}

EXTRA_DISTRIBUTIONS: dict[str, tuple[str, ...]] = {
    "text": (),
    "ocr": ("Pillow", "pypdfium2", "pytesseract"),
    "layout": ("openpyxl", "pdfplumber", "python-pptx"),
    "vision": ("Pillow", "torch", "torchvision", "transformers"),
    "media": ("faster-whisper", "scenedetect", "opencv-python"),
    "persistent": ("chromadb",),
    "hosted": ("openai",),
    "http": ("uvicorn",),
    "reranking": ("torch", "transformers"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extra", required=True, choices=sorted(EXTRA_IMPORTS))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    loaded: list[str] = []
    network_attempts = 0

    def deny_network(*_args: object, **_kwargs: object) -> None:
        nonlocal network_attempts
        network_attempts += 1
        raise RuntimeError("network access is forbidden during optional-boundary verification")

    with (
        mock.patch.object(socket.socket, "connect", deny_network),
        mock.patch.object(socket.socket, "connect_ex", deny_network),
        mock.patch.object(socket, "create_connection", deny_network),
    ):
        for module in EXTRA_IMPORTS[args.extra]:
            importlib.import_module(module)
            loaded.append(module)
    evidence = {
        "schema": "ragkit-install-profile/v1",
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "extra": args.extra,
        "loaded_imports": loaded,
        "provisioned_modules": [],
        "distributions": {
            distribution: metadata.version(distribution)
            for distribution in EXTRA_DISTRIBUTIONS[args.extra]
        },
        "model_downloads": 0,
        "network_attempts": network_attempts,
        "provider_calls": 0,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(f"optional boundary verified without provider or model work: {args.extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
