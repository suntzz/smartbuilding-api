"""
Descarga los pesos de los modelos del Reto 1 en la carpeta models/ (no se suben a git).

- YOLO11n (Ultralytics, COCO): detección de personas del Reto 1.
- YOLO11m (Ultralytics, COCO): personas, bolsos y vehículos de los retos 2 a 4.
- YOLO11n-pose (Ultralytics, COCO keypoints): postura para caídas (Reto 6).
- SegFormer B2 de ropa (mattmdjaga/segformer_b2_clothes, ONNX): segmentación de prendas.
- CLIP ViT-B/32 (OpenCLIP, LAION-2B): clasificación zero-shot.

Uso:
    python scripts/descargar_modelos.py
"""

import sys
import urllib.request
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
MODELOS = RAIZ / "models"

ARCHIVOS = {
    MODELOS / "yolo11n.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
    MODELOS / "yolo11m.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt",
    MODELOS / "yolo11n-pose.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt",
    **{
        MODELOS / "segformer_clothes" / nombre: f"https://huggingface.co/mattmdjaga/segformer_b2_clothes/resolve/main/onnx/{nombre}"
        for nombre in ("model.onnx", "config.json", "preprocessor_config.json")
    },
}


def main() -> None:
    for destino, url in ARCHIVOS.items():
        if destino.exists():
            print(f"✓ {destino.relative_to(RAIZ)}")
            continue
        destino.parent.mkdir(parents=True, exist_ok=True)
        print(f"↓ {destino.relative_to(RAIZ)}")
        urllib.request.urlretrieve(url, destino)

    import open_clip

    open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k", cache_dir=str(MODELOS / "clip")
    )
    print("✓ CLIP ViT-B/32")

    from rapidocr_onnxruntime import RapidOCR

    RapidOCR()
    print("✓ RapidOCR (placas y etiquetas)")


if __name__ == "__main__":
    sys.exit(main())
