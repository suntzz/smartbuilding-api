"""
Precalcula los embeddings de texto de CLIP (OpenAI ViT-B/32) para la cámara web del navegador.

El navegador solo descarga el modelo de visión (Xenova/clip-vit-base-patch32); las frases
de cada clase se codifican aquí una vez, con las mismas descripciones del motor de Python.

Uso:
    python scripts/generar_textos_clip_web.py
"""

import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import open_clip  # noqa: E402
import torch  # noqa: E402

from app.vision.demografia_ia import PROMPTS  # noqa: E402

SALIDA = RAIZ / "app" / "static" / "clip_textos.js"


def main() -> None:
    modelo, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32-quickgelu", pretrained="openai", cache_dir=str(RAIZ / "models" / "clip")
    )
    modelo.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32-quickgelu")
    textos = {}
    with torch.no_grad():
        for pregunta, clases in PROMPTS.items():
            embs = []
            for frases in clases.values():
                e = modelo.encode_text(tokenizer(frases))
                e = e / e.norm(dim=-1, keepdim=True)
                e = e.mean(dim=0)
                e = e / e.norm()
                embs.append([round(float(v), 5) for v in e])
            textos[pregunta] = {"clases": list(clases), "embeddings": embs}
    SALIDA.write_text("window.CLIP_TEXTOS = " + json.dumps(textos, ensure_ascii=False) + ";\n", encoding="utf-8")
    print(f"{SALIDA.relative_to(RAIZ)}: {sum(len(v['clases']) for v in textos.values())} clases")


if __name__ == "__main__":
    main()
