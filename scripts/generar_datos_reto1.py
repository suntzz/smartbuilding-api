"""
Ejecuta el pipeline de IA del Reto 1 sobre el video de prueba y actualiza los datos
precalculados que usa la página estática (app/static/retos_data.js).

Uso:
    python scripts/descargar_modelos.py      # una sola vez
    python scripts/generar_datos_reto1.py
"""

import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from app.vision.demografia_ia import AnalizadorDemografiaIA  # noqa: E402

VIDEO = "data/videos/reto1_demografia_vestimenta.mp4"
DATOS = RAIZ / "app" / "static" / "retos_data.js"
PREFIJO = "window.RETOS_DATA = "


def main() -> None:
    reporte = AnalizadorDemografiaIA().analizar_video(RAIZ / VIDEO)
    frames = reporte.pop("frames")

    texto = DATOS.read_text(encoding="utf-8")
    datos = json.loads(texto[len(PREFIJO):].rstrip().rstrip(";"))
    datos["retos"]["1"] = {
        "video": VIDEO,
        "fps": reporte["fps"],
        "frames": frames,
        "summary": reporte,
    }
    DATOS.write_text(PREFIJO + json.dumps(datos, ensure_ascii=False) + ";\n", encoding="utf-8")

    print(f"{reporte['total_personas_detectadas']} personas · {reporte['motor']}")
    for p in reporte["personas"]:
        sup, inf = p["prenda_superior"], p["prenda_inferior"]
        print(
            f"  #{p['person_id']}: {p['sexo_aparente']}, {p['rango_edad']} · "
            f"{sup['tipo']} {sup['color']} · {inf['tipo']} {inf['color']} · conf {p['confianza']}"
        )


if __name__ == "__main__":
    main()
