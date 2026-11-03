"""
Ejecuta los motores de los retos 2 a 7 sobre sus videos de prueba y actualiza los datos
precalculados de la página estática (app/static/retos_data.js).

Uso:
    python scripts/descargar_modelos.py         # una sola vez
    python scripts/generar_datos_retos.py       # todos los retos
    python scripts/generar_datos_retos.py 3 5   # solo algunos
"""

import json
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from app.vision.abandono import DetectorAbandono  # noqa: E402
from app.vision.caidas import DetectorCaidas  # noqa: E402
from app.vision.casilleros import MonitorCasilleros  # noqa: E402
from app.vision.conteo import ContadorFlujo  # noqa: E402
from app.vision.merodeo import DetectorMerodeo  # noqa: E402
from app.vision.placas import LectorPlacas  # noqa: E402

DATOS = RAIZ / "app" / "static" / "retos_data.js"
PREFIJO = "window.RETOS_DATA = "

RETOS = {
    "2": ("data/videos/reto2_conteo_flujo.mp4", ContadorFlujo),
    "3": ("data/videos/reto3_merodeo.mp4", DetectorMerodeo),
    "4": ("data/videos/reto4_objeto_abandonado.mp4", DetectorAbandono),
    "5": ("data/videos/reto5_lectura_placas.mp4", LectorPlacas),
    "6": ("data/videos/reto6_caidas_riesgo.mp4", DetectorCaidas),
    "7": ("data/videos/reto7_casilleros.mp4", MonitorCasilleros),
}


def a_json(valor):
    """Convierte los tipos de NumPy (int, float, bool) a tipos de Python."""
    if hasattr(valor, "item"):
        return valor.item()
    raise TypeError(f"{type(valor).__name__} no es serializable")


def main(elegidos) -> None:
    texto = DATOS.read_text(encoding="utf-8")
    datos = json.loads(texto[len(PREFIJO):].rstrip().rstrip(";"))

    for clave in elegidos:
        video, motor = RETOS[clave]
        inicio = time.time()
        resultado = motor().analizar_video(RAIZ / video)
        datos["retos"][clave] = {"video": video, **resultado}
        resumen = resultado["summary"]
        print(f"Reto {clave} · {time.time() - inicio:.0f} s · {resumen['motor']}")
        for k, v in resumen.items():
            if isinstance(v, (int, float)) and k.startswith("total"):
                print(f"   {k}: {v}")

    DATOS.write_text(PREFIJO + json.dumps(datos, ensure_ascii=False, separators=(",", ":"), default=a_json) + ";\n", encoding="utf-8")
    print(f"Actualizado {DATOS.relative_to(RAIZ)} ({DATOS.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main(sys.argv[1:] or list(RETOS))
