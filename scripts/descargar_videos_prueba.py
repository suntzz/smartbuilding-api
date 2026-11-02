"""
Script utilitario para descargar videos completos de repositorios públicos
de visión por computador (Intel IoT DevKit Sample Videos y ABODA Dataset).

Nota: El repositorio ya incluye en `data/videos/` los clips H.264 optimizados
para los 7 retos (`reto1_...` a `reto7_...`), por lo que este script es opcional.
"""

from pathlib import Path
import urllib.request

BASE_DIR = Path(__file__).resolve().parent.parent
VIDEOS_RAW_DIR = BASE_DIR / "data" / "videos" / "benchmarks_raw"

FUENTES_VIDEOS = {
    "intel_people_detection.mp4": (
        "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/people-detection.mp4"
    ),
    "intel_face_demographics_walking.mp4": (
        "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/face-demographics-walking.mp4"
    ),
    "intel_one_by_one_person.mp4": (
        "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/one-by-one-person-detection.mp4"
    ),
    "intel_car_detection.mp4": (
        "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/car-detection.mp4"
    ),
    "aboda_video1_abandoned_object.avi": (
        "https://raw.githubusercontent.com/kevinlin311tw/ABODA/master/video1.avi"
    ),
}


def descargar_benchmarks() -> list[str]:
    VIDEOS_RAW_DIR.mkdir(parents=True, exist_ok=True)
    descargados = []
    for nombre, url in FUENTES_VIDEOS.items():
        destino = VIDEOS_RAW_DIR / nombre
        if not destino.exists():
            print(f"Descargando {nombre} desde {url} ...")
            urllib.request.urlretrieve(url, str(destino))
        descargados.append(str(destino))
    return descargados


if __name__ == "__main__":
    archivos = descargar_benchmarks()
    print(f"Videos descargados ({len(archivos)}):")
    for a in archivos:
        print(" -", a)
