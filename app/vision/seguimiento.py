"""
Utilidades comunes de los retos 2 a 7: lectura de video y seguimiento con YOLO11 + ByteTrack.
"""

from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
MODELOS = RAIZ / "models"
TRACKER = Path(__file__).resolve().parent / "bytetrack_largo.yaml"

# Clases COCO que se usan en los retos
PERSONA = 0
BOLSOS = {24: "morral", 26: "bolso", 28: "maleta"}
VEHICULOS = {2: "carro", 3: "moto", 5: "bus", 7: "camion"}

_modelos: dict[str, object] = {}


def cargar_yolo(nombre: str = "yolo11m.pt"):
    """Carga (una sola vez) un modelo YOLO desde la carpeta models/."""
    if nombre not in _modelos:
        from ultralytics import YOLO

        ruta = MODELOS / nombre
        _modelos[nombre] = YOLO(str(ruta) if ruta.exists() else nombre)
    return _modelos[nombre]


def info_video(ruta) -> dict:
    cap = cv2.VideoCapture(str(ruta))
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {ruta}")
    info = {
        "fps": round(cap.get(cv2.CAP_PROP_FPS) or 15.0, 2),
        "ancho": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "alto": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return info


def leer_frames(ruta):
    cap = cv2.VideoCapture(str(ruta))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield i, frame
        i += 1
    cap.release()


def seguir(ruta, clases, modelo: str = "yolo11m.pt", conf: float = 0.25, imgsz: int = 960):
    """
    Recorre el video con detección + seguimiento.
    Produce (indice_frame, imagen, detecciones), donde cada detección es
    {"id": int | None, "clase": int, "conf": float, "bbox": [x, y, w, h]}.
    """
    yolo = cargar_yolo(modelo)
    yolo.predictor = None  # cada video empieza con un tracker nuevo
    resultados = yolo.track(
        str(ruta), stream=True, classes=list(clases), conf=conf, imgsz=imgsz,
        persist=True, tracker=str(TRACKER), verbose=False,
    )
    for i, r in enumerate(resultados):
        dets = []
        if r.boxes is not None and len(r.boxes):
            xyxy = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else [None] * len(xyxy)
            for (x1, y1, x2, y2), c, s, tid in zip(xyxy, r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy(), ids):
                dets.append({
                    "id": None if tid is None else int(tid),
                    "clase": int(c),
                    "conf": round(float(s), 3),
                    "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                })
        yield i, r.orig_img, dets


def renumerar(ids_vistos: dict, tid: int) -> int:
    """Convierte los IDs internos del tracker en una numeración consecutiva 1, 2, 3…"""
    if tid not in ids_vistos:
        ids_vistos[tid] = len(ids_vistos) + 1
    return ids_vistos[tid]


def pie(bbox) -> tuple[float, float]:
    x, y, w, h = bbox
    return x + w / 2, y + h


def centro(bbox) -> tuple[float, float]:
    x, y, w, h = bbox
    return x + w / 2, y + h / 2


def iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    ih = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def fondo_mediana(ruta, muestras: int = 40) -> np.ndarray:
    """Fondo de la escena como la mediana de varios frames repartidos en el video."""
    cap = cv2.VideoCapture(str(ruta))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for idx in np.linspace(0, max(0, n - 1), min(muestras, max(1, n))).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()
    return np.median(np.stack(frames), axis=0).astype(np.uint8)
