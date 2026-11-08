"""
Rutas de visión por computador (Reto 1: demografía y vestimenta).

Son opcionales: app/main.py las registra solo si OpenCV y NumPy están instalados
(pip install -r requirements-vision.txt). La API de residentes funciona sin ellas.
"""

from collections import deque
from pathlib import Path
import time

import cv2
import numpy as np
from fastapi import APIRouter, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.vision.demografia_vestimenta import DetectorDemografiaVestimenta

BASE_DIR = Path(__file__).resolve().parent.parent
VIDEOS_DIR = BASE_DIR / "data" / "videos"
STATIC_DIR = Path(__file__).resolve().parent / "static"

MAX_HISTORIAL_ANALISIS = 100
MAX_IMAGEN_BYTES = 10 * 1024 * 1024  # 10 MB

router = APIRouter(tags=["visión"])
historial_analisis_demografia: deque[dict] = deque(maxlen=MAX_HISTORIAL_ANALISIS)


def registrar_vision(app: FastAPI) -> None:
    """Agrega a la API las rutas de visión y los archivos estáticos del dashboard."""
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    if VIDEOS_DIR.exists():
        app.mount("/data/videos", StaticFiles(directory=VIDEOS_DIR), name="videos")
    app.include_router(router)


@router.get("/dashboard", response_class=HTMLResponse)
def get_dashboard():
    html_path = STATIC_DIR / "dashboard_reto1.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard no encontrado")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@router.get("/vision/videos")
def listar_videos_disponibles():
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    archivos = sorted(
        [
            {
                "nombre": p.name,
                "tamano_kb": round(p.stat().st_size / 1024, 1),
            }
            for p in VIDEOS_DIR.iterdir()
            if p.suffix.lower() in {".mp4", ".avi", ".mov"}
        ],
        key=lambda x: x["nombre"],
    )
    return {
        "videos_locales": archivos,
        "datasets_referencia": [
            {
                "nombre": "Intel IoT DevKit Sample Videos",
                "repo": "https://github.com/intel-iot-devkit/sample-videos",
                "videos": [
                    "people-detection.mp4",
                    "face-demographics-walking.mp4",
                    "one-by-one-person-detection.mp4",
                ],
            },
            {
                "nombre": "PA-100K / PETA / RAP (Pedestrian Attribute Recognition)",
                "repo": "https://github.com/xh-liu/HydraPlus-Net",
                "descripcion": "Benchmarks estándar para atributos de peatones (edad, sexo, vestimenta).",
            },
        ],
    }


@router.post("/vision/demografia-vestimenta/analizar-video")
def analizar_video_demografia(
    video: str = Query(
        default="reto1_demografia_vestimenta.mp4",
        description="Nombre del archivo de video dentro de data/videos/",
    ),
    cada_n_frames: int = Query(default=2, ge=1, le=15),
    min_frames_persona: int = Query(default=3, ge=1, le=30),
    motor: str = Query(
        default="clasico",
        pattern="^(clasico|ia)$",
        description="'clasico' (OpenCV + reglas) o 'ia' (YOLO11 + SegFormer + CLIP)",
    ),
):
    ruta_video = VIDEOS_DIR / Path(video).name
    if not ruta_video.exists():
        raise HTTPException(
            status_code=404, detail=f"Video no encontrado: {video}"
        )

    if motor == "ia":
        try:
            from app.vision.demografia_ia import AnalizadorDemografiaIA

            analizador = AnalizadorDemografiaIA(frames_entre_analisis=cada_n_frames)
        except (ImportError, FileNotFoundError) as err:
            raise HTTPException(status_code=503, detail=f"Motor IA no disponible: {err}")
        resultado = analizador.analizar_video(ruta_video, min_observaciones=min_frames_persona)
        resultado.pop("frames")
    else:
        detector = DetectorDemografiaVestimenta()
        resultado = detector.analizar_video(
            ruta_video,
            cada_n_frames=cada_n_frames,
            min_frames_persona=min_frames_persona,
        )
    historial_analisis_demografia.append(resultado)
    return resultado


@router.post("/vision/demografia-vestimenta/analizar-imagen")
async def analizar_imagen_demografia(archivo: UploadFile = File(...)):
    contenido = await archivo.read(MAX_IMAGEN_BYTES + 1)
    if not contenido:
        raise HTTPException(status_code=400, detail="El archivo de imagen está vacío")
    if len(contenido) > MAX_IMAGEN_BYTES:
        raise HTTPException(
            status_code=413,
            detail="La imagen supera el tamaño máximo permitido de 10 MB",
        )

    arr = np.frombuffer(contenido, dtype=np.uint8)
    frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise HTTPException(status_code=400, detail="Formato de imagen inválido")

    detector = DetectorDemografiaVestimenta()
    salida = detector.procesar_frame(frame_bgr, frame_idx=0, dibujar=False)
    return {
        "reto": "1. Demografía y vestimenta",
        "archivo": archivo.filename,
        "personas_en_frame": salida["personas_en_frame"],
        "personas": salida["personas"],
    }


@router.get("/vision/demografia-vestimenta/eventos")
def obtener_eventos_demografia():
    return {
        "total_analisis": len(historial_analisis_demografia),
        "historial": list(historial_analisis_demografia),
    }


def _generar_mjpeg_demografia(ruta_video: Path):
    detector = DetectorDemografiaVestimenta()
    cap = cv2.VideoCapture(str(ruta_video))
    if not cap.isOpened():
        return

    ret, primer_frame = cap.read()
    if not ret or primer_frame is None:
        cap.release()
        return

    detector.establecer_fondo(primer_frame)
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                detector.reiniciar()
                ret, primer_frame = cap.read()
                if not ret:
                    break
                detector.establecer_fondo(primer_frame)
                frame_idx = 0
                continue

            frame_idx += 1
            res = detector.procesar_frame(frame, frame_idx=frame_idx, dibujar=True)
            anotado = res["frame_anotado"]
            ok, buffer = cv2.imencode(".jpg", anotado, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if ok:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                )
            time.sleep(0.06)
    finally:
        cap.release()


@router.get("/vision/demografia-vestimenta/stream")
def stream_video_demografia(
    video: str = Query(default="reto1_demografia_vestimenta.mp4"),
):
    ruta_video = VIDEOS_DIR / Path(video).name
    if not ruta_video.exists():
        raise HTTPException(status_code=404, detail=f"Video no encontrado: {video}")

    return StreamingResponse(
        _generar_mjpeg_demografia(ruta_video),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
