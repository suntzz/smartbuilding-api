from pathlib import Path
import time
import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse

from app.datos import cargar_residentes
from app.vision.caidas_riesgo import DetectorCaidasRiesgo
from app.vision.conteo_entradas_salidas import ContadorEntradasSalidas
from app.vision.correspondencia_casillero import DetectorCorrespondenciaCasillero
from app.vision.demografia_vestimenta import DetectorDemografiaVestimenta
from app.vision.lectura_placas import LectorPlacasVehiculares
from app.vision.merodeo import DetectorMerodeo
from app.vision.objeto_abandonado import DetectorObjetoAbandonado

BASE_DIR = Path(__file__).resolve().parent.parent
VIDEOS_DIR = BASE_DIR / "data" / "videos"
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="SmartBuilding API - Suite Completa de Visión por Computador (Retos 1 al 7)",
    description=(
        "API REST para administración de edificios residenciales (Conjunto Residencial "
        "San Carlos) con los 7 retos de visión por computador integrados."
    ),
    version="2.0.0",
)

residentes = cargar_residentes()
historial_analisis_demografia: list[dict] = []


@app.get("/")
def root():
    return {"message": "Bienvenido a SmartBuilding API"}


@app.get("/residentes")
def get_residentes():
    return residentes


@app.get("/residentes/{residente_id}")
def get_residente_por_id(residente_id: int):
    for residente in residentes:
        if residente["id"] == residente_id:
            return residente
    raise HTTPException(status_code=404, detail="Residente no encontrado")


@app.get("/dashboard", response_class=HTMLResponse)
def get_dashboard():
    html_path = STATIC_DIR / "dashboard_completo.html"
    if not html_path.exists():
        html_path = STATIC_DIR / "dashboard_reto1.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/vision/videos")
def listar_videos_disponibles():
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    archivos = sorted(
        [
            {
                "nombre": p.name,
                "tamano_kb": round(p.stat().st_size / 1024, 1),
            }
            for p in VIDEOS_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in {".mp4", ".avi", ".mov"}
        ],
        key=lambda x: x["nombre"],
    )
    return {
        "videos_locales": archivos,
        "datasets_referencia": [
            {
                "nombre": "Intel IoT DevKit Sample Videos",
                "repo": "https://github.com/intel-iot-devkit/sample-videos",
                "retos": [1, 2, 3],
            },
            {
                "nombre": "ABODA - Abandoned Object Dataset",
                "repo": "https://github.com/kevinlin311tw/ABODA",
                "retos": [3, 4],
            },
            {
                "nombre": "PA-100K / PETA / RAP (Pedestrian Attribute Recognition)",
                "repo": "https://github.com/xh-liu/HydraPlus-Net",
                "retos": [1],
            },
            {
                "nombre": "UR Fall Detection Dataset (URFD) / Le2i",
                "repo": "http://fenix.ur.edu.pl/~mkepski/ds/uf.html",
                "retos": [6],
            },
        ],
    }


def _resolver_video(nombre_video: str) -> Path:
    ruta = VIDEOS_DIR / Path(nombre_video).name
    if not ruta.exists():
        raise HTTPException(
            status_code=404, detail=f"Video no encontrado: {nombre_video}"
        )
    return ruta


# =====================================================================
# RETO 1: Demografía y vestimenta
# =====================================================================
@app.post("/vision/demografia-vestimenta/analizar-video")
@app.post("/vision/reto1/demografia-vestimenta")
def analizar_video_demografia(
    video: str = Query(default="reto1_demografia_vestimenta.mp4"),
    cada_n_frames: int = Query(default=2, ge=1, le=15),
    min_frames_persona: int = Query(default=3, ge=1, le=30),
):
    ruta = _resolver_video(video)
    detector = DetectorDemografiaVestimenta()
    resultado = detector.analizar_video(
        ruta, cada_n_frames=cada_n_frames, min_frames_persona=min_frames_persona
    )
    historial_analisis_demografia.append(resultado)
    return resultado


@app.post("/vision/demografia-vestimenta/analizar-imagen")
async def analizar_imagen_demografia(archivo: UploadFile = File(...)):
    contenido = await archivo.read()
    if not contenido:
        raise HTTPException(status_code=400, detail="El archivo de imagen está vacío")

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


@app.get("/vision/demografia-vestimenta/eventos")
def obtener_eventos_demografia():
    return {
        "total_analisis": len(historial_analisis_demografia),
        "historial": historial_analisis_demografia,
    }


# =====================================================================
# RETO 2: Conteo de entradas y salidas
# =====================================================================
@app.post("/vision/reto2/conteo-entradas-salidas")
def analizar_reto2_conteo(
    video: str = Query(default="reto2_conteo_flujo.mp4"),
    cada_n_frames: int = Query(default=2, ge=1, le=10),
):
    ruta = _resolver_video(video)
    contador = ContadorEntradasSalidas()
    return contador.analizar_video(ruta, cada_n_frames=cada_n_frames)


# =====================================================================
# RETO 3: Merodeo
# =====================================================================
@app.post("/vision/reto3/merodeo")
def analizar_reto3_merodeo(
    video: str = Query(default="reto3_merodeo.mp4"),
    cada_n_frames: int = Query(default=2, ge=1, le=10),
):
    ruta = _resolver_video(video)
    detector = DetectorMerodeo()
    return detector.analizar_video(ruta, cada_n_frames=cada_n_frames)


# =====================================================================
# RETO 4: Objeto abandonado
# =====================================================================
@app.post("/vision/reto4/objeto-abandonado")
def analizar_reto4_objeto_abandonado(
    video: str = Query(default="reto4_objeto_abandonado.mp4"),
    cada_n_frames: int = Query(default=2, ge=1, le=10),
):
    ruta = _resolver_video(video)
    detector = DetectorObjetoAbandonado()
    return detector.analizar_video(ruta, cada_n_frames=cada_n_frames)


# =====================================================================
# RETO 5: Lectura de placas
# =====================================================================
@app.post("/vision/reto5/lectura-placas")
def analizar_reto5_lectura_placas(
    video: str = Query(default="reto5_lectura_placas.mp4"),
    cada_n_frames: int = Query(default=2, ge=1, le=10),
):
    ruta = _resolver_video(video)
    lector = LectorPlacasVehiculares(residentes_lista=residentes)
    return lector.analizar_video(ruta, cada_n_frames=cada_n_frames)


# =====================================================================
# RETO 6: Caídas y comportamientos de riesgo
# =====================================================================
@app.post("/vision/reto6/caidas-riesgo")
def analizar_reto6_caidas_riesgo(
    video: str = Query(default="reto6_caidas_riesgo.mp4"),
    cada_n_frames: int = Query(default=1, ge=1, le=10),
):
    ruta = _resolver_video(video)
    detector = DetectorCaidasRiesgo()
    return detector.analizar_video(ruta, cada_n_frames=cada_n_frames)


# =====================================================================
# RETO 7: Correspondencia en casillero
# =====================================================================
@app.post("/vision/reto7/correspondencia-casillero")
def analizar_reto7_casilleros(
    video: str = Query(default="reto7_casilleros.mp4"),
    cada_n_frames: int = Query(default=3, ge=1, le=15),
):
    ruta = _resolver_video(video)
    detector = DetectorCorrespondenciaCasillero(residentes_lista=residentes)
    return detector.analizar_video(ruta, cada_n_frames=cada_n_frames)


# =====================================================================
# EJECUCIÓN INTEGRAL DE LOS 7 RETOS + STREAMING MJPEG
# =====================================================================
@app.post("/vision/analizar-todos")
def analizar_todos_los_retos():
    return {
        "edificio": "Conjunto Residencial San Carlos",
        "retos": {
            "reto_1": DetectorDemografiaVestimenta().analizar_video(
                _resolver_video("reto1_demografia_vestimenta.mp4"), cada_n_frames=3
            ),
            "reto_2": ContadorEntradasSalidas().analizar_video(
                _resolver_video("reto2_conteo_flujo.mp4"), cada_n_frames=3
            ),
            "reto_3": DetectorMerodeo().analizar_video(
                _resolver_video("reto3_merodeo.mp4"), cada_n_frames=3
            ),
            "reto_4": DetectorObjetoAbandonado().analizar_video(
                _resolver_video("reto4_objeto_abandonado.mp4"), cada_n_frames=3
            ),
            "reto_5": LectorPlacasVehiculares(residentes_lista=residentes).analizar_video(
                _resolver_video("reto5_lectura_placas.mp4"), cada_n_frames=2
            ),
            "reto_6": DetectorCaidasRiesgo().analizar_video(
                _resolver_video("reto6_caidas_riesgo.mp4"), cada_n_frames=1
            ),
            "reto_7": DetectorCorrespondenciaCasillero(
                residentes_lista=residentes
            ).analizar_video(_resolver_video("reto7_casilleros.mp4"), cada_n_frames=4),
        },
    }


MAPA_STREAM_RETO = {
    1: ("reto1_demografia_vestimenta.mp4", DetectorDemografiaVestimenta),
    2: ("reto2_conteo_flujo.mp4", ContadorEntradasSalidas),
    3: ("reto3_merodeo.mp4", DetectorMerodeo),
    4: ("reto4_objeto_abandonado.mp4", DetectorObjetoAbandonado),
    5: ("reto5_lectura_placas.mp4", LectorPlacasVehiculares),
    6: ("reto6_caidas_riesgo.mp4", DetectorCaidasRiesgo),
    7: ("reto7_casilleros.mp4", DetectorCorrespondenciaCasillero),
}


def _generar_mjpeg_generico(ruta_video: Path, constructor_detector):
    detector = constructor_detector()
    cap = cv2.VideoCapture(str(ruta_video))
    if not cap.isOpened():
        return

    ret, primer_frame = cap.read()
    if not ret or primer_frame is None:
        cap.release()
        return

    if hasattr(detector, "establecer_fondo"):
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
                if hasattr(detector, "establecer_fondo"):
                    detector.establecer_fondo(primer_frame)
                frame_idx = 0
                continue

            frame_idx += 1
            res = detector.procesar_frame(frame, frame_idx=frame_idx, dibujar=True)
            anotado = res.get("frame_anotado", frame)
            ok, buffer = cv2.imencode(".jpg", anotado, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if ok:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                )
            time.sleep(0.06)
    finally:
        cap.release()


@app.get("/vision/demografia-vestimenta/stream")
def stream_video_demografia(
    video: str = Query(default="reto1_demografia_vestimenta.mp4"),
):
    ruta = _resolver_video(video)
    return StreamingResponse(
        _generar_mjpeg_generico(ruta, DetectorDemografiaVestimenta),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/vision/stream/{reto_id}")
def stream_por_reto(reto_id: int):
    if reto_id not in MAPA_STREAM_RETO:
        raise HTTPException(status_code=404, detail="Reto no existente (use 1 a 7)")
    nombre_video, cls_detector = MAPA_STREAM_RETO[reto_id]
    ruta = _resolver_video(nombre_video)
    return StreamingResponse(
        _generar_mjpeg_generico(ruta, cls_detector),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
