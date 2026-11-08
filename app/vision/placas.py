"""
Reto 5 · Lectura de placas.

- OCR: RapidOCR (detector de texto DBNet + reconocedor CRNN en ONNX) sobre cada frame.
  De noche se realza el contraste con CLAHE antes de leer.
- Formato colombiano: carro ABC123 y moto ABC12D. Se corrigen las confusiones típicas
  según la posición (0↔O, 1↔I, 5↔S, 8↔B, 2↔Z, 6↔G) y se descarta lo que no es placa.
- Cada lectura se asocia a un vehículo por continuidad de posición; la placa final se
  decide por votación carácter a carácter entre todos sus frames, así hay una sola
  lectura por vehículo aunque el OCR se equivoque en alguno.
- La carrocería se ubica restando el fondo (mediana de los frames con la misma luz).
"""

import re
from collections import Counter, defaultdict

import cv2
import numpy as np

from app.residentes_csv import cargar_residentes

from .seguimiento import fondo_mediana, info_video, leer_frames

A_LETRA = {"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "8": "B"}
A_NUMERO = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "G": "6", "B": "8", "T": "7"}
FORMATOS = {"carro": "LLLNNN", "moto": "LLLNNL"}

_ocr = None


def cargar_ocr():
    global _ocr
    if _ocr is None:
        from rapidocr_onnxruntime import RapidOCR

        _ocr = RapidOCR()
    return _ocr


def normalizar_placa(texto: str):
    """
    Devuelve (placa, tipo_vehiculo) si el texto encaja en un formato de placa; si no, None.
    Si encaja en varios formatos gana el que necesita menos correcciones (MTK48D es moto, no MTK480).
    """
    limpio = re.sub(r"[^A-Z0-9]", "", texto.upper())
    if len(limpio) != 6:
        return None
    opciones = []
    for tipo, patron in FORMATOS.items():
        corregido, cambios = [], 0
        for c, p in zip(limpio, patron):
            nuevo = A_LETRA.get(c, c) if p == "L" else A_NUMERO.get(c, c)
            if (p == "L" and not nuevo.isalpha()) or (p == "N" and not nuevo.isdigit()):
                break
            cambios += nuevo != c
            corregido.append(nuevo)
        else:
            opciones.append((cambios, "".join(corregido), tipo))
    if not opciones:
        return None
    _, placa, tipo = min(opciones)
    return placa, tipo


def votar(lecturas):
    """Placa final por votación carácter a carácter, pesada por la confianza del OCR."""
    votos = [defaultdict(float) for _ in range(6)]
    tipos = Counter()
    for placa, tipo, conf in lecturas:
        tipos[tipo] += conf
        for k, c in enumerate(placa):
            votos[k][c] += conf
    placa = "".join(max(v, key=v.get) for v in votos)
    return placa, tipos.most_common(1)[0][0]


def directorio_placas():
    por_placa = {}
    for r in cargar_residentes():
        for n in ("1", "2"):
            placa = (r.get(f"placa_{n}") or "").strip().upper()
            if placa and placa != "LIBRE":
                info = por_placa.setdefault(placa, {
                    "torre": r["tower"], "apartamento": r["apartment"],
                    "parqueadero": r.get(f"parqueadero_{n}"), "residentes": [],
                })
                info["residentes"].append(r["full_name"])
    return por_placa


class LectorPlacas:
    def __init__(self, umbral_noche: float = 70.0, distancia_max_px: float = 90.0,
                 segundos_memoria: float = 1.0, lecturas_minimas: int = 2, cada_frames: int = 2):
        self.umbral_noche = umbral_noche
        self.distancia_max_px = distancia_max_px
        self.segundos_memoria = segundos_memoria
        self.lecturas_minimas = lecturas_minimas
        self.cada_frames = cada_frames

    def _leer(self, frame, noche):
        imagen = frame
        if noche:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
            imagen = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        resultado, _ = cargar_ocr()(imagen)
        lecturas = []
        for caja, texto, conf in resultado or []:
            placa = normalizar_placa(texto)
            if placa is None or float(conf) < 0.5:
                continue
            pts = np.array(caja, dtype=float)
            x1, y1 = pts.min(axis=0)
            x2, y2 = pts.max(axis=0)
            lecturas.append({"placa": placa[0], "tipo": placa[1], "conf": float(conf),
                             "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]})
        return lecturas

    def _carroceria(self, frame, fondo, placa_bbox):
        """Caja del vehículo: zona distinta del fondo que contiene la placa."""
        if fondo is None:
            return None
        dif = cv2.absdiff(frame, fondo).max(axis=2)
        mascara = cv2.morphologyEx((dif > 30).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(mascara)
        x, y, w, h = placa_bbox
        cx, cy = min(etiquetas.shape[1] - 1, x + w // 2), min(etiquetas.shape[0] - 1, y + h // 2)
        k = etiquetas[cy, cx]
        if k == 0:
            return None
        bx, by, bw, bh, area = stats[k]
        return [int(bx), int(by), int(bw), int(bh)] if area > w * h * 1.5 else None

    def analizar_video(self, ruta) -> dict:
        info = info_video(ruta)
        fps = info["fps"]
        directorio = directorio_placas()

        brillos = [float(f.mean()) for _, f in leer_frames(ruta)]
        fondos = {}
        for noche in (False, True):
            indices = [k for k, b in enumerate(brillos) if (b < self.umbral_noche) == noche]
            if indices:
                fondos[noche] = fondo_mediana_indices(ruta, indices)

        vehiculos, frames = [], []
        for i, frame in leer_frames(ruta):
            noche = brillos[i] < self.umbral_noche
            lecturas = self._leer(frame, noche)
            visibles = []
            for lec in lecturas:
                cx, cy = lec["bbox"][0] + lec["bbox"][2] / 2, lec["bbox"][1] + lec["bbox"][3] / 2
                candidatos = [
                    v for v in vehiculos
                    if (i - v["ultimo_frame"]) <= self.segundos_memoria * fps
                    and np.hypot(cx - v["centro"][0], cy - v["centro"][1]) < self.distancia_max_px
                ]
                if candidatos:
                    v = min(candidatos, key=lambda v: np.hypot(cx - v["centro"][0], cy - v["centro"][1]))
                else:
                    v = {"vehiculo_id": len(vehiculos) + 1, "lecturas": [], "primer_frame": i, "noche": noche}
                    vehiculos.append(v)
                v["lecturas"].append((lec["placa"], lec["tipo"], lec["conf"]))
                v["centro"], v["ultimo_frame"] = (cx, cy), i
                placa, tipo = votar(v["lecturas"])
                v["placa"], v["tipo"] = placa, tipo
                confirmada = len(v["lecturas"]) >= self.lecturas_minimas
                if confirmada and "frame_lectura" not in v:
                    v["frame_lectura"] = i
                registro = directorio.get(placa)
                visibles.append({
                    "vehiculo_id": v["vehiculo_id"],
                    "placa": placa,
                    "lectura_frame": lec["placa"],
                    "confianza_ocr": round(lec["conf"], 2),
                    "confirmada": confirmada,
                    "tipo_vehiculo": tipo,
                    "condicion_luz": "noche" if noche else "dia",
                    "bbox_placa": lec["bbox"],
                    "bbox_vehiculo": self._carroceria(frame, fondos.get(noche), lec["bbox"]),
                    "autorizado": registro is not None,
                    "residente": ", ".join(registro["residentes"][:2]) if registro else "Visitante / No registrado",
                    "torre": registro["torre"] if registro else None,
                    "apartamento": registro["apartamento"] if registro else None,
                    "parqueadero": registro["parqueadero"] if registro else None,
                })
            if i % self.cada_frames == 0:
                frames.append({"frame": i, "time": round(i / fps, 2), "vehiculos": visibles,
                               "condicion_luz": "noche" if noche else "dia"})

        lecturas_unicas = []
        for v in vehiculos:
            if len(v["lecturas"]) < self.lecturas_minimas:
                continue
            registro = directorio.get(v["placa"])
            confianza = float(np.mean([c for p, _, c in v["lecturas"] if p == v["placa"]] or [0]))
            lecturas_unicas.append({
                "vehiculo_id": v["vehiculo_id"],
                "placa": v["placa"],
                "tipo_vehiculo": v["tipo"],
                "condicion_luz": "noche" if v["noche"] else "dia",
                "frames_observados": len(v["lecturas"]),
                "lecturas_distintas": dict(Counter(p for p, _, _ in v["lecturas"])),
                "confianza": round(confianza, 2),
                "primer_frame": v["primer_frame"],
                "ultimo_frame": v["ultimo_frame"],
                "frame_lectura": v.get("frame_lectura"),
                "autorizado": registro is not None,
                "residente": ", ".join(registro["residentes"]) if registro else "Visitante / No registrado",
                "torre": registro["torre"] if registro else None,
                "apartamento": registro["apartamento"] if registro else None,
                "parqueadero": registro["parqueadero"] if registro else None,
            })

        resumen = {
            "reto": "5. Lectura de placas",
            "motor": "RapidOCR (DBNet + CRNN) · CLAHE nocturno · votación por vehículo",
            "video": str(ruta).split("/")[-1],
            "resolucion": [info["ancho"], info["alto"]],
            "fps": fps,
            "total_frames_video": info["total_frames"],
            "total_vehiculos_leidos": len(lecturas_unicas),
            "lecturas_dia": sum(1 for v in lecturas_unicas if v["condicion_luz"] == "dia"),
            "lecturas_noche": sum(1 for v in lecturas_unicas if v["condicion_luz"] == "noche"),
            "lecturas_unicas": lecturas_unicas,
        }
        return {"summary": resumen, "frames": frames, "fps": fps}


def fondo_mediana_indices(ruta, indices, muestras: int = 30):
    elegidos = set(np.linspace(0, len(indices) - 1, min(muestras, len(indices))).astype(int).tolist())
    objetivo = {indices[k] for k in elegidos}
    frames = [f for i, f in leer_frames(ruta) if i in objetivo]
    return np.median(np.stack(frames), axis=0).astype(np.uint8)
