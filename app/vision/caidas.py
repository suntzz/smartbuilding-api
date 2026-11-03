"""
Reto 6 · Caídas y comportamientos de riesgo.

Personas: YOLO11-pose (caja + esqueleto). Si el modelo no encuentra personas en el video
(por ejemplo, figuras sintéticas), se usa sustracción de fondo y cada silueta es una persona.

Postura de cada persona, comparando con su propia altura de pie:
- de_pie:   altura ≥ 82 % de la de pie y caja vertical.
- sentado:  62–82 % de la altura, caja aún vertical.
- agachado: < 62 % de la altura, caja aún vertical.
- acostado: caja horizontal (ancho/alto ≥ 1.3) o tronco inclinado más de 60° (con esqueleto).

Una caída es un paso rápido (≤ 1 s) de estar erguido a acostado. Se confirma si la persona
sigue en el suelo 2 s sin levantarse. Sentarse o agacharse nunca generan alerta.
"""

import cv2
import numpy as np

from .seguimiento import cargar_yolo, centro, fondo_mediana, info_video, iou, leer_frames

ESTADOS = ["de_pie", "sentado", "agachado", "acostado", "posible_caida_en_observacion", "caida_confirmada"]


def angulo_tronco(kp):
    """Inclinación del tronco respecto a la vertical (grados), con hombros y caderas de COCO."""
    if kp is None:
        return None
    hombros, caderas = kp[[5, 6]], kp[[11, 12]]
    if (hombros[:, 2] < 0.4).any() or (caderas[:, 2] < 0.4).any():
        return None
    v = hombros[:, :2].mean(axis=0) - caderas[:, :2].mean(axis=0)
    return float(np.degrees(np.arctan2(abs(v[0]), abs(v[1]) + 1e-6)))


class DetectorCaidas:
    def __init__(self, segundos_caida: float = 1.0, segundos_confirmacion: float = 2.0,
                 area_minima: int = 1500, cada_frames: int = 2):
        self.segundos_caida = segundos_caida
        self.segundos_confirmacion = segundos_confirmacion
        self.area_minima = area_minima
        self.cada_frames = cada_frames

    def _personas_yolo(self, ruta):
        yolo = cargar_yolo("yolo11n-pose.pt")
        yolo.predictor = None
        por_frame = []
        for r in yolo.predict(str(ruta), stream=True, conf=0.35, verbose=False):
            dets = []
            if r.boxes is not None and len(r.boxes):
                kps = r.keypoints.data.cpu().numpy() if r.keypoints is not None else [None] * len(r.boxes)
                for (x1, y1, x2, y2), kp in zip(r.boxes.xyxy.cpu().numpy(), kps):
                    dets.append({"bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)], "kp": kp})
            por_frame.append(dets)
        return por_frame

    def _personas_silueta(self, ruta):
        fondo = fondo_mediana(ruta)
        por_frame = []
        for _, f in leer_frames(ruta):
            mascara = (cv2.absdiff(f, fondo).max(axis=2) > 40).astype(np.uint8)
            mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
            n, _, stats, _ = cv2.connectedComponentsWithStats(mascara)
            por_frame.append([
                {"bbox": [int(v) for v in stats[k][:4]], "kp": None}
                for k in range(1, n) if stats[k][4] >= self.area_minima
            ])
        return por_frame

    def _postura(self, p, bbox, kp):
        x, y, w, h = bbox
        ratio = w / max(1, h)
        angulo = angulo_tronco(kp)
        if ratio >= 1.3 or (angulo is not None and angulo > 60):
            return "acostado", ratio, angulo
        if ratio < 0.8:
            p["altura_de_pie"] = max(p["altura_de_pie"], h)
        rel = h / max(1, p["altura_de_pie"])
        if rel >= 0.82:
            return "de_pie", ratio, angulo
        return ("sentado" if rel >= 0.62 else "agachado"), ratio, angulo

    def analizar_video(self, ruta) -> dict:
        info = info_video(ruta)
        fps = info["fps"]
        por_frame = self._personas_yolo(ruta)
        motor = "YOLO11-pose"
        if sum(len(d) for d in por_frame) < 0.05 * max(1, len(por_frame)):
            por_frame = self._personas_silueta(ruta)
            motor = "sustracción de fondo (siluetas)"

        personas, alertas, frames = [], [], []
        conteo = {e: 0 for e in ESTADOS}
        for i, dets in enumerate(por_frame):
            usadas = set()
            visibles = []
            for d in dets:
                # Asociación con la persona activa más parecida (solapamiento o cercanía)
                libres = [p for p in personas if p["ultimo_frame"] >= i - 0.3 * fps and p["id"] not in usadas]
                def parecido(p):
                    c1, c2 = centro(p["bbox"]), centro(d["bbox"])
                    return iou(p["bbox"], d["bbox"]) + (1 if np.hypot(c1[0] - c2[0], c1[1] - c2[1]) < 120 else 0)
                candidatas = [p for p in libres if parecido(p) > 0]
                if candidatas:
                    p = max(candidatas, key=parecido)
                else:
                    p = {"id": len(personas) + 1, "altura_de_pie": d["bbox"][3], "historial": [],
                         "estado": "de_pie", "frame_suelo": None, "alertado": False, "primer_frame": i}
                    personas.append(p)
                usadas.add(p["id"])
                p["bbox"], p["ultimo_frame"] = d["bbox"], i

                postura, ratio, angulo = self._postura(p, d["bbox"], d["kp"])
                p["historial"].append((i, postura, d["bbox"][1]))
                estado = postura
                if postura == "acostado":
                    if p["frame_suelo"] is None:
                        # ¿Venía erguido hace menos de 1 s? Entonces es una caída, no acostarse
                        antes = [h for h in p["historial"] if i - fps * self.segundos_caida <= h[0] < i]
                        p["frame_suelo"] = i
                        p["caida"] = any(h[1] == "de_pie" for h in antes)
                    en_suelo = (i - p["frame_suelo"]) / fps
                    if p["caida"]:
                        estado = "caida_confirmada" if en_suelo >= self.segundos_confirmacion else "posible_caida_en_observacion"
                        if estado == "caida_confirmada" and not p["alertado"]:
                            p["alertado"] = True
                            alertas.append({
                                "person_id": p["id"],
                                "frame_caida": p["frame_suelo"],
                                "frame_alerta": i,
                                "tiempo": round(i / fps, 2),
                                "estado": "caida_confirmada",
                                "segundos_en_suelo": round(en_suelo, 1),
                                "ratio_ancho_alto": round(ratio, 2),
                                "bbox": d["bbox"],
                                "mensaje": f"Persona #{p['id']} se cayó y no se levanta ({en_suelo:.0f} s en el suelo).",
                            })
                else:
                    p["frame_suelo"] = None
                p["estado"] = estado
                conteo[estado] += 1
                visibles.append({
                    "person_id": p["id"],
                    "bbox": d["bbox"],
                    "estado": estado,
                    "alerta": estado == "caida_confirmada",
                    "ratio_ancho_alto": round(ratio, 2),
                    "altura_relativa": round(d["bbox"][3] / max(1, p["altura_de_pie"]), 2),
                    "angulo_tronco": None if angulo is None else round(angulo),
                    "segundos_en_suelo": 0 if p["frame_suelo"] is None else round((i - p["frame_suelo"]) / fps, 1),
                })
            if i % self.cada_frames == 0:
                frames.append({"frame": i, "time": round(i / fps, 2), "personas": visibles,
                               "alerta_activa": any(v["alerta"] for v in visibles)})

        resumen_personas = []
        for p in personas:
            vistas = [h[1] for h in p["historial"]]
            resumen_personas.append({
                "person_id": p["id"],
                "primer_frame": p["primer_frame"],
                "ultimo_frame": p["ultimo_frame"],
                "posturas": sorted(set(vistas), key=vistas.index),
                "cayo": p["alertado"],
            })
        resumen = {
            "reto": "6. Caídas y comportamientos de riesgo",
            "motor": motor + " · postura relativa a la altura de pie",
            "video": str(ruta).split("/")[-1],
            "resolucion": [info["ancho"], info["alto"]],
            "fps": fps,
            "total_frames_video": info["total_frames"],
            "total_personas": len(personas),
            "total_alertas_caida": len(alertas),
            "alertas": alertas,
            "personas": resumen_personas,
            "conteo_posturas_observadas": conteo,
        }
        return {"summary": resumen, "frames": frames, "fps": fps}
