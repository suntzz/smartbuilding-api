"""
Reto 6: Caídas y comportamientos de riesgo.
Detecta cuándo una persona se cae y no se levanta, sin confundirlo con
sentarse o agacharse.
"""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np


class DetectorCaidasRiesgo:
    """
    Evalúa la relación de aspecto (w/h), la reducción relativa de altura
    respecto a la postura erguida y el tiempo de permanencia en el suelo
    para distinguir entre:
    - 'de_pie_o_caminando'
    - 'agachado' (reducción temporal de altura sin postura horizontal prolongada)
    - 'sentado' (reducción moderada de altura manteniendo torso compacto vertical)
    - 'caida_confirmada' (postura horizontal en suelo w/h >= 1.35 que NO se levanta)
    """

    def __init__(self, umbral_frames_no_se_levanta: int = 10) -> None:
        self.umbral_frames_no_se_levanta = umbral_frames_no_se_levanta
        self.reiniciar()

    def reiniciar(self) -> None:
        self.bg_lab: np.ndarray | None = None
        self.tracks: dict[int, dict] = {}
        self.next_id: int = 1
        self.alertas_caida: list[dict] = []
        self.ids_alertados: set[int] = set()

    def establecer_fondo(self, frame_bgr: np.ndarray) -> None:
        blur = cv2.GaussianBlur(frame_bgr, (5, 5), 0)
        self.bg_lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB).astype(np.float32)

    def clasificar_postura(
        self,
        w: int,
        h: int,
        altura_erguida_ref: int,
        frames_horizontal: int,
    ) -> dict:
        """
        Clasifica la postura de una persona a partir de sus dimensiones actuales,
        su altura de referencia erguida y el número de frames en posición horizontal.
        """
        ratio_wh = float(w) / max(1.0, float(h))
        ratio_h_ref = float(h) / max(1.0, float(altura_erguida_ref))

        if ratio_wh >= 1.35 and ratio_h_ref <= 0.55:
            if frames_horizontal >= self.umbral_frames_no_se_levanta:
                estado = "caida_confirmada"
                alerta = True
            else:
                estado = "posible_caida_en_observacion"
                alerta = False
        elif 0.52 <= ratio_h_ref <= 0.78 and ratio_wh < 1.15:
            if ratio_h_ref >= 0.62:
                estado = "sentado"
            else:
                estado = "agachado"
            alerta = False
        elif ratio_h_ref < 0.52 and ratio_wh < 1.35:
            estado = "agachado"
            alerta = False
        else:
            estado = "de_pie_o_caminando"
            alerta = False

        return {
            "estado": estado,
            "alerta": alerta,
            "ratio_ancho_alto": round(ratio_wh, 2),
            "ratio_altura_erguida": round(ratio_h_ref, 2),
            "frames_en_suelo": frames_horizontal,
        }

    def procesar_frame(
        self, frame_bgr: np.ndarray, frame_idx: int = 0, dibujar: bool = False
    ) -> dict:
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("El frame no puede estar vacío")

        h_img, w_img = frame_bgr.shape[:2]
        blur = cv2.GaussianBlur(frame_bgr, (5, 5), 0)
        lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB).astype(np.float32)

        if self.bg_lab is None or self.bg_lab.shape != lab.shape:
            self.bg_lab = lab.copy()
            return {"frame_idx": frame_idx, "personas": [], "alerta_activa": False}

        diff = np.abs(lab - self.bg_lab)
        fg_mask = ((diff[:, :, 0] > 22.0) | (diff[:, :, 1] > 12.0) | (diff[:, :, 2] > 12.0)).astype(
            np.uint8
        ) * 255
        fg_mask = cv2.morphologyEx(
            fg_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)),
        )

        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cajas: list[tuple[int, int, int, int]] = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 900:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            cajas.append((x, y, w, h))

        usados: set[int] = set()
        personas_frame: list[dict] = []

        for x, y, w, h in cajas:
            cx, cy = x + w // 2, y + h // 2
            mejor_id = None
            mejor_dist = 1e9
            for tid, trk in self.tracks.items():
                if tid in usados or (frame_idx - trk["last_frame"]) > 8:
                    continue
                d = float(np.hypot(cx - trk["cx"], cy - trk["cy"]))
                if d < 140.0 and d < mejor_dist:
                    mejor_dist = d
                    mejor_id = tid

            if mejor_id is None:
                mejor_id = self.next_id
                self.next_id += 1
                self.tracks[mejor_id] = {
                    "person_id": mejor_id,
                    "cx": cx,
                    "cy": cy,
                    "bbox": [x, y, w, h],
                    "max_h": max(h, 165),
                    "frames_horizontal": 0,
                    "last_frame": frame_idx,
                    "historial_estados": set(),
                }

            usados.add(mejor_id)
            trk = self.tracks[mejor_id]
            trk["cx"] = cx
            trk["cy"] = cy
            trk["bbox"] = [x, y, w, h]
            trk["last_frame"] = frame_idx
            if h > trk["max_h"] and (w / max(1.0, float(h))) < 0.85:
                trk["max_h"] = h

            if (w / max(1.0, float(h))) >= 1.35:
                trk["frames_horizontal"] += 1
            else:
                trk["frames_horizontal"] = 0

            eval_postura = self.clasificar_postura(
                w, h, trk["max_h"], trk["frames_horizontal"]
            )
            estado = eval_postura["estado"]
            trk["historial_estados"].add(estado)

            if eval_postura["alerta"] and mejor_id not in self.ids_alertados:
                self.ids_alertados.add(mejor_id)
                self.alertas_caida.append(
                    {
                        "person_id": mejor_id,
                        "frame_alerta": frame_idx,
                        "estado": "caida_confirmada",
                        "frames_en_suelo": trk["frames_horizontal"],
                        "ratio_ancho_alto": eval_postura["ratio_ancho_alto"],
                        "mensaje": (
                            f"ALERTA CRÍTICA: Persona #{mejor_id} sufrió una caída "
                            f"y no se levanta tras {trk['frames_horizontal']} frames."
                        ),
                    }
                )

            personas_frame.append(
                {
                    "person_id": mejor_id,
                    "bbox": [x, y, w, h],
                    **eval_postura,
                }
            )

        salida = {
            "frame_idx": frame_idx,
            "personas": personas_frame,
            "alerta_activa": any(p["alerta"] for p in personas_frame),
            "total_alertas_caida": len(self.alertas_caida),
        }
        if dibujar:
            salida["frame_anotado"] = self.anotar_frame(frame_bgr, personas_frame)
        return salida

    def anotar_frame(
        self, frame_bgr: np.ndarray, personas: list[dict]
    ) -> np.ndarray:
        canvas = frame_bgr.copy()
        colores = {
            "de_pie_o_caminando": (80, 225, 100),
            "sentado": (255, 200, 50),
            "agachado": (50, 210, 255),
            "posible_caida_en_observacion": (0, 140, 255),
            "caida_confirmada": (30, 30, 255),
        }
        for p in personas:
            x, y, w, h = p["bbox"]
            est = p["estado"]
            col = colores.get(est, (200, 200, 200))
            cv2.rectangle(canvas, (x, y), (x + w, y + h), col, 2)
            txt = f"ID #{p['person_id']} | {est.upper()} (w/h={p['ratio_ancho_alto']})"
            cv2.rectangle(
                canvas, (x, max(0, y - 24)), (x + 310, y), (15, 20, 30), -1
            )
            cv2.putText(
                canvas,
                txt,
                (x + 5, max(16, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                col,
                1,
                cv2.LINE_AA,
            )
        return canvas

    def analizar_video(
        self, ruta_video: str | Path, cada_n_frames: int = 1
    ) -> dict:
        ruta = Path(ruta_video)
        if not ruta.exists():
            raise FileNotFoundError(f"Video no encontrado: {ruta}")

        cap = cv2.VideoCapture(str(ruta))
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {ruta}")

        self.reiniciar()
        ret, primer_frame = cap.read()
        if not ret or primer_frame is None:
            cap.release()
            raise ValueError("Video vacío")

        self.establecer_fondo(primer_frame)
        frame_idx = 0
        conteo_posturas: dict[str, int] = {
            "de_pie_o_caminando": 0,
            "agachado": 0,
            "sentado": 0,
            "posible_caida_en_observacion": 0,
            "caida_confirmada": 0,
        }

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frame_idx += 1
            if frame_idx % max(1, cada_n_frames) != 0:
                continue
            res = self.procesar_frame(frame, frame_idx=frame_idx, dibujar=False)
            for p in res["personas"]:
                conteo_posturas[p["estado"]] = conteo_posturas.get(p["estado"], 0) + 1

        cap.release()
        return {
            "reto": "6. Caídas y comportamientos de riesgo",
            "video": ruta.name,
            "total_alertas_caida": len(self.alertas_caida),
            "alertas": self.alertas_caida,
            "conteo_posturas_observadas": conteo_posturas,
        }
