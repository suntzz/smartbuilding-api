"""
Reto 4: Objeto abandonado.
Alerta cuando alguien deja un bolso, una maleta o un paquete y se aleja.
"""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np


class DetectorObjetoAbandonado:
    """
    Detecta objetos dejados en escena (bolso, maleta, paquete) comparando el fondo
    de referencia y rastreando la distancia a la persona más cercana.
    Solo emite alerta cuando la persona se aleja más allá del umbral espacial
    y el objeto permanece inmóvil durante un umbral de frames.
    """

    def __init__(
        self,
        umbral_frames_abandono: int = 12,
        distancia_alejamiento_px: float = 135.0,
    ) -> None:
        self.umbral_frames_abandono = umbral_frames_abandono
        self.distancia_alejamiento_px = distancia_alejamiento_px
        self.reiniciar()

    def reiniciar(self) -> None:
        self.bg_lab: np.ndarray | None = None
        self.objetos_tracks: dict[int, dict] = {}
        self.next_obj_id: int = 1
        self.alertas_abandono: list[dict] = []
        self.objetos_alertados: set[int] = set()

    def establecer_fondo(self, frame_bgr: np.ndarray) -> None:
        blur = cv2.GaussianBlur(frame_bgr, (5, 5), 0)
        self.bg_lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB).astype(np.float32)

    def _clasificar_tipo_objeto(self, w: int, h: int) -> str:
        ratio = w / max(1.0, float(h))
        area = w * h
        if area > 4200 or ratio < 0.82:
            return "maleta"
        if ratio > 1.25:
            return "bolso"
        return "paquete"

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
            return {
                "frame_idx": frame_idx,
                "personas": [],
                "objetos": [],
                "alerta_activa": False,
            }

        diff = np.abs(lab - self.bg_lab)
        diff_l = diff[:, :, 0]
        diff_ab = np.sqrt(diff[:, :, 1] ** 2 + diff[:, :, 2] ** 2)
        fg_mask = ((diff_l > 25.0) | (diff_ab > 13.5)).astype(np.uint8) * 255

        fg_mask = cv2.morphologyEx(
            fg_mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )
        fg_mask = cv2.morphologyEx(
            fg_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (9, 13)),
        )

        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        personas_cajas: list[tuple[int, int, int, int]] = []
        objetos_cajas: list[tuple[int, int, int, int]] = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 320:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            # Personas: siluetas altas o de gran área
            if h >= int(h_img * 0.23) or area >= int(h_img * w_img * 0.025):
                personas_cajas.append((x, y, w, h))
            # Objetos candidatos (bolso, maleta, paquete en suelo)
            elif 320 <= area < int(h_img * w_img * 0.025) and y > int(h_img * 0.25):
                objetos_cajas.append((x, y, w, h))

        usados_obj: set[int] = set()
        objetos_estado: list[dict] = []

        for ox, oy, ow, oh in objetos_cajas:
            ocx, ocy = ox + ow // 2, oy + oh // 2
            mejor_id = None
            mejor_dist = 1e9
            for oid, otrk in self.objetos_tracks.items():
                if oid in usados_obj or (frame_idx - otrk["last_frame"]) > 20:
                    continue
                d = float(np.hypot(ocx - otrk["cx"], ocy - otrk["cy"]))
                if d < 45.0 and d < mejor_dist:
                    mejor_dist = d
                    mejor_id = oid

            if mejor_id is None:
                mejor_id = self.next_obj_id
                self.next_obj_id += 1
                self.objetos_tracks[mejor_id] = {
                    "objeto_id": mejor_id,
                    "tipo": self._clasificar_tipo_objeto(ow, oh),
                    "cx": ocx,
                    "cy": ocy,
                    "bbox": [ox, oy, ow, oh],
                    "first_frame": frame_idx,
                    "last_frame": frame_idx,
                    "frames_estatico": 0,
                    "frames_sin_propietario": 0,
                    "estado": "con_propietario",
                    "alerta": False,
                }

            usados_obj.add(mejor_id)
            otrk = self.objetos_tracks[mejor_id]
            despl = float(np.hypot(ocx - otrk["cx"], ocy - otrk["cy"]))
            otrk["cx"] = ocx
            otrk["cy"] = ocy
            otrk["bbox"] = [ox, oy, ow, oh]
            otrk["last_frame"] = frame_idx
            if despl < 12.0:
                otrk["frames_estatico"] += 1
            else:
                otrk["frames_estatico"] = max(1, otrk["frames_estatico"] - 1)

            # Calcular distancia a la persona más cercana
            if personas_cajas:
                dist_min_persona = min(
                    float(
                        np.hypot(
                            ocx - (px + pw // 2),
                            ocy - (py + int(ph * 0.75)),
                        )
                    )
                    for px, py, pw, ph in personas_cajas
                )
            else:
                dist_min_persona = 9999.0

            propietario_lejos = dist_min_persona >= self.distancia_alejamiento_px
            if propietario_lejos:
                otrk["frames_sin_propietario"] += 1
            else:
                otrk["frames_sin_propietario"] = 0

            if (
                propietario_lejos
                and otrk["frames_sin_propietario"] >= self.umbral_frames_abandono
            ):
                otrk["estado"] = "abandonado"
                otrk["alerta"] = True
                if mejor_id not in self.objetos_alertados:
                    self.objetos_alertados.add(mejor_id)
                    self.alertas_abandono.append(
                        {
                            "objeto_id": mejor_id,
                            "tipo_objeto": otrk["tipo"],
                            "frame_alerta": frame_idx,
                            "ubicacion": [ocx, ocy],
                            "distancia_propietario_px": round(dist_min_persona, 1),
                            "mensaje": (
                                f"Alerta de objeto abandonado: {otrk['tipo']} #{mejor_id} "
                                f"dejado sin supervisión."
                            ),
                        }
                    )
            elif propietario_lejos:
                otrk["estado"] = "propietario_alejandose"
                otrk["alerta"] = False
            else:
                otrk["estado"] = "con_propietario"
                otrk["alerta"] = False

            objetos_estado.append(
                {
                    "objeto_id": mejor_id,
                    "tipo": otrk["tipo"],
                    "bbox": [ox, oy, ow, oh],
                    "estado": otrk["estado"],
                    "alerta": otrk["alerta"],
                    "distancia_persona_px": round(dist_min_persona, 1),
                    "frames_sin_propietario": otrk["frames_sin_propietario"],
                }
            )

        salida = {
            "frame_idx": frame_idx,
            "personas": [
                {"bbox": [px, py, pw, ph]} for px, py, pw, ph in personas_cajas
            ],
            "objetos": objetos_estado,
            "alerta_activa": any(o["alerta"] for o in objetos_estado),
            "total_alertas": len(self.alertas_abandono),
        }
        if dibujar:
            salida["frame_anotado"] = self.anotar_frame(
                frame_bgr, personas_cajas, objetos_estado
            )
        return salida

    def anotar_frame(
        self,
        frame_bgr: np.ndarray,
        personas_cajas: list[tuple[int, int, int, int]],
        objetos: list[dict],
    ) -> np.ndarray:
        canvas = frame_bgr.copy()
        for px, py, pw, ph in personas_cajas:
            cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (240, 180, 50), 2)
            cv2.putText(
                canvas,
                "Persona",
                (px, max(16, py - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (240, 180, 50),
                1,
                cv2.LINE_AA,
            )

        for obj in objetos:
            ox, oy, ow, oh = obj["bbox"]
            col = (35, 35, 255) if obj["alerta"] else (0, 215, 255)
            cv2.rectangle(canvas, (ox, oy), (ox + ow, oy + oh), col, 2)
            if obj["alerta"]:
                cv2.circle(
                    canvas,
                    (ox + ow // 2, oy + oh // 2),
                    max(ow, oh),
                    (35, 35, 255),
                    2,
                )
            txt = f"{obj['tipo'].upper()} ({obj['estado'].upper()})"
            cv2.putText(
                canvas,
                txt,
                (max(6, ox - 10), max(18, oy - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                col,
                2,
                cv2.LINE_AA,
            )
        return canvas

    def analizar_video(
        self, ruta_video: str | Path, cada_n_frames: int = 2
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
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frame_idx += 1
            if frame_idx % max(1, cada_n_frames) != 0:
                continue
            self.procesar_frame(frame, frame_idx=frame_idx, dibujar=False)

        cap.release()
        return {
            "reto": "4. Objeto abandonado",
            "video": ruta.name,
            "total_alertas_abandono": len(self.alertas_abandono),
            "alertas": self.alertas_abandono,
        }
