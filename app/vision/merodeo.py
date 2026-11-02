"""
Reto 3: Merodeo.
Alerta cuando una persona permanece o deambula en una zona de forma anómala,
sin alertar a quien espera, conversa o simplemente pasa.
"""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np


class DetectorMerodeo:
    """
    Analizador de trayectorias para detección de merodeo anómalo (deambulación errática),
    diferenciándolo de:
    - 'paso_normal' (tránsito directo o corta permanencia)
    - 'esperando' (permanencia quieta sin deambulación errática)
    - 'conversando' (dos o más personas próximas interactuando estáticamente)
    """

    def __init__(
        self,
        umbral_frames_merodeo: int = 16,
        distancia_min_deambulacion_px: float = 95.0,
        eficiencia_max_merodeo: float = 0.62,
        distancia_conversacion_px: float = 165.0,
    ) -> None:
        self.umbral_frames_merodeo = umbral_frames_merodeo
        self.distancia_min_deambulacion_px = distancia_min_deambulacion_px
        self.eficiencia_max_merodeo = eficiencia_max_merodeo
        self.distancia_conversacion_px = distancia_conversacion_px
        self.reiniciar()

    def reiniciar(self) -> None:
        self.bg_lab: np.ndarray | None = None
        self.tracks: dict[int, dict] = {}
        self.next_id: int = 1
        self.alertas_merodeo: list[dict] = []
        self.ids_alertados: set[int] = set()

    def establecer_fondo(self, frame_bgr: np.ndarray) -> None:
        blur = cv2.GaussianBlur(frame_bgr, (5, 5), 0)
        self.bg_lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB).astype(np.float32)

    def _detectar_personas(
        self, frame_bgr: np.ndarray
    ) -> list[tuple[int, int, int, int]]:
        h_img, w_img = frame_bgr.shape[:2]
        blur = cv2.GaussianBlur(frame_bgr, (5, 5), 0)
        lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB).astype(np.float32)

        if self.bg_lab is None or self.bg_lab.shape != lab.shape:
            self.bg_lab = lab.copy()
            return []

        diff = np.abs(lab - self.bg_lab)
        diff_l = diff[:, :, 0]
        diff_ab = np.sqrt(diff[:, :, 1] ** 2 + diff[:, :, 2] ** 2)
        fg_mask = ((diff_l > 24.0) | (diff_ab > 13.0)).astype(np.uint8) * 255

        fg_mask = cv2.morphologyEx(
            fg_mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )
        fg_mask = cv2.morphologyEx(
            fg_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (13, 25)),
        )

        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cajas: list[tuple[int, int, int, int]] = []
        area_min = max(1400, int(h_img * w_img * 0.009))

        for cnt in contours:
            if cv2.contourArea(cnt) < area_min:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if h < int(h_img * 0.16):
                continue

            if w > h * 0.80 and w > int(w_img * 0.22):
                sub_mask = fg_mask[y : y + h, x : x + w]
                perfil_x = np.sum(sub_mask > 0, axis=0).astype(np.float32)
                perfil_x = cv2.GaussianBlur(
                    perfil_x.reshape(1, -1), (1, 15), 0
                ).flatten()
                m1, m2 = int(w * 0.25), int(w * 0.75)
                if m2 > m1 + 10:
                    valle = int(np.argmin(perfil_x[m1:m2])) + m1
                    if perfil_x[valle] < 0.72 * min(
                        float(np.max(perfil_x[:valle])),
                        float(np.max(perfil_x[valle:])),
                    ):
                        cajas.append((x, y, valle, h))
                        cajas.append((x + valle, y, w - valle, h))
                        continue

            cajas.append((x, y, w, h))

        cajas.sort(key=lambda b: b[0])
        return cajas

    def clasificar_comportamiento_trayectoria(
        self,
        puntos: list[tuple[int, int]],
        en_proximidad_social: bool = False,
    ) -> dict:
        """
        Clasifica una trayectoria de puntos (x, y) en:
        - 'paso_normal'
        - 'esperando'
        - 'conversando'
        - 'merodeo' (ALERTA)
        """
        n = len(puntos)
        if n < 2:
            return {
                "comportamiento": "paso_normal",
                "alerta": False,
                "distancia_recorrida": 0.0,
                "desplazamiento_neto": 0.0,
                "eficiencia": 1.0,
                "cambios_direccion": 0,
            }

        dist_total = 0.0
        cambios_dir = 0
        dx_prev = 0.0
        dy_prev = 0.0

        for i in range(1, n):
            dx = float(puntos[i][0] - puntos[i - 1][0])
            dy = float(puntos[i][1] - puntos[i - 1][1])
            paso = float(np.hypot(dx, dy))
            dist_total += paso
            if paso > 4.0:
                if (dx_prev != 0.0 or dy_prev != 0.0) and (
                    dx * dx_prev + dy * dy_prev
                ) < -8.0:
                    cambios_dir += 1
                dx_prev, dy_prev = dx, dy

        despl_neto = float(
            np.hypot(puntos[-1][0] - puntos[0][0], puntos[-1][1] - puntos[0][1])
        )
        eficiencia = despl_neto / max(1.0, dist_total)

        if en_proximidad_social and n >= 8 and eficiencia < 0.75:
            comportamiento = "conversando"
            alerta = False
        elif n < self.umbral_frames_merodeo:
            comportamiento = "paso_normal"
            alerta = False
        elif dist_total < self.distancia_min_deambulacion_px:
            comportamiento = "esperando"
            alerta = False
        elif (
            eficiencia <= self.eficiencia_max_merodeo
            or cambios_dir >= 1
        ):
            comportamiento = "merodeo"
            alerta = True
        else:
            comportamiento = "paso_normal"
            alerta = False

        return {
            "comportamiento": comportamiento,
            "alerta": alerta,
            "distancia_recorrida": round(dist_total, 1),
            "desplazamiento_neto": round(despl_neto, 1),
            "eficiencia": round(eficiencia, 2),
            "cambios_direccion": cambios_dir,
        }

    def procesar_frame(
        self, frame_bgr: np.ndarray, frame_idx: int = 0, dibujar: bool = False
    ) -> dict:
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("El frame no puede estar vacío")

        cajas = self._detectar_personas(frame_bgr)
        usados: set[int] = set()
        activos_ids: list[int] = []

        for x, y, w, h in cajas:
            cx, cy = x + w // 2, y + h // 2
            mejor_id = None
            mejor_dist = 1e9
            for tid, trk in self.tracks.items():
                if tid in usados or (frame_idx - trk["last_frame"]) > 15:
                    continue
                dist = float(np.hypot(cx - trk["cx"], cy - trk["cy"]))
                if dist < 130.0 and dist < mejor_dist:
                    mejor_dist = dist
                    mejor_id = tid

            if mejor_id is None:
                mejor_id = self.next_id
                self.next_id += 1
                self.tracks[mejor_id] = {
                    "person_id": mejor_id,
                    "first_frame": frame_idx,
                    "last_frame": frame_idx,
                    "cx": cx,
                    "cy": cy,
                    "bbox": [x, y, w, h],
                    "puntos": [(cx, cy)],
                    "comportamiento": "paso_normal",
                    "alerta": False,
                }

            usados.add(mejor_id)
            activos_ids.append(mejor_id)
            trk = self.tracks[mejor_id]
            trk["last_frame"] = frame_idx
            trk["cx"] = cx
            trk["cy"] = cy
            trk["bbox"] = [x, y, w, h]
            trk["puntos"].append((cx, cy))

        # Determinar quiénes están conversando (2+ personas cercanas)
        en_conversacion: set[int] = set()
        for i in range(len(activos_ids)):
            for j in range(i + 1, len(activos_ids)):
                t1 = self.tracks[activos_ids[i]]
                t2 = self.tracks[activos_ids[j]]
                d = float(np.hypot(t1["cx"] - t2["cx"], t1["cy"] - t2["cy"]))
                if d <= self.distancia_conversacion_px:
                    en_conversacion.add(activos_ids[i])
                    en_conversacion.add(activos_ids[j])

        personas_frame: list[dict] = []
        for tid in activos_ids:
            trk = self.tracks[tid]
            eval_comp = self.clasificar_comportamiento_trayectoria(
                trk["puntos"], en_proximidad_social=(tid in en_conversacion)
            )
            trk["comportamiento"] = eval_comp["comportamiento"]
            trk["alerta"] = eval_comp["alerta"]

            if eval_comp["alerta"] and tid not in self.ids_alertados:
                self.ids_alertados.add(tid)
                self.alertas_merodeo.append(
                    {
                        "person_id": tid,
                        "frame_alerta": frame_idx,
                        "comportamiento": "merodeo",
                        "distancia_recorrida": eval_comp["distancia_recorrida"],
                        "eficiencia_trayectoria": eval_comp["eficiencia"],
                        "mensaje": f"Alerta de merodeo: Persona #{tid} deambula de forma anómala en la zona.",
                    }
                )

            personas_frame.append(
                {
                    "person_id": tid,
                    "bbox": trk["bbox"],
                    "comportamiento": trk["comportamiento"],
                    "alerta": trk["alerta"],
                    "metricas": eval_comp,
                    "puntos": trk["puntos"][-35:],
                }
            )

        salida = {
            "frame_idx": frame_idx,
            "personas": personas_frame,
            "alerta_activa": any(p["alerta"] for p in personas_frame),
            "total_alertas_acumuladas": len(self.alertas_merodeo),
        }
        if dibujar:
            salida["frame_anotado"] = self.anotar_frame(frame_bgr, personas_frame)
        return salida

    def anotar_frame(
        self, frame_bgr: np.ndarray, personas: list[dict]
    ) -> np.ndarray:
        canvas = frame_bgr.copy()
        colores = {
            "paso_normal": (80, 220, 100),
            "esperando": (255, 200, 60),
            "conversando": (255, 170, 50),
            "merodeo": (40, 40, 255),
        }
        for p in personas:
            x, y, w, h = p["bbox"]
            comp = p["comportamiento"]
            col = colores.get(comp, (200, 200, 200))
            cv2.rectangle(canvas, (x, y), (x + w, y + h), col, 2)
            pts = p["puntos"]
            for i in range(1, len(pts)):
                cv2.line(canvas, pts[i - 1], pts[i], col, 2)
            etiqueta = f"ID #{p['person_id']} | {comp.upper()}"
            cv2.rectangle(
                canvas, (x, max(0, y - 24)), (x + 195, y), (18, 22, 32), -1
            )
            cv2.putText(
                canvas,
                etiqueta,
                (x + 5, max(15, y - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                col,
                1,
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
        comportamientos_vistos: dict[str, int] = {
            "paso_normal": 0,
            "esperando": 0,
            "conversando": 0,
            "merodeo": 0,
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
                comportamientos_vistos[p["comportamiento"]] += 1

        cap.release()
        return {
            "reto": "3. Merodeo",
            "video": ruta.name,
            "total_alertas_merodeo": len(self.alertas_merodeo),
            "alertas": self.alertas_merodeo,
            "conteo_estados_observados": comportamientos_vistos,
        }
