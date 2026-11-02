"""
Reto 2: Conteo de entradas y salidas.
Cuenta cuántas personas cruzan una o varias líneas virtuales en cada sentido
(entrada vs salida), soportando múltiples líneas y detección de horas pico
con alta densidad de peatones.
"""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np


class ContadorEntradasSalidas:
    """
    Rastreador multi-persona con cruce de una o varias líneas virtuales
    y detección de flujo en hora pico.
    """

    def __init__(
        self,
        lineas_y_norm: dict[str, float] | None = None,
        umbral_hora_pico: int = 3,
    ) -> None:
        self.lineas_y_norm = lineas_y_norm or {
            "linea_acceso_principal": 0.62,
            "linea_distribucion_interior": 0.42,
        }
        self.umbral_hora_pico = umbral_hora_pico
        self.reiniciar()

    def reiniciar(self) -> None:
        self.bg_lab: np.ndarray | None = None
        self.tracks: dict[int, dict] = {}
        self.next_id: int = 1
        self.total_entradas: int = 0
        self.total_salidas: int = 0
        self.conteo_por_linea: dict[str, dict[str, int]] = {
            nombre: {"entradas": 0, "salidas": 0}
            for nombre in self.lineas_y_norm
        }
        self.eventos_cruce: list[dict] = []
        self.max_simultaneos: int = 0
        self.frames_hora_pico: int = 0

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
        fg_mask = (
            (diff_l > 26.0) | (diff_ab > 14.5) | ((diff_l > 17.0) & (diff_ab > 9.0))
        ).astype(np.uint8) * 255

        if np.mean(fg_mask > 0) < 0.010:
            self.bg_lab = 0.94 * self.bg_lab + 0.06 * lab

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
        area_min = max(1100, int(h_img * w_img * 0.006))

        for cnt in contours:
            if cv2.contourArea(cnt) < area_min:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if h < int(h_img * 0.14):
                continue

            # Separación de grupos densos en hora pico (división horizontal si hay 2 o 3 personas juntas)
            if w > h * 0.75 and w > int(w_img * 0.20):
                sub_mask = fg_mask[y : y + h, x : x + w]
                perfil_x = np.sum(sub_mask > 0, axis=0).astype(np.float32)
                perfil_x = cv2.GaussianBlur(
                    perfil_x.reshape(1, -1), (1, 15), 0
                ).flatten()
                mitad_izq = int(w * 0.25)
                mitad_der = int(w * 0.75)
                if mitad_der > mitad_izq + 10:
                    valle = int(np.argmin(perfil_x[mitad_izq:mitad_der])) + mitad_izq
                    if perfil_x[valle] < 0.70 * min(
                        float(np.max(perfil_x[:valle])),
                        float(np.max(perfil_x[valle:])),
                    ):
                        cajas.append((x, y, valle, h))
                        cajas.append((x + valle, y, w - valle, h))
                        continue

            cajas.append((x, y, w, h))

        cajas.sort(key=lambda b: b[0])
        return cajas

    def procesar_frame(
        self, frame_bgr: np.ndarray, frame_idx: int = 0, dibujar: bool = False
    ) -> dict:
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("El frame no puede estar vacío")

        h_img, w_img = frame_bgr.shape[:2]
        cajas = self._detectar_personas(frame_bgr)
        simultaneos = len(cajas)
        if simultaneos > self.max_simultaneos:
            self.max_simultaneos = simultaneos
        es_hora_pico = simultaneos >= self.umbral_hora_pico
        if es_hora_pico:
            self.frames_hora_pico += 1

        lineas_px = {
            nombre: int(y_norm * h_img)
            for nombre, y_norm in self.lineas_y_norm.items()
        }

        usados: set[int] = set()
        activos_frame: list[dict] = []

        for x, y, w, h in cajas:
            cx, cy = x + w // 2, y + int(h * 0.65)
            mejor_id = None
            mejor_dist = 1e9
            for tid, trk in self.tracks.items():
                if tid in usados or (frame_idx - trk["last_frame"]) > 12:
                    continue
                dist = float(np.hypot(cx - trk["cx"], cy - trk["cy"]))
                if dist < max(110.0, max(w, h) * 0.85) and dist < mejor_dist:
                    mejor_dist = dist
                    mejor_id = tid

            if mejor_id is None:
                mejor_id = self.next_id
                self.next_id += 1
                self.tracks[mejor_id] = {
                    "person_id": mejor_id,
                    "first_y": cy,
                    "prev_y": cy,
                    "cx": cx,
                    "cy": cy,
                    "bbox": [x, y, w, h],
                    "last_frame": frame_idx,
                    "trayectoria": [(cx, cy)],
                    "lineas_cruzadas": set(),
                }

            usados.add(mejor_id)
            trk = self.tracks[mejor_id]
            prev_y = trk["cy"]
            trk["prev_y"] = prev_y
            trk["cx"] = cx
            trk["cy"] = cy
            trk["bbox"] = [x, y, w, h]
            trk["last_frame"] = frame_idx
            trk["trayectoria"].append((cx, cy))
            if len(trk["trayectoria"]) > 25:
                trk["trayectoria"] = trk["trayectoria"][-25:]

            # Verificar cruce de cada línea configurada
            y_origen = trk["trayectoria"][0][1]
            for nombre_linea, y_linea in lineas_px.items():
                clave_e = f"{nombre_linea}:entrada"
                clave_s = f"{nombre_linea}:salida"
                # Cruce hacia arriba (desde abajo de la línea hacia arriba -> Entrada al edificio)
                if (
                    y_origen > y_linea + 6
                    and cy <= y_linea
                    and clave_e not in trk["lineas_cruzadas"]
                ):
                    trk["lineas_cruzadas"].add(clave_e)
                    self.conteo_por_linea[nombre_linea]["entradas"] += 1
                    if nombre_linea == "linea_acceso_principal":
                        self.total_entradas += 1
                    self.eventos_cruce.append(
                        {
                            "person_id": mejor_id,
                            "linea": nombre_linea,
                            "sentido": "entrada",
                            "frame": frame_idx,
                            "hora_pico": es_hora_pico,
                        }
                    )
                # Cruce hacia abajo (desde arriba de la línea hacia abajo -> Salida del edificio)
                elif (
                    y_origen < y_linea - 6
                    and cy >= y_linea
                    and clave_s not in trk["lineas_cruzadas"]
                ):
                    trk["lineas_cruzadas"].add(clave_s)
                    self.conteo_por_linea[nombre_linea]["salidas"] += 1
                    if nombre_linea == "linea_acceso_principal":
                        self.total_salidas += 1
                    self.eventos_cruce.append(
                        {
                            "person_id": mejor_id,
                            "linea": nombre_linea,
                            "sentido": "salida",
                            "frame": frame_idx,
                            "hora_pico": es_hora_pico,
                        }
                    )

            activos_frame.append(
                {
                    "person_id": mejor_id,
                    "bbox": [x, y, w, h],
                    "centroide": [cx, cy],
                    "trayectoria": list(trk["trayectoria"]),
                }
            )

        salida = {
            "frame_idx": frame_idx,
            "personas_simultaneas": simultaneos,
            "es_hora_pico": es_hora_pico,
            "total_entradas": self.total_entradas,
            "total_salidas": self.total_salidas,
            "ocupacion_neta": max(0, self.total_entradas - self.total_salidas),
            "conteo_por_linea": self.conteo_por_linea,
        }
        if dibujar:
            salida["frame_anotado"] = self.anotar_frame(
                frame_bgr, activos_frame, lineas_px, es_hora_pico
            )
        return salida

    def anotar_frame(
        self,
        frame_bgr: np.ndarray,
        activos: list[dict],
        lineas_px: dict[str, int],
        es_hora_pico: bool,
    ) -> np.ndarray:
        canvas = frame_bgr.copy()
        h_img, w_img = canvas.shape[:2]

        for idx, (nombre, y_px) in enumerate(lineas_px.items()):
            color_lin = (0, 230, 180) if idx == 0 else (255, 190, 40)
            cv2.line(canvas, (0, y_px), (w_img, y_px), color_lin, 2)
            info_l = self.conteo_por_linea[nombre]
            txt = f"{nombre}: Entradas={info_l['entradas']} | Salidas={info_l['salidas']}"
            cv2.putText(
                canvas,
                txt,
                (14, max(22, y_px - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color_lin,
                2,
                cv2.LINE_AA,
            )

        for p in activos:
            x, y, w, h = p["bbox"]
            cx, cy = p["centroide"]
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (56, 189, 248), 2)
            cv2.circle(canvas, (cx, cy), 4, (0, 255, 255), -1)
            pts = p["trayectoria"]
            for i in range(1, len(pts)):
                cv2.line(canvas, pts[i - 1], pts[i], (56, 189, 248), 2)
            cv2.putText(
                canvas,
                f"ID #{p['person_id']}",
                (x, max(18, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Panel superior de métricas
        cv2.rectangle(canvas, (10, 10), (390, 74), (15, 20, 32), -1)
        cv2.putText(
            canvas,
            f"ENTRADAS: {self.total_entradas}   SALIDAS: {self.total_salidas}",
            (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (80, 250, 160),
            2,
            cv2.LINE_AA,
        )
        estado_txt = (
            "HORA PICO DETECTADA (Alta densidad)"
            if es_hora_pico
            else f"Flujo normal ({len(activos)} en escena)"
        )
        color_est = (60, 80, 255) if es_hora_pico else (180, 210, 240)
        cv2.putText(
            canvas,
            estado_txt,
            (20, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color_est,
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
            "reto": "2. Conteo de entradas y salidas",
            "video": ruta.name,
            "total_entradas": self.total_entradas,
            "total_salidas": self.total_salidas,
            "ocupacion_actual": max(0, self.total_entradas - self.total_salidas),
            "conteo_por_linea": self.conteo_por_linea,
            "max_personas_simultaneas": self.max_simultaneos,
            "hubo_hora_pico": self.frames_hora_pico > 0,
            "frames_en_hora_pico": self.frames_hora_pico,
            "eventos_cruce": self.eventos_cruce,
        }
