"""
Reto 7: Correspondencia en casillero.
Detecta cuándo hay paquetes o recibos dejados en el casillero del apartamento
y lo reporta con número de casillero, número de apartamento y residentes asociados.
"""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

from app.datos import cargar_residentes


CASILLEROS_COORDS_DEFAULT = [
    {"casillero": "C101", "apartamento": 101, "torre": 1, "bbox": [40, 60, 125, 115]},
    {"casillero": "C102", "apartamento": 102, "torre": 1, "bbox": [185, 60, 125, 115]},
    {"casillero": "C103", "apartamento": 103, "torre": 1, "bbox": [330, 60, 125, 115]},
    {"casillero": "C104", "apartamento": 104, "torre": 1, "bbox": [475, 60, 125, 115]},
    {"casillero": "C201", "apartamento": 201, "torre": 1, "bbox": [40, 210, 125, 115]},
    {"casillero": "C202", "apartamento": 202, "torre": 1, "bbox": [185, 210, 125, 115]},
    {"casillero": "C203", "apartamento": 203, "torre": 1, "bbox": [330, 210, 125, 115]},
    {"casillero": "C204", "apartamento": 204, "torre": 1, "bbox": [475, 210, 125, 115]},
]


class DetectorCorrespondenciaCasillero:
    """
    Inspecciona visualmente la grilla de casilleros del edificio, detecta si un
    casillero contiene un 'paquete' o un 'recibo' y cruza el número de casillero
    con los residentes registrados en `residentes_san_carlos.csv`.
    """

    def __init__(
        self,
        casilleros_config: list[dict] | None = None,
        residentes_lista: list[dict] | None = None,
    ) -> None:
        self.casilleros_config = casilleros_config or CASILLEROS_COORDS_DEFAULT
        self.residentes = (
            residentes_lista if residentes_lista is not None else cargar_residentes()
        )
        self.mapa_casilleros: dict[str, list[str]] = {}
        for r in self.residentes:
            cod = str(r.get("casillero", f"C{r['apartment']}")).strip().upper()
            self.mapa_casilleros.setdefault(cod, []).append(r["full_name"])
        self.reiniciar()

    def reiniciar(self) -> None:
        self.bg_bgr: np.ndarray | None = None
        self.ultimo_estado_casilleros: list[dict] = []

    def establecer_fondo(self, frame_bgr: np.ndarray) -> None:
        self.bg_bgr = frame_bgr.copy()

    def _analizar_celda_casillero(
        self, roi_actual: np.ndarray, roi_fondo: np.ndarray | None = None
    ) -> dict:
        """
        Determina si el interior de un casillero está 'vacio' o contiene un
        'paquete' (caja volumétrica) o 'recibo' (sobre/factura plana blanca).
        """
        h, w = roi_actual.shape[:2]
        interior = roi_actual[8 : h - 6, 8 : w - 8]
        if roi_fondo is not None and roi_fondo.shape == roi_actual.shape:
            int_bg = roi_fondo[8 : h - 6, 8 : w - 8]
            diff = cv2.absdiff(interior, int_bg)
            mask_obj = (np.max(diff, axis=2) > 28).astype(np.uint8) * 255
        else:
            # Comparar contra el fondo oscuro del casillero vacío (brillo > 55)
            gris = cv2.cvtColor(interior, cv2.COLOR_BGR2GRAY)
            mask_obj = (gris > 58).astype(np.uint8) * 255

        ocupacion = float(np.mean(mask_obj > 0))
        if ocupacion < 0.06:
            return {"ocupado": False, "tipo_correspondencia": None}

        ys, xs = np.where(mask_obj > 0)
        alto_rel = (ys.max() - ys.min() + 1) / max(1.0, float(interior.shape[0]))
        pixeles_obj = interior[mask_obj > 0]
        med_bgr = np.median(pixeles_obj, axis=0)
        b, g, r = float(med_bgr[0]), float(med_bgr[1]), float(med_bgr[2])

        # Si el objeto es alto (> 38% de la celda) o tiene tono cartón/kraft (r > b + 25), es paquete
        if alto_rel > 0.36 or (r > b + 25 and g > b + 15):
            tipo = "paquete"
        else:
            tipo = "recibo"

        return {"ocupado": True, "tipo_correspondencia": tipo}

    def procesar_frame(
        self, frame_bgr: np.ndarray, frame_idx: int = 0, dibujar: bool = False
    ) -> dict:
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("El frame no puede estar vacío")

        if self.bg_bgr is None:
            self.bg_bgr = frame_bgr.copy()

        casilleros_estado: list[dict] = []
        notificaciones: list[dict] = []

        for cfg in self.casilleros_config:
            cod = cfg["casillero"]
            apto = cfg["apartamento"]
            torre = cfg["torre"]
            x, y, cw, ch = cfg["bbox"]

            roi_act = frame_bgr[y : y + ch, x : x + cw]
            roi_bg = (
                self.bg_bgr[y : y + ch, x : x + cw]
                if self.bg_bgr is not None
                else None
            )
            info = self._analizar_celda_casillero(roi_act, roi_bg)
            residentes_apto = self.mapa_casilleros.get(cod, [])

            item = {
                "casillero": cod,
                "apartamento": apto,
                "torre": torre,
                "bbox": [x, y, cw, ch],
                "ocupado": info["ocupado"],
                "tipo_correspondencia": info["tipo_correspondencia"],
                "residentes": residentes_apto,
            }
            casilleros_estado.append(item)
            if info["ocupado"]:
                notificaciones.append(
                    {
                        "casillero": cod,
                        "apartamento": apto,
                        "torre": torre,
                        "tipo_correspondencia": info["tipo_correspondencia"],
                        "residentes_notificados": residentes_apto,
                        "mensaje": (
                            f"Casillero {cod} (Apto {apto}): Se detectó "
                            f"{info['tipo_correspondencia']} pendiente de recoger."
                        ),
                    }
                )

        self.ultimo_estado_casilleros = casilleros_estado
        salida = {
            "frame_idx": frame_idx,
            "casilleros_con_correspondencia": len(notificaciones),
            "reporte_apartamentos": notificaciones,
            "estado_casilleros": casilleros_estado,
        }
        if dibujar:
            salida["frame_anotado"] = self.anotar_frame(frame_bgr, casilleros_estado)
        return salida

    def anotar_frame(
        self, frame_bgr: np.ndarray, casilleros_estado: list[dict]
    ) -> np.ndarray:
        canvas = frame_bgr.copy()
        for c in casilleros_estado:
            x, y, cw, ch = c["bbox"]
            if c["ocupado"]:
                tipo = c["tipo_correspondencia"]
                col = (40, 175, 255) if tipo == "paquete" else (80, 240, 140)
                cv2.rectangle(canvas, (x, y), (x + cw, y + ch), col, 3)
                etiqueta = f"Apto {c['apartamento']}: {tipo.upper()}"
                cv2.rectangle(
                    canvas, (x + 4, y + 6), (x + cw - 4, y + 28), (15, 20, 28), -1
                )
                cv2.putText(
                    canvas,
                    etiqueta,
                    (x + 8, y + 21),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.40,
                    col,
                    1,
                    cv2.LINE_AA,
                )
            else:
                cv2.rectangle(
                    canvas, (x, y), (x + cw, y + ch), (100, 110, 125), 1
                )
        return canvas

    def analizar_video(
        self, ruta_video: str | Path, cada_n_frames: int = 3
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
        ultimo_res = self.procesar_frame(primer_frame, frame_idx=0, dibujar=False)

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frame_idx += 1
            if frame_idx % max(1, cada_n_frames) != 0:
                continue
            ultimo_res = self.procesar_frame(
                frame, frame_idx=frame_idx, dibujar=False
            )

        cap.release()
        return {
            "reto": "7. Correspondencia en casillero",
            "video": ruta.name,
            "total_casilleros_ocupados": ultimo_res["casilleros_con_correspondencia"],
            "correspondencia_detectada": ultimo_res["reporte_apartamentos"],
            "estado_todos_los_casilleros": ultimo_res["estado_casilleros"],
        }
