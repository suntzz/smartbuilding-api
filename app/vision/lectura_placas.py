"""
Reto 5: Lectura de placas.
Lee las placas de carros y motos que pasan por la cámara, de día y de noche,
con una sola lectura consolidada por vehículo y cruce automático contra el
directorio de residentes del edificio (Conjunto Residencial San Carlos).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import cv2
import numpy as np

from app.datos import cargar_residentes


CARACTERES_PLACA = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def _construir_plantillas_ocr() -> dict[str, np.ndarray]:
    """Genera plantillas normalizadas (28x18) de caracteres alfanuméricos para OCR de placas."""
    plantillas: dict[str, np.ndarray] = {}
    for ch in CARACTERES_PLACA:
        lienzo = np.zeros((36, 26), dtype=np.uint8)
        cv2.putText(
            lienzo,
            ch,
            (3, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            255,
            2,
            cv2.LINE_AA,
        )
        ys, xs = np.where(lienzo > 40)
        if len(xs) > 0 and len(ys) > 0:
            recorte = lienzo[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
            plantillas[ch] = cv2.resize(recorte, (18, 28))
    return plantillas


PLANTILLAS_OCR = _construir_plantillas_ocr()


def mejorar_iluminacion_nocturna(frame_bgr: np.ndarray) -> tuple[np.ndarray, str]:
    """
    Detecta si la escena es de día o de noche por luminancia media y aplica
    CLAHE en espacio LAB cuando es de noche para realzar la placa.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        raise ValueError("El frame no puede estar vacío")

    gris = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    brillo = float(np.mean(gris))
    condicion = "noche" if brillo < 85.0 else "dia"

    if condicion == "noche":
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l_ch)
        mejorado = cv2.cvtColor(cv2.merge((l_eq, a_ch, b_ch)), cv2.COLOR_LAB2BGR)
        return mejorado, condicion
    return frame_bgr, condicion


def leer_texto_placa_bgr(placa_bgr: np.ndarray) -> str:
    """
    Segmenta y reconoce los 6 caracteres alfanuméricos de un recorte de placa
    usando umbralización Otsu, componentes conexos y correlación con plantillas.
    """
    if placa_bgr is None or placa_bgr.size == 0:
        raise ValueError("El recorte de placa no puede estar vacío")

    gris = cv2.cvtColor(placa_bgr, cv2.COLOR_BGR2GRAY)
    gris = cv2.resize(gris, (240, 72), interpolation=cv2.INTER_CUBIC)
    _, binaria = cv2.threshold(
        gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    binaria[:6, :] = 0
    binaria[-6:, :] = 0
    binaria[:, :8] = 0
    binaria[:, -8:] = 0

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binaria)
    cajas_chars: list[tuple[int, int, int, int]] = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if 18 <= h <= 64 and 5 <= w <= 85 and area >= 40:
            if w > 44:
                mitad = w // 2
                cajas_chars.append((int(x), int(y), mitad, int(h)))
                cajas_chars.append((int(x + mitad), int(y), int(w - mitad), int(h)))
            else:
                cajas_chars.append((int(x), int(y), int(w), int(h)))

    cajas_chars.sort(key=lambda c: c[0])
    caracteres: list[str] = []

    for idx, (x, y, w, h) in enumerate(cajas_chars[:6]):
        roi = binaria[y : y + h, x : x + w]
        ys, xs = np.where(roi > 0)
        if len(xs) > 0 and len(ys) > 0:
            roi = roi[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        roi_norm = cv2.resize(roi, (18, 28))

        if idx in {0, 1, 2}:
            candidatos = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        elif idx in {3, 4}:
            candidatos = "0123456789"
        else:
            candidatos = CARACTERES_PLACA

        mejor_ch = "?"
        mejor_score = -1e9
        for ch in candidatos:
            tpl = PLANTILLAS_OCR[ch]
            score = float(np.sum((roi_norm > 110) == (tpl > 110)))
            if score > mejor_score:
                mejor_score = score
                mejor_ch = ch
        caracteres.append(mejor_ch)

    return "".join(caracteres)


class LectorPlacasVehiculares:
    """
    Detecta vehículos (carros y motos), localiza su placa de día y de noche,
    consolida UNA SOLA lectura por vehículo y verifica si pertenece a un residente.
    """

    def __init__(self, residentes_lista: list[dict] | None = None) -> None:
        self.residentes = (
            residentes_lista if residentes_lista is not None else cargar_residentes()
        )
        self.mapa_placas: dict[str, dict] = {}
        for r in self.residentes:
            for campo_p, campo_parq in [
                ("placa_1", "parqueadero_1"),
                ("placa_2", "parqueadero_2"),
            ]:
                placa = str(r.get(campo_p, "")).strip().upper()
                if placa and placa != "LIBRE":
                    self.mapa_placas[placa] = {
                        "autorizado": True,
                        "residente": r["full_name"],
                        "torre": r["tower"],
                        "apartamento": r["apartment"],
                        "parqueadero": r.get(campo_parq, ""),
                    }
        self.reiniciar()

    def reiniciar(self) -> None:
        self.vehiculos_tracks: dict[int, dict] = {}
        self.next_vehiculo_id: int = 1

    def _detectar_placas_en_frame(
        self, frame_bgr: np.ndarray
    ) -> list[dict]:
        """Localiza placas amarillas/blancas en el frame y clasifica el vehículo portador."""
        mejorado, condicion_luz = mejorar_iluminacion_nocturna(frame_bgr)
        hsv = cv2.cvtColor(mejorado, cv2.COLOR_BGR2HSV)
        mask_amarilla = cv2.inRange(
            hsv, np.array([14, 75, 95]), np.array([38, 255, 255])
        )
        contours, _ = cv2.findContours(
            mask_amarilla, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        hallazgos: list[dict] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 900:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = w / max(1.0, float(h))
            if not (2.0 <= ratio <= 4.8):
                continue

            recorte_placa = frame_bgr[y : y + h, x : x + w]
            texto = leer_texto_placa_bgr(recorte_placa)
            if len(texto) == 6:
                tipo_vehiculo = "moto" if texto[-1].isalpha() else "carro"
                hallazgos.append(
                    {
                        "bbox_placa": [x, y, w, h],
                        "texto_placa": texto,
                        "tipo_vehiculo": tipo_vehiculo,
                        "condicion_luz": condicion_luz,
                    }
                )
        return hallazgos

    def procesar_frame(
        self, frame_bgr: np.ndarray, frame_idx: int = 0, dibujar: bool = False
    ) -> dict:
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("El frame no puede estar vacío")

        hallazgos = self._detectar_placas_en_frame(frame_bgr)
        detecciones_frame: list[dict] = []

        for h in hallazgos:
            x, y, w, ph = h["bbox_placa"]
            cx, cy = x + w // 2, y + ph // 2
            texto = h["texto_placa"]

            # Asociar con un vehículo activo si comparte >= 4 caracteres de la placa
            vid_match = None
            for vid, vtrk in self.vehiculos_tracks.items():
                sim = sum(
                    1
                    for a, b in zip(texto, vtrk["placa_consolidada"])
                    if a == b
                )
                if sim >= 4 and (frame_idx - vtrk["last_frame"]) <= 35:
                    vid_match = vid
                    break

            if vid_match is None:
                vid_match = self.next_vehiculo_id
                self.next_vehiculo_id += 1
                self.vehiculos_tracks[vid_match] = {
                    "vehiculo_id": vid_match,
                    "votos_placa": Counter(),
                    "tipo_vehiculo": h["tipo_vehiculo"],
                    "condicion_luz": h["condicion_luz"],
                    "first_frame": frame_idx,
                    "last_frame": frame_idx,
                    "cx": cx,
                    "cy": cy,
                    "bbox_placa": h["bbox_placa"],
                    "placa_consolidada": texto,
                }

            vtrk = self.vehiculos_tracks[vid_match]
            vtrk["votos_placa"][texto] += 1
            vtrk["placa_consolidada"] = vtrk["votos_placa"].most_common(1)[0][0]
            vtrk["tipo_vehiculo"] = (
                "moto"
                if (
                    len(vtrk["placa_consolidada"]) == 6
                    and vtrk["placa_consolidada"][-1].isalpha()
                )
                else "carro"
            )
            vtrk["condicion_luz"] = h["condicion_luz"]
            vtrk["last_frame"] = frame_idx
            vtrk["cx"] = cx
            vtrk["cy"] = cy
            vtrk["bbox_placa"] = h["bbox_placa"]

            info_res = self.mapa_placas.get(
                vtrk["placa_consolidada"],
                {
                    "autorizado": False,
                    "residente": "Visitante / No registrado",
                    "torre": None,
                    "apartamento": None,
                    "parqueadero": None,
                },
            )

            detecciones_frame.append(
                {
                    "vehiculo_id": vid_match,
                    "placa": vtrk["placa_consolidada"],
                    "tipo_vehiculo": vtrk["tipo_vehiculo"],
                    "condicion_luz": vtrk["condicion_luz"],
                    "bbox_placa": vtrk["bbox_placa"],
                    **info_res,
                }
            )

        salida = {
            "frame_idx": frame_idx,
            "vehiculos_en_frame": detecciones_frame,
            "total_vehiculos_unicos": len(self.vehiculos_tracks),
        }
        if dibujar:
            salida["frame_anotado"] = self.anotar_frame(frame_bgr, detecciones_frame)
        return salida

    def anotar_frame(
        self, frame_bgr: np.ndarray, detecciones: list[dict]
    ) -> np.ndarray:
        canvas = frame_bgr.copy()
        for d in detecciones:
            x, y, w, h = d["bbox_placa"]
            col = (50, 230, 100) if d["autorizado"] else (40, 170, 255)
            cv2.rectangle(canvas, (x - 4, y - 4), (x + w + 4, y + h + 4), col, 2)
            txt1 = f"Vehiculo #{d['vehiculo_id']} [{d['tipo_vehiculo'].upper()} - {d['condicion_luz'].upper()}]: {d['placa']}"
            txt2 = (
                f"Residente: {d['residente']} (Apto {d['apartamento']} - Parq {d['parqueadero']})"
                if d["autorizado"]
                else "Visitante / No registrado en edificio"
            )
            cv2.rectangle(canvas, (15, 15), (620, 72), (15, 20, 30), -1)
            cv2.putText(
                canvas,
                txt1,
                (25, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.54,
                col,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                txt2,
                (25, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (230, 235, 245),
                1,
                cv2.LINE_AA,
            )
        return canvas

    def obtener_lecturas_unicas(self) -> list[dict]:
        """Devuelve exactamente UNA lectura consolidada por cada vehículo que pasó."""
        lecturas: list[dict] = []
        for vid, vtrk in sorted(self.vehiculos_tracks.items()):
            placa = vtrk["placa_consolidada"]
            info_res = self.mapa_placas.get(
                placa,
                {
                    "autorizado": False,
                    "residente": "Visitante / No registrado",
                    "torre": None,
                    "apartamento": None,
                    "parqueadero": None,
                },
            )
            lecturas.append(
                {
                    "vehiculo_id": vid,
                    "placa": placa,
                    "tipo_vehiculo": vtrk["tipo_vehiculo"],
                    "condicion_luz": vtrk["condicion_luz"],
                    "frames_observados": sum(vtrk["votos_placa"].values()),
                    "primer_frame": vtrk["first_frame"],
                    "ultimo_frame": vtrk["last_frame"],
                    **info_res,
                }
            )
        return lecturas

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
        lecturas = self.obtener_lecturas_unicas()
        return {
            "reto": "5. Lectura de placas",
            "video": ruta.name,
            "total_vehiculos_leidos": len(lecturas),
            "lecturas_unicas": lecturas,
        }
