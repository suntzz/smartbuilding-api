"""
Reto 1: Demografía y vestimenta.
Para cada persona que pasa frente a la cámara, estima:
- Sexo aparente (Masculino / Femenino)
- Rango de edad (Infantil (0-14), Joven (15-29), Adulto (30-54), Adulto Mayor (55+))
- Color de la prenda superior e inferior
- Tipo de prenda superior (camiseta, camisa, chaqueta, buso, blusa)
  e inferior (pantalon, jean, pantaloneta, falda)
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import cv2
import numpy as np


COLORES_VALIDOS = {
    "negro",
    "blanco",
    "gris",
    "rojo",
    "azul",
    "azul oscuro",
    "verde",
    "amarillo",
    "naranja",
    "morado",
    "rosado",
    "cafe",
    "beige",
}

TIPOS_PRENDA_SUPERIOR = {"camiseta", "camisa", "chaqueta", "buso", "blusa", "vestido"}
TIPOS_PRENDA_INFERIOR = {"pantalon", "jean", "pantaloneta", "falda", "vestido"}
RANGOS_EDAD = {
    "Infantil (0-14)",
    "Joven (15-29)",
    "Adulto (30-54)",
    "Adulto Mayor (55+)",
}
SEXOS_APARENTES = {"Masculino", "Femenino"}


def detectar_mascara_piel(bgr: np.ndarray) -> np.ndarray:
    """Devuelve una máscara binaria (0 o 255) de píxeles con tono de piel."""
    if bgr is None or bgr.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask_ycrcb = cv2.inRange(ycrcb, np.array([50, 135, 80]), np.array([255, 173, 127]))
    mask_hsv = cv2.inRange(hsv, np.array([0, 25, 70]), np.array([22, 170, 255]))
    return cv2.bitwise_and(mask_ycrcb, mask_hsv)


def clasificar_color_bgr(
    region_bgr: np.ndarray, mascara: np.ndarray | None = None
) -> dict:
    """
    Clasifica el color dominante de una región BGR de vestimenta en español,
    filtrando píxeles de piel y de fondo si se provee máscara.
    """
    if region_bgr is None or region_bgr.size == 0:
        raise ValueError("La región de imagen no puede estar vacía")

    hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
    piel = detectar_mascara_piel(region_bgr)
    no_piel = cv2.bitwise_not(piel)

    if mascara is not None and mascara.shape[:2] == region_bgr.shape[:2]:
        kernel_erode = np.ones((3, 3), np.uint8)
        mascara_limpia = cv2.erode(mascara, kernel_erode, iterations=1)
        if cv2.countNonZero(mascara_limpia) < 25:
            mascara_limpia = mascara
        mask_final = cv2.bitwise_and(mascara_limpia, no_piel)
        if cv2.countNonZero(mask_final) < 20:
            mask_final = mascara_limpia
    else:
        mask_final = no_piel

    pixeles_bgr = region_bgr[mask_final > 0]
    pixeles_hsv = hsv[mask_final > 0]

    if len(pixeles_bgr) < 15:
        pixeles_bgr = region_bgr.reshape(-1, 3)
        pixeles_hsv = hsv.reshape(-1, 3)

    med_bgr = np.median(pixeles_bgr, axis=0).astype(int)
    b, g, r = int(med_bgr[0]), int(med_bgr[1]), int(med_bgr[2])
    hex_color = f"#{r:02X}{g:02X}{b:02X}"

    h = float(np.median(pixeles_hsv[:, 0]))
    s = float(np.median(pixeles_hsv[:, 1]))
    v = float(np.median(pixeles_hsv[:, 2]))

    if v < 62:
        nombre = "negro"
    elif s < 42:
        if v >= 182:
            nombre = "blanco"
        else:
            nombre = "gris"
    elif s < 78 and 10 <= h <= 32 and v >= 135:
        nombre = "beige"
    else:
        if h < 8 or h >= 168:
            nombre = "rosado" if (s < 110 and v > 175) else "rojo"
        elif 8 <= h < 20:
            nombre = "naranja" if v >= 145 else "cafe"
        elif 20 <= h < 35:
            nombre = "amarillo" if v >= 140 else "cafe"
        elif 35 <= h < 85:
            nombre = "verde"
        elif 85 <= h < 135:
            nombre = "azul oscuro" if v < 105 else "azul"
        elif 135 <= h < 156:
            nombre = "morado"
        else:
            nombre = "rosado"

    return {
        "color": nombre,
        "hex": hex_color,
        "rgb": [r, g, b],
    }


def clasificar_tipo_prenda_superior(
    torso_bgr: np.ndarray,
    mascara_torso: np.ndarray | None = None,
    color_superior: str = "",
) -> str:
    """
    Clasifica el tipo de prenda superior ('camiseta', 'camisa', 'chaqueta', 'buso', 'blusa')
    usando proporción de piel en brazos laterales, contraste central y textura.
    """
    if torso_bgr is None or torso_bgr.size == 0:
        raise ValueError("El recorte de torso no puede estar vacío")

    h, w = torso_bgr.shape[:2]
    if h < 6 or w < 6:
        return "camiseta"

    piel = detectar_mascara_piel(torso_bgr)
    if mascara_torso is not None and mascara_torso.shape[:2] == (h, w):
        piel = cv2.bitwise_and(piel, mascara_torso)

    ancho_lat = max(2, int(w * 0.24))
    lat_izq = piel[int(h * 0.25) : int(h * 0.90), :ancho_lat]
    lat_der = piel[int(h * 0.25) : int(h * 0.90), w - ancho_lat :]
    if mascara_torso is not None and mascara_torso.shape[:2] == (h, w):
        fg_izq = max(1, cv2.countNonZero(mascara_torso[int(h * 0.25) : int(h * 0.90), :ancho_lat]))
        fg_der = max(1, cv2.countNonZero(mascara_torso[int(h * 0.25) : int(h * 0.90), w - ancho_lat :]))
        ratio_piel_brazos = float(
            (cv2.countNonZero(lat_izq) / fg_izq + cv2.countNonZero(lat_der) / fg_der) / 2.0
        )
    else:
        ratio_piel_brazos = float((np.mean(lat_izq > 0) + np.mean(lat_der > 0)) / 2.0)

    cuello = piel[: max(2, int(h * 0.28)), int(w * 0.35) : int(w * 0.65)]
    ratio_piel_escote = float(np.mean(cuello > 0))

    gris = cv2.cvtColor(torso_bgr, cv2.COLOR_BGR2GRAY)
    franja_central = gris[int(h * 0.15) : int(h * 0.85), int(w * 0.42) : int(w * 0.58)]
    franja_media = gris[int(h * 0.15) : int(h * 0.85), int(w * 0.22) : int(w * 0.78)]
    sobel_x = cv2.Sobel(franja_central, cv2.CV_64F, 1, 0, ksize=3)
    energia_vertical_centro = float(np.mean(np.abs(sobel_x)))
    std_torso = float(np.std(franja_media))

    if ratio_piel_brazos > 0.16:
        if ratio_piel_escote > 0.25:
            return "blusa"
        return "camiseta"

    if energia_vertical_centro > 28.0 and (
        color_superior in {"negro", "gris", "azul oscuro", "cafe"} or std_torso > 36.0
    ):
        return "chaqueta"

    if color_superior in {"blanco", "azul", "beige"} or energia_vertical_centro > 20.0:
        return "camisa"

    if ratio_piel_escote > 0.20:
        return "blusa"

    if color_superior in {"negro", "gris", "morado", "verde", "rojo"} and std_torso < 24.0:
        return "buso"

    return "camiseta"


def clasificar_tipo_prenda_inferior(
    piernas_bgr: np.ndarray,
    mascara_piernas: np.ndarray | None = None,
    color_inferior: str = "",
) -> str:
    """
    Clasifica el tipo de prenda inferior ('pantalon', 'jean', 'pantaloneta', 'falda')
    evaluando la geometría de silueta (campana de falda vs columnas de pantalón),
    piel en pantorrillas dentro de la máscara y color/textura.
    """
    if piernas_bgr is None or piernas_bgr.size == 0:
        raise ValueError("El recorte de piernas no puede estar vacío")

    h, w = piernas_bgr.shape[:2]
    if h < 6 or w < 6:
        return "pantalon"

    piel = detectar_mascara_piel(piernas_bgr)
    if mascara_piernas is not None and mascara_piernas.shape[:2] == (h, w):
        piel = cv2.bitwise_and(piel, mascara_piernas)
        sub_mask_inf = mascara_piernas[int(h * 0.55) :, int(w * 0.18) : int(w * 0.82)]
        fg_inf = max(1, cv2.countNonZero(sub_mask_inf))
        piel_inf = cv2.countNonZero(piel[int(h * 0.55) :, int(w * 0.18) : int(w * 0.82)])
        ratio_piel_pantorrillas = float(piel_inf / fg_inf)

        # Comparar ancho de la silueta en la parte alta/media (muslos/falda: 8% a 48%)
        # contra la parte baja (tobillos/pantorrillas: 65% a 90%)
        zona_alta = mascara_piernas[int(h * 0.08) : int(h * 0.48), :]
        zona_baja = mascara_piernas[int(h * 0.65) : int(h * 0.90), :]
        ancho_alto = float(np.mean(np.sum(zona_alta > 0, axis=1))) if zona_alta.size else 1.0
        ancho_bajo = float(np.mean(np.sum(zona_baja > 0, axis=1))) if zona_bajo_ok(zona_baja) else ancho_alto
        ratio_campana = ancho_alto / max(1.0, ancho_bajo)

        # Verificar si existe hueco central entre las dos piernas (típico de pantalón/jean)
        franja_entre_piernas = mascara_piernas[
            int(h * 0.25) : int(h * 0.75), int(w * 0.43) : int(w * 0.57)
        ]
        ocupacion_central = (
            float(np.mean(franja_entre_piernas > 0))
            if franja_entre_piernas.size
            else 1.0
        )

        # Una falda tiene silueta acampanada arriba y estrecha en tobillos, sin hueco entre muslos
        if ratio_campana >= 1.42 and ocupacion_central > 0.78 and ancho_alto > w * 0.48:
            return "falda"
    else:
        mitad_inf_piel = piel[int(h * 0.55) :, int(w * 0.18) : int(w * 0.82)]
        ratio_piel_pantorrillas = float(np.mean(mitad_inf_piel > 0))

    if ratio_piel_pantorrillas > 0.28:
        if color_inferior in {"blanco", "beige", "rosado", "gris"}:
            return "falda"
        return "pantaloneta"

    if color_inferior in {"azul", "azul oscuro"}:
        return "jean"

    return "pantalon"


def zona_bajo_ok(zona_baja: np.ndarray) -> bool:
    return zona_baja is not None and zona_baja.size > 0 and np.any(zona_baja > 0)


def estimar_demografia_persona(
    persona_bgr: np.ndarray,
    mascara_persona: np.ndarray | None = None,
    tipo_superior: str = "",
    tipo_inferior: str = "",
) -> dict:
    """
    Estima el sexo aparente ('Masculino' | 'Femenino') y el rango de edad
    ('Infantil (0-14)' | 'Joven (15-29)' | 'Adulto (30-54)' | 'Adulto Mayor (55+)')
    a partir de la silueta de cabeza/cuello/hombros, cobertura capilar y vestimenta.
    """
    if persona_bgr is None or persona_bgr.size == 0:
        raise ValueError("La imagen de la persona no puede estar vacía")

    h, w = persona_bgr.shape[:2]
    cabeza_cuello = persona_bgr[: max(4, int(h * 0.25)), :]
    hsv_cab = cv2.cvtColor(cabeza_cuello, cv2.COLOR_BGR2HSV)
    h_cab = cabeza_cuello.shape[0]

    # Analizar constricción del cuello vs ancho de cabeza en la máscara (cabello corto vs largo)
    ratio_cuello_cabeza = 0.75
    ratio_hombro_cadera = 1.05
    if mascara_persona is not None and mascara_persona.shape[:2] == (h, w):
        franja_cabeza = mascara_persona[int(h * 0.03) : int(h * 0.12), :]
        franja_cuello = mascara_persona[int(h * 0.13) : int(h * 0.20), :]
        ancho_cab = float(np.mean(np.sum(franja_cabeza > 0, axis=1))) if franja_cabeza.size else 10.0
        ancho_cue = float(np.mean(np.sum(franja_cuello > 0, axis=1))) if franja_cuello.size else 7.5
        if ancho_cab > 3.0:
            ratio_cuello_cabeza = ancho_cue / ancho_cab

        franja_hombros = mascara_persona[int(h * 0.22) : int(h * 0.35), :]
        franja_cadera = mascara_persona[int(h * 0.48) : int(h * 0.62), :]
        ancho_hombros = float(np.sum(np.any(franja_hombros > 0, axis=0)))
        ancho_cadera = float(np.sum(np.any(franja_cadera > 0, axis=0)))
        if ancho_cadera > 4:
            ratio_hombro_cadera = ancho_hombros / ancho_cadera

    puntaje_femenino = 0.0
    if tipo_inferior == "falda":
        puntaje_femenino += 2.6
    if tipo_superior == "blusa":
        puntaje_femenino += 2.2
    if ratio_cuello_cabeza > 1.02 and w < 96:
        puntaje_femenino += 1.4
    elif ratio_cuello_cabeza < 0.85 or w >= 98:
        puntaje_femenino -= 1.1

    if ratio_hombro_cadera < 0.94:
        puntaje_femenino += 1.2
    elif ratio_hombro_cadera > 1.08:
        puntaje_femenino -= 1.0

    sexo_aparente = "Femenino" if puntaje_femenino >= 1.2 else "Masculino"

    # Estimación de rango de edad
    coronilla = hsv_cab[: max(2, int(h_cab * 0.45)), int(w * 0.35) : int(w * 0.65)]
    if mascara_persona is not None and mascara_persona.shape[:2] == (h, w):
        mask_cor = mascara_persona[: max(2, int(h_cab * 0.45)), int(w * 0.35) : int(w * 0.65)]
        pix_cor = coronilla[mask_cor > 0]
    else:
        pix_cor = coronilla.reshape(-1, 3)

    if len(pix_cor) > 5:
        brillo_coronilla = float(np.median(pix_cor[:, 2]))
        sat_coronilla = float(np.median(pix_cor[:, 1]))
    else:
        brillo_coronilla = 90.0
        sat_coronilla = 50.0

    if h < 95 and (w / max(h, 1)) > 0.48:
        rango_edad = "Infantil (0-14)"
    elif brillo_coronilla > 170 and sat_coronilla < 38:
        rango_edad = "Adulto Mayor (55+)"
    elif tipo_superior in {"camisa", "chaqueta"}:
        rango_edad = "Adulto (30-54)"
    else:
        rango_edad = "Joven (15-29)"

    confianza = round(min(0.96, 0.80 + min(0.15, abs(puntaje_femenino - 0.6) * 0.05)), 2)
    return {
        "sexo_aparente": sexo_aparente,
        "rango_edad": rango_edad,
        "confianza": confianza,
    }


def analizar_recorte_persona(
    persona_bgr: np.ndarray,
    mascara_persona: np.ndarray | None = None,
    person_id: int = 1,
) -> dict:
    """
    Ejecuta el análisis completo de demografía y vestimenta sobre el recorte
    de una persona detectada.
    """
    if persona_bgr is None or persona_bgr.size == 0:
        raise ValueError("El recorte de la persona no puede estar vacío")

    h, w = persona_bgr.shape[:2]
    if h < 12 or w < 8:
        raise ValueError("Dimensiones del recorte insuficientes para análisis")

    # Partición anatómica:
    # Torso (prenda superior): 21% - 54%
    # Piernas completas (para forma falda/pantalón): 54% - 90%
    # Zona cromática prenda inferior (54% - 76%, evita zapatos y piso en la parte inferior):
    y_torso_ini = max(1, int(h * 0.21))
    y_torso_fin = max(y_torso_ini + 2, int(h * 0.54))
    y_piernas_ini = y_torso_fin
    y_piernas_color_fin = max(y_piernas_ini + 2, int(h * 0.76))
    y_piernas_fin = max(y_piernas_color_fin + 2, int(h * 0.90))

    x_pad = max(1, int(w * 0.14))
    x_pad_color = max(1, int(w * 0.20))

    torso_bgr = persona_bgr[y_torso_ini:y_torso_fin, x_pad : w - x_pad]
    torso_color_bgr = persona_bgr[y_torso_ini:y_torso_fin, x_pad_color : w - x_pad_color]
    piernas_bgr = persona_bgr[y_piernas_ini:y_piernas_fin, x_pad : w - x_pad]
    piernas_color_bgr = persona_bgr[
        y_piernas_ini:y_piernas_color_fin, x_pad_color : w - x_pad_color
    ]

    mascara_torso = None
    mascara_torso_color = None
    mascara_piernas = None
    mascara_piernas_color = None
    if mascara_persona is not None and mascara_persona.shape[:2] == (h, w):
        mascara_torso = mascara_persona[y_torso_ini:y_torso_fin, x_pad : w - x_pad]
        mascara_torso_color = mascara_persona[
            y_torso_ini:y_torso_fin, x_pad_color : w - x_pad_color
        ]
        mascara_piernas = mascara_persona[y_piernas_ini:y_piernas_fin, x_pad : w - x_pad]
        mascara_piernas_color = mascara_persona[
            y_piernas_ini:y_piernas_color_fin, x_pad_color : w - x_pad_color
        ]

    color_sup_info = clasificar_color_bgr(torso_color_bgr, mascara_torso_color)
    color_inf_info = clasificar_color_bgr(piernas_color_bgr, mascara_piernas_color)

    tipo_sup = clasificar_tipo_prenda_superior(
        torso_bgr, mascara_torso, color_sup_info["color"]
    )
    tipo_inf = clasificar_tipo_prenda_inferior(
        piernas_bgr, mascara_piernas, color_inf_info["color"]
    )

    demografia = estimar_demografia_persona(
        persona_bgr, mascara_persona, tipo_sup, tipo_inf
    )

    return {
        "person_id": int(person_id),
        "sexo_aparente": demografia["sexo_aparente"],
        "rango_edad": demografia["rango_edad"],
        "prenda_superior": {
            "tipo": tipo_sup,
            "color": color_sup_info["color"],
            "hex": color_sup_info["hex"],
        },
        "prenda_inferior": {
            "tipo": tipo_inf,
            "color": color_inf_info["color"],
            "hex": color_inf_info["hex"],
        },
        "confianza": demografia["confianza"],
    }


class DetectorDemografiaVestimenta:
    """
    Pipeline de visión por computador para detectar, rastrear (multi-object tracking)
    y extraer demografía y vestimenta de cada persona que pasa frente a la cámara.
    """

    def __init__(self) -> None:
        self.hog = None
        if hasattr(cv2, "HOGDescriptor"):
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.face_cascade = None
        if hasattr(cv2, "data") and hasattr(cv2, "CascadeClassifier"):
            cascade_path = (
                Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            )
            if cascade_path.exists():
                self.face_cascade = cv2.CascadeClassifier(str(cascade_path))
        self.bg_lab: np.ndarray | None = None
        self.tracks: dict[int, dict] = {}
        self.next_id: int = 1

    def reiniciar(self) -> None:
        self.bg_lab = None
        self.tracks = {}
        self.next_id = 1

    def establecer_fondo(self, frame_bgr: np.ndarray) -> None:
        blur = cv2.GaussianBlur(frame_bgr, (5, 5), 0)
        self.bg_lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB).astype(np.float32)

    def _segmentar_siluetas_fondo(
        self, frame_bgr: np.ndarray
    ) -> tuple[list[tuple[int, int, int, int]], np.ndarray]:
        """
        Extrae cajas delimitadoras (x, y, w, h) y máscara de primer plano
        suprimiendo sombras del suelo mediante diferencia cromática + luminancia en LAB.
        """
        h_img, w_img = frame_bgr.shape[:2]
        blur = cv2.GaussianBlur(frame_bgr, (5, 5), 0)
        lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB).astype(np.float32)

        if self.bg_lab is None or self.bg_lab.shape != lab.shape:
            self.bg_lab = lab.copy()
            return [], np.zeros((h_img, w_img), dtype=np.uint8)

        diff = np.abs(lab - self.bg_lab)
        diff_l = diff[:, :, 0]
        diff_ab = np.sqrt(diff[:, :, 1] ** 2 + diff[:, :, 2] ** 2)

        fg_mask = (
            (diff_l > 28.0) | (diff_ab > 15.0) | ((diff_l > 18.0) & (diff_ab > 9.5))
        ).astype(np.uint8) * 255

        if np.mean(fg_mask > 0) < 0.012:
            self.bg_lab = 0.92 * self.bg_lab + 0.08 * lab

        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 29))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close)

        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cajas: list[tuple[int, int, int, int]] = []
        area_min = max(1400, int(h_img * w_img * 0.008))

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < area_min:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if h < int(h_img * 0.16):
                continue

            if w > h * 0.78 and w > int(w_img * 0.22):
                sub_mask = fg_mask[y : y + h, x : x + w]
                perfil_x = np.sum(sub_mask > 0, axis=0).astype(np.float32)
                perfil_x = cv2.GaussianBlur(
                    perfil_x.reshape(1, -1), (1, 15), 0
                ).flatten()
                mitad_izq = int(w * 0.28)
                mitad_der = int(w * 0.72)
                if mitad_der > mitad_izq + 10:
                    valle_rel = (
                        int(np.argmin(perfil_x[mitad_izq:mitad_der])) + mitad_izq
                    )
                    pico_izq = float(np.max(perfil_x[:valle_rel]))
                    pico_der = float(np.max(perfil_x[valle_rel:]))
                    if perfil_x[valle_rel] < 0.65 * min(pico_izq, pico_der):
                        cajas.append((x, y, valle_rel, h))
                        cajas.append((x + valle_rel, y, w - valle_rel, h))
                        continue

            cajas.append((x, y, w, h))

        cajas.sort(key=lambda b: b[0])
        return cajas, fg_mask

    def _detectar_personas_estatico(
        self, frame_bgr: np.ndarray
    ) -> tuple[list[tuple[int, int, int, int]], np.ndarray]:
        """
        Detección para imágenes individuales sin fondo previo.
        """
        h_img, w_img = frame_bgr.shape[:2]
        cajas: list[tuple[int, int, int, int]] = []

        if self.face_cascade is not None:
            gris = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gris, scaleFactor=1.1, minNeighbors=5, minSize=(28, 28)
            )
            for fx, fy, fw, fh in faces:
                bx = max(0, int(fx - fw * 1.1))
                by = max(0, int(fy - fh * 0.35))
                bw = min(w_img - bx, int(fw * 3.2))
                bh = min(h_img - by, int(fh * 6.5))
                if bh > int(h_img * 0.22):
                    cajas.append((bx, by, bw, bh))

        if not cajas and self.hog is not None:
            rects, _ = self.hog.detectMultiScale(
                frame_bgr, winStride=(8, 8), padding=(16, 16), scale=1.05
            )
            for rx, ry, rw, rh in rects:
                if rh > int(h_img * 0.20):
                    cajas.append((int(rx), int(ry), int(rw), int(rh)))

        borde_pixels = np.concatenate(
            [
                frame_bgr[0, :, :],
                frame_bgr[-1, :, :],
                frame_bgr[:, 0, :],
                frame_bgr[:, -1, :],
            ],
            axis=0,
        )
        color_fondo = np.median(borde_pixels, axis=0).astype(np.float32)
        diff_borde = np.linalg.norm(
            frame_bgr.astype(np.float32) - color_fondo, axis=2
        )
        fg_mask = (diff_borde > 28.0).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 21))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        if not cajas:
            contours, _ = cv2.findContours(
                fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for cnt in contours:
                if cv2.contourArea(cnt) < max(900, int(h_img * w_img * 0.01)):
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                if h >= int(h_img * 0.20):
                    cajas.append((x, y, w, h))

        cajas.sort(key=lambda b: b[0])
        return cajas, fg_mask

    def procesar_frame(
        self, frame_bgr: np.ndarray, frame_idx: int = 0, dibujar: bool = False
    ) -> dict:
        """
        Detecta y rastrea todas las personas en un frame, estimando demografía y vestimenta.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("El frame no puede estar vacío")

        h_img, w_img = frame_bgr.shape[:2]
        if self.bg_lab is not None:
            cajas, fg_mask = self._segmentar_siluetas_fondo(frame_bgr)
        else:
            cajas, fg_mask = self._detectar_personas_estatico(frame_bgr)

        detecciones_frame: list[dict] = []
        usados_tracks: set[int] = set()

        for x, y, w, h in cajas:
            cx, cy = x + w // 2, y + h // 2
            mejor_id = None
            mejor_dist = 1e9
            for tid, trk in self.tracks.items():
                if tid in usados_tracks:
                    continue
                if frame_idx - trk["last_frame"] > 18:
                    continue
                dist = float(np.hypot(cx - trk["cx"], cy - trk["cy"]))
                if dist < max(115.0, max(w, h) * 0.85) and dist < mejor_dist:
                    mejor_dist = dist
                    mejor_id = tid

            if mejor_id is None:
                mejor_id = self.next_id
                self.next_id += 1
                self.tracks[mejor_id] = {
                    "person_id": mejor_id,
                    "first_frame": frame_idx,
                    "last_frame": frame_idx,
                    "frames_seen": 0,
                    "cx": cx,
                    "cy": cy,
                    "bbox": [x, y, w, h],
                    "votos_sexo": Counter(),
                    "votos_edad": Counter(),
                    "votos_sup_tipo": Counter(),
                    "votos_sup_color": Counter(),
                    "votos_inf_tipo": Counter(),
                    "votos_inf_color": Counter(),
                    "hex_sup": "#333333",
                    "hex_inf": "#333333",
                    "confianzas": [],
                }

            usados_tracks.add(mejor_id)
            trk = self.tracks[mejor_id]
            trk["last_frame"] = frame_idx
            trk["frames_seen"] += 1
            trk["cx"] = cx
            trk["cy"] = cy
            trk["bbox"] = [x, y, w, h]

            recorte = frame_bgr[y : y + h, x : x + w]
            sub_mask = fg_mask[y : y + h, x : x + w]
            analisis = analizar_recorte_persona(recorte, sub_mask, person_id=mejor_id)

            en_borde = (
                x <= 8
                or y <= 8
                or (x + w) >= (w_img - 8)
                or (y + h) >= (h_img - 8)
            )
            peso = 1 if en_borde else 4

            trk["votos_sexo"][analisis["sexo_aparente"]] += peso
            trk["votos_edad"][analisis["rango_edad"]] += peso
            trk["votos_sup_tipo"][analisis["prenda_superior"]["tipo"]] += peso
            trk["votos_sup_color"][analisis["prenda_superior"]["color"]] += peso
            trk["votos_inf_tipo"][analisis["prenda_inferior"]["tipo"]] += peso
            trk["votos_inf_color"][analisis["prenda_inferior"]["color"]] += peso
            if not en_borde or trk["frames_seen"] == 1:
                trk["hex_sup"] = analisis["prenda_superior"]["hex"]
                trk["hex_inf"] = analisis["prenda_inferior"]["hex"]
            trk["confianzas"].append(analisis["confianza"])

            consolidado = self._consolidar_track(trk)
            consolidado["bbox"] = [int(x), int(y), int(w), int(h)]
            detecciones_frame.append(consolidado)

        salida = {
            "frame_idx": frame_idx,
            "personas_en_frame": len(detecciones_frame),
            "personas": detecciones_frame,
        }
        if dibujar:
            salida["frame_anotado"] = self.anotar_frame(frame_bgr, detecciones_frame)
        return salida

    def _consolidar_track(self, trk: dict) -> dict:
        return {
            "person_id": trk["person_id"],
            "sexo_aparente": trk["votos_sexo"].most_common(1)[0][0],
            "rango_edad": trk["votos_edad"].most_common(1)[0][0],
            "prenda_superior": {
                "tipo": trk["votos_sup_tipo"].most_common(1)[0][0],
                "color": trk["votos_sup_color"].most_common(1)[0][0],
                "hex": trk["hex_sup"],
            },
            "prenda_inferior": {
                "tipo": trk["votos_inf_tipo"].most_common(1)[0][0],
                "color": trk["votos_inf_color"].most_common(1)[0][0],
                "hex": trk["hex_inf"],
            },
            "frames_observados": trk["frames_seen"],
            "primer_frame": trk["first_frame"],
            "ultimo_frame": trk["last_frame"],
            "confianza": round(float(np.mean(trk["confianzas"])), 2),
        }

    def anotar_frame(
        self, frame_bgr: np.ndarray, detecciones: list[dict]
    ) -> np.ndarray:
        """Dibuja cajas delimitadoras y etiquetas de demografía y vestimenta en el frame."""
        canvas = frame_bgr.copy()
        for det in detecciones:
            x, y, w, h = det["bbox"]
            color_box = (
                (245, 175, 40)
                if det["sexo_aparente"] == "Masculino"
                else (190, 105, 255)
            )
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color_box, 2)

            y_div = y + int(h * 0.54)
            cv2.line(canvas, (x, y_div), (x + w, y_div), (200, 200, 200), 1)

            etiqueta_1 = f"ID #{det['person_id']} | {det['sexo_aparente']} | {det['rango_edad']}"
            sup = det["prenda_superior"]
            inf = det["prenda_inferior"]
            etiqueta_2 = f"Sup: {sup['tipo']} ({sup['color']})"
            etiqueta_3 = f"Inf: {inf['tipo']} ({inf['color']})"

            y_txt = max(55, y - 8)
            cv2.rectangle(
                canvas,
                (x, y_txt - 52),
                (min(canvas.shape[1] - 2, x + 285), y_txt + 4),
                (18, 22, 32),
                -1,
            )
            cv2.putText(
                canvas,
                etiqueta_1,
                (x + 5, y_txt - 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                etiqueta_2,
                (x + 5, y_txt - 19),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (150, 240, 180),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                etiqueta_3,
                (x + 5, y_txt - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (160, 210, 255),
                1,
                cv2.LINE_AA,
            )
        return canvas

    def analizar_video(
        self,
        ruta_video: str | Path,
        cada_n_frames: int = 3,
        min_frames_persona: int = 3,
    ) -> dict:
        """
        Procesa un archivo de video completo y devuelve el reporte consolidado de todas
        las personas que pasaron frente a la cámara con su demografía y vestimenta.
        """
        ruta = Path(ruta_video)
        if not ruta.exists():
            raise FileNotFoundError(f"Video no encontrado: {ruta}")

        cap = cv2.VideoCapture(str(ruta))
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el archivo de video: {ruta}")

        self.reiniciar()
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 12.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        ret, primer_frame = cap.read()
        if not ret or primer_frame is None:
            cap.release()
            raise ValueError("El video no contiene frames válidos")

        self.establecer_fondo(primer_frame)

        frame_idx = 0
        frames_procesados = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frame_idx += 1
            if frame_idx % max(1, cada_n_frames) != 0:
                continue
            self.procesar_frame(frame, frame_idx=frame_idx, dibujar=False)
            frames_procesados += 1

        cap.release()

        personas_consolidadas = [
            self._consolidar_track(trk)
            for trk in self.tracks.values()
            if trk["frames_seen"] >= min_frames_persona
        ]
        personas_consolidadas.sort(key=lambda p: p["person_id"])

        conteo_sexo = Counter(p["sexo_aparente"] for p in personas_consolidadas)
        conteo_edad = Counter(p["rango_edad"] for p in personas_consolidadas)
        conteo_color_sup = Counter(
            p["prenda_superior"]["color"] for p in personas_consolidadas
        )
        conteo_tipo_sup = Counter(
            p["prenda_superior"]["tipo"] for p in personas_consolidadas
        )
        conteo_color_inf = Counter(
            p["prenda_inferior"]["color"] for p in personas_consolidadas
        )
        conteo_tipo_inf = Counter(
            p["prenda_inferior"]["tipo"] for p in personas_consolidadas
        )

        return {
            "reto": "1. Demografía y vestimenta",
            "video": ruta.name,
            "resolucion": f"{ancho}x{alto}",
            "fps": round(fps, 2),
            "total_frames_video": total_frames,
            "frames_analizados": frames_procesados,
            "total_personas_detectadas": len(personas_consolidadas),
            "personas": personas_consolidadas,
            "resumen_demografico": {
                "por_sexo_aparente": dict(conteo_sexo),
                "por_rango_edad": dict(conteo_edad),
                "prenda_superior_tipos": dict(conteo_tipo_sup),
                "prenda_superior_colores": dict(conteo_color_sup),
                "prenda_inferior_tipos": dict(conteo_tipo_inf),
                "prenda_inferior_colores": dict(conteo_color_inf),
            },
        }
