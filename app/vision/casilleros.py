"""
Reto 7 · Correspondencia en casillero.

1. Se ubican los casilleros en el primer frame (contornos rectangulares) y se lee la
   etiqueta de cada uno con OCR ("C101 (Apto 101)"). Sin etiqueta legible se usa el
   orden de la cuadrícula y el directorio de residentes.
2. El interior de cada casillero se compara con su imagen vacía de referencia. Si cambia
   una parte suficiente durante varios frames seguidos, hay correspondencia.
3. Tipo: lo que aparece se clasifica por forma y color: alto y café/cartón → paquete;
   plano y claro → recibo o sobre.
4. Se avisa a los residentes del apartamento; si el casillero vuelve a quedar vacío se
   registra que la recogieron.
"""

import re

import cv2
import numpy as np

from app.residentes_csv import cargar_residentes

from .seguimiento import info_video, leer_frames


def detectar_casilleros(frame, area_minima_frac: float = 0.02):
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bordes = cv2.Canny(gris, 40, 120)
    contornos, _ = cv2.findContours(bordes, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    area_min = area_minima_frac * frame.shape[0] * frame.shape[1]
    rects = []
    for c in contornos:
        x, y, w, h = cv2.boundingRect(c)
        if w * h >= area_min and 0.6 < w / h < 1.7:
            rects.append((x, y, w, h))
    # Cada casillero tiene borde exterior e interior: se deja el interior (el más pequeño)
    finales = []
    for r in sorted(rects, key=lambda r: r[2] * r[3]):
        cx, cy = r[0] + r[2] / 2, r[1] + r[3] / 2
        if all(abs(cx - (f[0] + f[2] / 2)) > 15 or abs(cy - (f[1] + f[3] / 2)) > 15 for f in finales):
            finales.append(r)
    filas = sorted(finales, key=lambda r: (round(r[1] / 40), r[0]))
    return [list(map(int, r)) for r in filas]


def leer_etiquetas(frame, casilleros):
    try:
        from .placas import cargar_ocr

        resultado, _ = cargar_ocr()(frame)
    except Exception:
        return [None] * len(casilleros)
    etiquetas = []
    for x, y, w, h in casilleros:
        mejor = None
        for caja, texto, _ in resultado or []:
            tx, ty = np.array(caja, dtype=float).mean(axis=0)
            if x - 10 <= tx <= x + w + 10 and y - 40 <= ty <= y + 5:
                m = re.search(r"C\s*(\d{3}).*?(\d{3})", texto.upper())
                if m:
                    mejor = (f"C{m.group(1)}", int(m.group(2)))
        etiquetas.append(mejor)
    return etiquetas


def clasificar(recorte_bgr, mascara):
    ys, xs = np.nonzero(mascara)
    if len(xs) == 0:
        return None, None
    x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
    ancho, alto = x2 - x1 + 1, y2 - y1 + 1
    hsv = cv2.cvtColor(recorte_bgr, cv2.COLOR_BGR2HSV)[mascara > 0]
    tono, saturacion, brillo = np.median(hsv, axis=0)
    es_carton = 5 <= tono <= 30 and saturacion > 70
    if alto / ancho < 0.35 or (not es_carton and saturacion < 50 and brillo > 150):
        tipo = "recibo"
    else:
        tipo = "paquete"
    return tipo, [int(x1), int(y1), int(ancho), int(alto)]


class MonitorCasilleros:
    def __init__(self, fraccion_ocupado: float = 0.04, frames_confirmacion: int = 3, cada_frames: int = 2):
        self.fraccion_ocupado = fraccion_ocupado
        self.frames_confirmacion = frames_confirmacion
        self.cada_frames = cada_frames

    def analizar_video(self, ruta) -> dict:
        info = info_video(ruta)
        fps = info["fps"]
        residentes = cargar_residentes()
        por_casillero = {}
        for r in residentes:
            por_casillero.setdefault(r.get("casillero"), []).append(r)

        frames_video = leer_frames(ruta)
        _, primero = next(frames_video)
        cajas = detectar_casilleros(primero)
        etiquetas = leer_etiquetas(primero, cajas)
        conocidos = sorted(k for k in por_casillero if k)

        casilleros = []
        for k, (caja, etiqueta) in enumerate(zip(cajas, etiquetas)):
            codigo, apto = etiqueta if etiqueta else (conocidos[k] if k < len(conocidos) else f"C{k + 1}", None)
            dueños = por_casillero.get(codigo, [])
            if apto is None and dueños:
                apto = dueños[0]["apartment"]
            x, y, w, h = caja
            m = max(3, int(min(w, h) * 0.08))
            interior = (x + m, y + m, w - 2 * m, h - 2 * m)
            ix, iy, iw, ih = interior
            casilleros.append({
                "casillero": codigo, "apartamento": apto,
                "torre": dueños[0]["tower"] if dueños else None,
                "residentes": [d["full_name"] for d in dueños],
                "bbox": caja, "interior": interior,
                "referencia": cv2.GaussianBlur(primero[iy:iy + ih, ix:ix + iw], (5, 5), 0),
                "racha": 0, "ocupado": False, "tipo": None, "objeto": None, "desde": None,
            })

        eventos, frames = [], []

        def procesar(i, frame):
            estados = []
            for c in casilleros:
                ix, iy, iw, ih = c["interior"]
                recorte = frame[iy:iy + ih, ix:ix + iw]
                dif = cv2.absdiff(cv2.GaussianBlur(recorte, (5, 5), 0), c["referencia"]).max(axis=2)
                mascara = cv2.morphologyEx((dif > 40).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
                hay_algo = mascara.mean() >= self.fraccion_ocupado
                c["racha"] = c["racha"] + 1 if hay_algo != c["ocupado"] else 0
                if c["racha"] >= self.frames_confirmacion:
                    c["ocupado"], c["racha"] = hay_algo, 0
                    if hay_algo:
                        c["tipo"], caja = clasificar(recorte, mascara)
                        c["desde"] = i
                        eventos.append({
                            "tipo_evento": "nueva_correspondencia",
                            "casillero": c["casillero"], "apartamento": c["apartamento"], "torre": c["torre"],
                            "tipo_correspondencia": c["tipo"], "frame": i, "tiempo": round(i / fps, 2),
                            "residentes_notificados": c["residentes"],
                            "mensaje": f"Casillero {c['casillero']} (Apto {c['apartamento']}): llegó un {c['tipo']}.",
                        })
                    else:
                        eventos.append({
                            "tipo_evento": "retirada",
                            "casillero": c["casillero"], "apartamento": c["apartamento"], "torre": c["torre"],
                            "tipo_correspondencia": c["tipo"], "frame": i, "tiempo": round(i / fps, 2),
                            "mensaje": f"Casillero {c['casillero']} (Apto {c['apartamento']}): recogieron el {c['tipo']}.",
                        })
                        c["tipo"] = None
                if c["ocupado"]:
                    _, caja = clasificar(recorte, mascara)
                    c["objeto"] = None if caja is None else [caja[0] + ix, caja[1] + iy, caja[2], caja[3]]
                else:
                    c["objeto"] = None
                estados.append({
                    "casillero": c["casillero"], "apartamento": c["apartamento"], "torre": c["torre"],
                    "bbox": c["bbox"], "ocupado": c["ocupado"], "tipo_correspondencia": c["tipo"],
                    "bbox_objeto": c["objeto"], "residentes": c["residentes"],
                })
            if i % self.cada_frames == 0:
                frames.append({"frame": i, "time": round(i / fps, 2), "casilleros": estados})

        procesar(0, primero)
        for i, frame in frames_video:
            procesar(i, frame)

        pendientes = [
            {
                "casillero": c["casillero"], "apartamento": c["apartamento"], "torre": c["torre"],
                "tipo_correspondencia": c["tipo"], "desde_frame": c["desde"],
                "residentes_notificados": c["residentes"],
                "mensaje": f"Casillero {c['casillero']} (Apto {c['apartamento']}): {c['tipo']} pendiente de recoger.",
            }
            for c in casilleros if c["ocupado"]
        ]
        resumen = {
            "reto": "7. Correspondencia en casillero",
            "motor": "Cuadrícula por contornos + OCR de etiquetas · cambio frente al casillero vacío",
            "video": str(ruta).split("/")[-1],
            "resolucion": [info["ancho"], info["alto"]],
            "fps": fps,
            "total_frames_video": info["total_frames"],
            "total_casilleros": len(casilleros),
            "total_casilleros_ocupados": len(pendientes),
            "correspondencia_detectada": pendientes,
            "eventos": eventos,
            "estado_todos_los_casilleros": [
                {"casillero": c["casillero"], "apartamento": c["apartamento"], "ocupado": c["ocupado"],
                 "tipo_correspondencia": c["tipo"]}
                for c in casilleros
            ],
        }
        return {"summary": resumen, "frames": frames, "fps": fps}
