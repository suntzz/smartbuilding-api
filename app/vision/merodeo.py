"""
Reto 3 · Merodeo.

Cada persona se sigue con YOLO11 + ByteTrack y se mide su trayectoria (punto de los pies):
- tiempo en la zona,
- distancia recorrida medida en "alturas de cuerpo" (no depende de la distancia a la cámara),
- eficiencia = desplazamiento neto / distancia recorrida (≈1 al pasar, baja al ir y volver),
- velocidad reciente y si hay otra persona cerca.

Estados:
- paso_normal: camina con una dirección clara.
- esperando: casi quieta y sola. No alerta.
- conversando: casi quieta junto a otra persona. No alerta.
- merodeo: lleva tiempo en la zona y está sola, y además se mueve mucho sin avanzar
  (va y vuelve) o ya pasó varias veces por el mismo lugar.
  Si se mantiene ~0.6 s se genera la alerta (una por persona).

Una reidentificación ligera por apariencia (histograma de color) une a quien sale y
vuelve a entrar poco después, para que su tiempo y recorrido se acumulen.
"""

import itertools

import cv2
import numpy as np

from .seguimiento import PERSONA, info_video, pie, renumerar, seguir


def histograma(imagen, bbox):
    if imagen is None:
        return None
    x, y, w, h = bbox
    recorte = imagen[max(0, y + int(h * 0.15)): y + int(h * 0.6), max(0, x): x + w]
    if recorte.size == 0:
        return None
    hsv = cv2.cvtColor(recorte, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    return cv2.normalize(hist, hist).flatten()


class DetectorMerodeo:
    def __init__(
        self,
        tiempo_minimo_s: float = 3.0,
        recorrido_minimo_alturas: float = 2.0,
        eficiencia_maxima: float = 0.6,
        velocidad_quieto: float = 0.25,
        persistencia_alerta_s: float = 0.6,
        memoria_reid_s: float = 20.0,
        umbral_reid: float = 0.2,
        frames_firma: int = 8,
        pasadas_sospechosas: int = 3,
        cada_frames: int = 2,
    ):
        self.tiempo_minimo_s = tiempo_minimo_s
        self.recorrido_minimo_alturas = recorrido_minimo_alturas
        self.eficiencia_maxima = eficiencia_maxima
        self.velocidad_quieto = velocidad_quieto
        self.persistencia_alerta_s = persistencia_alerta_s
        self.memoria_reid_s = memoria_reid_s
        self.umbral_reid = umbral_reid
        self.frames_firma = frames_firma
        self.pasadas_sospechosas = pasadas_sospechosas
        self.cada_frames = cada_frames

    def _nuevo_visitante(self, frame, visitantes):
        v = {
            "id": next(self._ids), "hists": [], "firma": None, "primer_frame": frame,
            "ultimo_frame": frame, "puntos": [], "alturas": [], "regresos": 0, "estado": "paso_normal",
            "frames_merodeo": 0, "alertado": False, "estados": {},
        }
        visitantes[v["id"]] = v
        return v

    def _reidentificar(self, v, frame, visitantes, activos, fps):
        """
        Con la apariencia media de los primeros frames decide si el visitante es alguien
        que salió hace poco. Si lo es, se une a esa persona y devuelve la persona original.
        """
        firma = np.mean(v["hists"], axis=0).astype(np.float32)
        mejor, distancia = None, self.umbral_reid
        for o in visitantes.values():
            if o is v or o["firma"] is None or o["id"] in activos:
                continue
            if (frame - o["ultimo_frame"]) / fps > self.memoria_reid_s or o["ultimo_frame"] >= v["primer_frame"]:
                continue
            d = cv2.compareHist(firma, o["firma"], cv2.HISTCMP_BHATTACHARYYA)
            if d < distancia:
                mejor, distancia = o, d
        if mejor is None:
            v["firma"] = firma
            return v
        mejor["puntos"] += v["puntos"]
        mejor["alturas"] += v["alturas"]
        mejor["ultimo_frame"] = v["ultimo_frame"]
        mejor["regresos"] += 1
        for k, n in v["estados"].items():
            mejor["estados"][k] = mejor["estados"].get(k, 0) + n
        del visitantes[v["id"]]
        return mejor

    def _medidas(self, v, fps):
        pts = np.array(v["puntos"], dtype=float)
        altura = float(np.median(v["alturas"])) or 1.0
        if len(pts) >= 5:
            nucleo = np.ones(5) / 5
            suav = np.stack([np.convolve(pts[:, k], nucleo, mode="valid") for k in (0, 1)], axis=1)
        else:
            suav = pts
        muestras = suav[::3] if len(suav) > 3 else suav
        recorrido = float(np.sum(np.linalg.norm(np.diff(muestras, axis=0), axis=1))) if len(muestras) > 1 else 0.0
        desplazamiento = float(np.linalg.norm(suav[-1] - suav[0])) if len(suav) > 1 else 0.0
        ventana = suav[-int(fps):] if len(suav) > int(fps) else suav
        vector = (ventana[-1] - ventana[0]) / altura * fps / max(1, len(ventana) - 1)
        velocidad = float(np.linalg.norm(vector))
        return {
            "vector": vector,
            "altura": altura,
            "recorrido_alturas": recorrido / altura,
            "recorrido_px": recorrido,
            "eficiencia": desplazamiento / recorrido if recorrido > 0 else 1.0,
            "velocidad": velocidad,
        }

    def analizar_video(self, ruta) -> dict:
        info = info_video(ruta)
        fps = info["fps"]
        tracks = {}        # id del tracker -> visitante
        visitantes = {}
        alias = {}         # visitante provisional -> persona con la que se unió
        self._ids = itertools.count(1)
        alertas, frames = [], []
        conteo_estados = {"paso_normal": 0, "esperando": 0, "conversando": 0, "merodeo": 0}

        for i, imagen, dets in seguir(ruta, [PERSONA]):
            personas = [d for d in dets if d["id"] is not None]
            for d in personas:
                if d["id"] not in tracks:
                    tracks[d["id"]] = self._nuevo_visitante(i, visitantes)
                v = tracks[d["id"]]
                v["ultimo_frame"] = i
                v["puntos"].append(pie(d["bbox"]))
                v["alturas"].append(d["bbox"][3])
                if v["firma"] is None:
                    h = histograma(imagen, d["bbox"])
                    if h is not None:
                        v["hists"].append(h)
            activos = {tracks[d["id"]]["id"] for d in personas}
            for d in personas:
                v = tracks[d["id"]]
                if v["firma"] is None and len(v["hists"]) >= self.frames_firma:
                    original = self._reidentificar(v, i, visitantes, activos - {v["id"]}, fps)
                    if original is not v:
                        alias[v["id"]] = original["id"]
                        tracks[d["id"]] = original

            visibles = []
            medidas = {d["id"]: self._medidas(tracks[d["id"]], fps) for d in personas}
            for d in personas:
                v = tracks[d["id"]]
                m = medidas[d["id"]]
                tiempo = (i - v["primer_frame"]) / fps
                fx, fy = pie(d["bbox"])
                # Acompañado: cerca de alguien que se mueve igual (conversan o caminan juntos);
                # cruzarse con quien va en otra dirección no cuenta
                acompanado = any(
                    o is not d
                    and np.hypot(*(np.subtract(pie(o["bbox"]), (fx, fy)))) < 1.1 * m["altura"]
                    and np.linalg.norm(m["vector"] - medidas[o["id"]]["vector"]) < 0.35
                    for o in personas
                )
                if m["velocidad"] < self.velocidad_quieto and len(v["puntos"]) > fps:
                    estado = "conversando" if acompanado else "esperando"
                elif tiempo >= self.tiempo_minimo_s and not acompanado and (
                    (m["recorrido_alturas"] >= self.recorrido_minimo_alturas and m["eficiencia"] <= self.eficiencia_maxima)
                    or v["regresos"] + 1 >= self.pasadas_sospechosas
                ):
                    estado = "merodeo"
                else:
                    estado = "paso_normal"

                v["frames_merodeo"] = v["frames_merodeo"] + 1 if estado == "merodeo" else 0
                # Varias pasadas ya son evidencia suficiente; la trayectoria necesita sostenerse
                alerta = v["frames_merodeo"] >= self.persistencia_alerta_s * fps or (
                    estado == "merodeo" and v["regresos"] + 1 >= self.pasadas_sospechosas
                )
                if alerta and not v["alertado"]:
                    v["alertado"] = True
                    alertas.append({
                        "_visitante": v["id"],
                        "person_id": v["id"],
                        "frame_alerta": i,
                        "tiempo": round(i / fps, 2),
                        "comportamiento": "merodeo",
                        "tiempo_en_zona_s": round(tiempo, 1),
                        "distancia_recorrida": round(m["recorrido_px"], 1),
                        "recorrido_alturas": round(m["recorrido_alturas"], 1),
                        "eficiencia_trayectoria": round(m["eficiencia"], 2),
                        "regresos": v["regresos"],
                        "bbox": d["bbox"],
                    })
                v["estado"] = estado
                v["estados"][estado] = v["estados"].get(estado, 0) + 1
                conteo_estados[estado] += 1
                visibles.append({
                    "person_id": v["id"],
                    "bbox": d["bbox"],
                    "comportamiento": estado,
                    "alerta": v["alertado"] and estado == "merodeo",
                    "tiempo_en_zona_s": round(tiempo, 1),
                    "eficiencia": round(m["eficiencia"], 2),
                    "puntos": [[int(x), int(y)] for x, y in v["puntos"][-40::2]],
                })

            if i % self.cada_frames == 0:
                frames.append({
                    "frame": i,
                    "time": round(i / fps, 2),
                    "personas": visibles,
                    "alerta_activa": any(p["alerta"] for p in visibles),
                    "total_alertas": len(alertas),
                })

        # Numeración final consecutiva, resolviendo a quién se unió cada visitante provisional
        def raiz(pid):
            while pid in alias:
                pid = alias[pid]
            return pid

        orden = {pid: n for n, pid in enumerate(sorted(visitantes), start=1)}
        numero = lambda pid: orden[raiz(pid)]
        for f in frames:
            for pr in f["personas"]:
                pr["person_id"] = numero(pr["person_id"])
        for al in alertas:
            al["person_id"] = numero(al["person_id"])
            regresos = visitantes[raiz(al.pop("_visitante"))]["regresos"]
            al["regresos"] = regresos
            al["mensaje"] = (
                f"Persona #{al['person_id']} deambula en la zona: {al['tiempo_en_zona_s']:.0f} s y recorrió "
                f"{al['recorrido_alturas']:.1f} veces su altura sin avanzar"
                + (f"; ha pasado {regresos + 1} veces" if regresos else "")
                + "."
            )

        personas_resumen = []
        for v in visitantes.values():
            dominante = max(v["estados"], key=v["estados"].get) if v["estados"] else "paso_normal"
            personas_resumen.append({
                "person_id": orden[v["id"]],
                "clasificacion": "merodeo" if v["alertado"] else dominante,
                "alerta": v["alertado"],
                "primer_frame": v["primer_frame"],
                "ultimo_frame": v["ultimo_frame"],
                "tiempo_en_zona_s": round((v["ultimo_frame"] - v["primer_frame"]) / fps, 1),
                "regresos": v["regresos"],
            })

        resumen = {
            "reto": "3. Merodeo",
            "motor": "YOLO11m + ByteTrack · análisis de trayectoria + reidentificación por apariencia",
            "video": str(ruta).split("/")[-1],
            "resolucion": [info["ancho"], info["alto"]],
            "fps": fps,
            "total_frames_video": info["total_frames"],
            "total_personas": len(visitantes),
            "total_alertas_merodeo": len(alertas),
            "alertas": alertas,
            "personas": personas_resumen,
            "conteo_estados_observados": conteo_estados,
            "parametros": {
                "tiempo_minimo_s": self.tiempo_minimo_s,
                "recorrido_minimo_alturas": self.recorrido_minimo_alturas,
                "eficiencia_maxima": self.eficiencia_maxima,
            },
        }
        return {"summary": resumen, "frames": frames, "fps": fps}
