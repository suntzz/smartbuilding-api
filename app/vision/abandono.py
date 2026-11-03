"""
Reto 4 · Objeto abandonado.

YOLO11 detecta personas y bolsos (morral, bolso, maleta). Las detecciones de bolsos se
unen por posición en un mismo objeto aunque el modelo cambie de clase entre frames, y el
objeto se conserva unos segundos sin detección (los objetos quietos parpadean).

Cuando el objeto queda quieto, su propietario es la persona más cercana en ese momento.
El objeto está supervisado mientras haya alguien cerca: se mide sobre el piso, de la base
del objeto a los pies de cada persona, en alturas de esa persona. Si el
propietario se aleja y nadie lo acompaña durante varios segundos, se genera la alerta.
Si el propietario regresa o alguien lo recoge, el objeto deja de estar abandonado.
"""

from collections import Counter

import numpy as np

from .seguimiento import BOLSOS, PERSONA, centro, info_video, iou, pie, renumerar, seguir


def distancia_relativa(objeto_bbox, persona_bbox) -> float:
    """
    Distancia entre la base del objeto y los pies de la persona, en alturas de esa persona.
    Medir sobre el piso evita que alguien al fondo del pasillo parezca cercano.
    """
    ox, oy, ow, oh = objeto_bbox
    px, py = pie(persona_bbox)
    dx = max(ox - px, 0, px - (ox + ow))
    return float(np.hypot(dx, (oy + oh) - py)) / max(1, persona_bbox[3])


def sin_duplicados(dets):
    """El modelo a veces ve el mismo bolso como morral y maleta a la vez: se deja el más seguro."""
    finales = []
    for d in sorted(dets, key=lambda d: -d["conf"]):
        if all(iou(d["bbox"], f["bbox"]) < 0.3 for f in finales):
            finales.append(d)
    return finales


class DetectorAbandono:
    def __init__(
        self,
        segundos_quieto: float = 1.0,
        segundos_alerta: float = 3.0,
        segundos_memoria: float = 3.0,
        distancia_supervision_alturas: float = 0.8,
        movimiento_maximo_px: float = 12.0,
        cada_frames: int = 2,
    ):
        self.segundos_quieto = segundos_quieto
        self.segundos_alerta = segundos_alerta
        self.segundos_memoria = segundos_memoria
        self.distancia_supervision_alturas = distancia_supervision_alturas
        self.movimiento_maximo_px = movimiento_maximo_px
        self.cada_frames = cada_frames

    def analizar_video(self, ruta) -> dict:
        info = info_video(ruta)
        fps = info["fps"]
        ids_personas = {}
        objetos = []
        alertas, frames = [], []
        siguiente_objeto = 1

        for i, _, dets in seguir(ruta, [PERSONA, *BOLSOS], conf=0.25):
            personas = []
            for d in dets:
                if d["clase"] == PERSONA and d["id"] is not None:
                    personas.append({"person_id": renumerar(ids_personas, d["id"]), "bbox": d["bbox"]})

            # Asociar cada detección de bolso al objeto más cercano (sin importar la clase)
            for d in sin_duplicados([d for d in dets if d["clase"] in BOLSOS]):
                c = centro(d["bbox"])
                candidatos = [
                    (np.hypot(c[0] - o["centro"][0], c[1] - o["centro"][1]), o)
                    for o in objetos if o["activo"] and o["visto_en"] != i
                ]
                cercanos = [(dist, o) for dist, o in candidatos if dist < max(30.0, 0.8 * max(d["bbox"][2:]))]
                if cercanos:
                    o = min(cercanos, key=lambda t: t[0])[1]
                else:
                    o = {
                        "objeto_id": siguiente_objeto, "votos": Counter(), "historial": [], "activo": True,
                        "quieto_desde": None, "propietario": None,
                        "frames_sin_supervision": 0, "alertado": False, "frame_abandono": None,
                        "primer_frame": i,
                    }
                    siguiente_objeto += 1
                    objetos.append(o)
                o["votos"][BOLSOS[d["clase"]]] += d["conf"]
                o["bbox"], o["centro"], o["visto_en"], o["ultimo_frame"] = d["bbox"], c, i, i
                o["historial"].append((i, c))

            visibles = []
            for o in objetos:
                if not o["activo"]:
                    continue
                if (i - o["ultimo_frame"]) / fps > self.segundos_memoria:
                    o["activo"] = False
                    continue
                # ¿Está quieto? Se compara su posición en el último segundo
                recientes = [c for f, c in o["historial"] if i - f <= fps * self.segundos_quieto]
                quieto = (
                    (i - o["primer_frame"]) >= fps * self.segundos_quieto
                    and len(recientes) >= 2
                    and np.ptp(np.array(recientes), axis=0).max() <= self.movimiento_maximo_px
                )
                distancias = [(distancia_relativa(o["bbox"], p["bbox"]), p) for p in personas]
                if quieto and o["quieto_desde"] is None:
                    o["quieto_desde"] = i
                    if distancias:
                        dist, dueno = min(distancias, key=lambda t: t[0])
                        if dist < 1.5:
                            o["propietario"] = dueno["person_id"]
                if not quieto and o["quieto_desde"] is not None and not o["alertado"]:
                    o["quieto_desde"] = None  # lo levantaron o lo movieron

                umbral = self.distancia_supervision_alturas
                cercana = min((dist for dist, _ in distancias), default=float("inf"))
                dueno_visible = next((dist for dist, p in distancias if p["person_id"] == o["propietario"]), None)

                if o["quieto_desde"] is None:
                    estado = "en_movimiento"
                    o["frames_sin_supervision"] = 0
                elif cercana <= umbral:
                    estado = "con_propietario" if dueno_visible is not None and dueno_visible <= umbral else "supervisado"
                    if o["alertado"] and estado == "con_propietario":
                        o["alertado"] = False  # el dueño volvió por él
                    o["frames_sin_supervision"] = 0
                else:
                    o["frames_sin_supervision"] += 1
                    estado = "propietario_alejandose"
                    if o["frames_sin_supervision"] >= self.segundos_alerta * fps and o["propietario"] is not None:
                        estado = "abandonado"
                        if not o["alertado"]:
                            o["alertado"], o["frame_abandono"] = True, i
                            tipo = o["votos"].most_common(1)[0][0]
                            alertas.append({
                                "objeto_id": o["objeto_id"],
                                "tipo_objeto": tipo,
                                "frame_alerta": i,
                                "tiempo": round(i / fps, 2),
                                "ubicacion": [int(o["centro"][0]), int(o["centro"][1])],
                                "bbox": o["bbox"],
                                "propietario_id": o["propietario"],
                                "segundos_sin_supervision": round(o["frames_sin_supervision"] / fps, 1),
                                "mensaje": (
                                    f"{tipo.capitalize()} #{o['objeto_id']} abandonado: la persona #{o['propietario']} "
                                    f"lo dejó y se alejó hace {o['frames_sin_supervision'] / fps:.0f} s."
                                ),
                            })
                if o["alertado"]:
                    estado = "abandonado"

                visibles.append({
                    "objeto_id": o["objeto_id"],
                    "tipo": o["votos"].most_common(1)[0][0],
                    "bbox": o["bbox"],
                    "estado": estado,
                    "alerta": estado == "abandonado",
                    "propietario_id": o["propietario"],
                    "distancia_persona_alturas": None if cercana == float("inf") else round(cercana, 2),
                    "segundos_sin_supervision": round(o["frames_sin_supervision"] / fps, 1),
                })

            if i % self.cada_frames == 0:
                frames.append({
                    "frame": i,
                    "time": round(i / fps, 2),
                    "personas": personas,
                    "objetos": visibles,
                    "alerta_activa": any(o["alerta"] for o in visibles),
                })

        # Solo se reportan los objetos que estuvieron quietos (los que se llevan puestos no cuentan)
        resumen_objetos = [
            {
                "objeto_id": o["objeto_id"],
                "tipo": o["votos"].most_common(1)[0][0],
                "primer_frame": o["primer_frame"],
                "ultimo_frame": o["ultimo_frame"],
                "propietario_id": o["propietario"],
                "abandonado": o["frame_abandono"] is not None,
                "frame_abandono": o["frame_abandono"],
            }
            for o in objetos if o["quieto_desde"] is not None or o["frame_abandono"] is not None
        ]
        resumen = {
            "reto": "4. Objeto abandonado",
            "motor": "YOLO11m + ByteTrack · objeto quieto sin supervisión",
            "video": str(ruta).split("/")[-1],
            "resolucion": [info["ancho"], info["alto"]],
            "fps": fps,
            "total_frames_video": info["total_frames"],
            "total_personas": len(ids_personas),
            "total_alertas_abandono": len(alertas),
            "alertas": alertas,
            "objetos": resumen_objetos,
            "parametros": {
                "segundos_alerta": self.segundos_alerta,
                "distancia_supervision_alturas": self.distancia_supervision_alturas,
            },
        }
        return {"summary": resumen, "frames": frames, "fps": fps}
