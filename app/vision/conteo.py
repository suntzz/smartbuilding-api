"""
Reto 2 · Conteo de entradas y salidas.

Cada persona se detecta y sigue con YOLO11 + ByteTrack. Se usa el punto de los pies
(centro inferior de la caja) y se cuenta un cruce cuando ese punto pasa de un lado
de la línea al otro con margen (histéresis), así el temblor de la caja no genera
cruces falsos. Se pueden configurar varias líneas; cada una cuenta entradas y salidas
por separado. La hora pico se marca cuando hay muchas personas a la vez en escena.
"""

from .seguimiento import PERSONA, info_video, pie, renumerar, seguir

# Líneas en coordenadas relativas (0–1). Entrar es cruzar hacia el lado izquierdo
# del segmento p1 → p2 (para una línea horizontal de izquierda a derecha: hacia arriba).
LINEAS = [
    {"id": "linea_acceso_principal", "nombre": "Acceso principal", "p1": [0.0, 0.80], "p2": [1.0, 0.80]},
    {"id": "linea_pasillo_interior", "nombre": "Pasillo interior", "p1": [0.04, 0.62], "p2": [0.96, 0.62]},
]


def lado(linea, punto, ancho, alto, margen):
    """+1 / -1 según el lado de la línea; 0 si el punto está dentro del margen."""
    x1, y1 = linea["p1"][0] * ancho, linea["p1"][1] * alto
    x2, y2 = linea["p2"][0] * ancho, linea["p2"][1] * alto
    dx, dy = x2 - x1, y2 - y1
    largo = (dx * dx + dy * dy) ** 0.5 or 1.0
    distancia = (dx * (punto[1] - y1) - dy * (punto[0] - x1)) / largo
    # Fuera del tramo de la línea no se cuenta
    t = ((punto[0] - x1) * dx + (punto[1] - y1) * dy) / (largo * largo)
    if t < -0.05 or t > 1.05 or abs(distancia) < margen:
        return 0
    return 1 if distancia > 0 else -1


class ContadorFlujo:
    def __init__(self, lineas=None, umbral_hora_pico: int = 3, margen_px: float = 6.0, cada_frames: int = 2):
        self.lineas = lineas or LINEAS
        self.umbral_hora_pico = umbral_hora_pico
        self.margen_px = margen_px
        self.cada_frames = cada_frames

    def analizar_video(self, ruta) -> dict:
        info = info_video(ruta)
        ancho, alto, fps = info["ancho"], info["alto"], info["fps"]
        ids = {}
        ultimo_lado = {}      # (persona, linea) -> último lado definido
        trayectorias = {}
        conteo = {l["id"]: {"entradas": 0, "salidas": 0} for l in self.lineas}
        eventos, frames = [], []
        max_simultaneas, frames_pico = 0, 0
        principal = self.lineas[0]["id"]

        for i, _, dets in seguir(ruta, [PERSONA]):
            personas = [d for d in dets if d["id"] is not None]
            simultaneas = len(personas)
            es_pico = simultaneas >= self.umbral_hora_pico
            max_simultaneas = max(max_simultaneas, simultaneas)
            frames_pico += es_pico

            visibles = []
            for d in personas:
                pid = renumerar(ids, d["id"])
                punto = pie(d["bbox"])
                tray = trayectorias.setdefault(pid, [])
                tray.append([int(punto[0]), int(punto[1])])
                del tray[:-24]

                for linea in self.lineas:
                    s = lado(linea, punto, ancho, alto, self.margen_px)
                    if s == 0:
                        continue
                    previo = ultimo_lado.get((pid, linea["id"]))
                    ultimo_lado[(pid, linea["id"])] = s
                    if previo is None or previo == s:
                        continue
                    sentido = "entrada" if s < 0 else "salida"
                    conteo[linea["id"]]["entradas" if sentido == "entrada" else "salidas"] += 1
                    eventos.append({
                        "person_id": pid,
                        "linea": linea["id"],
                        "nombre_linea": linea["nombre"],
                        "sentido": sentido,
                        "frame": i,
                        "tiempo": round(i / fps, 2),
                        "hora_pico": es_pico,
                    })
                visibles.append({"person_id": pid, "bbox": d["bbox"], "trayectoria": list(tray)})

            if i % self.cada_frames == 0:
                frames.append({
                    "frame": i,
                    "time": round(i / fps, 2),
                    "personas": visibles,
                    "total_entradas": conteo[principal]["entradas"],
                    "total_salidas": conteo[principal]["salidas"],
                    "conteo_por_linea": {k: dict(v) for k, v in conteo.items()},
                    "es_hora_pico": es_pico,
                    "personas_simultaneas": simultaneas,
                })

        entradas, salidas = conteo[principal]["entradas"], conteo[principal]["salidas"]
        resumen = {
            "reto": "2. Conteo de entradas y salidas",
            "motor": "YOLO11m + ByteTrack · cruce de línea con histéresis",
            "video": str(ruta).split("/")[-1],
            "resolucion": [ancho, alto],
            "fps": fps,
            "total_frames_video": info["total_frames"],
            "lineas": self.lineas,
            "total_personas_unicas": len(ids),
            "total_entradas": entradas,
            "total_salidas": salidas,
            "ocupacion_actual": max(0, entradas - salidas),
            "conteo_por_linea": conteo,
            "max_personas_simultaneas": max_simultaneas,
            "umbral_hora_pico": self.umbral_hora_pico,
            "hubo_hora_pico": frames_pico > 0,
            "frames_en_hora_pico": frames_pico,
            "segundos_en_hora_pico": round(frames_pico / fps, 1),
            "eventos_cruce": eventos,
        }
        return {"summary": resumen, "frames": frames, "fps": fps}
