"""
Genera el video de prueba del Reto 6 (data/videos/reto6_caidas_riesgo.mp4): tres personas
dibujadas con esqueleto articulado (cabeza, tronco, brazos y piernas) sobre la sala vacía.

1. La primera entra caminando, se agacha a recoger algo, se levanta y sale.
2. La segunda se sienta en una silla, se levanta y sale.
3. La tercera entra caminando, se cae de golpe y se queda en el suelo.

Uso:
    python scripts/generar_video_caidas.py
"""

import math
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
VIDEO = RAIZ / "data" / "videos" / "reto6_caidas_riesgo.mp4"
FONDO = RAIZ / "data" / "videos" / "reto6_fondo.png"   # la sala vacía, sin personas ni silla
FPS = 15
SUELO = 368            # y de los pies sobre el piso
ALTURA = 190           # altura de pie en píxeles
PIEL = (150, 180, 222)
SILLA = (330, SUELO)   # base de la silla


def direccion(angulo_grados, sentido):
    """Vector unitario: 0° hacia abajo, positivo hacia donde mira la persona."""
    a = math.radians(angulo_grados)
    return np.array([math.sin(a) * sentido, math.cos(a)])


def esqueleto(pose, sentido):
    """Articulaciones con la cadera en el origen, a partir de los ángulos de la pose."""
    H = ALTURA
    k = {"cadera": np.zeros(2)}
    tronco = -direccion(pose["tronco"], -sentido)          # hacia arriba, inclinado hacia adelante
    k["cuello"] = k["cadera"] + tronco * 0.30 * H
    k["cabeza"] = k["cuello"] + tronco * 0.085 * H
    for lado in ("a", "b"):
        rodilla = k["cadera"] + direccion(pose["muslo_" + lado], sentido) * 0.245 * H
        tobillo = rodilla + direccion(pose["pierna_" + lado], sentido) * 0.245 * H
        k["rodilla_" + lado], k["tobillo_" + lado] = rodilla, tobillo
        k["pie_" + lado] = tobillo + np.array([0.07 * H * sentido, 0])
        codo = k["cuello"] + direccion(pose["brazo_" + lado], sentido) * 0.17 * H
        k["codo_" + lado] = codo
        k["mano_" + lado] = codo + direccion(pose["antebrazo_" + lado], sentido) * 0.16 * H
    return k


def pose_caminar(fase):
    s = math.sin(fase)
    return {
        "tronco": 4, "muslo_a": 24 * s, "pierna_a": 24 * s - 18 * max(0, -s),
        "muslo_b": -24 * s, "pierna_b": -24 * s - 18 * max(0, s),
        "brazo_a": -22 * s, "antebrazo_a": -22 * s + 20, "brazo_b": 22 * s, "antebrazo_b": 22 * s + 20,
    }


POSE_DE_PIE = pose_caminar(0)
POSE_AGACHADO = {
    "tronco": 48, "muslo_a": 100, "pierna_a": -18, "muslo_b": 92, "pierna_b": -12,
    "brazo_a": 40, "antebrazo_a": 25, "brazo_b": 30, "antebrazo_b": 15,
}
POSE_SENTADO = {
    "tronco": 6, "muslo_a": 90, "pierna_a": 2, "muslo_b": 86, "pierna_b": -4,
    "brazo_a": 25, "antebrazo_a": 80, "brazo_b": 18, "antebrazo_b": 75,
}


def mezclar(p1, p2, t):
    t = 0.5 - 0.5 * math.cos(math.pi * min(1, max(0, t)))
    return {k: p1[k] * (1 - t) + p2[k] * t for k in p1}


def colocar(k, x, rotacion=0.0, pivote=None, sentido=1):
    """Lleva el esqueleto a la escena: pies en el suelo, opcionalmente girado (caída)."""
    puntos = {n: v.copy() for n, v in k.items()}
    pies = max(v[1] for v in puntos.values())
    for v in puntos.values():
        v += np.array([x, SUELO - pies])
    if rotacion:
        cx, cy = pivote if pivote is not None else (x, SUELO)
        a = math.radians(rotacion) * sentido
        c, s = math.cos(a), math.sin(a)
        for v in puntos.values():
            dx, dy = v[0] - cx, v[1] - cy
            v[:] = (cx + dx * c - dy * s, cy + dx * s + dy * c)
        # El cuerpo no atraviesa el piso
        fondo = max(v[1] for v in puntos.values())
        for v in puntos.values():
            v[1] -= max(0, fondo - SUELO)
    return puntos


def dibujar_persona(img, p, ropa):
    camisa, pantalon, zapatos, pelo = ropa
    capa = np.zeros_like(img)
    mascara = np.zeros(img.shape[:2], np.uint8)
    pt = lambda v: tuple(int(round(c)) for c in v)
    H = ALTURA

    def segmento(a, b, color, grosor):
        cv2.line(capa, pt(a), pt(b), color, grosor, cv2.LINE_AA)
        cv2.line(mascara, pt(a), pt(b), 255, grosor, cv2.LINE_AA)

    oscurecer = lambda c: tuple(int(x * 0.78) for x in c)
    # Extremidades del lado lejano, más oscuras
    segmento(p["cadera"], p["rodilla_b"], oscurecer(pantalon), int(0.075 * H))
    segmento(p["rodilla_b"], p["tobillo_b"], oscurecer(pantalon), int(0.065 * H))
    segmento(p["tobillo_b"], p["pie_b"], zapatos, int(0.045 * H))
    segmento(p["cuello"], p["codo_b"], oscurecer(camisa), int(0.055 * H))
    segmento(p["codo_b"], p["mano_b"], oscurecer(PIEL), int(0.045 * H))
    # Tronco
    segmento(p["cadera"], p["cuello"], camisa, int(0.13 * H))
    segmento(p["cadera"], p["cadera"] + (p["cuello"] - p["cadera"]) * 0.15, pantalon, int(0.13 * H))
    # Lado cercano
    segmento(p["cadera"], p["rodilla_a"], pantalon, int(0.08 * H))
    segmento(p["rodilla_a"], p["tobillo_a"], pantalon, int(0.07 * H))
    segmento(p["tobillo_a"], p["pie_a"], zapatos, int(0.05 * H))
    segmento(p["cuello"], p["codo_a"], camisa, int(0.06 * H))
    segmento(p["codo_a"], p["mano_a"], PIEL, int(0.048 * H))
    # Cabeza con pelo
    r = int(0.068 * H)
    cv2.circle(capa, pt(p["cabeza"]), r, PIEL, -1, cv2.LINE_AA)
    cv2.circle(mascara, pt(p["cabeza"]), r, 255, -1, cv2.LINE_AA)
    cv2.ellipse(capa, pt(p["cabeza"] - (p["cabeza"] - p["cuello"]) * 0.25), (r, r), 0, 180, 360, pelo, -1, cv2.LINE_AA)

    # Sombra suave en el piso y bordes ligeramente difuminados
    xs = [v[0] for v in p.values()]
    sombra = np.zeros(img.shape[:2], np.float32)
    cv2.ellipse(sombra, (int(np.mean(xs)), SUELO + 2), (int((max(xs) - min(xs)) / 2 + 18), 7), 0, 0, 360, 1, -1)
    sombra = cv2.GaussianBlur(sombra, (21, 21), 0)[..., None] * 0.35
    img[:] = (img * (1 - sombra)).astype(np.uint8)
    alfa = cv2.GaussianBlur(mascara, (3, 3), 0).astype(np.float32)[..., None] / 255
    img[:] = (img * (1 - alfa) + capa * alfa).astype(np.uint8)


def dibujar_silla(img):
    x, y = SILLA
    madera, oscuro = (52, 74, 104), (34, 48, 70)
    asiento = int(ALTURA * 0.25)
    cv2.rectangle(img, (x - 30, y - asiento - 6), (x + 30, y - asiento + 4), madera, -1, cv2.LINE_AA)
    for dx in (-26, 24):
        cv2.line(img, (x + dx, y - asiento), (x + dx, y), oscuro, 5, cv2.LINE_AA)
    cv2.line(img, (x - 28, y - asiento), (x - 30, y - asiento - 70), oscuro, 6, cv2.LINE_AA)
    cv2.rectangle(img, (x - 34, y - asiento - 72), (x - 26, y - asiento - 40), madera, -1, cv2.LINE_AA)


def guion(f):
    """Lo que pasa en el frame f: lista de (esqueleto colocado, ropa)."""
    escena = []
    # Persona 1: camina, se agacha (recoge algo), se levanta y sale
    if 8 <= f < 96:
        sentido, ropa = 1, ((120, 72, 40), (70, 50, 40), (30, 30, 30), (30, 26, 24))
        if f < 40:
            x, pose = -40 + (f - 8) * 7.5, pose_caminar((f - 8) * 0.55)
        elif f < 70:
            x = 200
            t = (f - 40) / 6 if f < 46 else 1 if f < 62 else 1 - (f - 62) / 6
            pose = mezclar(POSE_DE_PIE, POSE_AGACHADO, t)
        else:
            x, pose = 200 - (f - 70) * 9, pose_caminar((f - 70) * 0.55)
            sentido = -1
        escena.append((colocar(esqueleto(pose, sentido), x), ropa))
    # Persona 2: llega a la silla, se sienta, se levanta y sale
    if 100 <= f < 205:
        ropa = ((52, 48, 160), (40, 36, 34), (24, 24, 24), (20, 30, 50))
        silla_x = SILLA[0] + 4
        if f < 128:
            x, pose, sentido = 800 - (f - 100) * 16.5, pose_caminar((f - 100) * 0.55), -1
            x = max(x, silla_x)
        elif f < 176:
            x, sentido = silla_x, 1
            t = (f - 128) / 7 if f < 135 else 1 if f < 168 else 1 - (f - 168) / 8
            pose = mezclar(POSE_DE_PIE, POSE_SENTADO, t)
        else:
            x, pose, sentido = silla_x + (f - 176) * 16, pose_caminar((f - 176) * 0.55), 1
        k = colocar(esqueleto(pose, sentido), x)
        if pose is not None and f >= 128 and f < 176:
            # Sentado, la cadera queda a la altura del asiento
            asiento = SUELO - ALTURA * 0.25 - 4
            dy = asiento - k["cadera"][1]
            if dy < 0:
                for v in k.values():
                    v[1] += dy * min(1, (f - 128) / 7 if f < 168 else (176 - f) / 8)
        escena.append((k, ropa))
    # Persona 3: entra caminando, cae de golpe y no se levanta
    if 212 <= f:
        ropa = ((225, 225, 225), (90, 90, 96), (36, 36, 36), (40, 36, 34))
        x_caida = 470
        if f < 236:
            x = 800 - (f - 212) * 13.75
            escena.append((colocar(esqueleto(pose_caminar((f - 212) * 0.55), -1), x), ropa))
        else:
            t = min(1, (f - 236) / 7)
            angulo = 88 * (t ** 1.8)
            pose = mezclar(POSE_DE_PIE, {**POSE_DE_PIE, "muslo_a": 12, "pierna_a": -10, "brazo_a": 70, "antebrazo_a": 95,
                                         "brazo_b": 50, "antebrazo_b": 80}, t)
            k = colocar(esqueleto(pose, -1), x_caida, rotacion=angulo, pivote=(x_caida, SUELO), sentido=1)
            escena.append((k, ropa))
    return escena


def main() -> None:
    if not shutil.which("ffmpeg"):
        sys.exit("Falta ffmpeg para codificar el video en H.264 (brew install ffmpeg)")
    if not FONDO.exists():
        # La primera vez se guarda la sala vacía del video original
        captura = cv2.VideoCapture(str(VIDEO))
        ok, frame = captura.read()
        captura.release()
        if not ok:
            sys.exit(f"No se pudo leer el fondo de {VIDEO}")
        cv2.imwrite(str(FONDO), frame)
    fondo = cv2.imread(str(FONDO))
    dibujar_silla(fondo)

    total = 300
    salida = VIDEO.with_suffix(".tmp.mp4")
    escritor = cv2.VideoWriter(str(salida), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (fondo.shape[1], fondo.shape[0]))
    rng = np.random.default_rng(6)
    for f in range(total):
        img = fondo.copy()
        for k, ropa in guion(f):
            dibujar_persona(img, k, ropa)
        ruido = rng.normal(0, 2.0, img.shape)
        escritor.write(np.clip(img + ruido, 0, 255).astype(np.uint8))
    escritor.release()
    # H.264 para que se reproduzca en cualquier navegador
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(salida), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "23", "-movflags", "+faststart", str(VIDEO)], check=True)
    salida.unlink()
    print(f"{VIDEO.relative_to(RAIZ)}: {total} frames a {FPS} fps")


if __name__ == "__main__":
    main()
