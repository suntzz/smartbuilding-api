"""
Reto 1 con modelos preentrenados: demografía y vestimenta.

Reemplaza las reglas manuales del detector clásico por tres modelos:
- YOLO11n + ByteTrack: detecta personas y mantiene un ID por persona.
- SegFormer B2 (mattmdjaga/segformer_b2_clothes): segmenta prendas, piel y cabello
  para medir el color real de cada prenda y distinguir pantalón, falda o vestido.
- CLIP ViT-B/32 (OpenCLIP, LAION-2B): clasificación zero-shot de sexo aparente,
  rango de edad y tipo de prenda.

Las predicciones de cada frame se acumulan por persona (suma de log-probabilidades
ponderada por el tamaño del recorte), así un frame borroso no cambia el resultado.

Los pesos se descargan con `python scripts/descargar_modelos.py` en la carpeta `models/`.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from app.vision.demografia_vestimenta import (
    COLORES_VALIDOS,
    RANGOS_EDAD,
    SEXOS_APARENTES,
    TIPOS_PRENDA_INFERIOR,
    TIPOS_PRENDA_SUPERIOR,
)

DIR_MODELOS = Path(__file__).resolve().parents[2] / "models"

# Etiquetas del modelo SegFormer de ropa (dataset ATR)
SEG_SUPERIOR = (4, 17)  # Upper-clothes, Scarf
SEG_FALDA = 5
SEG_PANTALON = 6
SEG_VESTIDO = 7
SEG_PIERNAS = (12, 13)
SEG_BRAZOS = (14, 15)

# Paleta de referencia (RGB). Cada color tiene varios prototipos para cubrir
# tonos claros, oscuros y en sombra.
PALETA = {
    "negro": [(28, 28, 30), (45, 42, 48)],
    "blanco": [(236, 236, 232), (210, 212, 215)],
    "gris": [(128, 128, 128), (165, 165, 168), (95, 98, 100)],
    "rojo": [(190, 35, 40), (140, 30, 35)],
    "azul": [(55, 95, 185), (140, 170, 215), (95, 130, 175)],
    "azul oscuro": [(30, 40, 80), (45, 55, 95)],
    "verde": [(50, 125, 65), (95, 115, 80), (60, 75, 50)],
    "amarillo": [(225, 195, 50)],
    "naranja": [(230, 125, 40)],
    "morado": [(110, 55, 140)],
    "rosado": [(230, 150, 175)],
    "cafe": [(105, 70, 45), (140, 100, 70)],
    "beige": [(205, 185, 150), (180, 160, 120)],
}

# Descripciones en inglés para CLIP (se promedian varias por clase)
PROMPTS = {
    "sexo": {
        "Masculino": ["a photo of a man", "a man walking", "a photo of a male person"],
        "Femenino": ["a photo of a woman", "a woman walking", "a photo of a female person"],
    },
    "edad": {
        "Infantil (0-14)": ["a photo of a child", "a photo of a kid"],
        "Joven (15-29)": ["a photo of a young adult in their twenties", "a photo of a teenager"],
        "Adulto (30-54)": ["a photo of a middle-aged adult", "a photo of a person in their forties"],
        "Adulto Mayor (55+)": ["a photo of an elderly person", "a photo of a senior with gray hair"],
    },
    "superior": {
        "camiseta": ["a person wearing a t-shirt", "a person wearing a short-sleeved t-shirt"],
        "camisa": ["a person wearing a button-up collared shirt", "a person wearing a dress shirt"],
        "chaqueta": ["a person wearing a jacket", "a person wearing a coat or blazer"],
        "buso": ["a person wearing a hoodie", "a person wearing a sweater"],
        "blusa": ["a woman wearing a blouse", "a person wearing a blouse"],
    },
    "inferior": {
        "pantalon": ["a person wearing trousers", "a person wearing dress pants"],
        "jean": ["a person wearing blue jeans", "a person wearing denim jeans"],
        "pantaloneta": ["a person wearing shorts", "a person wearing short pants"],
        "falda": ["a person wearing a skirt", "a woman wearing a skirt"],
    },
}


def _rgb_a_lab(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.uint8).reshape(-1, 1, 3)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)


_NOMBRES_PALETA = [n for n, protos in PALETA.items() for _ in protos]
_LAB_PALETA = _rgb_a_lab(np.array([p for protos in PALETA.values() for p in protos]))


def nombrar_color(rgb) -> str:
    """Devuelve el nombre en español del color de la paleta más cercano en espacio Lab."""
    lab = _rgb_a_lab(np.array([rgb]))[0]
    dist = np.linalg.norm(_LAB_PALETA - lab, axis=1)
    return _NOMBRES_PALETA[int(np.argmin(dist))]


def balance_blancos(rgb: np.ndarray) -> np.ndarray:
    """Corrige el tinte y la poca luz de la cámara llevando los tonos más claros a blanco."""
    # Paredes y pisos claros (percentil 90) se toman como referencia de blanco;
    # el percentil 98 suele caer en lámparas o ventanas saturadas.
    ref = np.percentile(rgb.reshape(-1, 3), 90, axis=0).astype(np.float32)
    ganancia = np.clip(225.0 / np.maximum(ref, 1.0), 0.8, 1.8)
    return np.clip(rgb.astype(np.float32) * ganancia, 0, 255).astype(np.uint8)


def color_dominante(pixeles_rgb: np.ndarray, k: int = 3) -> tuple[np.ndarray, float] | None:
    """Agrupa los píxeles con k-means en Lab y devuelve el centro del grupo más grande y su peso."""
    if pixeles_rgb is None or len(pixeles_rgb) < 30:
        return None
    if len(pixeles_rgb) > 3000:
        idx = np.random.default_rng(0).choice(len(pixeles_rgb), 3000, replace=False)
        pixeles_rgb = pixeles_rgb[idx]
    lab = _rgb_a_lab(pixeles_rgb)
    k = min(k, len(lab))
    criterios = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, etiquetas, _ = cv2.kmeans(lab, k, None, criterios, 3, cv2.KMEANS_PP_CENTERS)
    conteo = np.bincount(etiquetas.ravel(), minlength=k)
    mayor = int(np.argmax(conteo))
    # Percentil 70: representa la cara iluminada de la prenda y no la sombra
    rgb_centro = np.percentile(pixeles_rgb[etiquetas.ravel() == mayor], 70, axis=0)
    return rgb_centro, float(conteo[mayor] / len(lab))


class AnalizadorDemografiaIA:
    """Pipeline de demografía y vestimenta con YOLO11 + SegFormer + CLIP."""

    def __init__(self, dir_modelos: str | Path = DIR_MODELOS, frames_entre_analisis: int = 2):
        import onnxruntime as ort
        import open_clip
        import torch
        from ultralytics import YOLO

        dir_modelos = Path(dir_modelos)
        ruta_yolo = dir_modelos / "yolo11n.pt"
        ruta_seg = dir_modelos / "segformer_clothes" / "model.onnx"
        for ruta in (ruta_yolo, ruta_seg):
            if not ruta.exists():
                raise FileNotFoundError(
                    f"Falta el modelo {ruta}. Ejecuta: python scripts/descargar_modelos.py"
                )

        self.torch = torch
        self.frames_entre_analisis = max(1, frames_entre_analisis)
        self.yolo = YOLO(str(ruta_yolo))
        self.seg = ort.InferenceSession(str(ruta_seg), providers=["CPUExecutionProvider"])
        self.seg_entrada = self.seg.get_inputs()[0].name

        self.clip, _, self.clip_prep = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k", cache_dir=str(dir_modelos / "clip")
        )
        self.clip.eval()
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self.texto = {}
        with torch.no_grad():
            for pregunta, clases in PROMPTS.items():
                embs = []
                for frases in clases.values():
                    e = self.clip.encode_text(tokenizer(frases))
                    e = e / e.norm(dim=-1, keepdim=True)
                    e = e.mean(dim=0)
                    embs.append(e / e.norm())
                self.texto[pregunta] = (list(clases), torch.stack(embs))

    # ------------------------------------------------------------------ modelos

    def segmentar(self, recorte_rgb: np.ndarray) -> np.ndarray:
        """Devuelve el mapa de etiquetas de prendas con el tamaño del recorte."""
        h, w = recorte_rgb.shape[:2]
        x = cv2.resize(recorte_rgb, (512, 512), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        x = x.transpose(2, 0, 1)[None]
        logits = self.seg.run(None, {self.seg_entrada: x})[0][0]  # (18, 128, 128)
        logits = np.stack([cv2.resize(c, (w, h), interpolation=cv2.INTER_LINEAR) for c in logits])
        return np.argmax(logits, axis=0).astype(np.uint8)

    def clip_probabilidades(self, imagenes_rgb: list[np.ndarray], pregunta: str) -> np.ndarray:
        """Probabilidades zero-shot de CLIP para un lote de imágenes y una pregunta."""
        from PIL import Image

        clases, texto = self.texto[pregunta]
        lote = self.torch.stack([self.clip_prep(Image.fromarray(im)) for im in imagenes_rgb])
        with self.torch.no_grad():
            img = self.clip.encode_image(lote)
            img = img / img.norm(dim=-1, keepdim=True)
            logits = 100.0 * img @ texto.T
        return logits.softmax(dim=-1).numpy()

    # ----------------------------------------------------------------- análisis

    def analizar_recorte(self, recorte_rgb: np.ndarray) -> dict:
        """Analiza una persona recortada y devuelve evidencias acumulables."""
        etiquetas = self.segmentar(recorte_rgb)
        h, w = etiquetas.shape
        area = float(h * w)

        def frac(ids) -> float:
            ids = ids if isinstance(ids, tuple) else (ids,)
            return float(np.isin(etiquetas, ids).sum()) / area

        sup = np.isin(etiquetas, SEG_SUPERIOR)
        vestido = etiquetas == SEG_VESTIDO
        inf_ids = (SEG_PANTALON, SEG_FALDA, SEG_VESTIDO)
        inf = np.isin(etiquetas, inf_ids)
        # Si la persona lleva vestido, el color superior sale de la mitad alta del vestido
        if sup.sum() < 40 and vestido.sum() > 40:
            filas = np.where(vestido.any(axis=1))[0]
            corte = filas.min() + (filas.max() - filas.min()) // 2
            sup = vestido.copy()
            sup[corte:] = False

        recorte_sup = self._recortar_mascara(recorte_rgb, sup)
        recorte_inf = self._recortar_mascara(recorte_rgb, inf)

        p_sexo = self.clip_probabilidades([recorte_rgb], "sexo")[0]
        p_edad = self.clip_probabilidades([recorte_rgb], "edad")[0]
        p_sup = self.clip_probabilidades([recorte_sup if recorte_sup is not None else recorte_rgb], "superior")[0]
        p_inf = self.clip_probabilidades([recorte_inf if recorte_inf is not None else recorte_rgb], "inferior")[0]

        return {
            "peso": max(1.0, area / 4000.0),
            "p_sexo": p_sexo,
            "p_edad": p_edad,
            "p_sup": p_sup,
            "p_inf": p_inf,
            "frac_falda": frac(SEG_FALDA),
            "frac_pantalon": frac(SEG_PANTALON),
            "frac_vestido": frac(SEG_VESTIDO),
            "frac_piernas": frac(SEG_PIERNAS),
            "frac_brazos": frac(SEG_BRAZOS),
            "color_sup": color_dominante(recorte_rgb[sup]),
            "color_inf": color_dominante(recorte_rgb[inf]),
        }

    @staticmethod
    def _recortar_mascara(img: np.ndarray, mascara: np.ndarray) -> np.ndarray | None:
        if mascara.sum() < 60:
            return None
        ys, xs = np.where(mascara)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        if (y1 - y0) < 12 or (x1 - x0) < 8:
            return None
        return img[y0:y1, x0:x1]

    # ------------------------------------------------------------ consolidación

    @staticmethod
    def _mejor(clases: list[str], log_prob: np.ndarray) -> tuple[str, float]:
        p = np.exp(log_prob - log_prob.max())
        p /= p.sum()
        i = int(np.argmax(p))
        return clases[i], float(p[i])

    def _consolidar(self, obs: list[dict]) -> dict:
        peso_total = sum(o["peso"] for o in obs)

        def acumular(clave: str) -> np.ndarray:
            # Log del promedio ponderado de probabilidades: robusto a frames atípicos
            # y con una confianza interpretable (no se dispara a 100 %).
            return np.log(sum(o["peso"] * o[clave] for o in obs) / peso_total + 1e-6)

        sexo, c_sexo = self._mejor(list(PROMPTS["sexo"]), acumular("p_sexo"))
        log_edad = acumular("p_edad")
        # En cámaras cenitales los adultos se ven pequeños y CLIP los confunde con niños:
        # "Infantil" solo gana si su evidencia es claramente mayor.
        log_edad[list(PROMPTS["edad"]).index("Infantil (0-14)")] -= 1.2
        edad, c_edad = self._mejor(list(PROMPTS["edad"]), log_edad)

        prom = lambda k: sum(o["peso"] * o[k] for o in obs) / peso_total  # noqa: E731
        f_falda, f_pant, f_vest = prom("frac_falda"), prom("frac_pantalon"), prom("frac_vestido")
        f_piernas, f_brazos = prom("frac_piernas"), prom("frac_brazos")

        # Tipo inferior: la segmentación decide entre pantalón, falda y vestido;
        # CLIP afina entre pantalón, jean y pantaloneta.
        clases_inf = list(PROMPTS["inferior"])
        log_inf = acumular("p_inf")
        if f_vest > max(f_falda, f_pant) and f_vest > 0.04:
            tipo_inf, c_inf = "vestido", min(0.99, 0.6 + f_vest)
        elif f_falda > f_pant * 0.8 and f_falda > 0.03:
            tipo_inf, c_inf = "falda", min(0.99, 0.6 + f_falda)
        else:
            log_inf[clases_inf.index("falda")] = -1e9
            if f_piernas > 0.05:  # piernas descubiertas debajo del pantalón
                log_inf[clases_inf.index("pantaloneta")] += 2.0
            tipo_inf, c_inf = self._mejor(clases_inf, log_inf)

        if tipo_inf == "vestido":
            tipo_sup, c_sup = "vestido", c_inf
        else:
            clases_sup = list(PROMPTS["superior"])
            log_sup = acumular("p_sup")
            if f_brazos > 0.04:  # brazos descubiertos: manga corta
                for t in ("chaqueta", "buso"):
                    log_sup[clases_sup.index(t)] -= 2.0
            tipo_sup, c_sup = self._mejor(clases_sup, log_sup)

        color_sup, hex_sup = self._color_ganador([o["color_sup"] for o in obs], [o["peso"] for o in obs])
        color_inf, hex_inf = self._color_ganador([o["color_inf"] for o in obs], [o["peso"] for o in obs])
        if tipo_inf == "vestido":
            color_inf, hex_inf = color_sup, hex_sup
        if tipo_inf == "pantalon" and color_inf in {"azul", "azul oscuro"}:
            tipo_inf = "jean"

        return {
            "sexo_aparente": sexo,
            "rango_edad": edad,
            "prenda_superior": {"tipo": tipo_sup, "color": color_sup, "hex": hex_sup},
            "prenda_inferior": {"tipo": tipo_inf, "color": color_inf, "hex": hex_inf},
            "confianza": round(float(np.mean([c_sexo, c_edad, c_sup, c_inf])), 2),
            "confianza_detalle": {
                "sexo": round(c_sexo, 2),
                "edad": round(c_edad, 2),
                "prenda_superior": round(c_sup, 2),
                "prenda_inferior": round(c_inf, 2),
            },
        }

    @staticmethod
    def _color_ganador(colores: list, pesos: list[float]) -> tuple[str, str]:
        votos: Counter = Counter()
        suma_rgb = defaultdict(lambda: np.zeros(3))
        suma_peso: Counter = Counter()
        for c, p in zip(colores, pesos):
            if c is None:
                continue
            rgb, cobertura = c
            nombre = nombrar_color(rgb)
            votos[nombre] += p * cobertura
            suma_rgb[nombre] += np.asarray(rgb) * p * cobertura
            suma_peso[nombre] += p * cobertura
        if not votos:
            return "gris", "#808080"
        nombre = votos.most_common(1)[0][0]
        r, g, b = (suma_rgb[nombre] / suma_peso[nombre]).astype(int)
        return nombre, f"#{r:02X}{g:02X}{b:02X}"

    # -------------------------------------------------------------------- video

    def analizar_video(self, ruta_video: str | Path, min_observaciones: int = 4) -> dict:
        """Procesa un video y devuelve el reporte por persona y los cuadros por frame."""
        ruta = Path(ruta_video)
        if not ruta.exists():
            raise FileNotFoundError(f"Video no encontrado: {ruta}")
        cap = cv2.VideoCapture(str(ruta))
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el archivo de video: {ruta}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 12.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        observaciones: dict[int, list[dict]] = defaultdict(list)
        cajas_por_frame: list[dict] = []
        apariciones: dict[int, list[int]] = defaultdict(list)
        frame_idx = 0
        analizados = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frame_idx += 1
            res = self.yolo.track(
                frame, persist=True, classes=[0], conf=0.35, iou=0.5,
                tracker="bytetrack.yaml", verbose=False,
            )[0]
            if frame_idx % self.frames_entre_analisis != 0:
                continue
            analizados += 1
            rgb = balance_blancos(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cajas = []
            if res.boxes is not None and res.boxes.id is not None:
                for (x1, y1, x2, y2), tid, conf in zip(
                    res.boxes.xyxy.numpy(), res.boxes.id.numpy().astype(int), res.boxes.conf.numpy()
                ):
                    x1, y1, x2, y2 = (int(v) for v in (x1, y1, x2, y2))
                    w, h = x2 - x1, y2 - y1
                    cajas.append({"track": int(tid), "bbox": [x1, y1, w, h]})
                    apariciones[int(tid)].append(frame_idx)
                    if w < 24 or h < 60:
                        continue  # muy pequeño para estimar atributos
                    pad_x, pad_y = int(w * 0.08), int(h * 0.04)
                    recorte = rgb[max(0, y1 - pad_y):min(alto, y2 + pad_y), max(0, x1 - pad_x):min(ancho, x2 + pad_x)]
                    obs = self.analizar_recorte(recorte)
                    obs["peso"] *= float(conf)
                    observaciones[int(tid)].append(obs)
            cajas_por_frame.append({"frame": frame_idx, "time": round(frame_idx / fps, 2), "cajas": cajas})
        cap.release()

        # IDs consecutivos para las personas con suficientes observaciones
        validos = [t for t in sorted(observaciones, key=lambda t: apariciones[t][0]) if len(observaciones[t]) >= min_observaciones]
        id_persona = {t: i + 1 for i, t in enumerate(validos)}
        personas = []
        for t in validos:
            p = {"person_id": id_persona[t], **self._consolidar(observaciones[t])}
            p["frames_observados"] = len(observaciones[t])
            p["primer_frame"] = apariciones[t][0]
            p["ultimo_frame"] = apariciones[t][-1]
            personas.append(p)
        por_id = {p["person_id"]: p for p in personas}

        frames = []
        for f in cajas_por_frame:
            ps = []
            for c in f["cajas"]:
                pid = id_persona.get(c["track"])
                if pid is None:
                    continue
                base = {k: v for k, v in por_id[pid].items() if k != "confianza_detalle"}
                ps.append({**base, "bbox": c["bbox"]})
            frames.append({"frame": f["frame"], "time": f["time"], "personas": ps})

        conteo = lambda f: dict(Counter(f(p) for p in personas))  # noqa: E731
        return {
            "reto": "1. Demografía y vestimenta",
            "motor": "YOLO11n + ByteTrack + SegFormer-B2 (ropa) + CLIP ViT-B/32",
            "video": ruta.name,
            "resolucion": f"{ancho}x{alto}",
            "fps": round(fps, 2),
            "total_frames_video": total_frames,
            "frames_analizados": analizados,
            "total_personas_detectadas": len(personas),
            "personas": personas,
            "frames": frames,
            "resumen_demografico": {
                "por_sexo_aparente": conteo(lambda p: p["sexo_aparente"]),
                "por_rango_edad": conteo(lambda p: p["rango_edad"]),
                "prenda_superior_tipos": conteo(lambda p: p["prenda_superior"]["tipo"]),
                "prenda_superior_colores": conteo(lambda p: p["prenda_superior"]["color"]),
                "prenda_inferior_tipos": conteo(lambda p: p["prenda_inferior"]["tipo"]),
                "prenda_inferior_colores": conteo(lambda p: p["prenda_inferior"]["color"]),
            },
        }


assert set(PALETA) == COLORES_VALIDOS
assert set(PROMPTS["sexo"]) == SEXOS_APARENTES
assert set(PROMPTS["edad"]) == RANGOS_EDAD
assert set(PROMPTS["superior"]) <= TIPOS_PRENDA_SUPERIOR
assert set(PROMPTS["inferior"]) <= TIPOS_PRENDA_INFERIOR
