# SmartBuilding API — Reto 1: Demografía y Vestimenta 🏢🎥

🌐 **Demo en vivo:** https://suntzz.github.io/smartbuilding-api/

Backend para la administración de edificios residenciales (**Conjunto Residencial San Carlos**) desarrollado con **Python**, **FastAPI** y **OpenCV**, incorporando el **Reto 1 de Visión por Computador: Demografía y vestimenta**.

> **Nota de ramas:**
> - **`main`**: Implementación base de residentes + **Reto 1 (Demografía y vestimenta)** con videos reales, dashboard interactivo y suite de pruebas `pytest`.
> - **`feature/retos-vision-completos`**: Implementación completa de los **7 retos de visión por computador** (Demografía y vestimenta, Conteo de entradas y salidas, Merodeo, Objeto abandonado, Lectura de placas, Caídas y comportamientos de riesgo, y Correspondencia en casillero).

---

## 🎯 Reto 1: Demografía y Vestimenta

Para cada persona que pasa frente a la cámara de seguridad del edificio, el módulo [`app/vision/demografia_vestimenta.py`](app/vision/demografia_vestimenta.py) detecta, rastrea (`person_id`) y estima:

1. **Sexo aparente**: `Masculino` | `Femenino`
2. **Rango de edad**: `Infantil (0-14)` | `Joven (15-29)` | `Adulto (30-54)` | `Adulto Mayor (55+)`
3. **Prenda superior**:
   - **Tipo**: `camiseta`, `camisa`, `chaqueta`, `buso`, `blusa`
   - **Color dominante**: `negro`, `blanco`, `gris`, `rojo`, `azul`, `azul oscuro`, `verde`, `amarillo`, `naranja`, `morado`, `rosado`, `cafe`, `beige` + código hexadecimal (`#RRGGBB`)
4. **Prenda inferior**:
   - **Tipo**: `pantalon`, `jean`, `pantaloneta`, `falda`
   - **Color dominante**: nombre en español + código hexadecimal (`#RRGGBB`)

---

## 📹 Videos de Prueba y Bases de Datos de Referencia

El repositorio incluye clips reales de cámaras de vigilancia en `data/videos/`:
- `data/videos/reto1_demografia_vestimenta.mp4`: Cámara cenital/lobby con múltiples peatones de cuerpo completo (derivado de [`intel-iot-devkit/sample-videos`](https://github.com/intel-iot-devkit/sample-videos) — `people-detection.mp4`).
- `data/videos/reto1_corredor_frontal.mp4`: Cámara de pasillo frontal (`face-demographics-walking.mp4`).

---

---

## 🧠 Motor IA del Reto 1 (YOLO11 + SegFormer + CLIP)

El módulo [`app/vision/demografia_ia.py`](app/vision/demografia_ia.py) reemplaza las reglas manuales por modelos preentrenados:

| Paso | Modelo | Para qué |
|---|---|---|
| Detección y seguimiento | [YOLO11n](https://docs.ultralytics.com/models/yolo11/) + ByteTrack | Un ID estable por persona |
| Segmentación de ropa | [SegFormer B2 clothes](https://huggingface.co/mattmdjaga/segformer_b2_clothes) (ONNX) | Color real de cada prenda; pantalón, falda o vestido; mangas y piernas descubiertas |
| Atributos | [CLIP ViT-B/32](https://github.com/mlfoundations/open_clip) (zero-shot) | Sexo aparente, rango de edad, tipo de prenda |

Las predicciones de todos los frames se promedian por persona; el color se nombra en espacio Lab tras un balance de blancos de la escena.

```bash
pip install -r requirements.txt
python scripts/descargar_modelos.py       # ~700 MB en models/ (no se sube a git)
python scripts/generar_datos_reto1.py     # actualiza app/static/retos_data.js para la página
```

En la API: `POST /vision/demografia-vestimenta/analizar-video?motor=ia`.

**Cámara web y foto en el navegador** ([`app/static/motor_web.js`](app/static/motor_web.js)): el mismo enfoque corre en el navegador con [transformers.js](https://huggingface.co/docs/transformers.js): YOLOS-tiny detecta a la persona, SegFormer segmenta su ropa y CLIP clasifica los atributos. Los textos de CLIP se precalculan con `python scripts/generar_textos_clip_web.py`. La primera vez se descargan ~135 MB (quedan en caché) y nada sale del navegador. Si solo se ve el torso, la prenda inferior se reporta como "no visible".

**Resultado en el video del corredor (no usado para ajustar nada):** el detector clásico encontraba 6 de 7 personas y acertaba el sexo en 3; el motor IA encuentra las 7 y acierta el sexo en las 7, además de reconocer vestidos. Limitación conocida: con poca luz, las prendas blancas o celestes pueden salir como gris o beige.

---

## 🚀 Instalación y Ejecución

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Ejecutar pruebas unitarias y de integración (TDD)
pytest -v

# 3. Iniciar el servidor FastAPI
uvicorn app.main:app --reload
```

- **Dashboard Interactivo con Video en Vivo**: `http://127.0.0.1:8000/dashboard`
- **Documentación Swagger OpenAPI**: `http://127.0.0.1:8000/docs`
