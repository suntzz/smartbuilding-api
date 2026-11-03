# SmartBuilding API — 7 retos de visión por computador 🏢🎥

🌐 **Demo en vivo:** https://suntzz.github.io/smartbuilding-api/

Backend para la administración de edificios residenciales (**Conjunto Residencial San Carlos**) desarrollado con **Python**, **FastAPI** y **OpenCV**, incorporando los **7 retos de visión por computador**: demografía y vestimenta, conteo de entradas y salidas, merodeo, objeto abandonado, lectura de placas, caídas y correspondencia en casillero.

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

## 🛰️ Retos 2 a 7

Cada reto tiene su motor en `app/vision/` y se prueba sobre su video en `data/videos/`. No hay datos etiquetados para entrenar modelos propios, así que se combinan modelos preentrenados (detección, seguimiento, pose, OCR) con reglas explícitas y calibradas para cada problema.

| Reto | Motor | Cómo decide | Resultado en el video de prueba |
|---|---|---|---|
| 2 · Conteo | [`conteo.py`](app/vision/conteo.py) · YOLO11m + ByteTrack | Los pies de cada persona cruzan una o varias líneas con histéresis; la hora pico es ≥ 3 personas a la vez | 6 entradas y 3 salidas en la línea principal |
| 3 · Merodeo | [`merodeo.py`](app/vision/merodeo.py) · YOLO11m + ByteTrack + reidentificación por color | Tiempo en zona, recorrido en alturas de cuerpo, eficiencia de la trayectoria y pasadas repetidas; quien está quieto solo *espera* y quien está quieto con otra persona *conversa* | Alerta a quien va y vuelve y a quien pasa 3 veces; la pareja que conversa y quien solo pasa no generan alerta |
| 4 · Objeto abandonado | [`abandono.py`](app/vision/abandono.py) · YOLO11m | Morral, bolso o maleta quieto; dueño = persona más cercana al dejarlo; distancia medida sobre el piso; alerta tras 3 s sin nadie cerca | Una alerta, 3 s después de que el dueño se aleja |
| 5 · Placas | [`placas.py`](app/vision/placas.py) · RapidOCR (DBNet + CRNN) | CLAHE de noche, formato colombiano (ABC123 carro, ABC12D moto) con corrección de caracteres y votación por vehículo; se cruza con el directorio | 3 vehículos, una lectura por vehículo, de día y de noche |
| 6 · Caídas | [`caidas.py`](app/vision/caidas.py) · YOLO11-pose o siluetas | Postura relativa a la altura de pie de cada persona; caída = paso rápido de erguido a acostado y 2 s sin levantarse | Detecta la caída; sentarse y agacharse no alertan |
| 7 · Casilleros | [`casilleros.py`](app/vision/casilleros.py) · contornos + OCR de etiquetas | Cada casillero frente a su imagen vacía; paquete o recibo por forma y color; avisa a los residentes del apartamento | 4 casilleros con correspondencia, con su número de apartamento |

```bash
python scripts/descargar_modelos.py      # YOLO11m, YOLO11-pose, OCR…
python scripts/generar_datos_retos.py    # actualiza los datos de la página (o: ... 3 5 para algunos retos)
pytest tests/test_retos.py -v
```

Limitaciones: los videos de los retos 5, 6 y 7 son animaciones sintéticas; con cámaras reales hay que calibrar las líneas del conteo, la zona de merodeo y el tamaño de los casilleros, y conviene un detector de placas dedicado. La reidentificación por color puede fallar si cambia mucho la luz (en el reto 3 la misma persona aparece con dos IDs en sus dos vueltas, aunque se alerta en ambas).

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
