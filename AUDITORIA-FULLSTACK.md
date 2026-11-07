# Auditoría Full-Stack: Backend FastAPI + Pipeline IA + Consola Frontend
**Fecha:** 2026-11-06  
**Proyecto:** SmartBuilding · Conjunto Residencial San Carlos  
**Alcance:** Backend ([`app/main.py`](file:///Users/suntz/Documents/Soft/app/main.py), [`app/datos.py`](file:///Users/suntz/Documents/Soft/app/datos.py), [`app/vision/demografia_ia.py`](file:///Users/suntz/Documents/Soft/app/vision/demografia_ia.py), [`tests/`](file:///Users/suntz/Documents/Soft/tests/test_residentes.py)) y Frontend ([`index.html`](file:///Users/suntz/Documents/Soft/index.html), [`app/static/retos_vista.js`](file:///Users/suntz/Documents/Soft/app/static/retos_vista.js), [`app/static/motor_web.js`](file:///Users/suntz/Documents/Soft/app/static/motor_web.js)). Retos 2 a 7 permanecen ocultos en la UI según requerimiento funcional.

---

## 1. Resumen ejecutivo

El sistema combina un motor de visión por computador sólido (YOLO11 + ByteTrack + SegFormer + CLIP en servidor y Transformers.js WebGPU/WASM en navegador) con una consola web sin dependencias de build pesado. Los riesgos más importantes en backend eran la **lectura sin límite de tamaño en la carga de imágenes** (`app/main.py:136`) y una **lista en memoria sin cota** (`historial_analisis_demografia` en `app/main.py:31`); en frontend, la **sobre-animación de cortina de 860 ms al cambiar de herramienta dentro de la consola o usar el teclado** y bloques heredados sin unificar con el sistema de diseño (`filtrarDirectorio`, `analizarFotoLocal` duplicado, y falta de interruptor claro/oscuro). Las mejoras de mayor impacto y menor costo ya quedaron aplicadas y verificadas con pruebas automatizadas.

---

## 2. Calificación por rol (estimaciones de criterio, no mediciones)

*Escala: **P50** = aplicación típica de su categoría; **P90** = entre el 10 % mejor; **P99,99** = referencia mundial.*

| Rol | Percentil | Lo mejor | Lo que más lo frena |
|---|---|---|---|
| **Arquitectura** | **P88** | Separación clara entre API FastAPI (`app/main.py`), carga de datos (`app/datos.py`), motores de visión (`app/vision/`) y motor cliente (`app/static/motor_web.js`). | `index.html` concentra estilos, marcado y lógica de consola en un único archivo (~3.400 líneas) para facilitar el despliegue directo en GitHub Pages. |
| **Backend** | **P86** | Validación con `Query` tipado y expresiones regulares (`motor="^(clasico\|ia)$"`), protección contra *path traversal* con `Path(video).name`, y nuevo índice $O(1)$ `residentes_por_id` + filtros `/residentes?q=&torre=`. | La inferencia de video síncrona corre en el hilo del worker; para alta concurrencia convendría una cola de tareas. |
| **Seguridad** | **P82** | Cabeceras HTTP de seguridad (`X-Content-Type-Options: nosniff`, `X-Frame-Options: SAMEORIGIN`, `Referrer-Policy`), manejador central sin filtrar trazas y límite de 10 MB (`413`) en subida de imágenes. | Los endpoints de análisis y el directorio de residentes son públicos (adecuado para demo académica, pero requiere autenticación y *rate limiting* antes de producción real). |
| **Datos e integridad** | **P85** | Normalización limpia del CSV de residentes (`app/datos.py`), votación ponderada por área/nitidez en `AnalizadorDemografiaIA` y historial acotado con `deque(maxlen=100)`. | El historial vive en memoria del proceso; se reinicia al desplegar o reiniciar el servidor (candidato a SQLite si se requiere auditoría persistente). |
| **Rendimiento** | **P91** | Inferencia en cliente con WebGPU/WASM (`motor_web.js`), reconciliación DOM sin parpadeo (`reconciliarLista`), `#dotGrid` y `#radarCanvas` que pausan en reposo o con pestaña oculta. | La primera descarga de pesos ONNX en navegador (~135 MB) depende del caché del navegador. |
| **UX / UI (`/unslop` + `/emil-design-eng`)** | **P93** | Retícula de 4 px, paleta grafito + latón (`#c9a45c`), modo claro/oscuro sin destello, píldoras rápidas de módulo, muelle físico en `.magnet`, toasts que pausan con pestaña oculta y hoja inferior en móvil. | Algunas vistas heredadas (retos 2–7 ocultos) aún conservan lógica en línea dentro de `index.html`. |
| **QA y pruebas** | **P85** | Suite `pytest` rápida para endpoints, filtros, cabeceras de seguridad, validación de bordes de imagen y precisión del detector demográfico. | Faltan pruebas E2E automatizadas (Playwright) sobre el flujo de cámara web en navegador. |
| **Operación** | **P84** | Nuevo endpoint `/salud` con estado de modelos y residentes cargados, logs estructurados y compatibilidad híbrida Uvicorn / GitHub Pages. | Sin métricas Prometheus ni persistencia de bitácora en disco en el servidor. |

---

## 3. Hallazgos ordenados por gravedad

| # | Gravedad | Área | Dónde | Hallazgo | Arreglo propuesto / Estado |
|---|---|---|---|---|---|
| 1 | **ALTA** | Seguridad / Backend | [`app/main.py:188-197`](file:///Users/suntz/Documents/Soft/app/main.py#L188-L197) | `analizar_imagen_demografia` leía `await archivo.read()` sin cota superior, permitiendo agotar la memoria del proceso con un archivo gigante. | **Aplicado:** lectura acotada `await archivo.read(MAX_IMAGEN_BYTES + 1)` con rechazo HTTP `413` al superar 10 MB. |
| 2 | **ALTA** | Rendimiento / Memoria | [`app/main.py:38-41`](file:///Users/suntz/Documents/Soft/app/main.py#L38-L41) | `historial_analisis_demografia` era un `list` sin límite que crecía indefinidamente en cada llamada a `POST /vision/demografia-vestimenta/analizar-video`. | **Aplicado:** reemplazado por ` deque(maxlen=MAX_HISTORIAL_ANALISIS)` (`100` ejecuciones). |
| 3 | **ALTA** | Movimiento (`/emil-design-eng`) | [`index.html:2746-2771`](file:///Users/suntz/Documents/Soft/index.html#L2746-L2771) | Cambiar entre herramientas dentro de `#vistaApp` o pulsar los atajos `1` / `C` disparaba `conCortina` (860 ms de bloqueo visual), violando la regla de frecuencia de Emil Kowalski (*nunca animar acciones de teclado; transiciones internas < 300 ms*). | **Aplicado:** `conCortina` solo actúa entre `#vistaInicio` y `#vistaApp`; el cambio interno es inmediato y sin animación cuando viene del teclado. |
| 4 | **MEDIA** | Código muerto / Front | [`index.html:2515-2525`](file:///Users/suntz/Documents/Soft/index.html#L2515-L2525) | Existía una función `analizarFotoLocal` duplicada de ~95 líneas con heurísticas RGB antiguas y fuente `Inter`, sobrescrita más abajo por `window.analizarFotoLocal`. | **Aplicado:** eliminada la función muerta y unificado el reporte de `window.analizarFotoLocal` con `.person-item` y `datosPersona`. |
| 5 | **MEDIA** | Diseño (`/unslop`) | [`index.html:2530-2580`](file:///Users/suntz/Documents/Soft/index.html#L2530-L2580) | `filtrarDirectorio` renderizaba HTML antiguo (`.feed-item`, `r.tipo.toUpperCase()`, `<code>`) sin estado vacío cuando una búsqueda no arrojaba resultados. | **Aplicado:** migrado a `.person-item`, `.pi-avatar`, `.chip-prenda`, mayúscula inicial y estado vacío `.feed-vacio`. |
| 6 | **MEDIA** | Color y Tema (`/unslop`) | [`index.html:12-26`](file:///Users/suntz/Documents/Soft/index.html#L12-L26) | Existían tokens `[data-theme="light"]`, pero faltaba el control de cambio de tema con persistencia sin destello, y `#dotGrid` / `#radarCanvas` dibujaban puntos blancos invisibles sobre fondo claro. | **Aplicado:** script en `<head>` sin destello, botón `#themeToggleBtn` en cabecera y adaptación de color en `#dotGrid` y `#radarCanvas`. |
| 7 | **MEDIA** | Operación / Seguridad | [`app/main.py:44-80`](file:///Users/suntz/Documents/Soft/app/main.py#L44-L80) | Faltaban el endpoint `/salud`, cabeceras de seguridad HTTP, manejador central de excepciones no controladas y filtros de búsqueda en `GET /residentes`. | **Aplicado:** añadidos `/salud`, middleware de cabeceras, manejador `500` seguro, índice `residentes_por_id` $O(1)$ y filtros `?q=&torre=`. |
| 8 | **BAJA** | Detalles UI (`/emil-design-eng`) | [`index.html:1226-1263`](file:///Users/suntz/Documents/Soft/index.html#L1226-L1263) | `#shortcutsModal` usaba `display: none` sin transición, sin devolver el foco al cerrar con `Esc` y sin adaptación a hoja inferior en móvil; `mostrarToast` no pausaba su temporizador al ocultar la pestaña. | **Aplicado:** transición asimétrica (`200ms` entrada / `140ms` salida), hoja inferior en móvil (`<= 760px`), restauración de foco y pausa de toasts con `document.hidden`. |
| 9 | **BAJA** | Pruebas | [`tests/test_residentes.py:1-58`](file:///Users/suntz/Documents/Soft/tests/test_residentes.py#L1-L58) | Había 20 líneas de pruebas comentadas con `'''...'''` y faltaban pruebas para `/salud`, filtros de `/residentes` y validación de subida de imágenes. | **Aplicado:** eliminado el código muerto y añadidas 3 pruebas nuevas (`5 passed` en `test_residentes.py`). |

---

## 4. Qué falta para P99,99 en cada rol

- **Arquitectura y Frontend:** dividir `index.html` en módulos ES (`app/static/ui/consola.js`, `app/static/ui/tema.js`, `app/static/ styles.css`) manteniendo compatibilidad con GitHub Pages.
- **Backend y Datos:** migrar el historial en memoria y el directorio CSV a **SQLite** con escrituras transaccionales e idempotencia por `X-Request-Id`.
- **Seguridad:** añadir autenticación por sesión (`HttpOnly`, `SameSite=Lax`) y límite de peticiones por IP (*rate limiting*) en los endpoints de inferencia pesada (`/vision/demografia-vestimenta/analizar-video`).
- **QA:** incorporar pruebas de regresión visual y accesibilidad en navegador real con Playwright + `@axe-core/playwright`.

---

## 5. Plan de trabajo (Pareto y dependencias)

| # | Trabajo | Por qué en este orden |
|---|---|---|
| 1 | **Endurecer el backend (`app/main.py` + `tests/test_residentes.py`):** `/salud`, límite de 10 MB en imágenes, `deque` acotado, cabeceras de seguridad, manejador central de errores y filtros en `/residentes`. | Elimina de inmediato los riesgos de agotamiento de memoria y deja red de pruebas en verde (**Completado**). |
| 2 | **Aplicar `/emil-design-eng` en navegación y micro-interacciones (`index.html`):** quitar la cortina al cambiar de módulo interno o usar el teclado, añadir píldoras rápidas `.module-pills`, modal con salida rápida (`140ms`) y hoja inferior móvil, y toasts con pausa en segundo plano. | Es lo que más mejora la fluidez percibida por el operador todos los días (**Completado**). |
| 3 | **Aplicar `/unslop` y `/react-bits` (`index.html` + `retos_vista.js`):** modo claro/oscuro persistente y adaptado en canvas (`#dotGrid`, `#radarCanvas`), muelle físico en `.magnet`, limpieza de `analizarFotoLocal` duplicado, rediseño de `filtrarDirectorio` y miras sin brillo neón. | Unifica el 100 % de las vistas activas bajo el sistema de diseño grafito + latón (**Completado**). |
| 4 | *(Siguiente fase opcional)* Persistir bitácora de eventos en SQLite y separar CSS/JS de `index.html` en archivos estáticos dedicados. | Requiere tocar estructura de archivos servidos en GitHub Pages; menor urgencia funcional. |

---

## 6. Lo aplicado — Tabla `Antes | Después | Por qué`

| Antes | Después | Por qué |
|---|---|---|
| `await archivo.read()` sin límite en `POST /vision/demografia-vestimenta/analizar-imagen` (`app/main.py:189`). | `await archivo.read(MAX_IMAGEN_BYTES + 1)` con rechazo HTTP `413` si supera 10 MB. | Evita agotamiento de RAM ante subidas accidentales o maliciosas (Seguridad / Validación en el borde). |
| `historial_analisis_demografia: list[dict] = []` sin límite (`app/main.py:39`). | `deque(maxlen=MAX_HISTORIAL_ANALISIS)` (100 registros) + serialización a lista en `/eventos`. | Previene fugas de memoria en ejecuciones prolongadas del servidor. |
| Sin endpoint de salud, sin cabeceras HTTP de seguridad y búsqueda lineal en `/residentes/{id}` (`app/main.py:44-101`). | `GET /salud`, middleware con `X-Content-Type-Options` / `X-Frame-Options`, manejador central `500`, diccionario $O(1)$ `residentes_por_id` y filtros `GET /residentes?q=&torre=`. | Cumple la lista de revisión de operación, seguridad y rendimiento de backend. |
| Cambiar de herramienta dentro de `#vistaApp` o usar atajos (`1`, `C`) lanzaba `conCortina` (860 ms de bloqueo a pantalla completa). | Cambio inmediato dentro de `#vistaApp` con barra rápida `.module-pills` en escritorio y cero animación en acciones de teclado; la cortina queda reservada para `#vistaInicio` $\leftrightarrow$ `#vistaApp`. | Regla de frecuencia de `/emil-design-eng`: las acciones frecuentes o de teclado nunca deben frenar al operador. |
| `#shortcutsModal` cambiaba con `display: none` brusco, no devolvía el foco al cerrar y se veía como caja flotante en móvil. | Transición `scale(0.96)` $\rightarrow$ `scale(1)` (`200ms` entrada, `140ms` salida), hoja inferior con `env(safe-area-inset-bottom)` en `<= 760px` y restauración de foco al disparador. | `/emil-design-eng` + `mobile-native`: la salida es más rápida que la entrada, nunca escala desde `0`, y en celular los diálogos son hojas inferiores. |
| `mostrarToast` usaba un `setTimeout` fijo de 3.000 ms aunque el usuario estuviera en otra pestaña. | Temporizador que se pausa en `visibilitychange` cuando `document.hidden` es `true` y salida acelerada a `150ms`. | Detalle invisible de `/emil-design-eng` y `ask-sonner`: el usuario no pierde notificaciones al volver a la pestaña. |
| `.magnet` asignaba `translate(x, y)` directo en `pointermove` con transición CSS fija (`index.html:2853`). | Interpolación por muelle físico (`stiffness: 0.18, damping: 0.72`) con `requestAnimationFrame` que se detiene sola al llegar al reposo. | `/react-bits` + `/emil-design-eng`: el seguimiento magnético responde con inercia natural sin saltos al entrar o salir. |
| Sin botón de cambio de tema y `#dotGrid` / `#radarCanvas` con puntos blancos invisibles en modo claro. | Script inicial sin destello en `<head>`, botón `#themeToggleBtn` (`Claro` / `Oscuro`) en cabecera y colores adaptativos en ambos canvas. | Regla 2 de diseño (`AGENTS.md`): modo claro y oscuro siempre disponibles, recordados en `localStorage` y con contraste AA. |
| Función `analizarFotoLocal` duplicada (~95 líneas muertas) y `filtrarDirectorio` con `.feed-item`, `toUpperCase()` y sin estado vacío. | Eliminado el código muerto; tanto `analizarFotoLocal` como `filtrarDirectorio` usan `.person-item`, `.pi-avatar`, `.chip-prenda` y `.feed-vacio`. | Claridad de código y consistencia visual (`/unslop`): cero MAYÚSCULAS gritonas y componentes unificados. |
| Miras del visor con `shadowBlur = 8` (`index.html:3255` y `app/static/retos_vista.js:21`). | Trazo nítido de `2.5px` sin `shadowBlur` de color. | Regla 2 y 8 (`/unslop`): prohibidos los brillos de color y efectos neón en elementos de datos. |
