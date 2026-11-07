# Auditoría de Frontend y Reducción de Ruido Visual — SmartBuilding
**Fecha:** 7 de noviembre de 2026  
**Alcance:** Auditoría de interfaz, densidad cognitiva (Ley de Hick), rendimiento de animaciones (`/emil-design-eng`), composición editorial (`/design-taste-frontend`), uso comedido de `/react-bits` y ergonomía (`/ui-ux-pro-max`) en `index.html`.

---

## 1. Resumen ejecutivo
La iteración anterior acumuló demasiados elementos simultáneos en la primera pantalla (un escenario dividido con diagrama SVG animado, tarjeta rotatoria, 3 contadores, más una rejilla Bento de 4 tarjetas con 10 etiquetas tipo chip y dos lienzos `<canvas>` superpuestos). Eso rompió la regla principal de `/design-taste-frontend` (*"One section per screen: the hero alone on the first screen, 60 to 80 words"*) y la regla 8 de diseño (*"Cada pantalla tiene un elemento rico y todo lo demás en calma"*). La mejora de mayor impacto consiste en **restaurar el espacio negativo editorial en la portada** y **compactar las cabeceras de la consola** para que el video y las personas detectadas sean los únicos protagonistas.

---

## 2. Calificación por rol (antes de las correcciones de esta ronda)
*Estimaciones de criterio sobre el estado previo (`3f39d50`), no mediciones automatizadas.*

| Rol | Percentil | Lo mejor | Lo que más lo frena |
|---|---|---|---|
| **Diseño visual (`/design-taste-frontend`)** | **P68** | Paleta cálida ORYZO (`#100904` / `#ffedd7`) con excelente contraste AAA (`16.4:1`). | Sobreacumulación en `#vistaInicio`: Hero + Diagrama SVG + 4 tarjetas Bento + 10 chips en una sola vista. |
| **Ingeniería de movimiento (`/emil-design-eng`)** | **P74** | Curvas `--ease-out`, salida más rápida que entrada en modales/toasts y deslizador `#pillGlider`. | Dos `<canvas>` animados a la vez (`#dotGrid` con ondas senoidales + `#radarCanvas`) y actualización de `--mouse-x` en contenedores padres con muchos hijos. |
| **UX y Ley de Hick (`/ui-ux-pro-max`)** | **P72** | Puntos 2 al 7 ocultos y filtros de 1 clic en Monitor y Directorio. | Cabecera con 4 controles a la derecha y exceso de bloques apilados encima de la lista de personas (`kpi-row` grande + `quick-filters`). |
| **Rendimiento percibido y foco operativo** | **P75** | Motor IA en WebGPU/WASM sin servidor externo y reconciliación del DOM sin parpadeos. | Decoración tipo HUD de ciencia ficción (`.scanlines`, `.hud-crosshair`, `.hud-corner`, `.serial-edge`) que compite con el video real. |

---

## 3. Hallazgos ordenados por gravedad

| # | Gravedad | Área | Dónde | Hallazgo | Arreglo propuesto |
|---|---|---|---|---|---|
| 1 | **ALTA** | Composición / Ley de Hick | `index.html:1498-1635` | `#vistaInicio` mezcla un Hero de 2 columnas (con diagrama SVG y tarjeta cambiante) + una rejilla Bento de 4 tarjetas y 10 chips. Supera las 160 palabras y satura la vista inicial. | Dejar en `#vistaInicio` **una sola composición editorial a pantalla completa**: título, frase clara (< 35 palabras), acceso directo a los 4 módulos en una fila limpia y 2 botones de acción. |
| 2 | **ALTA** | Visuales / Ruido | `index.html:229-239`, `index.html:3382-3404` | Tres efectos de fondo compiten al mismo tiempo: hebras senoidales (`FloatingLines`), puntos (`DotGrid`) y el barrido de radar (`#radarCanvas`). | Eliminar las ondas senoidales y el radar superpuesto; dejar únicamente la retícula sutil de puntos (`DotGrid`) con respuesta elástica al cursor y parada automática en reposo. |
| 3 | **MEDIA** | Rendimiento CSS (`/emil-design-eng`) | `index.html:3275-3288` | El evento `pointermove` actualiza `--mouse-x` y `--mouse-y` directamente en `.card` y `.person-item`, forzando recálculo de estilos heredados en todos los nodos hijos en cada fotograma. | Quitar el seguimiento por frame en `.person-item` y usar transiciones CSS puras de borde/superficie en `:hover` (`@media (hover: hover) and (pointer: fine)`). |
| 4 | **MEDIA** | Decoración sobre el video | `index.html:1710-1718`, `index.html:1765-1773` | El visor de video incluye `.scanlines`, `.hud-crosshair` central y 4 esquinas `.hud-corner` que ensucian la imagen real de la cámara. | Ocultar las miras falsas, líneas de escaneo y etiqueta vertical `.serial-edge`; dejar el video limpio con una sola píldora discreta de estado (`En vivo`). |
| 5 | **MEDIA** | Densidad vertical en Monitor | `index.html:1742-1755` | En la columna derecha del Monitor se apilan: cabecera, 2 tarjetas KPI grandes y 4 botones de filtro antes de ver la primera persona detectada. | Integrar los KPIs como contadores compactos en línea junto a los filtros rápidos para ganar espacio vertical para la lista de personas. |
| 6 | **BAJA** | Cabecera (`header`) | `index.html:1485-1501` | La barra superior muestra 4 elementos a la derecha (`Reloj`, `Movimiento`, `Oscuro`, `Atajos`), restando limpieza al encabezado. | Dejar solo el reloj en vivo y el botón de tema (`Oscuro / Claro`); mover el acceso a `Atajos (?)` al pie de página. |

---

## 4. Qué falta para P99,99 en cada rol
- **Diseño visual (`/design-taste-frontend`):** Una sola idea por pantalla, bordes izquierdo y derecho idénticos, cero adornos encima del video y tipografía con jerarquía por tamaño y contraste.
- **Ingeniería de movimiento (`/emil-design-eng`):** Animar únicamente `transform` y `opacity`, cero recálculo de variables CSS heredadas en listas largas, y respuesta táctil `scale(0.97)` en `160ms`.
- **UX (`/ui-ux-pro-max` + Ley de Hick):** Máximo 1 acción principal y 1 secundaria por bloque; filtros de 1 clic integrados en una sola barra compacta.

---

## 5. Plan aplicado por Pareto

| # | Trabajo | Por qué en este orden |
|---|---|---|
| 1 | Simplificar `#vistaInicio` a una sola pantalla editorial limpia con acceso directo a las 4 herramientas | Elimina de raíz la saturación visual de la portada (80 % del impacto percibido). |
| 2 | Quitar `#radarCanvas`, las hebras senoidales, `.scanlines`, `.hud-crosshair`, `.hud-corner` y `.serial-edge` | Limpia el lienzo y el reproductor de video de adornos innecesarios. |
| 3 | Compactar la cabecera de `header` y fusionar KPIs + filtros rápidos en el panel de `Personas detectadas` | Despeja la consola operativa y deja más altura útil a la lista en tiempo real. |
| 4 | Optimizar el `pointermove` según `/emil-design-eng` (sin mutar variables CSS en contenedores con decenas de hijos) | Garantiza 60 fps constantes durante la reproducción del video y el uso de la cámara web. |


---

## 6. Lo aplicado en esta ronda

| Antes | Después | Por qué |
|---|---|---|
| `#vistaInicio` con Hero de 2 columnas + diagrama SVG animado + tarjeta rotatoria + 3 métricas + rejilla Bento de 4 tarjetas y 10 chips (`index.html:1229-1259`) | Una sola pantalla editorial a altura completa (`min-height: calc(100svh - 68px)`) con espacio negativo generoso, titular `BlurText`, descripción de 21 palabras, barra `.hero-dock` de 4 accesos directos y 2 botones (`Abrir monitor` y `Probar mi cámara`) | Cumple la Regla 3 de `/design-taste-frontend` (*una sección por pantalla, menos de 80 palabras en portada*) y la Ley de Hick. |
| 3 animaciones de fondo compitiendo a la vez: `#dotGrid` + 5 ondas senoidales (`FloatingLines`) + `#radarCanvas` (`index.html:2898-2995`) | Un único fondo interactivo (`DotGrid` + `ClickSpark` de `/react-bits`) con muelle suave, aura cálida en el cursor y pausa automática tras `2.2 s` de inactividad | Elimina el ruido visual y deja un solo efecto de fondo con propósito (Regla 8 de `AGENTS.md`). |
| `pointermove` actualizando `--mouse-x` / `--mouse-y` en `.person-item` y botones en cada fotograma (`index.html:2889`) | Seguimiento restringido únicamente a los contenedores `.card`, usando `:hover` CSS puro en las filas `.person-item` (`index.html:644-646`) | Evita recalcular variables CSS heredadas en listas largas a 60 fps (`/emil-design-eng`). |
| Adornos falsos sobre el video: `.scanlines`, `.hud-crosshair`, `.hud-corner`, `.hud-br-tag` y `.serial-edge` (`index.html:125-128`, `index.html:529-532`) | Reproductor de video limpio; solo permanece la píldora funcional `En vivo` y las cajas reales del modelo de visión | El video de recepción y las detecciones reales no deben competir con decoración artificial. |
| 2 tarjetas KPI grandes apiladas sobre los filtros en `#cctvView` (`index.html:590-608`) | Contadores en línea tipo píldora (`0 Personas`, `0F / 0M`) seguidos de los 4 filtros rápidos (`Todas · En pantalla · Femenino · Masculino`) | Reduce la altura de cabecera del panel derecho y deja más espacio visible para la lista de personas. |
| 4 elementos en la esquina derecha de `<header>` (`index.html:1218-1228`) | Solo el reloj en vivo (`COT`) y el selector `Oscuro / Claro`; `Atajos (?)` pasa al pie de página (`index.html:1485`) | Despeja la barra superior y mantiene una jerarquía limpia. |
