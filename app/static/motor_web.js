// Motor IA del Reto 1 en el navegador: YOLOS + SegFormer (ropa) + CLIP, con transformers.js.
// Es el mismo enfoque de app/vision/demografia_ia.py: un detector de personas (YOLOS-tiny)
// recorta a la persona y sobre ese recorte se segmenta la ropa y se consulta a CLIP.
// Los modelos se descargan la primera vez (~135 MB) y el navegador los guarda en caché.

import {
  pipeline,
  AutoProcessor,
  CLIPVisionModelWithProjection,
  RawImage,
  env,
} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1';

env.allowLocalModels = false;

const MODELO_DET = 'Xenova/yolos-tiny';
const MODELO_SEG = 'Xenova/segformer_b2_clothes';
const MODELO_CLIP = 'Xenova/clip-vit-base-patch32';

// Etiquetas de SegFormer (dataset ATR)
const SUPERIOR = ['Upper-clothes', 'Scarf'];
const FALDA = 'Skirt';
const PANTALON = 'Pants';
const VESTIDO = 'Dress';
const PIERNAS = ['Left-leg', 'Right-leg'];
const ZAPATOS = ['Left-shoe', 'Right-shoe'];
const BRAZOS = ['Left-arm', 'Right-arm'];

// Misma paleta que el motor de Python (varios prototipos por color)
const PALETA = {
  'negro': [[28, 28, 30], [45, 42, 48]],
  'blanco': [[236, 236, 232], [210, 212, 215]],
  'gris': [[128, 128, 128], [165, 165, 168], [95, 98, 100]],
  'rojo': [[190, 35, 40], [140, 30, 35]],
  'azul': [[55, 95, 185], [140, 170, 215], [95, 130, 175]],
  'azul oscuro': [[30, 40, 80], [45, 55, 95]],
  'verde': [[50, 125, 65], [95, 115, 80], [60, 75, 50]],
  'amarillo': [[225, 195, 50]],
  'naranja': [[230, 125, 40]],
  'morado': [[110, 55, 140]],
  'rosado': [[230, 150, 175]],
  'cafe': [[105, 70, 45], [140, 100, 70]],
  'beige': [[205, 185, 150], [180, 160, 120]],
};

function rgbALab([r, g, b]) {
  const lin = c => { c /= 255; return c > 0.04045 ? ((c + 0.055) / 1.055) ** 2.4 : c / 12.92; };
  const R = lin(r), G = lin(g), B = lin(b);
  const f = t => (t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116);
  const x = f((R * 0.4124 + G * 0.3576 + B * 0.1805) / 0.95047);
  const y = f(R * 0.2126 + G * 0.7152 + B * 0.0722);
  const z = f((R * 0.0193 + G * 0.1192 + B * 0.9505) / 1.08883);
  return [116 * y - 16, 500 * (x - y), 200 * (y - z)];
}

const PROTOTIPOS = Object.entries(PALETA).flatMap(([nombre, protos]) => protos.map(p => [nombre, rgbALab(p)]));

export function nombrarColor(rgb) {
  const lab = rgbALab(rgb);
  let mejor = 'gris', dist = Infinity;
  for (const [nombre, p] of PROTOTIPOS) {
    const d = (lab[0] - p[0]) ** 2 + (lab[1] - p[1]) ** 2 + (lab[2] - p[2]) ** 2;
    if (d < dist) { dist = d; mejor = nombre; }
  }
  return mejor;
}

const hex = ([r, g, b]) => '#' + [r, g, b].map(v => Math.round(v).toString(16).padStart(2, '0')).join('').toUpperCase();

// Color dominante: k-means (k=3) en Lab y percentil 70 del grupo más grande (cara iluminada)
function colorDominante(pixeles) {
  if (pixeles.length < 30) return null;
  if (pixeles.length > 2000) {
    const paso = pixeles.length / 2000;
    pixeles = Array.from({ length: 2000 }, (_, i) => pixeles[Math.floor(i * paso)]);
  }
  const labs = pixeles.map(rgbALab);
  let centros = [labs[0], labs[Math.floor(labs.length / 2)], labs[labs.length - 1]];
  let asignacion = new Array(labs.length).fill(0);
  for (let it = 0; it < 10; it++) {
    asignacion = labs.map(l => {
      let mejor = 0, dist = Infinity;
      centros.forEach((c, k) => { const d = (l[0] - c[0]) ** 2 + (l[1] - c[1]) ** 2 + (l[2] - c[2]) ** 2; if (d < dist) { dist = d; mejor = k; } });
      return mejor;
    });
    centros = centros.map((c, k) => {
      const m = labs.filter((_, i) => asignacion[i] === k);
      return m.length ? [0, 1, 2].map(j => m.reduce((s, v) => s + v[j], 0) / m.length) : c;
    });
  }
  const conteo = [0, 0, 0];
  asignacion.forEach(k => conteo[k]++);
  const mayor = conteo.indexOf(Math.max(...conteo));
  const grupo = pixeles.filter((_, i) => asignacion[i] === mayor);
  const p70 = j => { const v = grupo.map(p => p[j]).sort((a, b) => a - b); return v[Math.floor(v.length * 0.7)]; };
  return { rgb: [p70(0), p70(1), p70(2)], cobertura: conteo[mayor] / labs.length };
}

let detector = null, seg = null, clipProc = null, clipVision = null;
let cargando = null;

export function cargar(alProgreso = () => {}) {
  if (cargando) return cargando;
  const progreso = {};
  const avisar = info => {
    if (info.status === 'progress' && info.total) {
      progreso[info.file] = [info.loaded, info.total];
      const [l, t] = Object.values(progreso).reduce((a, [x, y]) => [a[0] + x, a[1] + y], [0, 0]);
      alProgreso(l / t, t);
    }
  };
  cargando = (async () => {
    let device = 'wasm';
    if (navigator.gpu) {
      try { if (await navigator.gpu.requestAdapter()) device = 'webgpu'; } catch { /* sin WebGPU */ }
    }
    // ~135 MB con WebGPU (fp32 / fp16 / q4f16), ~130 MB en CPU (q8)
    const dtypeDet = device === 'webgpu' ? 'fp32' : 'q8';
    const dtypeSeg = device === 'webgpu' ? 'fp16' : 'q8';
    const dtypeClip = device === 'webgpu' ? 'q4f16' : 'q8';
    [detector, seg, clipProc, clipVision] = await Promise.all([
      pipeline('object-detection', MODELO_DET, { device, dtype: dtypeDet, progress_callback: avisar }),
      pipeline('image-segmentation', MODELO_SEG, { device, dtype: dtypeSeg, progress_callback: avisar }),
      AutoProcessor.from_pretrained(MODELO_CLIP, { progress_callback: avisar }),
      CLIPVisionModelWithProjection.from_pretrained(MODELO_CLIP, { device, dtype: dtypeClip, progress_callback: avisar }),
    ]);
    return device;
  })();
  cargando.catch(() => { cargando = null; });
  return cargando;
}

async function probabilidadesClip(imagen, pregunta) {
  const { clases, embeddings } = window.CLIP_TEXTOS[pregunta];
  const entradas = await clipProc(imagen);
  const { image_embeds } = await clipVision(entradas);
  const v = image_embeds.data;
  let norma = 0;
  for (let i = 0; i < v.length; i++) norma += v[i] * v[i];
  norma = Math.sqrt(norma);
  const logits = embeddings.map(e => { let s = 0; for (let i = 0; i < e.length; i++) s += e[i] * v[i]; return (100 * s) / norma; });
  const max = Math.max(...logits);
  const ex = logits.map(l => Math.exp(l - max));
  const suma = ex.reduce((a, b) => a + b, 0);
  return { clases, p: ex.map(x => x / suma) };
}

// Balance de blancos: el percentil 90 de la escena se lleva a casi blanco
function balanceBlancos(datos) {
  const canales = [[], [], []];
  for (let i = 0; i < datos.length; i += 16) for (let c = 0; c < 3; c++) canales[c].push(datos[i + c]);
  const ganancia = canales.map(v => { v.sort((a, b) => a - b); return Math.min(1.8, Math.max(0.8, 225 / Math.max(1, v[Math.floor(v.length * 0.9)]))); });
  const salida = new Uint8ClampedArray(datos.length);
  for (let i = 0; i < datos.length; i += 4) {
    for (let c = 0; c < 3; c++) salida[i + c] = datos[i + c] * ganancia[c];
    salida[i + 3] = 255;
  }
  return salida;
}

// Analiza un canvas: detecta a la persona principal, segmenta su ropa y devuelve
// la caja y una observación acumulable
export async function analizar(canvas) {
  const W = canvas.width, H = canvas.height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  const imagen = new RawImage(balanceBlancos(ctx.getImageData(0, 0, W, H).data), W, H, 4).rgb();

  // 1. Persona principal con YOLOS: entre las detecciones de buen tamaño, la más segura
  // (las cajas más grandes a veces agrupan a dos personas juntas)
  const detecciones = await detector(imagen, { threshold: 0.6, percentage: false });
  const personas = detecciones.filter(d => d.label === 'person');
  if (!personas.length) return null;
  const area = d => (d.box.xmax - d.box.xmin) * (d.box.ymax - d.box.ymin);
  const mayor = Math.max(...personas.map(area));
  const p = personas.filter(d => area(d) >= mayor * 0.35).reduce((a, b) => (a.score >= b.score ? a : b));
  const bx0 = Math.max(0, p.box.xmin), by0 = Math.max(0, p.box.ymin);
  const bx1 = Math.min(W, p.box.xmax), by1 = Math.min(H, p.box.ymax);
  if (bx1 - bx0 < 24 || by1 - by0 < 48) return null;

  // 2. Recorte con margen, como en el motor de Python
  const mx = (bx1 - bx0) * 0.08, my = (by1 - by0) * 0.04;
  const caja = [Math.max(0, bx0 - mx), Math.max(0, by0 - my), Math.min(W, bx1 + mx), Math.min(H, by1 + my)].map(Math.round);
  const recorte = await imagen.crop(caja);
  const w = recorte.width, h = recorte.height, pix = recorte.data; // RGB, 3 canales

  // 3. Segmentación de ropa sobre el recorte
  const mascaras = {};
  for (const s of await seg(recorte)) mascaras[s.label] = s.mask.data;
  const total = w * h;
  const contar = etiquetas => {
    let n = 0;
    for (const e of [].concat(etiquetas)) { const m = mascaras[e]; if (m) for (let i = 0; i < m.length; i++) if (m[i]) n++; }
    return n / total;
  };
  const pixelesDe = etiquetas => {
    const out = [];
    for (const e of [].concat(etiquetas)) {
      const m = mascaras[e];
      if (!m) continue;
      for (let i = 0; i < m.length; i += 3) if (m[i]) out.push([pix[i * 3], pix[i * 3 + 1], pix[i * 3 + 2]]);
    }
    return out;
  };
  const cajaDe = etiquetas => {
    let a = w, b = h, c = 0, d = 0, n = 0;
    for (const e of [].concat(etiquetas)) {
      const m = mascaras[e];
      if (!m) continue;
      for (let i = 0; i < m.length; i++) if (m[i]) { n++; const x = i % w, y = (i / w) | 0; if (x < a) a = x; if (x > c) c = x; if (y < b) b = y; if (y > d) d = y; }
    }
    return n > 60 && c - a > 8 && d - b > 12 ? [a, b, c + 1, d + 1] : null;
  };

  // Con la cámara web casi siempre se ve solo el torso: sin piernas, pantalón, falda
  // ni zapatos visibles, lo que el modelo marca como "vestido" es la prenda superior.
  const inferiorVisible = contar([PANTALON, FALDA, ...PIERNAS, ...ZAPATOS]) > 0.02;
  const superiorEtiquetas = inferiorVisible ? SUPERIOR : [...SUPERIOR, VESTIDO];
  const inferiorEtiquetas = inferiorVisible ? [PANTALON, FALDA, VESTIDO] : [];
  const cajaSup = cajaDe(superiorEtiquetas);
  const cajaInf = inferiorVisible ? cajaDe(inferiorEtiquetas) : null;

  const [sexo, edad, sup, inf] = await Promise.all([
    probabilidadesClip(recorte, 'sexo'),
    probabilidadesClip(recorte, 'edad'),
    probabilidadesClip(cajaSup ? await recorte.crop(cajaSup) : recorte, 'superior'),
    probabilidadesClip(cajaInf ? await recorte.crop(cajaInf) : recorte, 'inferior'),
  ]);
  const pixSup = pixelesDe(superiorEtiquetas);

  return {
    bbox: [Math.round(bx0), Math.round(by0), Math.round(bx1 - bx0), Math.round(by1 - by0)],
    peso: Math.max(1, total / 4000) * p.score,
    p_sexo: sexo.p, p_edad: edad.p, p_sup: sup.p, p_inf: inf.p,
    inferior_visible: inferiorVisible ? 1 : 0,
    frac_falda: contar(FALDA), frac_pantalon: contar(PANTALON), frac_vestido: inferiorVisible ? contar(VESTIDO) : 0,
    frac_piernas: contar(PIERNAS), frac_brazos: contar(BRAZOS),
    color_sup: colorDominante(pixSup),
    color_inf: inferiorVisible ? colorDominante(pixelesDe(inferiorEtiquetas)) : null,
  };
}

// Combina las observaciones de una persona (misma lógica que el motor de Python)
export function consolidar(obs) {
  const T = window.CLIP_TEXTOS;
  const total = obs.reduce((s, o) => s + o.peso, 0);
  const logProm = clave => obs[0][clave].map((_, k) => Math.log(obs.reduce((s, o) => s + o.peso * o[clave][k], 0) / total + 1e-6));
  const prom = clave => obs.reduce((s, o) => s + o.peso * o[clave], 0) / total;
  const mejor = (clases, lp) => {
    const m = Math.max(...lp);
    const p = lp.map(x => Math.exp(x - m));
    const s = p.reduce((a, b) => a + b, 0);
    const i = p.indexOf(Math.max(...p));
    return [clases[i], p[i] / s];
  };

  const [sexo, cSexo] = mejor(T.sexo.clases, logProm('p_sexo'));
  const lpEdad = logProm('p_edad');
  lpEdad[T.edad.clases.indexOf('Infantil (0-14)')] -= 1.2;
  const [edad, cEdad] = mejor(T.edad.clases, lpEdad);

  const fFalda = prom('frac_falda'), fPant = prom('frac_pantalon'), fVest = prom('frac_vestido');
  let tipoInf, cInf;
  const lpInf = logProm('p_inf');
  const inferiorVisible = prom('inferior_visible') >= 0.5;
  if (!inferiorVisible) [tipoInf, cInf] = ['no visible', null];
  else if (fVest > Math.max(fFalda, fPant) && fVest > 0.04) [tipoInf, cInf] = ['vestido', Math.min(0.99, 0.6 + fVest)];
  else if (fFalda > fPant * 0.8 && fFalda > 0.03) [tipoInf, cInf] = ['falda', Math.min(0.99, 0.6 + fFalda)];
  else {
    lpInf[T.inferior.clases.indexOf('falda')] = -1e9;
    if (prom('frac_piernas') > 0.05) lpInf[T.inferior.clases.indexOf('pantaloneta')] += 2;
    [tipoInf, cInf] = mejor(T.inferior.clases, lpInf);
  }

  let tipoSup, cSup;
  if (tipoInf === 'vestido') [tipoSup, cSup] = ['vestido', cInf];
  else {
    const lpSup = logProm('p_sup');
    if (prom('frac_brazos') > 0.04) for (const t of ['chaqueta', 'buso']) lpSup[T.superior.clases.indexOf(t)] -= 2;
    [tipoSup, cSup] = mejor(T.superior.clases, lpSup);
  }

  const colorGanador = clave => {
    const votos = {}, suma = {};
    for (const o of obs) {
      const c = o[clave];
      if (!c) continue;
      const n = nombrarColor(c.rgb), p = o.peso * c.cobertura;
      votos[n] = (votos[n] || 0) + p;
      suma[n] = (suma[n] || [0, 0, 0]).map((v, j) => v + c.rgb[j] * p);
    }
    const nombres = Object.keys(votos);
    if (!nombres.length) return ['gris', '#808080'];
    const n = nombres.reduce((a, b) => (votos[a] >= votos[b] ? a : b));
    return [n, hex(suma[n].map(v => v / votos[n]))];
  };
  let [colorSup, hexSup] = colorGanador('color_sup');
  let [colorInf, hexInf] = colorGanador('color_inf');
  if (tipoInf === 'vestido') [colorInf, hexInf] = [colorSup, hexSup];
  if (tipoInf === 'no visible') [colorInf, hexInf] = ['—', ''];
  if (tipoInf === 'pantalon' && (colorInf === 'azul' || colorInf === 'azul oscuro')) tipoInf = 'jean';

  return {
    sexo, edad,
    prenda_superior: { tipo: tipoSup, color: colorSup, hex: hexSup },
    prenda_inferior: { tipo: tipoInf, color: colorInf, hex: hexInf },
    confianza: Math.round((cInf === null ? (cSexo + cEdad + cSup) / 3 : (cSexo + cEdad + cSup + cInf) / 4) * 100) / 100,
  };
}

window.MotorWeb = { cargar, analizar, consolidar, nombrarColor };
window.dispatchEvent(new Event('motorweb-listo'));
