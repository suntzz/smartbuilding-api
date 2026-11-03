// Vistas de los retos 2 a 7: dibujo sobre el video, indicadores y registro de eventos
// que se van llenando a medida que avanza el video. Usa los datos precalculados por
// scripts/generar_datos_retos.py (window.RETOS_DATA) y las utilidades de index.html
// (reconciliarLista, ponerValor, mostrarToast, playBeep).
(() => {
  const COLOR = {
    verde: '#3fb27f', oro: '#c9a45c', rojo: '#e5484d', ambar: '#d9a13b',
    blanco: '#e8e8e8', gris: '#8a8a8a', naranja: '#e07b39', plata: '#b8b8b8'
  };
  const datos = id => window.RETOS_DATA && window.RETOS_DATA.retos[id];
  const reloj = s => `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(Math.floor(s % 60)).padStart(2, '0')}`;

  // ------------------------------------------------------------------ dibujo
  function caminoRedondo(ctx, x, y, w, h, r) {
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(x, y, Math.max(0, w), h, r);
    else ctx.rect(x, y, Math.max(0, w), h);
  }

  // Esquinas de mira alrededor de una caja (como en la cámara web)
  function mira(ctx, [x, y, w, h], color, { grosor = 2.5, latido = false, tenue = false } = {}) {
    const l = Math.max(10, Math.min(w, h) * 0.22);
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = grosor;
    ctx.lineCap = 'round';
    ctx.globalAlpha = tenue ? 0.55 : latido ? 0.6 + 0.4 * Math.sin(Date.now() / 140) : 1;
    ctx.shadowColor = color;
    ctx.shadowBlur = tenue ? 0 : 8;
    ctx.beginPath();
    [[x, y, 1, 1], [x + w, y, -1, 1], [x, y + h, 1, -1], [x + w, y + h, -1, -1]].forEach(([cx, cy, sx, sy]) => {
      ctx.moveTo(cx, cy + sy * l);
      ctx.lineTo(cx, cy);
      ctx.lineTo(cx + sx * l, cy);
    });
    ctx.stroke();
    ctx.shadowBlur = 0;
    ctx.globalAlpha = tenue ? 0.08 : 0.14;
    ctx.lineWidth = 1;
    ctx.strokeRect(x, y, w, h);
    ctx.restore();
  }

  // Etiqueta redondeada con punto de color: [titulo, detalle?, extra?]
  function etiqueta(ctx, bbox, lineas, color, { dentro = false } = {}) {
    const [x, y, w, h] = bbox;
    const cw = ctx.canvas.width, ch = ctx.canvas.height;
    ctx.save();
    ctx.font = '600 13px "IBM Plex Sans", sans-serif';
    const anchos = lineas.map((t, k) => {
      ctx.font = k === 0 ? '600 13px "IBM Plex Sans", sans-serif' : '11px "IBM Plex Mono", monospace';
      return ctx.measureText(t).width;
    });
    const ancho = Math.min(cw - 8, Math.max(...anchos) + 34);
    const alto = 12 + lineas.length * 16;
    const lx = Math.min(Math.max(4, x), cw - ancho - 4);
    let ly = dentro ? y + 6 : y - alto - 6;
    if (ly < 4) ly = Math.min(ch - alto - 4, y + h + 6);
    if (ly + alto > ch - 4) ly = Math.max(4, y + 6);
    ctx.fillStyle = 'rgba(10, 10, 10, 0.84)';
    ctx.strokeStyle = color + '88';
    ctx.lineWidth = 1;
    caminoRedondo(ctx, lx, ly, ancho, alto, 8);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(lx + 13, ly + 14, 3.5, 0, Math.PI * 2);
    ctx.fill();
    lineas.forEach((t, k) => {
      ctx.font = k === 0 ? '600 13px "IBM Plex Sans", sans-serif' : '11px "IBM Plex Mono", monospace';
      ctx.fillStyle = k === 0 ? '#ffffff' : '#bdbdbd';
      ctx.fillText(t, lx + 24, ly + 19 + k * 16);
    });
    ctx.restore();
  }

  // Anillo de progreso (cuenta regresiva hasta la alerta)
  function anillo(ctx, cx, cy, r, progreso, color) {
    ctx.save();
    ctx.lineWidth = 3;
    ctx.strokeStyle = 'rgba(255,255,255,0.15)';
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.stroke();
    ctx.strokeStyle = color;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.arc(cx, cy, r, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * Math.min(1, progreso));
    ctx.stroke();
    ctx.restore();
  }

  // Ondas que se expanden desde un punto (alerta activa)
  function radar(ctx, cx, cy, r, color) {
    ctx.save();
    for (let k = 0; k < 2; k++) {
      const t = ((Date.now() / 1400) + k / 2) % 1;
      ctx.globalAlpha = 0.7 * (1 - t);
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(cx, cy, r * (0.8 + t * 1.2), 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.restore();
  }

  function estela(ctx, puntos, color) {
    if (!puntos || puntos.length < 2) return;
    ctx.save();
    ctx.lineCap = 'round';
    ctx.lineWidth = 2;
    for (let k = 1; k < puntos.length; k++) {
      ctx.globalAlpha = 0.05 + 0.6 * (k / puntos.length);
      ctx.strokeStyle = color;
      ctx.beginPath();
      ctx.moveTo(...puntos[k - 1]);
      ctx.lineTo(...puntos[k]);
      ctx.stroke();
    }
    ctx.restore();
  }

  function chip(ctx, texto, x, y, color) {
    ctx.save();
    ctx.font = '600 11px "IBM Plex Mono", monospace';
    const w = ctx.measureText(texto).width + 16;
    ctx.fillStyle = 'rgba(10,10,10,0.82)';
    ctx.strokeStyle = color + '99';
    caminoRedondo(ctx, x, y, w, 20, 10);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = color;
    ctx.fillText(texto, x + 8, y + 14);
    ctx.restore();
    return w;
  }

  // ------------------------------------------------------------------ panel
  const ICONO = {
    arriba: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M12 19V5M5 12l7-7 7 7"/></svg>',
    abajo: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M12 5v14M5 12l7 7 7-7"/></svg>',
    bolso: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round"><rect x="5" y="8" width="14" height="12" rx="3"/><path d="M9 8V6a3 3 0 0 1 6 0v2M5 13h14"/></svg>',
    carro: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round"><path d="M5 16V11l2-5h10l2 5v5M3 16h18v3H3zM7 11h10"/><circle cx="7.5" cy="16" r="1"/><circle cx="16.5" cy="16" r="1"/></svg>',
    moto: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"><circle cx="6" cy="16" r="3"/><circle cx="18" cy="16" r="3"/><path d="M6 16l4-6h5l3 6M13 10l-1-3h3"/></svg>',
    paquete: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round"><path d="M4 8l8-4 8 4v8l-8 4-8-4z"/><path d="M4 8l8 4 8-4M12 12v8"/></svg>',
    recibo: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round"><rect x="3" y="6" width="18" height="12" rx="2"/><path d="M3 8l9 6 9-6"/></svg>',
    vacio: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="4" y="4" width="16" height="16" rx="2" stroke-dasharray="3 3"/></svg>'
  };

  function kpis(titulos) {
    document.getElementById('kpiContainer').innerHTML = titulos.map((t, k) => `
      <div class="kpi-card"><div class="num" id="kpiR${k}">0</div><div class="desc">${t}</div></div>`).join('');
  }
  const kpi = (k, valor) => {
    const el = document.getElementById('kpiR' + k);
    if (!el) return;
    el.classList.toggle('texto', typeof valor === 'string' && !/^[\d.\s/s%]+$/.test(valor));
    ponerValor(el, valor);
  };

  function vacio(feed, titulo, texto) {
    feed.innerHTML = `<div class="feed-vacio"><b>${titulo}</b><span>${texto}</span></div>`;
  }

  function banner(texto, tipo = 'alerta') {
    const b = document.getElementById('alertBanner');
    if (!texto) { b.style.display = 'none'; return; }
    b.dataset.tipo = tipo;
    document.getElementById('alertMsg').textContent = texto;
    b.style.display = 'flex';
  }

  // Línea de tiempo: marcas de los eventos sobre la barra del video
  function marcas(lista, totalFrames) {
    const cont = document.getElementById('marcasTiempo');
    if (!cont) return;
    cont.innerHTML = lista.map(m => `
      <button class="marca" style="left:${(100 * m.frame / totalFrames).toFixed(2)}%;--c:${m.color}" data-frame="${m.frame}" title="${m.titulo}" aria-label="${m.titulo}"></button>`).join('');
  }

  // Historia por persona/objeto a partir de los frames: [{frame, ...}]
  function historias(frames, lista, clave) {
    const mapa = new Map();
    frames.forEach(f => (f[lista] || []).forEach(p => {
      if (!mapa.has(p[clave])) mapa.set(p[clave], []);
      mapa.get(p[clave]).push({ frame: f.frame, ...p });
    }));
    return mapa;
  }
  const ultimoHasta = (hist, frame) => {
    let r = null;
    for (const h of hist || []) { if (h.frame <= frame) r = h; else break; }
    return r;
  };

  // Avisos (toast + sonido) una sola vez por evento; se reinician al rebobinar
  let avisados = new Set(), ultimoFrame = -1;
  function avisar(clave, texto, tipo = 'danger') {
    if (avisados.has(clave)) return;
    avisados.add(clave);
    if (typeof mostrarToast === 'function') mostrarToast(texto, tipo);
    if (typeof playBeep === 'function') playBeep(tipo === 'danger' ? 520 : 880, 0.12);
  }

  // ------------------------------------------------------------------ retos
  const ESTADO_MERODEO = {
    paso_normal: ['De paso', COLOR.blanco], esperando: ['Esperando', COLOR.oro],
    conversando: ['Conversando', COLOR.plata], merodeo: ['Merodeo', COLOR.rojo]
  };
  const ESTADO_OBJETO = {
    en_movimiento: ['Lo llevan consigo', COLOR.gris], con_propietario: ['Con su dueño', COLOR.verde],
    supervisado: ['Supervisado', COLOR.verde], propietario_alejandose: ['Dueño alejándose', COLOR.ambar],
    abandonado: ['Abandonado', COLOR.rojo]
  };
  const ESTADO_POSTURA = {
    de_pie: ['De pie', COLOR.verde], sentado: ['Sentado', COLOR.oro], agachado: ['Agachado', COLOR.oro],
    acostado: ['Acostado', COLOR.ambar], posible_caida_en_observacion: ['Posible caída', COLOR.naranja],
    caida_confirmada: ['Caída confirmada', COLOR.rojo]
  };

  const POSTURA_CORTA = {
    de_pie: 'De pie', sentado: 'Sentado', agachado: 'Agachado', acostado: 'Acostado',
    posible_caida_en_observacion: '¿Caída?', caida_confirmada: 'Caída'
  };

  const RETOS = {
    2: {
      panel: ['Analítica · Conteo de flujo', 'Cruces de línea', 'YOLO11m + ByteTrack · cruce de línea con histéresis'],
      kpis: ['Entradas', 'Salidas', 'Personas en escena'],
      vacio: ['Aún no hay cruces', 'Reproduce el video: cada vez que una persona cruce una línea aparecerá aquí con su sentido.'],
      marcas: s => s.eventos_cruce.filter(e => e.linea === s.lineas[0].id).map(e => ({
        frame: e.frame, color: e.sentido === 'entrada' ? COLOR.verde : COLOR.ambar,
        titulo: `Persona #${e.person_id}: ${e.sentido} (${reloj(e.tiempo)})`
      })),
      actualizar(frame, f, s, feed) {
        kpi(0, f.total_entradas);
        kpi(1, f.total_salidas);
        kpi(2, f.personas_simultaneas);
        banner(f.es_hora_pico ? `Hora pico · ${f.personas_simultaneas} personas en simultáneo` : '', 'aviso');
        const eventos = s.eventos_cruce.filter(e => e.frame <= frame).reverse();
        if (!eventos.length) return vacio(feed, ...this.vacio);
        reconciliarLista(feed, eventos, {
          clave: e => `${e.person_id}-${e.linea}-${e.frame}`,
          clase: e => `person-item cruce ${e.sentido}`,
          fijo: e => `<div class="pi-icono ${e.sentido}">${e.sentido === 'entrada' ? ICONO.arriba : ICONO.abajo}</div>`,
          datos: e => `
            <div class="pi-main">
              <div class="pi-top"><b class="pi-sexo">Persona #${e.person_id}</b>
                <span class="pi-estado ${e.sentido === 'entrada' ? 'vivo' : 'aviso'}">${e.sentido}</span>
                ${e.hora_pico ? '<span class="pi-estado pico">Hora pico</span>' : ''}</div>
              <div class="garments"><span class="chip-prenda">${e.nombre_linea}</span></div>
            </div>
            <div class="pi-conf"><span>${reloj(e.tiempo)}</span><small>F${e.frame}</small></div>`
        });
      },
      dibujar(ctx, frame, f, s) {
        const cw = ctx.canvas.width, ch = ctx.canvas.height;
        s.lineas.forEach((l, k) => {
          const [x1, y1, x2, y2] = [l.p1[0] * cw, l.p1[1] * ch, l.p2[0] * cw, l.p2[1] * ch];
          const reciente = s.eventos_cruce.find(e => e.linea === l.id && frame - e.frame >= 0 && frame - e.frame < 10);
          const color = reciente ? (reciente.sentido === 'entrada' ? COLOR.verde : COLOR.ambar) : k === 0 ? COLOR.oro : COLOR.plata;
          ctx.save();
          ctx.strokeStyle = color;
          ctx.lineWidth = reciente ? 4 : 2;
          ctx.shadowColor = color;
          ctx.shadowBlur = reciente ? 18 : 6;
          ctx.setLineDash([14, 8]);
          ctx.lineDashOffset = -Date.now() / 40;
          ctx.beginPath();
          ctx.moveTo(x1, y1);
          ctx.lineTo(x2, y2);
          ctx.stroke();
          ctx.restore();
          const c = f.conteo_por_linea[l.id] || { entradas: 0, salidas: 0 };
          chip(ctx, `${l.nombre.toUpperCase()}  ↑ ${c.entradas}  ↓ ${c.salidas}`, Math.max(8, x1 + 8), y1 - 26, color);
        });
        f.personas.forEach(p => {
          estela(ctx, p.trayectoria, COLOR.oro);
          mira(ctx, p.bbox, COLOR.blanco);
          chip(ctx, `#${p.person_id}`, p.bbox[0], Math.max(4, p.bbox[1] - 24), COLOR.blanco);
        });
      }
    },

    3: {
      panel: ['Analítica · Merodeo', 'Personas en la zona', 'YOLO11m + ByteTrack · trayectoria + reidentificación por apariencia'],
      kpis: ['Alertas', 'En escena', 'Máx. en zona'],
      vacio: ['Zona despejada', 'Reproduce el video: cada persona que entre aparece aquí con su comportamiento.'],
      iniciar(s, r) { this.hist = historias(r.frames, 'personas', 'person_id'); },
      marcas: s => s.alertas.map(a => ({ frame: a.frame_alerta, color: COLOR.rojo, titulo: `Merodeo: persona #${a.person_id}` })),
      actualizar(frame, f, s, feed) {
        const alertas = s.alertas.filter(a => a.frame_alerta <= frame);
        alertas.forEach(a => avisar(`m${a.person_id}`, a.mensaje));
        kpi(0, alertas.length);
        kpi(1, f.personas.length);
        kpi(2, `${Math.max(0, ...f.personas.map(p => p.tiempo_en_zona_s)).toFixed(1)} s`);
        banner(f.personas.some(p => p.alerta) ? 'Alerta de merodeo · una persona deambula en el pasillo' : '');
        const personas = s.personas.filter(p => p.primer_frame <= frame);
        if (!personas.length) return vacio(feed, ...this.vacio);
        reconciliarLista(feed, personas, {
          clave: p => p.person_id,
          clase: p => {
            const alerta = alertas.some(a => a.person_id === p.person_id);
            return 'person-item' + (alerta ? ' alerta' : '') + (p.ultimo_frame >= frame ? ' active-tracking' : '');
          },
          fijo: p => `<div class="pi-avatar" style="--sup:${COLOR.oro};--inf:#3a3122"><span>#${p.person_id}</span></div>`,
          datos: p => {
            const h = ultimoHasta(this.hist.get(p.person_id), frame);
            const alerta = alertas.find(a => a.person_id === p.person_id);
            const estado = alerta ? 'merodeo' : h ? h.comportamiento : 'paso_normal';
            const [nombre, color] = ESTADO_MERODEO[estado];
            const enEscena = p.ultimo_frame >= frame;
            return `
              <div class="pi-main">
                <div class="pi-top"><b class="pi-sexo" style="color:${color}">${nombre}</b>
                  <span class="pi-estado ${enEscena ? 'vivo' : ''}">${enEscena ? '<span class="radar-dot"></span>En escena' : 'Salió'}</span></div>
                <div class="garments">
                  <span class="chip-prenda">${h ? h.tiempo_en_zona_s.toFixed(1) : '0.0'} s en la zona</span>
                  ${alerta && alerta.regresos ? `<span class="chip-prenda">Pasó ${alerta.regresos + 1} veces</span>` : ''}
                  ${alerta ? `<span class="chip-prenda">Recorrió ${alerta.recorrido_alturas}× su altura</span>` : ''}
                </div>
              </div>
              <div class="pi-conf"><span>${reloj(p.primer_frame / s.fps)}</span><small>entró</small></div>`;
          }
        });
      },
      dibujar(ctx, frame, f) {
        f.personas.forEach(p => {
          const [nombre, color] = ESTADO_MERODEO[p.comportamiento];
          estela(ctx, p.puntos, color);
          mira(ctx, p.bbox, color, { latido: p.alerta });
          if (p.alerta) radar(ctx, p.bbox[0] + p.bbox[2] / 2, p.bbox[1] + p.bbox[3], p.bbox[2] * 0.6, COLOR.rojo);
          etiqueta(ctx, p.bbox, [`#${p.person_id} · ${nombre}`, `${p.tiempo_en_zona_s.toFixed(1)} s en la zona`], color);
        });
      }
    },

    4: {
      panel: ['Analítica · Objeto abandonado', 'Objetos vigilados', 'YOLO11m + ByteTrack · objeto quieto sin supervisión'],
      kpis: ['Objetos quietos', 'Alertas', 'Sin supervisión'],
      vacio: ['Sin objetos en el piso', 'Cuando alguien deje un bolso, una maleta o un paquete, aparecerá aquí con su dueño.'],
      iniciar(s, r) { this.hist = historias(r.frames, 'objetos', 'objeto_id'); },
      marcas: s => s.alertas.map(a => ({ frame: a.frame_alerta, color: COLOR.rojo, titulo: `Abandono: ${a.tipo_objeto} #${a.objeto_id}` })),
      actualizar(frame, f, s, feed) {
        const alertas = s.alertas.filter(a => a.frame_alerta <= frame);
        alertas.forEach(a => avisar(`a${a.objeto_id}`, a.mensaje));
        const objetos = s.objetos.filter(o => o.primer_frame <= frame);
        kpi(0, objetos.length);
        kpi(1, alertas.length);
        const sin = Math.max(0, ...f.objetos.map(o => o.segundos_sin_supervision));
        kpi(2, `${sin.toFixed(1)} s`);
        banner(f.objetos.some(o => o.alerta) ? 'Objeto abandonado · su dueño lo dejó y se alejó' : '');
        if (!objetos.length) return vacio(feed, ...this.vacio);
        const limite = s.parametros.segundos_alerta;
        reconciliarLista(feed, objetos, {
          clave: o => o.objeto_id,
          clase: o => {
            const h = ultimoHasta(this.hist.get(o.objeto_id), frame);
            return 'person-item' + (h && h.estado === 'abandonado' ? ' alerta' : '');
          },
          fijo: () => `<div class="pi-icono objeto">${ICONO.bolso}</div>`,
          datos: o => {
            const h = ultimoHasta(this.hist.get(o.objeto_id), frame) || { estado: 'en_movimiento', segundos_sin_supervision: 0 };
            const [nombre, color] = ESTADO_OBJETO[h.estado];
            const avance = Math.min(100, Math.round(100 * h.segundos_sin_supervision / limite));
            return `
              <div class="pi-main">
                <div class="pi-top"><b class="pi-sexo">${o.tipo[0].toUpperCase() + o.tipo.slice(1)} #${o.objeto_id}</b>
                  <span class="pi-estado" style="color:${color};border-color:${color}66">${nombre}</span></div>
                <div class="garments"><span class="chip-prenda">Dueño: persona #${o.propietario_id ?? '—'}</span></div>
              </div>
              <div class="pi-conf"><span>${h.segundos_sin_supervision.toFixed(1)}<small> s</small></span><i style="--v:${avance}%"></i><small>sin supervisión</small></div>`;
          }
        });
      },
      dibujar(ctx, frame, f, s) {
        const limite = s.parametros.segundos_alerta;
        f.objetos.forEach(o => {
          const dueno = f.personas.find(p => p.person_id === o.propietario_id);
          if (dueno && o.estado !== 'en_movimiento') {
            ctx.save();
            ctx.strokeStyle = COLOR.oro;
            ctx.setLineDash([6, 6]);
            ctx.globalAlpha = 0.7;
            ctx.beginPath();
            ctx.moveTo(o.bbox[0] + o.bbox[2] / 2, o.bbox[1] + o.bbox[3] / 2);
            ctx.lineTo(dueno.bbox[0] + dueno.bbox[2] / 2, dueno.bbox[1] + dueno.bbox[3] * 0.6);
            ctx.stroke();
            ctx.restore();
          }
        });
        f.personas.forEach(p => {
          const esDueno = f.objetos.some(o => o.propietario_id === p.person_id && o.estado !== 'en_movimiento');
          mira(ctx, p.bbox, esDueno ? COLOR.oro : COLOR.gris, { tenue: !esDueno, grosor: 2 });
          chip(ctx, esDueno ? `Dueño #${p.person_id}` : `#${p.person_id}`, p.bbox[0], Math.max(4, p.bbox[1] - 24), esDueno ? COLOR.oro : COLOR.gris);
        });
        f.objetos.forEach(o => {
          const [nombre, color] = ESTADO_OBJETO[o.estado];
          const cx = o.bbox[0] + o.bbox[2] / 2, cy = o.bbox[1] + o.bbox[3] / 2, r = Math.max(o.bbox[2], o.bbox[3]) * 0.75;
          mira(ctx, o.bbox, color, { latido: o.alerta });
          if (o.alerta) radar(ctx, cx, cy, r, COLOR.rojo);
          else if (o.estado === 'propietario_alejandose') anillo(ctx, cx, cy, r, o.segundos_sin_supervision / limite, COLOR.ambar);
          const detalle = o.estado === 'propietario_alejandose' || o.alerta ? `${o.segundos_sin_supervision.toFixed(1)} s sin supervisión` : null;
          etiqueta(ctx, [o.bbox[0], o.bbox[1] - (o.alerta ? r * 0.6 : 0), o.bbox[2], o.bbox[3]], [`${o.tipo[0].toUpperCase() + o.tipo.slice(1)} · ${nombre}`, ...(detalle ? [detalle] : [])], color);
        });
      }
    },

    5: {
      panel: ['Analítica · Portería vehicular', 'Placas leídas', 'RapidOCR (DBNet + CRNN) · CLAHE nocturno · votación por vehículo'],
      kpis: ['Vehículos', 'Residentes', 'Visitantes'],
      vacio: ['Esperando vehículos', 'Cada vehículo se lee una sola vez y se cruza con el directorio de residentes.'],
      marcas: s => s.lecturas_unicas.map(v => ({ frame: v.frame_lectura, color: v.autorizado ? COLOR.verde : COLOR.oro, titulo: `Placa ${v.placa}` })),
      actualizar(frame, f, s, feed) {
        const leidas = s.lecturas_unicas.filter(v => v.frame_lectura <= frame);
        leidas.forEach(v => avisar(`p${v.vehiculo_id}`, `Placa ${v.placa}: ${v.autorizado ? 'residente del apto ' + v.apartamento : 'visitante'}`, v.autorizado ? 'success' : 'warning'));
        kpi(0, leidas.length);
        kpi(1, leidas.filter(v => v.autorizado).length);
        kpi(2, leidas.filter(v => !v.autorizado).length);
        banner('');
        if (!leidas.length) return vacio(feed, ...this.vacio);
        reconciliarLista(feed, leidas.slice().reverse(), {
          clave: v => v.vehiculo_id,
          clase: v => 'person-item placa-item' + (f.vehiculos.some(x => x.vehiculo_id === v.vehiculo_id) ? ' active-tracking' : ''),
          fijo: v => `<div class="placa-co"><small>${v.tipo_vehiculo === 'moto' ? 'MOTO' : 'COLOMBIA'}</small><b>${v.placa.slice(0, 3)} ${v.placa.slice(3)}</b></div>`,
          datos: v => `
            <div class="pi-main">
              <div class="pi-top"><b class="pi-sexo">${v.autorizado ? `Apto ${v.apartamento} · Torre ${v.torre}` : 'Visitante'}</b>
                <span class="pi-estado ${v.autorizado ? 'vivo' : 'aviso'}">${v.autorizado ? 'Autorizado' : 'No registrado'}</span></div>
              <div class="garments">
                <span class="chip-prenda"><span class="ico">${ICONO[v.tipo_vehiculo === 'moto' ? 'moto' : 'carro']}</span>${v.tipo_vehiculo}</span>
                <span class="chip-prenda">${v.condicion_luz === 'noche' ? '☾ Noche' : '☀ Día'}</span>
                ${v.autorizado ? `<span class="chip-prenda">${v.parqueadero}</span>` : ''}
              </div>
              ${v.autorizado ? `<div class="pi-nota">${v.residente}</div>` : ''}
            </div>
            <div class="pi-conf"><span>${Math.round(v.confianza * 100)}<small>%</small></span><i style="--v:${Math.round(v.confianza * 100)}%"></i><small>${v.frames_observados} lecturas</small></div>`
        });
      },
      dibujar(ctx, frame, f) {
        if (f.condicion_luz === 'noche') chip(ctx, '☾ MODO NOCHE · CLAHE', ctx.canvas.width - 190, 12, COLOR.oro);
        f.vehiculos.forEach(v => {
          const color = v.autorizado ? COLOR.verde : COLOR.oro;
          if (v.bbox_vehiculo) mira(ctx, v.bbox_vehiculo, color, { grosor: 2 });
          const [x, y, w, h] = v.bbox_placa;
          ctx.save();
          ctx.strokeStyle = color;
          ctx.lineWidth = 2;
          ctx.shadowColor = color;
          ctx.shadowBlur = 12;
          caminoRedondo(ctx, x - 4, y - 4, w + 8, h + 8, 5);
          ctx.stroke();
          ctx.restore();
          etiqueta(ctx, v.bbox_vehiculo || v.bbox_placa, [
            `${v.placa} · ${v.tipo_vehiculo}`,
            v.autorizado ? `Apto ${v.apartamento} · ${v.parqueadero}` : 'Visitante no registrado'
          ], color);
        });
      }
    },

    6: {
      panel: ['Analítica · Caídas', 'Personas y postura', 'YOLO11-pose · postura relativa a la altura de pie'],
      kpis: ['Caídas', 'Personas', 'Postura actual'],
      vacio: ['Nadie en la sala', 'Cada persona aparece con su postura. Sentarse o agacharse no genera alerta; una caída sin levantarse sí.'],
      iniciar(s, r) { this.hist = historias(r.frames, 'personas', 'person_id'); },
      marcas: s => s.alertas.map(a => ({ frame: a.frame_alerta, color: COLOR.rojo, titulo: `Caída: persona #${a.person_id}` })),
      actualizar(frame, f, s, feed) {
        const alertas = s.alertas.filter(a => a.frame_alerta <= frame);
        alertas.forEach(a => avisar(`c${a.person_id}`, a.mensaje));
        const personas = s.personas.filter(p => p.primer_frame <= frame);
        kpi(0, alertas.length);
        kpi(1, personas.length);
        kpi(2, f.personas.length ? POSTURA_CORTA[f.personas[0].estado] : '—');
        banner(f.personas.some(p => p.alerta) ? 'Emergencia · una persona se cayó y no se levanta' : '');
        if (!personas.length) return vacio(feed, ...this.vacio);
        reconciliarLista(feed, personas.slice().reverse(), {
          clave: p => p.person_id,
          clase: p => 'person-item' + (alertas.some(a => a.person_id === p.person_id) ? ' alerta' : '') + (p.ultimo_frame >= frame ? ' active-tracking' : ''),
          fijo: p => `<div class="pi-avatar" style="--sup:${COLOR.verde};--inf:#1d3a2c"><span>#${p.person_id}</span></div>`,
          datos: p => {
            const hist = (this.hist.get(p.person_id) || []).filter(h => h.frame <= frame);
            const actual = hist[hist.length - 1];
            const secuencia = hist.map(h => h.estado).filter((e, k, a) => k === 0 || e !== a[k - 1]);
            const [nombre, color] = ESTADO_POSTURA[actual ? actual.estado : 'de_pie'];
            return `
              <div class="pi-main">
                <div class="pi-top"><b class="pi-sexo" style="color:${color}">${nombre}</b>
                  <span class="pi-estado ${p.ultimo_frame >= frame ? 'vivo' : ''}">${p.ultimo_frame >= frame ? '<span class="radar-dot"></span>En escena' : 'Salió'}</span></div>
                <div class="garments secuencia">${secuencia.map(e => `<span class="chip-prenda" style="--c:${ESTADO_POSTURA[e][1]}"><i></i>${ESTADO_POSTURA[e][0]}</span>`).join('<span class="flecha">→</span>')}</div>
              </div>
              <div class="pi-conf"><span>${actual ? Math.round(actual.altura_relativa * 100) : 100}<small>%</small></span><small>de su altura</small></div>`;
          }
        });
      },
      dibujar(ctx, frame, f) {
        f.personas.forEach(p => {
          const [nombre, color] = ESTADO_POSTURA[p.estado];
          const cx = p.bbox[0] + p.bbox[2] / 2, cy = p.bbox[1] + p.bbox[3] / 2;
          mira(ctx, p.bbox, color, { latido: p.alerta });
          if (p.alerta) radar(ctx, cx, cy, Math.max(p.bbox[2], p.bbox[3]) * 0.55, COLOR.rojo);
          else if (p.estado === 'posible_caida_en_observacion') anillo(ctx, cx, cy, Math.max(p.bbox[2], p.bbox[3]) * 0.6, p.segundos_en_suelo / 2, COLOR.naranja);
          const detalle = p.segundos_en_suelo ? `${p.segundos_en_suelo.toFixed(1)} s en el suelo` : `${Math.round(p.altura_relativa * 100)} % de su altura de pie`;
          etiqueta(ctx, p.bbox, [`#${p.person_id} · ${nombre}`, detalle], color);
        });
      }
    },

    7: {
      panel: ['Analítica · Correspondencia', 'Casilleros', 'Cuadrícula + OCR de etiquetas · cambio frente al casillero vacío'],
      kpis: ['Con correo', 'Paquetes', 'Recibos'],
      marcas: s => s.eventos.map(e => ({ frame: e.frame, color: e.tipo_correspondencia === 'paquete' ? COLOR.oro : COLOR.verde, titulo: e.mensaje })),
      actualizar(frame, f, s, feed) {
        s.eventos.filter(e => e.frame <= frame && e.tipo_evento === 'nueva_correspondencia')
          .forEach(e => avisar(`k${e.casillero}${e.frame}`, `Apto ${e.apartamento}: llegó un ${e.tipo_correspondencia}`, 'success'));
        const ocupados = f.casilleros.filter(c => c.ocupado);
        kpi(0, `${ocupados.length} / ${f.casilleros.length}`);
        kpi(1, ocupados.filter(c => c.tipo_correspondencia === 'paquete').length);
        kpi(2, ocupados.filter(c => c.tipo_correspondencia === 'recibo').length);
        banner('');
        feed.classList.add('modo-casilleros');
        reconciliarLista(feed, f.casilleros, {
          clave: c => c.casillero,
          clase: c => 'casillero-tile' + (c.ocupado ? ' ocupado ' + c.tipo_correspondencia : ''),
          fijo: c => `<div class="ct-cabeza"><b>${c.casillero}</b><small>Apto ${c.apartamento}</small></div>`,
          datos: c => {
            const llegada = c.ocupado && s.eventos.filter(e => e.casillero === c.casillero && e.frame <= frame).pop();
            return `
              <div class="ct-icono">${ICONO[c.ocupado ? c.tipo_correspondencia : 'vacio']}</div>
              <div class="ct-estado">${c.ocupado ? (c.tipo_correspondencia === 'paquete' ? 'Paquete' : 'Recibo') : 'Vacío'}</div>
              <small class="ct-nota">${c.ocupado ? `Avisado a ${c.residentes.length} residentes${llegada ? ' · ' + reloj(llegada.tiempo) : ''}` : 'Sin correspondencia'}</small>`;
          }
        });
      },
      dibujar(ctx, frame, f, s) {
        f.casilleros.forEach(c => {
          if (!c.ocupado) {
            ctx.save();
            ctx.strokeStyle = 'rgba(255,255,255,0.18)';
            ctx.setLineDash([4, 5]);
            ctx.strokeRect(...c.bbox);
            ctx.restore();
            return;
          }
          const color = c.tipo_correspondencia === 'paquete' ? COLOR.oro : COLOR.verde;
          const nueva = s.eventos.some(e => e.casillero === c.casillero && e.tipo_evento === 'nueva_correspondencia' && frame - e.frame >= 0 && frame - e.frame < 14);
          mira(ctx, c.bbox, color, { latido: nueva });
          if (c.bbox_objeto) {
            ctx.save();
            ctx.strokeStyle = color;
            ctx.setLineDash([4, 4]);
            ctx.strokeRect(...c.bbox_objeto);
            ctx.restore();
          }
          etiqueta(ctx, c.bbox, [`Apto ${c.apartamento} · ${c.tipo_correspondencia}`], color, { dentro: true });
        });
      }
    }
  };

  // Redibuja el video a ~30 fps mientras se ve el visor (las alertas laten aunque esté en pausa)
  let ultimoDibujo = 0;
  function bucle(t) {
    const visor = document.getElementById('cctvView');
    if (t - ultimoDibujo > 33 && visor && visor.style.display !== 'none' && !document.getElementById('vistaApp').hidden) {
      ultimoDibujo = t;
      if (typeof dibujarOverlay === 'function') dibujarOverlay();
    }
    requestAnimationFrame(bucle);
  }
  requestAnimationFrame(bucle);

  // Clic en una marca de la línea de tiempo: salta un poco antes del evento
  document.addEventListener('click', e => {
    const m = e.target.closest && e.target.closest('.marca');
    if (!m) return;
    const r = datos(currentReto);
    const video = document.getElementById('mainVideo');
    if (r && video) video.currentTime = Math.max(0, Number(m.dataset.frame) / r.fps - 0.6);
  });

  // ------------------------------------------------------------------ API para index.html
  window.VistasRetos = {
    tiene: id => id in RETOS,

    iniciar(id) {
      const vista = RETOS[id], r = datos(id);
      if (!vista || !r) return;
      avisados = new Set();
      ultimoFrame = -1;
      const [kicker, titulo, motor] = vista.panel;
      document.getElementById('panelCabecera').dataset.kicker = kicker;
      document.getElementById('panelTitulo').textContent = titulo;
      document.getElementById('motorIA').innerHTML = `<span>Modelo</span> ${r.summary.motor || motor}`;
      const feed = document.getElementById('feedContainer');
      feed.classList.remove('modo-casilleros');
      feed.innerHTML = '';
      kpis(vista.kpis);
      if (vista.iniciar) vista.iniciar(r.summary, r);
      marcas(vista.marcas(r.summary), r.summary.total_frames_video);
    },

    actualizar(id, frame, f) {
      const vista = RETOS[id], r = datos(id);
      if (!vista || !r || !f) return;
      if (frame < ultimoFrame - 5) avisados = new Set();  // se rebobinó
      ultimoFrame = frame;
      document.getElementById('metaRetoLabel').textContent = reloj(frame / r.fps);
      vista.actualizar(frame, f, r.summary, document.getElementById('feedContainer'));
    },

    dibujar(id, ctx, frame, f) {
      const vista = RETOS[id], r = datos(id);
      if (vista && r && f) vista.dibujar(ctx, frame, f, r.summary);
    },

    // Reto 1 con el mismo estilo de esquinas y etiquetas que los demás
    dibujarReto1(ctx, f) {
      (f.personas || []).forEach(p => {
        const color = p.sexo_aparente === 'Masculino' ? COLOR.blanco : COLOR.oro;
        mira(ctx, p.bbox, color);
        const inf = p.prenda_inferior;
        etiqueta(ctx, p.bbox, [
          `#${p.person_id} · ${p.sexo_aparente} · ${p.rango_edad}`,
          `Sup: ${p.prenda_superior.tipo} ${p.prenda_superior.color}`,
          `Inf: ${inf.tipo}${inf.tipo === 'no visible' ? '' : ' ' + inf.color}`
        ], color);
      });
    },

    marcasReto1() {
      const r = datos(1);
      if (!r) return;
      marcas(r.summary.personas.map(p => ({
        frame: p.primer_frame, color: COLOR.oro, titulo: `Persona #${p.person_id}: ${p.sexo_aparente}, ${p.rango_edad}`
      })), r.summary.total_frames_video);
    }
  };
})();
