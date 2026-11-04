/* Pointer front end: synced video + telemetry + log, and an answer player that
   points at evidence in all three views while it speaks. */

const $ = (s) => document.querySelector(s);
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const DPR = () => window.devicePixelRatio || 1;

const state = {
  meta: null, tele: null, signals: null, log: [], masks: null, status: null,
  active: null,          // current answer segment
  objects: [],           // object ids outlined in the video (persist after the answer for tracking)
  point: null,           // {t_ms, x, y, label, record, frame}
  history: [],
  token: 0,              // increments to abort an answer in progress
  scrubbing: false,
};

/* ------------------------------------------------------------------ charts */

const CHARTS = [
  { id: "force", title: "Finger force", unit: "N", series: [
    { key: "finger_L_cmd_force_N", name: "Left actuator", color: "--left" },
    { key: "contact_L_normal_N", name: "Left contact", color: "--left", dash: [4, 3] },
    { key: "finger_R_cmd_force_N", name: "Right actuator", color: "--right" },
    { key: "contact_R_normal_N", name: "Right contact", color: "--right", dash: [4, 3] },
  ]},
  { id: "temp", title: "Left driver temperature", unit: "°C", refs: [70, 80], series: [
    { key: "finger_L_temp_C", name: "Left driver temperature", color: "--left" },
  ]},
  { id: "aux", unit: "", swappable: true, series: [
    { key: "cube_z", name: "Cube height", color: "--cube" },
  ]},
];
const PADS = { l: 48, r: 12, t: 22, b: 18 };

function buildCharts() {
  const host = $("#charts");
  host.innerHTML = "";
  for (const c of CHARTS) {
    const div = document.createElement("div");
    div.className = "chart";
    div.style.flexGrow = c.id === "force" ? 1.4 : 1;
    div.innerHTML = `<div class="chart-label"></div><canvas></canvas>`;
    host.appendChild(div);
    c.el = div;
    c.canvas = div.querySelector("canvas");
    renderChartLabel(c);
  }
}

function renderChartLabel(c) {
  const lab = c.el.querySelector(".chart-label");
  lab.innerHTML = c.series.map((s) =>
    `<span class="key" data-key="${s.key}"><i style="background:${css(s.color)}"></i>${s.name}</span>`).join("");
}

function seriesRange(c) {
  let lo = Infinity, hi = -Infinity;
  for (const s of c.series) for (const tele of [state.tele, state.ghost]) {
    if (!tele) continue;
    for (const v of tele[s.key]) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  }
  for (const r of c.refs || []) { lo = Math.min(lo, r); hi = Math.max(hi, r); }
  if (lo === hi) { lo -= 1; hi += 1; }
  const step = niceStep((hi - lo) / 3);
  c.step = step;
  return [Math.floor(lo / step) * step, Math.ceil(hi / step) * step];
}

function niceStep(raw) {
  const p = Math.pow(10, Math.floor(Math.log10(raw)));
  const f = raw / p;
  return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10) * p;
}

function chartGeom(c) {
  const w = c.canvas.clientWidth, h = c.canvas.clientHeight;
  const PAD = { ...PADS, t: (c.el.querySelector(".chart-label").offsetHeight || 16) + 8 };
  const [lo, hi] = c.range;
  const dur = state.axisMs || state.meta.duration_ms;
  return {
    w, h, pad: PAD,
    x: (t) => PAD.l + (t / dur) * (w - PAD.l - PAD.r),
    y: (v) => PAD.t + (1 - (v - lo) / (hi - lo)) * (h - PAD.t - PAD.b),
  };
}

function valueAt(key, t) {
  const ts = state.tele.timestamp_ms;
  const i = Math.max(0, Math.min(ts.length - 1, Math.round(t / 10)));
  return state.tele[key][i];
}

function drawChart(c, nowMs) {
  const cv = c.canvas, d = DPR();
  const w = cv.clientWidth, h = cv.clientHeight;
  if (cv.width !== w * d || cv.height !== h * d) { cv.width = w * d; cv.height = h * d; }
  const g = cv.getContext("2d");
  g.setTransform(d, 0, 0, d, 0, 0);
  g.clearRect(0, 0, w, h);
  const G = chartGeom(c);
  const PAD = G.pad;
  const [lo, hi] = c.range;
  const lit = state.active && state.active.signal;

  // grid + axes
  g.font = `11px ${css("--sans")}`;
  g.fillStyle = css("--ink-3");
  g.strokeStyle = css("--rule");
  g.lineWidth = 1;
  const every = G.y(lo) - G.y(lo + c.step) < 16 ? 2 : 1;   // skip labels that would collide
  for (let v = lo, k = 0; v <= hi + c.step / 2; v += c.step, k++) {
    if (k % every) continue;
    const y = Math.round(G.y(v)) + 0.5;
    g.globalAlpha = 0.5;
    g.beginPath(); g.moveTo(PAD.l, y); g.lineTo(w - PAD.r, y); g.stroke();
    g.globalAlpha = 1;
    g.textAlign = "right"; g.textBaseline = "middle";
    g.fillText(fmt(v) + (c.unit ? ` ${c.unit}` : ""), PAD.l - 6, y);
  }
  g.textAlign = "center"; g.textBaseline = "alphabetic";
  for (let s = 0; s <= (state.axisMs || state.meta.duration_ms) / 1000; s++) {
    const x = G.x(s * 1000);
    g.fillText(`${s}s`, x, h - 4);
  }
  for (const r of c.refs || []) {
    const y = Math.round(G.y(r)) + 0.5;
    g.setLineDash([2, 4]); g.strokeStyle = css("--ink-3");
    g.beginPath(); g.moveTo(PAD.l, y); g.lineTo(w - PAD.r, y); g.stroke();
    g.setLineDash([]);
    g.textAlign = "right"; g.fillStyle = css("--ink-3");
    const isWarn = r === c.refs[0] && c.refs.length > 1;
    g.fillText(`${+r.toFixed(1)} °C ${isWarn ? "warning" : "derate"}`, w - PAD.r - 2, isWarn ? y + 11 : y - 4);
    g.textAlign = "center";
  }

  // the window the agent is reading right now
  const scan = state.scan;
  const scanHere = !lit && scan && scan.signals && c.series.some((s) => scan.signals.includes(s.key));
  if (scanHere) {
    g.fillStyle = "rgba(255, 77, 141, 0.10)";
    g.fillRect(G.x(scan.t0), PAD.t - 6, Math.max(2, G.x(scan.t1) - G.x(scan.t0)), h - PAD.b - PAD.t + 6);
  }

  // the other run, faint, for comparison
  if (state.ghost) {
    const gts = state.ghost.timestamp_ms;
    g.globalAlpha = 0.3; g.lineWidth = 1.2; g.setLineDash([3, 3]);
    for (const s of c.series) {
      g.strokeStyle = css(s.color) || css("--ink");
      g.beginPath();
      for (let i = 0; i < gts.length; i += 2) {
        const x = G.x(gts[i]), y = G.y(state.ghost[s.key][i]);
        i ? g.lineTo(x, y) : g.moveTo(x, y);
      }
      g.stroke();
    }
    g.setLineDash([]); g.globalAlpha = 1;
  }

  // series
  const ts = state.tele.timestamp_ms;
  for (const s of c.series) {
    const hot = lit ? lit === s.key : !!(scanHere && scan.signals.includes(s.key));
    const on = (!lit && !scanHere) || hot;
    g.strokeStyle = css(s.color) || css("--ink");
    g.globalAlpha = on ? 1 : 0.28;
    g.lineWidth = hot ? 2.4 : 1.5;
    g.setLineDash(s.dash || []);
    g.beginPath();
    for (let i = 0; i < ts.length; i += 2) {
      const x = G.x(ts[i]), y = G.y(state.tele[s.key][i]);
      i ? g.lineTo(x, y) : g.moveTo(x, y);
    }
    g.stroke();
    g.setLineDash([]);
  }
  g.globalAlpha = 1;

  // playhead
  const px = Math.round(G.x(nowMs)) + 0.5;
  g.strokeStyle = css("--ink-2"); g.lineWidth = 1;
  g.beginPath(); g.moveTo(px, PAD.t - 6); g.lineTo(px, h - PAD.b); g.stroke();

  // Pointer's cursor
  if (state.active) {
    const t = state.active.time_ms;
    const x = G.x(t);
    g.strokeStyle = css("--laser"); g.lineWidth = 2;
    g.beginPath(); g.moveTo(x, PAD.t - 6); g.lineTo(x, h - PAD.b); g.stroke();
    const s = c.series.find((s) => s.key === lit);
    if (s) {
      const y = G.y(valueAt(s.key, t));
      g.fillStyle = css("--laser");
      g.beginPath(); g.arc(x, y, 4.5, 0, Math.PI * 2); g.fill();
      g.font = `600 12px ${css("--sans")}`;
      g.textAlign = x > w - 110 ? "right" : "left";
      g.fillText(`${fmt(valueAt(s.key, t))}${c.unit ? " " + c.unit : ""}`, x + (g.textAlign === "left" ? 8 : -8), y - 8);
    }
  }
  c.el.querySelectorAll(".key").forEach((k) => k.classList.toggle("lit", k.dataset.key === lit));
}

function fmt(v) {
  const a = Math.abs(v);
  return a >= 100 ? v.toFixed(0) : a >= 10 ? v.toFixed(1) : a >= 1 ? +v.toFixed(1) + "" : v === 0 ? "0" : +v.toFixed(2) + "";
}

function chartAnchor(signal) {
  for (const c of CHARTS) {
    if (!c.series.some((s) => s.key === signal)) continue;
    const G = chartGeom(c), r = c.canvas.getBoundingClientRect();
    const t = state.active.time_ms;
    return { x: r.left + G.x(t), y: r.top + G.y(valueAt(signal, t)) };
  }
  return null;
}

function ensureSignalShown(signal) {
  if (!signal || signal === "none") return;
  if (CHARTS.some((c) => c.series.some((s) => s.key === signal))) return;
  const aux = CHARTS.find((c) => c.swappable);
  aux.series = [{ key: signal, name: niceName(signal), color: "--ink" }];
  aux.unit = (signal.match(/_(N|C|mm)$/) || [])[1]?.replace("C", "°C") || (signal.endsWith("_z") ? "m" : "");
  aux.range = seriesRange(aux);
  renderChartLabel(aux);
}

function niceName(key) {
  const d = state.signals[key] || key;
  return d.charAt(0).toUpperCase() + d.slice(1);
}

/* ------------------------------------------------------------------ video overlay */

const video = $("#video");
const overlay = $("#overlay");

function frameIdx() {
  return Math.min(state.meta.n_frames - 1, Math.round(video.currentTime * state.meta.fps));
}

function videoRect() {
  const wrap = $("#video-wrap").getBoundingClientRect();
  const v = video.getBoundingClientRect();
  return { left: v.left - wrap.left, top: v.top - wrap.top, width: v.width, height: v.height, abs: v };
}

function drawOverlay() {
  const r = videoRect(), d = DPR();
  Object.assign(overlay.style, { left: r.left + "px", top: r.top + "px", width: r.width + "px", height: r.height + "px" });
  if (overlay.width !== Math.round(r.width * d)) { overlay.width = Math.round(r.width * d); overlay.height = Math.round(r.height * d); }
  const g = overlay.getContext("2d");
  const k = r.width / state.meta.width;
  g.setTransform(d * k, 0, 0, d * k, 0, 0);
  g.clearRect(0, 0, state.meta.width, state.meta.height);
  const fi = frameIdx();
  const frame = state.masks.frames[String(fi)] || {};
  const laser = css("--laser");

  for (const oid of state.objects) {
    const rec = frame[oid];
    if (!rec || !rec.polygons) continue;
    outline(g, rec.polygons, laser, k);
    tag(g, rec.bbox, state.meta.objects[oid], laser, k);
  }
  const sv = state.segView;
  if (sv) {
    const col = css(`--a-${sv.agent}`) || css("--ink");
    for (const it of sv.items) {
      if (Math.abs(it.frame - fi) > 1) continue;
      for (const rec of it.records) {
        g.save();
        g.strokeStyle = col; g.lineWidth = 2 / k; g.setLineDash([5 / k, 3 / k]);
        g.fillStyle = col + "33";
        for (const poly of rec.polygons) {
          g.beginPath(); poly.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y))); g.closePath(); g.fill(); g.stroke();
        }
        g.restore();
        tag(g, rec.bbox, `${it.source === "sam" ? "SAM 3.1" : "Sim"}: ${it.prompt}`, col, k, "#1d2633");
      }
    }
  }
  const p = state.point;
  if (p) {
    if (p.record && p.record.polygons && p.frame === fi) {
      outline(g, p.record.polygons, laser, k);
      tag(g, p.record.bbox, p.label || "SAM 3.1", laser, k);
    } else if (Math.abs(p.t_ms - video.currentTime * 1000) < 60) {
      g.strokeStyle = laser; g.lineWidth = 2.5 / k;
      g.beginPath(); g.arc(p.x, p.y, 14 / k, 0, Math.PI * 2); g.stroke();
      g.fillStyle = laser; g.beginPath(); g.arc(p.x, p.y, 3 / k, 0, Math.PI * 2); g.fill();
      if (p.label) tag(g, [p.x + 14 / k, p.y - 14 / k], p.label, laser, k);
    }
  }
}

function outline(g, polys, color, k) {
  g.fillStyle = "rgba(255, 77, 141, 0.22)";
  g.strokeStyle = color;
  g.lineWidth = 2.2 / k;
  g.lineJoin = "round";
  g.shadowColor = "rgba(255, 77, 141, 0.8)";
  g.shadowBlur = 8;
  for (const poly of polys) {
    g.beginPath();
    poly.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y)));
    g.closePath(); g.fill(); g.stroke();
  }
  g.shadowBlur = 0;
}

function tag(g, bbox, text, color, k, ink = "#fff") {
  const fs = 12.5 / k;
  g.font = `600 ${fs}px ${css("--sans")}`;
  const tw = g.measureText(text).width;
  const x = bbox[0], y = Math.max(fs + 6 / k, bbox[1] - 6 / k);
  g.fillStyle = color;
  roundRect(g, x - 1 / k, y - fs - 3 / k, tw + 12 / k, fs + 7 / k, 4 / k);
  g.fill();
  g.fillStyle = ink;
  g.fillText(text, x + 5 / k, y);
}

function roundRect(g, x, y, w, h, r) {
  g.beginPath();
  g.moveTo(x + r, y); g.arcTo(x + w, y, x + w, y + h, r); g.arcTo(x + w, y + h, x, y + h, r);
  g.arcTo(x, y + h, x, y, r); g.arcTo(x, y, x + w, y, r); g.closePath();
}

function videoAnchor() {
  const r = videoRect().abs, k = r.width / state.meta.width;
  const fi = frameIdx();
  const p = state.point;
  if (p && p.x >= 0) {
    const c = p.record && p.frame === fi ? p.record.centroid : [p.x, p.y];
    return { x: r.left + c[0] * k, y: r.top + c[1] * k };
  }
  const frame = state.masks.frames[String(fi)] || {};
  for (const oid of state.objects) {
    const rec = frame[oid];
    if (rec && rec.centroid) return { x: r.left + rec.centroid[0] * k, y: r.top + rec.centroid[1] * k };
  }
  return null;
}

/* ------------------------------------------------------------------ timeline */

const tl = $("#timeline");

function drawTimeline() {
  const d = DPR(), w = tl.clientWidth, h = tl.clientHeight;
  if (tl.width !== w * d) { tl.width = w * d; tl.height = h * d; }
  const g = tl.getContext("2d");
  g.setTransform(d, 0, 0, d, 0, 0);
  g.clearRect(0, 0, w, h);
  const dur = state.meta.duration_ms;
  const X = (t) => 6 + (t / dur) * (w - 12);
  const mid = h / 2;
  g.strokeStyle = css("--rule"); g.lineWidth = 4; g.lineCap = "round";
  g.beginPath(); g.moveTo(X(0), mid); g.lineTo(X(dur), mid); g.stroke();
  const now = video.currentTime * 1000;
  g.strokeStyle = css("--ink-3");
  g.beginPath(); g.moveTo(X(0), mid); g.lineTo(X(now), mid); g.stroke();
  const rev = revealMs();
  for (const l of state.log) {
    if (l.level !== "WARN" && l.level !== "ERROR") continue;
    if (l.t_ms > rev) continue;
    g.fillStyle = l.level === "ERROR" ? css("--error") : css("--warn");
    g.fillRect(X(l.t_ms) - 1, mid - 9, 2, 6);
  }
  if (state.active) {
    g.fillStyle = css("--laser");
    const x = X(state.active.time_ms);
    g.beginPath(); g.moveTo(x - 5, mid + 12); g.lineTo(x + 5, mid + 12); g.lineTo(x, mid + 5); g.fill();
  }
  g.fillStyle = css("--ink");
  g.beginPath(); g.arc(X(now), mid, 6, 0, Math.PI * 2); g.fill();
}

function seekFromEvent(e) {
  const r = tl.getBoundingClientRect();
  const f = Math.min(1, Math.max(0, (e.clientX - r.left - 6) / (r.width - 12)));
  video.currentTime = (f * state.meta.duration_ms) / 1000;
}

/* ------------------------------------------------------------------ log */

function buildLog() {
  const ol = $("#log");
  ol.innerHTML = state.log.map((l) =>
    `<li class="${l.level}" data-n="${l.n}"><span class="n">${l.n}</span>` +
    `<span class="ts">${l.t_ms === null ? "" : (l.t_ms / 1000).toFixed(2)}</span>` +
    `<span class="lv">${l.level}</span><span class="src">${escapeHtml(l.src)}</span>` +
    `<span class="msg">${escapeHtml(l.msg)}</span></li>`).join("");
  lastReveal = { id: null, ms: -1 };
  [...ol.children].forEach((li) => li.classList.add("unplayed"));
  if (!ol.dataset.bound) { ol.dataset.bound = "1"; ol.addEventListener("scroll", () => drawThread(false)); }
}

/* The log fills in as the video plays, like a live terminal. Lines stay once shown, the whole
   log stays after one full play-through, and it opens fully when the agents start reading it. */
const revealed = {};           // episode id -> furthest time shown (ms); Infinity = whole log
let lastReveal = { id: null, ms: -1 };

function revealMs() { return revealed[state.meta.episode_id] ?? 0; }

function revealAll() { revealed[state.meta.episode_id] = Infinity; }

function advanceReveal(nowMs) {
  const id = state.meta.episode_id;
  const done = video.ended || nowMs >= state.meta.duration_ms - 40;
  revealed[id] = done ? Infinity : Math.max(revealed[id] ?? 0, nowMs);
}

function updateLogTime(nowMs) {
  advanceReveal(nowMs);
  const rev = revealMs(), ol = $("#log");
  const changed = lastReveal.id !== state.meta.episode_id || lastReveal.ms !== rev;
  const nearBottom = ol.scrollHeight - ol.scrollTop - ol.clientHeight < 40;
  let shown = 0, newest = null;
  for (const li of ol.children) {
    const l = state.log[+li.dataset.n - 1];
    if (!l) continue;
    const hidden = l.t_ms !== null && l.t_ms > rev;
    if (changed) {
      if (!hidden && li.classList.contains("unplayed")) { li.classList.add("fresh"); newest = li; }
      li.classList.toggle("unplayed", hidden);
    }
    if (!hidden) shown++;
    li.classList.toggle("future", !hidden && l.t_ms !== null && l.t_ms > nowMs + 1);
  }
  if (!changed) return;
  lastReveal = { id: state.meta.episode_id, ms: rev };
  $("#log-count").textContent = rev === Infinity ? `${state.log.length} lines, times in seconds`
    : `${shown} of ${state.log.length} lines so far, times in seconds`;
  if (newest && nearBottom && !state.active) ol.scrollTop = ol.scrollHeight;
}

function litLog(n) {
  const ol = $("#log");
  ol.querySelectorAll(".lit").forEach((li) => li.classList.remove("lit"));
  if (n < 1) return null;
  const li = ol.querySelector(`li[data-n="${n}"]`);
  if (li.classList.contains("unplayed")) { revealAll(); updateLogTime(video.currentTime * 1000); }
  li.classList.add("lit");
  ol.scrollTop = li.offsetTop - ol.clientHeight / 2 + li.clientHeight / 2;
  return li;
}

function logAnchor() {
  const li = $("#log li.lit");
  if (!li) return null;
  const r = li.getBoundingClientRect(), ol = $("#log").getBoundingClientRect();
  const y = Math.min(Math.max(r.top + r.height / 2, ol.top + 6), ol.bottom - 6);
  const ts = li.querySelector(".ts").getBoundingClientRect();
  return { x: Math.max(ts.left - 4, ol.left + 4), y };
}

function escapeHtml(s) {
  return s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
}

/* ------------------------------------------------------------------ laser thread */

let threadAnim = 0;

function drawThread(animate) {
  const svg = $("#thread");
  if (!state.active) { svg.innerHTML = ""; return; }
  const a = videoAnchor();
  const b = state.active.signal !== "none" ? chartAnchor(state.active.signal) : null;
  const c = logAnchor();
  const parts = [];
  const curve = (p, q) => {
    const dx = (q.x - p.x) * 0.5;
    return `M${p.x},${p.y} C${p.x + dx},${p.y} ${q.x - dx},${q.y} ${q.x},${q.y}`;
  };
  const curveV = (p, q) => {
    const dy = (q.y - p.y) * 0.55;
    return `M${p.x},${p.y} C${p.x},${p.y + dy} ${q.x},${q.y - dy} ${q.x},${q.y}`;
  };
  if (a && b) parts.push(curve(a, b));
  if (a && c) parts.push(curveV(a, c));
  else if (!a && b && c) parts.push(curve(c, b));
  const dots = [a, b, c].filter(Boolean);
  svg.innerHTML = parts.map((d) => `<path d="${d}"/>`).join("") +
    dots.map((p) => `<circle cx="${p.x}" cy="${p.y}" r="3.5"/>`).join("");
  if (animate && !matchMedia("(prefers-reduced-motion: reduce)").matches) {
    const id = ++threadAnim;
    svg.querySelectorAll("path").forEach((p) => {
      const L = p.getTotalLength();
      p.style.strokeDasharray = L;
      p.style.strokeDashoffset = L;
      p.animate([{ strokeDashoffset: L }, { strokeDashoffset: 0 }], { duration: 650, easing: "cubic-bezier(.3,.7,.2,1)", fill: "forwards" })
        .finished.then(() => { if (id === threadAnim) p.style.strokeDasharray = "none"; });
    });
  }
}

/* ------------------------------------------------------------------ render loop */

function frame() {
  const now = video.currentTime * 1000;
  for (const c of CHARTS) drawChart(c, now);
  drawOverlay();
  drawTimeline();
  $("#clock").textContent = `${(now / 1000).toFixed(2)} s`;
  updateLogTime(now);
  if (!video.paused) drawThread(false);
  requestAnimationFrame(frame);
}

/* ------------------------------------------------------------------ answers */

function setActive(seg) {
  state.active = seg;
  if (!seg) { litLog(-1); drawThread(false); return; }
  if (seg.objects.length) state.objects = seg.objects;
  ensureSignalShown(seg.signal);
  litLog(seg.log_line);
  if (seg.point && seg.point.x >= 0) {
    state.point = { ...seg.point, t_ms: seg.time_ms, record: null, frame: null };
    refine(state.point);
  } else {
    state.point = null;
  }
}

async function refine(p) {
  try {
    const r = await postJSON("/api/refine", { t_ms: p.t_ms, x: p.x, y: p.y });
    if (state.point === p && r.record && r.record.polygons) {
      p.record = r.record; p.frame = r.frame;
      drawThread(false);
    }
  } catch (e) { console.warn("refine failed", e); }
}

function seekTo(ms) {
  return new Promise((res) => {
    const t = ms / 1000;
    if (Math.abs(video.currentTime - t) < 0.01) return res();
    video.addEventListener("seeked", () => res(), { once: true });
    video.currentTime = t;
  });
}

let voice = null;
function pickVoice() {
  const vs = speechSynthesis.getVoices();
  voice = vs.find((v) => /Samantha|Ava|Google US English|Microsoft Aria/i.test(v.name)) ||
          vs.find((v) => v.lang === "en-US") || vs[0] || null;
}
speechSynthesis.onvoiceschanged = pickVoice;
pickVoice();

function speak(text, token, onWord) {
  return new Promise((res) => {
    if (!("speechSynthesis" in window) || token !== state.token) { onWord && onWord(Infinity); return res(); }
    const u = new SpeechSynthesisUtterance(text);
    if (voice) u.voice = voice;
    u.rate = 1.04;
    const starts = [...text.matchAll(/\S+/g)].map((m) => m.index);
    let gotBoundary = false, timer = null, shown = -1;
    const show = (i) => { if (onWord && i > shown) { shown = i; onWord(i); } };
    u.onboundary = (e) => {
      if (e.name && e.name !== "word") return;
      gotBoundary = true; clearInterval(timer);
      let i = 0; while (i + 1 < starts.length && starts[i + 1] <= e.charIndex) i++;
      show(i);
    };
    u.onstart = () => {
      show(0);
      setTimeout(() => {               // some voices never send word boundaries: estimate instead
        if (gotBoundary) return;
        const t0 = performance.now();
        timer = setInterval(() => show(Math.floor(((performance.now() - t0) / 1000) * 2.7) + 1), 120);
      }, 600);
    };
    const fallback = setTimeout(done, starts.length * 420 + 1800);
    function done() { clearTimeout(fallback); clearInterval(timer); show(Infinity); res(); }
    u.onend = u.onerror = done;
    speechSynthesis.speak(u);
  });
}

const UNIT = /^(newtons?|degrees?|seconds?|milliseconds?|millimeters?|mm|ms|N|°C|C|kg|percent|%)[.,;]?$/i;

function wordsHTML(text) {
  const toks = text.split(/\s+/).filter(Boolean);
  return toks.map((t, i) => {
    const num = /\d/.test(t) || (i > 0 && /\d/.test(toks[i - 1]) && UNIT.test(t));
    return `<span class="w${num ? " num" : ""}" data-w="${i}">${escapeHtml(t)}</span>`;
  }).join(" ");
}

function revealWords(el, upTo) {
  el.querySelectorAll(".w").forEach((w) => w.classList.toggle("on", +w.dataset.w <= upTo));
}

function renderAnswer(q, ans) {
  const box = $("#answer");
  const tk = ans.takeaway || {};
  const timing = ans.cached
    ? `replayed from an earlier board session (${(ans.total_ms / 1000).toFixed(0)} s live)`
    : `board session ${(ans.total_ms / 1000).toFixed(0)} s`;
  box.innerHTML =
    `<p class="q">${escapeHtml(q)}</p>` + verdictBanner(ans) +
    `<ol class="steps">` + ans.segments.map((s, i) =>
      `<li class="step" data-i="${i}"><button type="button" class="t" title="Show this moment">${(s.time_ms / 1000).toFixed(2)} s</button>` +
      `<span class="say">${wordsHTML(s.say)}</span></li>`).join("") + `</ol>` +
    citedHTML(ans) +
    (tk.root_cause ? `<div class="takeaway" id="takeaway"><h3>Takeaway</h3><p class="cause">${numHTML(tk.root_cause)}</p>` +
      (tk.fixes && tk.fixes.length ? `<h3>Fix</h3><ul class="fixes">${tk.fixes.map((f) => `<li>${numHTML(f)}</li>`).join("")}</ul>` : "") +
      testedHTML(ans) + `</div>` : "") +
    `<div class="answer-foot"><span class="meta">${ans.model}, ${timing}</span>` +
    `<button type="button" class="replay" id="replay">Replay answer</button></div>` +
    suggestHTML("Ask next");
  box.scrollTop = 0;
}

function numHTML(text) {
  return wordsHTML(text).replace(/class="w"/g, 'class="w on"').replace(/class="w num"/g, 'class="w num on"');
}

function citedHTML(ans) {
  const logs = [...new Set(ans.segments.map((s) => s.log_line).filter((n) => n > 0))].sort((a, b) => a - b);
  const sigs = [...new Set(ans.segments.map((s) => s.signal).filter((k) => k && k !== "none"))];
  const objs = [...new Set(ans.segments.flatMap((s) => s.objects))].map((o) => state.meta.objects[o]);
  const pts = ans.segments.filter((s) => s.point && s.point.x >= 0).map((s) => s.point.label || "a point");
  const chips = [
    ...logs.map((n) => `<span class="chip-ev">log line ${n}</span>`),
    ...sigs.map((k) => `<span class="chip-ev">${escapeHtml(niceName(k))}</span>`),
    ...objs.map((o) => `<span class="chip-ev">${escapeHtml(o)}, ${(state.eps.orig.masks.source || "").startsWith("sam") ? "SAM 3.1 track" : "simulator mask"}</span>`),
    ...pts.map((l) => `<span class="chip-ev">${escapeHtml(l)}, segmented by SAM 3.1</span>`),
  ];
  return chips.length ? `<div class="cited"><h3>Evidence Muse Spark pointed at</h3>${chips.join("")}</div>` : "";
}

const PATCH_LABELS = {
  hold_force_N: ["Hold force", " N"],
  speed_scale: ["Speed", "×"],
  transport_height_cm: ["Carry height", " cm"],
};

function patchText(params, defaults) {
  const rows = Object.entries(params).filter(([k, v]) => defaults[k] !== v && PATCH_LABELS[k])
    .map(([k, v]) => `${PATCH_LABELS[k][0]} <span class="num">${defaults[k]}${PATCH_LABELS[k][1]}</span> to <span class="num">${v}${PATCH_LABELS[k][1]}</span>`);
  return rows.join(", ") || "no change";
}

async function loadView(v) {
  const ep = state.eps[v];
  if (!ep) return;
  state.view = v;
  state.scan = null; state.segView = null;
  const other = state.eps[v === "orig" ? "fixed" : "orig"];
  Object.assign(state, { meta: ep.meta, tele: ep.telemetry, log: ep.log, masks: ep.masks, ghost: other ? other.telemetry : null });
  state.axisMs = Math.max(ep.meta.duration_ms, other ? other.meta.duration_ms : 0);
  state.active = null; state.point = null;
  drawThread(false);
  if (!video.src.endsWith(ep.video_url)) {
    video.src = ep.video_url;
    await new Promise((r) => video.addEventListener("loadeddata", r, { once: true }));
  }
  const tc = CHARTS.find((c) => c.id === "temp");
  tc.refs = [ep.meta.driver_soft_C ?? 70, ep.meta.driver_limit_C ?? 80];
  for (const c of CHARTS) c.range = seriesRange(c);
  buildLog();
  litLog(-1);
  const ok = ep.meta.outcome.startsWith("OK");
  const kind = v === "fixed" && ep.verdict ? ep.verdict.label + " fix: " : "";
  $("#ep-outcome").textContent = kind + (ok ? ep.meta.outcome.replace(/^OK: /, "") : ep.meta.outcome.replace(/^FAIL: /, "failed, "));
  $("#ep-outcome").classList.toggle("ok", ok && (!ep.verdict || ep.verdict.kind === "ok"));
  $("#ep-outcome").classList.toggle("warn", ok && ep.verdict && ep.verdict.kind === "warn");
  const fromSam = ep.masks.source && ep.masks.source.startsWith("sam");
  $("#mask-source").textContent = fromSam ? "Objects tracked by SAM 3.1" : "Object masks from simulator ground truth";
  document.querySelectorAll("#runs button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.v === v)));
  $("#ghost-note").textContent = other ? `Dashed: ${v === "orig" ? "fixed" : "original"} run` : "100 Hz";
}

function suggestHTML(title) {
  return `<div class="suggest"><h3>${title}</h3>` +
    state.suggestions.map((s) => `<button type="button" class="sq">${escapeHtml(s)}</button>`).join("") + `</div>`;
}

function showIdle() {
  boardIdle();
  findingsIdle();
  setTab("board");
}

function markStep(i) {
  const box = $("#answer");
  box.querySelectorAll(".step").forEach((el, j) => {
    el.classList.toggle("now", j === i);
    el.classList.toggle("done", i === -1 || j < i);
  });
  const tk = $("#takeaway");
  if (tk) tk.classList.toggle("now", i === "takeaway");
  const cur = i === "takeaway" ? tk : box.querySelector(`.step[data-i="${i}"]`);
  if (cur) {
    const top = cur.offsetTop - box.offsetTop;
    if (top < box.scrollTop || top + cur.offsetHeight > box.scrollTop + box.clientHeight)
      box.scrollTop = Math.max(0, top - 8);
  }
}

async function showStep(seg, token, animate = true) {
  setActive(seg);
  await seekTo(seg.time_ms);
  if (token !== state.token) return;
  drawThread(animate);
}

async function playAnswer(q, ans, token, render = true) {
  state.answer = { q, ans };
  if (state.view !== "orig") await loadView("orig");
  video.pause();
  for (let i = 0; i < ans.segments.length; i++) {
    if (token !== state.token) return;
    markStep(i);
    await showStep(ans.segments[i], token);
    const say = $(`.step[data-i="${i}"] .say`);
    await speak(ans.segments[i].say, token, (w) => say && revealWords(say, w));
    await sleep(250);
  }
  if (token !== state.token) return;
  const tk = ans.takeaway || {};
  if (tk.say || tk.root_cause) {
    state.active = null;      // keep the objects outlined, drop the cursor and thread
    litLog(-1);
    drawThread(false);
    markStep("takeaway");
    await speak(tk.say || tk.root_cause, token);
  }
  if (token === state.token) markStep(-1);
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const AGENT = {};   // id -> {name, short}
const PHASES = [["investigate", "Investigate"], ["cross_examine", "Cross-examine"], ["experiments", "Experiments"], ["decision", "Decision"]];
const STATUS_LABEL = { plausible: "Plausible", needs_test: "Needs a test", refuted: "Refuted" };

function agentBadge(id) {
  const a = AGENT[id] || { name: id, short: "?" };
  return `<span class="av av-${id}" aria-hidden="true">${a.short}</span>`;
}

function boardIdle() {
  $("#board").innerHTML = `<p class="empty">Ask why the run failed. Three investigators look into it at the same time, ` +
    `each from a different angle: hardware, grasp physics, and the environment. A skeptic cross-examines their ` +
    `hypotheses, and MuJoCo tests each proposed fix. The simulation decides who was right.</p>` + suggestHTML("Try asking");
}

function findingsIdle() {
  $("#answer").innerHTML = `<p class="empty">Findings appear here when the board reaches a decision: the explanation, ` +
    `pointed out on screen, the root cause, and the fixes that were tested.</p>`;
}

function setTab(t) {
  document.querySelectorAll(".tabs [role=tab]").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === t)));
  $("#board").hidden = t !== "board";
  $("#answer").hidden = t !== "findings";
  drawThread(false);
}

function startBoard(q, agents) {
  agents.forEach((a) => (AGENT[a.id] = a));
  const inv = agents.filter((a) => !["skeptic", "lead"].includes(a.id));
  $("#board").innerHTML =
    `<p class="q">${escapeHtml(q)}</p>` +
    `<div class="board-head"><ol class="phases" id="phases">${PHASES.map(([id, label]) => `<li data-p="${id}">${label}</li>`).join("")}</ol>` +
    `<ul class="roster" id="roster">${[...inv, AGENT.skeptic].map((a) =>
      `<li data-a="${a.id}" title="${a.name}">${agentBadge(a.id)}<span class="nm">${a.name}</span><span class="st" id="st-${a.id}">${a.id === "skeptic" ? "waiting" : "starting"}</span></li>`).join("")}</ul></div>` +
    `<ol class="feed" id="feed"></ol>` +
    `<p class="thinking" id="board-status">Investigating <span id="elapsed">0 s</span></p>`;
  $("#board-live").hidden = false;
  const t0 = performance.now();
  const id = setInterval(() => {
    const el = $("#elapsed");
    if (!el) return clearInterval(id);
    el.textContent = `${Math.round((performance.now() - t0) / 1000)} s`;
  }, 500);
  return () => { clearInterval(id); $("#board-live").hidden = true; };
}

function post(html, cls = "") {
  const feed = $("#feed"), box = $("#board");
  const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 60;
  const li = document.createElement("li");
  li.className = `post ${cls}`;
  li.innerHTML = html;
  feed.appendChild(li);
  if (atBottom) box.scrollTop = box.scrollHeight;
  return li;
}

function setStatus(agent, text, cls = "") {
  const el = $(`#st-${agent}`);
  if (el) { el.textContent = text; el.className = `st ${cls}`; }
}

function setPhase(p) {
  const order = PHASES.map(([id]) => id);
  document.querySelectorAll("#phases li").forEach((li) => {
    const i = order.indexOf(li.dataset.p), j = order.indexOf(p);
    li.className = i < j ? "done" : i === j ? "now" : "";
  });
  const label = { investigate: "Investigating", cross_examine: "The skeptic is cross-examining",
                  experiments: "Running experiments in MuJoCo", decision: "Writing the findings" }[p];
  const st = $("#board-status");
  if (st && label) st.firstChild.textContent = label + " ";
}

function pointAt(ui, toolName, agent) {
  state.active = null; drawThread(false);
  if (ui.masks) state.segView = { agent, items: ui.masks };   // stays until the next segmentation
  document.querySelectorAll("#log li.read").forEach((x) => x.classList.remove("read"));
  if (ui.lines && ui.lines.length) {
    ui.lines.forEach((n) => { const x = $(`#log li[data-n="${n}"]`); if (x) x.classList.add("read"); });
    const first = $(`#log li[data-n="${ui.lines[0]}"]`), ol = $("#log");
    if (first) ol.scrollTop = first.offsetTop - 8;
  }
  if (ui.signals) { ui.signals.forEach(ensureSignalShown); state.scan = { signals: ui.signals, t0: ui.t0, t1: ui.t1 }; }
  if (ui.times && ui.times.length && (ui.masks || !state.segView)) { video.pause(); seekTo(ui.times[ui.times.length - 1]); }
}

const TOOL_STATUS = { read_log: "reading the log", read_telemetry: "reading telemetry", view_frames: "looking at frames",
                      segment: "segmenting with SAM 3.1", finish: "forming a hypothesis" };

function onAgentStep(e) {
  post(`${agentBadge(e.agent)}<div class="pb"><p class="thought">${numHTML(e.thought)}</p>` +
       `<p class="act"><span class="tool">${escapeHtml(e.label)}</span>${e.summary ? ` <span class="res">${escapeHtml(e.summary)}</span>` : ""}</p></div>`,
       `step-post a-${e.agent}`);
  setStatus(e.agent, TOOL_STATUS[e.tool] || "working");
  if (e.tool !== "finish") pointAt(e.ui || {}, e.tool, e.agent);
}

function timeChip(ms) {
  return `<button type="button" class="tchip" data-t="${ms}">${(ms / 1000).toFixed(2)} s</button>`;
}

function onHypothesis(e) {
  state.hyps[e.agent] = e;
  const pct = Math.round(e.confidence * 100);
  post(`${agentBadge(e.agent)}<div class="pb card hyp a-${e.agent}" id="hyp-${e.agent}">` +
       `<p class="kind">${AGENT[e.agent].name} hypothesis <span class="conf"><i style="width:${pct}%"></i></span> <span class="pct">${pct}%</span></p>` +
       `<p class="claim">${numHTML(e.claim)}</p>` +
       (e.evidence.length ? `<ul class="evid">${e.evidence.map((x) => `<li>${timeChip(x.time_ms)} ${numHTML(x.text)}${x.log_line > 0 ? ` <span class="ln">log line ${x.log_line}</span>` : ""}</li>`).join("")}</ul>` : "") +
       `<p class="exp">${e.patch.changed ? `Proposed test: ${patchText(e.patch, e.defaults)}` : "No change proposed"}</p>` +
       `<p class="badge-row" id="hb-${e.agent}"></p></div>`, `hyp-post`);
  setStatus(e.agent, `hypothesis, ${pct}%`);
}

function onChallenge(e) {
  post(`${agentBadge("skeptic")}<div class="pb card chal">` +
       `<p class="kind">To ${AGENT[e.target].name} <span class="stat s-${e.status}">${STATUS_LABEL[e.status]}</span>` +
       `${e.guardrail ? ` <span class="stat s-guard" title="Code-enforced rule: evidence after the failure cannot be the cause">${e.overrode ? "Guardrail overrode" : "Guardrail"}</span>` : ""}</p>` +
       `<p class="txt">${numHTML(e.challenge || "No objection.")}</p></div>`, "chal-post");
  setStatus("skeptic", "cross-examining");
  const hb = $(`#hb-${e.target}`);
  if (hb) hb.innerHTML = `<span class="stat s-${e.status}">Skeptic: ${STATUS_LABEL[e.status]}</span>`;
  if (e.status === "refuted") setStatus(e.target, "refuted by the skeptic", "bad");
}

function onExperimentStart(e) {
  const li = post(`${agentBadge(e.agent)}<div class="pb exp-line running" id="exp-${e.agent}-${e.attempt}">` +
       `<p><b>Simulating ${AGENT[e.agent].name}'s fix${e.attempt > 1 ? `, attempt ${e.attempt}` : ""}:</b> ${patchText(e.params, e.defaults)}</p>` +
       `<p class="v">Re-running the episode in MuJoCo</p></div>`, "exp-post");
  setStatus(e.agent, "in the simulator");
  return li;
}

function onExperimentResult(e) {
  state.fixRuns[e.run.meta.episode_id] = e.run;
  const box = $(`#exp-${e.agent}-${e.attempt}`);
  if (box) {
    box.classList.remove("running"); box.classList.add(e.verdict.kind);
    box.querySelector(".v").innerHTML = `${numHTML(e.verdict.text)} <button type="button" class="view-run" data-run="${e.run.meta.episode_id}">Watch this run</button>`;
  }
  setStatus(e.agent, { ok: "fix verified", warn: "fix only partly worked", err: "refuted by simulation" }[e.verdict.kind],
            { ok: "good", warn: "meh", err: "bad" }[e.verdict.kind]);
}

function onRevise(e) {
  post(`${agentBadge(e.agent)}<div class="pb"><p class="thought">${numHTML(e.thought)}</p>` +
       `<p class="act"><span class="tool">Revised the fix</span> <span class="res">${patchText(e.params, e.defaults)}</span></p></div>`, `step-post a-${e.agent}`);
}

function onDecision(e) {
  state.decision = e;
  const OUT = { supported: "good", "refuted by simulation": "bad", "refuted by the skeptic": "bad", "refuted by the guardrail": "bad", "only partly fixed it": "meh", "not tested": "" };
  post(`${agentBadge("lead")}<div class="pb card decision ${e.status}">` +
       `<p class="kind">Decision</p><p class="claim">${escapeHtml(e.summary)}</p>` +
       `<ul class="outcomes">${Object.entries(e.outcomes).map(([a, o]) =>
         `<li>${agentBadge(a)} <span>${AGENT[a].name}</span> <span class="o ${OUT[o] || ""}">${o}</span></li>`).join("")}</ul>` +
       `<p class="txt">The explanation and fix are in Findings.</p></div>`, "dec-post");
  for (const [a, o] of Object.entries(e.outcomes)) setStatus(a, o, OUT[o] || "");
  setStatus("skeptic", "done");
}

async function revealMystery(token) {
  const m = state.eps.orig.meta;
  if (!m.mystery || !state.decision) return;
  const key = await fetch(`/api/mystery/${state.episodeId}/answer`).then((r) => r.json());
  if (token !== state.token) return;
  const d = state.decision;
  const found = d.winner === key.expected_agent;
  const verified = d.status === "verified";
  state.reveal = { key, found, verified };
  const verdict = found && verified ? "The board found the planted cause and verified a fix."
    : found ? "The board found the planted cause, but no fix verified cleanly."
    : `The board picked the ${AGENT[d.winner].name} investigator; the planted cause was in the ${AGENT[key.expected_agent].name} investigator's area.`;
  post(`<span class="av av-key" aria-hidden="true">?</span><div class="pb card reveal ${found ? "right" : "wrong"}">` +
       `<p class="kind">Answer key, opened after the decision</p>` +
       `<p class="claim">${numHTML(key.planted)}</p>` +
       `<p class="txt"><b>${escapeHtml(verdict)}</b></p>` +
       (key.decoys.length ? `<p class="exp">Decoy warnings planted in the log: ${key.decoys.map(escapeHtml).join("; ")}</p>` : "") +
       `</div>`, "reveal-post");
  const fb = $("#answer .banner");
  if (fb) fb.insertAdjacentHTML("afterend", revealLine());
}

function revealLine() {
  const r = state.reveal;
  if (!r) return "";
  return `<p class="reveal-line ${r.found ? "right" : "wrong"}">Answer key: ${r.found ? "correct" : "missed"}. ${numHTML(r.key.planted)}</p>`;
}

function verdictBanner(ans) {
  if (!ans.winner) return "";
  const who = AGENT[ans.winner] ? AGENT[ans.winner].name : ans.winner;
  const t = { verified: `${who} investigator's hypothesis, verified in simulation`,
              partial: `${who} investigator's hypothesis; its best fix only partly worked`,
              unverified: `${who} investigator's hypothesis; no fix verified` }[ans.status] || "";
  return `<p class="banner ${ans.status}">${agentBadge(ans.winner)} ${t}</p>` + revealLine();
}

function testedHTML(ans) {
  const ex = ans.experiments || [];
  if (!ex.length) return "";
  return `<div class="fixtest" id="fixtest"><h3>Tested in MuJoCo</h3><ol class="attempts">` +
    ex.map((x) => `<li class="attempt ${x.verdict.kind}"><p class="a-head">${agentBadge(x.agent)} ${patchText(x.params, Object.fromEntries(Object.entries(state.meta.patchable || {}).map(([k, v]) => [k, v.default])))}</p>` +
      `<p class="a-verdict">${numHTML(x.verdict.text)} <button type="button" class="view-run" data-run="${x.run_id}">Watch this run</button></p></li>`).join("") +
    `</ol></div>`;
}

async function announce(token) {
  const ans = state.answer && state.answer.ans;
  if (!ans || ans.announced || token !== state.token) return;
  ans.announced = true;
  if (!ans.final_run_id || !state.fixRuns[ans.final_run_id]) return;
  const run = state.fixRuns[ans.final_run_id];
  const n = (ans.experiments || []).length;
  const who = AGENT[ans.winner] ? AGENT[ans.winner].name : "one";
  const lead = ans.status === "verified"
    ? `The board ran ${n} experiment${n > 1 ? "s" : ""}, and the ${who} investigator's fix is verified. `
    : `The board ran ${n} experiment${n > 1 ? "s" : ""} and none was clean. `;
  state.eps.fixed = run;
  $("#runs").hidden = false;
  const ft = $("#fixtest"), box = $("#answer");
  if (ft) box.scrollTop = ft.offsetTop - box.offsetTop - 8;
  state.view = null;
  await loadView("fixed");
  video.currentTime = 2.0; video.play();
  speak(lead + run.verdict.say, token);
}

async function ask(q) {
  q = q.trim();
  if (!q) return;
  const token = ++state.token;
  speechSynthesis.cancel();
  setActive(null);
  state.scan = null; state.fixRuns = {}; state.hyps = {}; state.decision = null; state.answer = null; state.reveal = null;
  if (state.view !== "orig") await loadView("orig");
  state.eps.fixed = null; $("#runs").hidden = true;
  setTab("board");
  findingsIdle();
  revealAll();
  let stop = () => {};
  try {
    const r = await fetch("/api/investigate", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ episode_id: state.episodeId, question: q,
                             current_ms: Math.round(video.currentTime * 1000), history: state.history }),
    });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || `Request failed with ${r.status}`);
    const reader = r.body.getReader(), dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        const chunk = buf.slice(0, i); buf = buf.slice(i + 2);
        if (!chunk.startsWith("data: ") || token !== state.token) continue;
        const e = JSON.parse(chunk.slice(6));
        switch (e.type) {
          case "start": stop = startBoard(q, e.agents); if (e.replayed) $("#board .q").insertAdjacentHTML("afterend", `<p class="replayed">Replaying an earlier board session for this question.</p>`); break;
          case "phase": setPhase(e.phase); if (e.phase !== "investigate") { state.scan = null; state.segView = null; document.querySelectorAll("#log li.read").forEach((x) => x.classList.remove("read")); } break;
          case "agent_step": onAgentStep(e); break;
          case "hypothesis": onHypothesis(e); break;
          case "agent_error": post(`${agentBadge(e.agent)}<div class="pb"><p class="err">${escapeHtml(e.message)}</p></div>`); setStatus(e.agent, "failed", "bad"); break;
          case "challenge": onChallenge(e); break;
          case "skeptic_note": if (e.note) post(`${agentBadge("skeptic")}<div class="pb"><p class="thought">${numHTML(e.note)}</p></div>`, "step-post a-skeptic"); break;
          case "experiment_skipped": post(`${agentBadge(e.agent)}<div class="pb"><p class="act">Not simulated: ${escapeHtml(e.reason)}</p></div>`, "skip-post"); break;
          case "experiment_start": onExperimentStart(e); break;
          case "experiment_result": onExperimentResult(e); break;
          case "experiment_error": post(`<div class="pb"><p class="err">${escapeHtml(e.message)}</p></div>`); break;
          case "revise": onRevise(e); break;
          case "decision": onDecision(e); revealMystery(token); break;
          case "answer": {
            stop();
            const st = $("#board-status"); if (st) st.remove();
            const ans = e.answer;
            if (!ans.segments || !ans.segments.length) throw new Error("Muse Spark returned no findings. Try rephrasing the question.");
            state.history.push({ q, a: ans.segments.map((s) => s.say).join(" ") + " " + ((ans.takeaway || {}).say || "") });
            state.answer = { q, ans };
            renderAnswer(q, ans);
            setTab("findings");
            state.playing = true;
            playAnswer(q, ans, token, false).then(() => { state.playing = false; announce(token); });
            break;
          }
          case "error": throw new Error(e.message);
        }
      }
    }
  } catch (err) {
    stop();
    if (token !== state.token) return;
    const msg = `<p class="err">${escapeHtml(err.message)}</p>`;
    const st = $("#board-status");
    if (st) st.outerHTML = msg; else $("#board").insertAdjacentHTML("beforeend", msg);
    setTab("board");
  }
}

function onAnswerClick(e) {
  const sq = e.target.closest(".sq");
  if (sq) { ask(sq.textContent); return; }
  const tc = e.target.closest(".tchip");
  if (tc) {
    state.token++; speechSynthesis.cancel();
    (state.view !== "orig" ? loadView("orig") : Promise.resolve()).then(() => { video.pause(); seekTo(+tc.dataset.t); });
    return;
  }
  if (!e.target.closest(".view-run") && !state.answer) return;
  const vr = e.target.closest(".view-run");
  if (vr) {
    state.token++; speechSynthesis.cancel();
    const run = state.fixRuns[vr.dataset.run];
    if (!run) return;
    state.eps.fixed = run;
    $("#runs").hidden = false;
    state.view = null;
    loadView("fixed").then(() => { video.currentTime = 2.0; video.play(); });
    return;
  }
  if (e.target.closest("#replay")) {
    const token = ++state.token;
    speechSynthesis.cancel();
    playAnswer(state.answer.q, state.answer.ans, token);
    return;
  }
  const step = e.target.closest(".step");
  if (step) {
    const token = ++state.token;
    speechSynthesis.cancel();
    const i = +step.dataset.i;
    markStep(i);
    const say = step.querySelector(".say");
    (state.view !== "orig" ? loadView("orig") : Promise.resolve())
      .then(() => showStep(state.answer.ans.segments[i], token))
      .then(() => speak(state.answer.ans.segments[i].say, token, (w) => revealWords(say, w)));
  }
}

async function postJSON(url, body) {
  const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || `${url} failed with ${r.status}`);
  return j;
}

/* ------------------------------------------------------------------ push to talk */

const rec = { ctx: null, stream: null, node: null, chunks: [], on: false };

async function startRec() {
  if (rec.on) return;
  rec.on = true;
  state.token++;               // barge-in: stop any answer that is playing
  speechSynthesis.cancel();
  $("#mic").classList.add("rec");
  try {
    rec.stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } });
    rec.ctx = new AudioContext({ sampleRate: 16000 });
    const src = rec.ctx.createMediaStreamSource(rec.stream);
    rec.node = rec.ctx.createScriptProcessor(4096, 1, 1);
    rec.chunks = [];
    rec.node.onaudioprocess = (e) => rec.chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
    src.connect(rec.node); rec.node.connect(rec.ctx.destination);
    if (!rec.on) stopRec();
  } catch (e) {
    rec.on = false;
    $("#mic").classList.remove("rec");
    setTab("board"); $("#board").innerHTML = `<p class="err">Microphone unavailable: ${escapeHtml(e.message)}. Type your question instead.</p>`;
  }
}

async function stopRec() {
  if (!rec.on) return;
  rec.on = false;
  $("#mic").classList.remove("rec");
  if (!rec.ctx) return;
  const rate = rec.ctx.sampleRate;
  rec.node.disconnect();
  rec.stream.getTracks().forEach((t) => t.stop());
  await rec.ctx.close();
  rec.ctx = null;
  let pcm = concat(rec.chunks);
  if (pcm.length < rate * 0.3) return;
  if (rate !== 16000 && rate !== 24000) { pcm = resample(pcm, rate, 16000); }
  const wav = encodeWav(pcm, rate === 24000 ? 24000 : 16000);
  setTab("board"); $("#board").innerHTML = `<p class="thinking">Transcribing with Muse Voice Transcribe</p>`;
  try {
    const r = await fetch("/api/transcribe", { method: "POST", headers: { "Content-Type": "audio/wav" }, body: wav });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || "Transcription failed");
    if (!j.transcript) throw new Error("No speech detected. Hold the mic a little longer and try again.");
    $("#question").value = "";
    ask(j.transcript);
  } catch (e) {
    $("#board").innerHTML = `<p class="err">${escapeHtml(e.message)}</p>`;
  }
}

function concat(chunks) {
  const out = new Float32Array(chunks.reduce((n, c) => n + c.length, 0));
  let o = 0; for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}

function resample(x, from, to) {
  const n = Math.round((x.length * to) / from), out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const p = (i * from) / to, j = Math.floor(p), f = p - j;
    out[i] = (x[j] || 0) * (1 - f) + (x[j + 1] || 0) * f;
  }
  return out;
}

function encodeWav(pcm, rate) {
  const buf = new ArrayBuffer(44 + pcm.length * 2), v = new DataView(buf);
  const w = (o, s) => [...s].forEach((ch, i) => v.setUint8(o + i, ch.charCodeAt(0)));
  w(0, "RIFF"); v.setUint32(4, 36 + pcm.length * 2, true); w(8, "WAVE");
  w(12, "fmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, rate, true); v.setUint32(28, rate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  w(36, "data"); v.setUint32(40, pcm.length * 2, true);
  for (let i = 0; i < pcm.length; i++) v.setInt16(44 + i * 2, Math.max(-1, Math.min(1, pcm[i])) * 0x7fff, true);
  return new Blob([buf], { type: "audio/wav" });
}

/* ------------------------------------------------------------------ wiring */

function bind() {
  const mic = $("#mic");
  mic.addEventListener("pointerdown", (e) => { e.preventDefault(); startRec(); });
  window.addEventListener("pointerup", stopRec);
  window.addEventListener("keydown", (e) => {
    const typing = document.activeElement === $("#question");
    if (e.code === "Space" && !typing && !e.repeat) { e.preventDefault(); startRec(); }
    if (!typing && e.code === "ArrowRight") video.currentTime += 1 / state.meta.fps;
    if (!typing && e.code === "ArrowLeft") video.currentTime -= 1 / state.meta.fps;
    if (!typing && e.code === "KeyK") togglePlay();
  });
  window.addEventListener("keyup", (e) => { if (e.code === "Space" && document.activeElement !== $("#question")) stopRec(); });

  $("#ask-form").addEventListener("submit", (e) => { e.preventDefault(); ask($("#question").value); $("#question").value = ""; });
  showIdle();
  $("#answer").addEventListener("click", onAnswerClick);
  $("#board").addEventListener("click", onAnswerClick);
  document.querySelectorAll(".tabs [role=tab]").forEach((b) => b.addEventListener("click", () => setTab(b.dataset.tab)));

  $("#play").addEventListener("click", togglePlay);
  $("#runs").addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b || b.dataset.v === state.view) return;
    state.token++; speechSynthesis.cancel();
    loadView(b.dataset.v);
  });
  video.addEventListener("play", () => $("#play-icon").setAttribute("d", "M7 5h4v14H7zM13 5h4v14h-4z"));
  video.addEventListener("pause", () => { $("#play-icon").setAttribute("d", "M8 5v14l11-7z"); drawThread(false); });
  video.addEventListener("ended", () => video.pause());

  tl.addEventListener("pointerdown", (e) => { state.scrubbing = true; tl.setPointerCapture(e.pointerId); video.pause(); seekFromEvent(e); });
  tl.addEventListener("pointermove", (e) => { if (state.scrubbing) seekFromEvent(e); });
  tl.addEventListener("pointerup", () => { state.scrubbing = false; drawThread(false); });
  video.addEventListener("seeked", () => drawThread(false));
  window.addEventListener("resize", () => drawThread(false));
}

function togglePlay() {
  state.token++;
  speechSynthesis.cancel();
  if (video.paused) {
    if (video.ended || video.currentTime >= video.duration - 0.05) video.currentTime = 0;
    video.play();
  } else video.pause();
}

function fillEpisodes(eps) {
  const built = eps.filter((e) => !e.mystery), myst = eps.filter((e) => e.mystery);
  const opt = (e) => `<option value="${e.id}">${escapeHtml(e.title)}</option>`;
  $("#episode").innerHTML = `<optgroup label="Recorded failures">${built.map(opt).join("")}</optgroup>` +
    (myst.length ? `<optgroup label="Mystery episodes">${myst.map(opt).join("")}</optgroup>` : "");
}

async function newMystery() {
  const btn = $("#mystery");
  btn.disabled = true;
  btn.textContent = "Simulating, 0 s";
  state.token++; speechSynthesis.cancel();
  const t0 = performance.now();
  const tick = setInterval(() => { btn.textContent = `Simulating, ${Math.round((performance.now() - t0) / 1000)} s`; }, 300);
  setTab("board");
  $("#board").innerHTML = `<p class="thinking">MuJoCo is simulating a new failure. The fault type, its parameters and ` +
    `two decoy warnings are drawn at random, and nobody, including the agents, knows the answer.</p>`;
  try {
    const r = await postJSON("/api/mystery", {});
    fillEpisodes(r.episodes);
    $("#episode").value = r.id;
    await loadEpisode(r.id);
  } catch (e) {
    $("#board").innerHTML = `<p class="err">${escapeHtml(e.message)}</p>`;
  } finally {
    clearInterval(tick);
    btn.disabled = false; btn.textContent = "New mystery";
  }
}

async function loadEpisode(id) {
  state.token++; speechSynthesis.cancel();
  const ep = await fetch(`/api/episode?id=${encodeURIComponent(id)}`).then((r) => r.json());
  state.episodeId = id;
  state.eps = { orig: ep, fixed: null };
  state.fixRuns = {}; state.hyps = {}; state.decision = null; state.history = []; state.answer = null; state.reveal = null;
  state.objects = []; state.active = null; state.point = null; state.scan = null;
  state.suggestions = ep.suggestions || [];
  state.signals = ep.signals;
  const aux = CHARTS.find((c) => c.swappable);
  aux.series = [{ key: "cube_z", name: "Object height", color: "--cube" }]; aux.unit = "";
  renderChartLabel(aux);
  $("#runs").hidden = true;
  state.view = null;
  await loadView("orig");
  video.currentTime = 0;
  showIdle();
}

async function init() {
  const [status, list] = await Promise.all([fetch("/api/status").then((r) => r.json()), fetch("/api/episodes").then((r) => r.json())]);
  state.status = status;
  const sel = $("#episode");
  fillEpisodes(list.episodes);
  sel.value = list.default;
  sel.addEventListener("change", () => loadEpisode(sel.value));
  $("#mystery").addEventListener("click", newMystery);
  const chips = [];
  if (status.mock) chips.push(`<span class="chip mock">Mock answers</span>`);
  else if (status.test_model) chips.push(`<span class="chip mock">Test model, not Muse Spark</span>`);
  else if (!status.has_key) chips.push(`<span class="chip off">No API key</span>`);
  chips.push(`<span class="chip">${status.models.reasoning}</span>`, `<span class="chip">${status.models.segmentation}</span>`, `<span class="chip">${status.models.speech}</span>`);
  $("#models").innerHTML = chips.join("");
  const first = await fetch(`/api/episode?id=${list.default}`).then((r) => r.json());
  Object.assign(state, { meta: first.meta, tele: first.telemetry, log: first.log, masks: first.masks,
                         signals: first.signals, suggestions: first.suggestions });
  for (const c of CHARTS) c.range = seriesRange(c);
  buildCharts();
  bind();
  await loadEpisode(list.default);
  requestAnimationFrame(frame);
}

init();
