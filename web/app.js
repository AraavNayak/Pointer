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

const SUGGESTIONS = [
  "Why did the gripper drop the cube?",
  "When did the left finger start to fail?",
  "Was the right finger at fault?",
  "What should we change so this doesn't happen again?",
];

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
const PAD = { l: 48, r: 12, t: 22, b: 18 };

function buildCharts() {
  const host = $("#charts");
  host.innerHTML = "";
  for (const c of CHARTS) {
    const div = document.createElement("div");
    div.className = "chart";
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
  for (const s of c.series) for (const v of state.tele[s.key]) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
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
  const [lo, hi] = c.range;
  const dur = state.meta.duration_ms;
  return {
    w, h,
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
  const [lo, hi] = c.range;
  const lit = state.active && state.active.signal;

  // grid + axes
  g.font = `11px ${css("--sans")}`;
  g.fillStyle = css("--ink-3");
  g.strokeStyle = css("--rule");
  g.lineWidth = 1;
  for (let v = lo; v <= hi + c.step / 2; v += c.step) {
    const y = Math.round(G.y(v)) + 0.5;
    g.globalAlpha = 0.5;
    g.beginPath(); g.moveTo(PAD.l, y); g.lineTo(w - PAD.r, y); g.stroke();
    g.globalAlpha = 1;
    g.textAlign = "right"; g.textBaseline = "middle";
    g.fillText(fmt(v) + (c.unit ? ` ${c.unit}` : ""), PAD.l - 6, y);
  }
  g.textAlign = "center"; g.textBaseline = "alphabetic";
  for (let s = 0; s <= state.meta.duration_ms / 1000; s++) {
    const x = G.x(s * 1000);
    g.fillText(`${s}s`, x, h - 4);
  }
  for (const r of c.refs || []) {
    const y = Math.round(G.y(r)) + 0.5;
    g.setLineDash([2, 4]); g.strokeStyle = css("--ink-3");
    g.beginPath(); g.moveTo(PAD.l, y); g.lineTo(w - PAD.r, y); g.stroke();
    g.setLineDash([]);
    g.textAlign = "right"; g.fillStyle = css("--ink-3");
    g.fillText(r === 70 ? "70 °C warning" : r === 80 ? "80 °C derate" : String(r), w - PAD.r - 2, y - 4);
    g.textAlign = "center";
  }

  // series
  const ts = state.tele.timestamp_ms;
  for (const s of c.series) {
    const on = !lit || lit === s.key;
    g.strokeStyle = css(s.color) || css("--ink");
    g.globalAlpha = on ? 1 : 0.28;
    g.lineWidth = lit === s.key ? 2.4 : 1.5;
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

function tag(g, bbox, text, color, k) {
  const fs = 12.5 / k;
  g.font = `600 ${fs}px ${css("--sans")}`;
  const tw = g.measureText(text).width;
  const x = bbox[0], y = Math.max(fs + 6 / k, bbox[1] - 6 / k);
  g.fillStyle = color;
  roundRect(g, x - 1 / k, y - fs - 3 / k, tw + 12 / k, fs + 7 / k, 4 / k);
  g.fill();
  g.fillStyle = "#fff";
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
  for (const l of state.log) {
    if (l.level !== "WARN" && l.level !== "ERROR") continue;
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
    `<li class="${l.level}" data-n="${l.n}"><span class="n">${l.n}</span><span>${escapeHtml(l.text)}</span></li>`).join("");
  $("#log-count").textContent = `${state.log.length} lines`;
  ol.addEventListener("scroll", () => drawThread(false));
}

function updateLogTime(nowMs) {
  for (const li of $("#log").children) {
    const l = state.log[+li.dataset.n - 1];
    const future = l.t_ms !== null && l.t_ms > nowMs + 1;
    li.classList.toggle("future", future);
  }
}

function litLog(n) {
  const ol = $("#log");
  ol.querySelectorAll(".lit").forEach((li) => li.classList.remove("lit"));
  if (n < 1) return null;
  const li = ol.querySelector(`li[data-n="${n}"]`);
  li.classList.add("lit");
  ol.scrollTop = li.offsetTop - ol.clientHeight / 2 + li.clientHeight / 2;
  return li;
}

function logAnchor() {
  const li = $("#log li.lit");
  if (!li) return null;
  const r = li.getBoundingClientRect(), ol = $("#log").getBoundingClientRect();
  const y = Math.min(Math.max(r.top + r.height / 2, ol.top + 6), ol.bottom - 6);
  const text = li.children[1].getBoundingClientRect();
  return { x: Math.max(text.left - 4, ol.left + 4), y };
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

function speak(text, token) {
  return new Promise((res) => {
    if (!("speechSynthesis" in window) || token !== state.token) return res();
    const u = new SpeechSynthesisUtterance(text);
    if (voice) u.voice = voice;
    u.rate = 1.04;
    const words = text.split(/\s+/).length;
    const fallback = setTimeout(res, words * 420 + 1800);
    u.onend = u.onerror = () => { clearTimeout(fallback); res(); };
    speechSynthesis.speak(u);
  });
}

async function playAnswer(q, ans, token) {
  const cap = $("#caption");
  cap.innerHTML = `<p class="q">${escapeHtml(q)}</p>` +
    ans.segments.map((s, i) => `<p class="seg" data-i="${i}">${escapeHtml(s.say)}</p>`).join("") +
    (ans.root_cause ? `<p class="root"><b>Root cause</b> ${escapeHtml(ans.root_cause)}</p>` : "") +
    `<p class="meta">${ans.model} · ${(ans.latency_ms / 1000).toFixed(1)} s</p>`;
  video.pause();
  for (let i = 0; i < ans.segments.length; i++) {
    if (token !== state.token) return;
    const seg = ans.segments[i];
    cap.querySelectorAll(".seg").forEach((p, j) => {
      p.classList.toggle("now", j === i);
      p.classList.toggle("done", j < i);
    });
    setActive(seg);
    await seekTo(seg.time_ms);
    drawThread(true);
    await speak(seg.say, token);
    await sleep(250);
  }
  if (token === state.token) cap.querySelectorAll(".seg").forEach((p) => { p.classList.remove("now"); p.classList.add("done"); });
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function ask(q) {
  q = q.trim();
  if (!q) return;
  const token = ++state.token;
  speechSynthesis.cancel();
  setActive(null);
  const cap = $("#caption");
  cap.innerHTML = `<p class="q">${escapeHtml(q)}</p><p class="thinking">Looking at the video, telemetry and log</p>`;
  try {
    const ans = await postJSON("/api/ask", { question: q, current_ms: Math.round(video.currentTime * 1000), history: state.history });
    if (token !== state.token) return;
    if (!ans.segments || !ans.segments.length) throw new Error("Muse Spark returned no segments. Try rephrasing the question.");
    state.history.push({ q, a: ans.segments.map((s) => s.say).join(" ") });
    await playAnswer(q, ans, token);
  } catch (e) {
    if (token === state.token) cap.innerHTML = `<p class="q">${escapeHtml(q)}</p><p class="err">${escapeHtml(e.message)}</p>`;
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
    $("#caption").innerHTML = `<p class="err">Microphone unavailable: ${escapeHtml(e.message)}. Type your question instead.</p>`;
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
  $("#caption").innerHTML = `<p class="thinking">Transcribing</p>`;
  try {
    const r = await fetch("/api/transcribe", { method: "POST", headers: { "Content-Type": "audio/wav" }, body: wav });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || "Transcription failed");
    if (!j.transcript) throw new Error("No speech detected. Hold the mic a little longer and try again.");
    $("#question").value = "";
    ask(j.transcript);
  } catch (e) {
    $("#caption").innerHTML = `<p class="err">${escapeHtml(e.message)}</p>`;
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
  $("#suggest").innerHTML = SUGGESTIONS.map((s) => `<button type="button">${s}</button>`).join("");
  $("#suggest").addEventListener("click", (e) => { if (e.target.tagName === "BUTTON") ask(e.target.textContent); });

  $("#play").addEventListener("click", togglePlay);
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
    if (video.ended || video.currentTime >= state.meta.duration_ms / 1000 - 0.05) video.currentTime = 0;
    video.play();
  } else video.pause();
}

async function init() {
  const [status, ep] = await Promise.all([fetch("/api/status").then((r) => r.json()), fetch("/api/episode").then((r) => r.json())]);
  Object.assign(state, { status, meta: ep.meta, tele: ep.telemetry, signals: ep.signals, log: ep.log, masks: ep.masks });
  $("#ep-name").textContent = ep.meta.task;
  $("#ep-outcome").textContent = ep.meta.outcome.replace(/^FAIL: /, "Failed: ");
  const fromSam = ep.masks.source && ep.masks.source.startsWith("sam");
  $("#mask-source").textContent = fromSam ? "Objects tracked by SAM 3.1" : "Object masks from simulator ground truth";
  const chips = [];
  if (status.mock) chips.push(`<span class="chip mock">Mock answers</span>`);
  else if (!status.has_key) chips.push(`<span class="chip off">No API key</span>`);
  chips.push(`<span class="chip">${status.models.reasoning}</span>`, `<span class="chip">${status.models.segmentation}</span>`, `<span class="chip">${status.models.speech}</span>`);
  $("#models").innerHTML = chips.join("");
  for (const c of CHARTS) c.range = seriesRange(c);
  buildCharts();
  buildLog();
  bind();
  requestAnimationFrame(frame);
}

init();
