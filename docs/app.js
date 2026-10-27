/* HBICP dashboard: caricamento dati (manifest -> export), grafici, popup del glossario, tracciatore. */
(function () {
"use strict";
const GLOSS = window.GLOSSARY || {};
const MI = window.MODEL_INFO;
let D = { runs: [], results: [] };       // dati attualmente mostrati
let SRC = { kind: "none" };              // {kind: export|demo|file|none, entry?, note?, name?}
let MANIFEST = null;
const exportCache = {};
const $ = (id) => document.getElementById(id);
const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const fmt = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v)) ? "–" :
  (Math.abs(v) !== 0 && (Math.abs(v) < 1e-3 || Math.abs(v) >= 1e5) ? Number(v).toExponential(2) : Number(v).toFixed(d));
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const charts = {};
const DEV = /^(localhost|127\.0\.0\.1|\[::1\])$/.test(location.hostname) || /[?&]dev\b/.test(location.search);

/* ---------------------------------------------------------------- termini del glossario */
const pretty = (t) => esc(t).replace(/_([A-Za-z0-9]+)/g, "<sub>$1</sub>");
const tm = (id, label) => `<span class="term" data-term="${id}" tabindex="0">${label}</span>`;
function shortDef(id) {
  const g = GLOSS[id]; if (!g) return "";
  const s = g.def.split(/(?<=[.!?])\s+(?=[A-ZÀ-Ý])/)[0];
  return s.length > 150 ? s.slice(0, 147) + "…" : s;
}
function wrap(text, w = 52) {
  const out = []; let line = "";
  text.split(" ").forEach((word) => { if ((line + " " + word).trim().length > w) { out.push(line); line = word; } else line = (line + " " + word).trim(); });
  if (line) out.push(line);
  return out;
}
let typesetTimer = null;
function typeset(root) {
  if (!(window.MathJax && MathJax.typesetPromise)) return Promise.resolve();
  return MathJax.typesetPromise(root ? [root] : undefined).catch(() => {});
}
function scheduleTypeset() { clearTimeout(typesetTimer); typesetTimer = setTimeout(() => typeset(document.querySelector("main")), 30); }
function waitMathJax() {
  if (window.MathJax && MathJax.typesetPromise) { typeset(document.body).then(() => { if (popFor) place(popFor); }); return; }
  setTimeout(waitMathJax, 120);
}

/* popup unico, con delegazione degli eventi su document */
const pop = document.createElement("div");
pop.id = "termPop"; pop.setAttribute("role", "tooltip"); pop.hidden = true;
document.body.appendChild(pop);
let popFor = null, hideTimer = null, shownAt = 0;
function popHTML(id) {
  const g = GLOSS[id];
  if (!g) return `<b>${esc(id)}</b><p>Voce di glossario mancante.</p>`;
  const rows = [];
  if (g.range) rows.push(`<div class="pr"><span>Valori</span><span>${esc(g.range)}</span></div>`);
  if (g.good) rows.push(`<div class="pr"><span>Atteso</span><span>${esc(g.good)}</span></div>`);
  rows.push(`<div class="pr src"><span>Fonte</span><span>${esc(g.source || "")}</span></div>`);
  return `<div class="pn">${pretty(g.name)}</div><p>${esc(g.def)}</p>${g.formula ? `<div class="pf">\\[ ${g.formula} \\]</div>` : ""}${rows.join("")}`;
}
function place(el) {
  const r = el.getBoundingClientRect(), w = pop.offsetWidth, h = pop.offsetHeight, m = 8;
  let left = Math.min(Math.max(m, r.left), innerWidth - w - m);
  let top = r.bottom + 6;
  if (top + h > innerHeight - m) top = r.top - h - 6;
  if (top < m) top = Math.max(m, Math.min(innerHeight - h - m, r.bottom + 6));
  pop.style.left = left + "px"; pop.style.top = top + "px";
}
function showTerm(el) {
  clearTimeout(hideTimer);
  if (popFor === el && !pop.hidden) return;
  popFor = el; shownAt = Date.now();
  pop.innerHTML = popHTML(el.dataset.term);
  pop.hidden = false; place(el);
  el.setAttribute("aria-describedby", "termPop");
  if (pop.querySelector(".pf")) typeset(pop).then(() => { if (popFor === el) place(el); });
}
function hideTerm() {
  clearTimeout(hideTimer);
  if (popFor) popFor.removeAttribute("aria-describedby");
  pop.hidden = true; popFor = null;
}
const termOf = (t) => (t && t.closest ? t.closest(".term") : null);
document.addEventListener("mouseover", (e) => { const t = termOf(e.target); if (t) showTerm(t); else if (pop.contains(e.target)) clearTimeout(hideTimer); });
document.addEventListener("mouseout", (e) => {
  const t = termOf(e.target); if (!t && !pop.contains(e.target)) return;
  const to = e.relatedTarget; if (to && (termOf(to) === t || pop.contains(to))) return;
  clearTimeout(hideTimer); hideTimer = setTimeout(hideTerm, 140);
});
document.addEventListener("focusin", (e) => { const t = termOf(e.target); if (t) showTerm(t); });
document.addEventListener("focusout", (e) => { if (termOf(e.target)) hideTimer = setTimeout(hideTerm, 100); });
document.addEventListener("click", (e) => {                       // tap su touch: apre/chiude; tap fuori: chiude
  const t = termOf(e.target);
  if (t) { if (popFor === t && Date.now() - shownAt > 400) hideTerm(); else showTerm(t); }
  else if (!pop.contains(e.target)) hideTerm();
});
document.addEventListener("keydown", (e) => { if (e.key === "Escape") hideTerm(); });
addEventListener("scroll", () => { if (popFor) place(popFor); }, { passive: true });
addEventListener("resize", () => { if (popFor) place(popFor); });

/* controllo di sviluppo: data-term senza voce di glossario */
function checkTerms() {
  if (!DEV) return;
  const missing = new Set();
  document.querySelectorAll("[data-term]").forEach((el) => { if (!GLOSS[el.dataset.term]) missing.add(el.dataset.term); });
  if (missing.size) console.warn("[glossario] data-term senza voce:", [...missing].join(", "));
  else console.info("[glossario] tutti i data-term hanno una voce (" + document.querySelectorAll("[data-term]").length + " occorrenze)");
}

/* ---------------------------------------------------------------- metodi e grafici */
const METHOD = {
  "model(A)": { c: "--s1", label: "Modello (Head A)", g: "headA" },
  "model(B)": { c: "--s7", label: "Modello (Head B)", g: "headB" },
  persistence: { c: "--s2", label: "Persistenza", g: "persistence" },
  dmd: { c: "--s3", label: "DMD ridge w=1", g: "dmd1" },
  dmd4: { c: "--s4", label: "DMD ridge w=4", g: "dmd4" },
  hl_oracle: { c: "--s5", label: "HL + stato oracolo", g: "hl" },
  hl_fair_o0: { c: "--s9", label: "HL equo (stato da o_0)", g: "hl" },
  hl_fair_fit: { c: "--s10", label: "HL equo (stato fittato)", g: "hl" },
  oracle: { c: "--s6", label: "Oracolo", dash: true, g: "oracle" },
};
const base = (k) => k.split("/").pop();
const mInfo = (k) => METHOD[base(k)] || { c: "--s8", label: k };
const mLabel = (k) => (k.includes("/") ? k.split("/")[0] + " · " : "") + mInfo(k).label;
const mColor = (k) => css(mInfo(k).c);
const isModel = (k) => base(k).startsWith("model");
const mTerm = (k) => tm(mInfo(k).g || "headA", esc(mLabel(k)));

function applyDefaults() {
  Chart.defaults.color = css("--ink-2");
  Chart.defaults.borderColor = css("--rule");
  Chart.defaults.font.family = css("--f-body");
  Chart.defaults.font.size = 12;
  Chart.defaults.animation = false;
  Chart.defaults.maintainAspectRatio = false;
  Chart.defaults.plugins.legend.display = false;
  const tt = Chart.defaults.plugins.tooltip;
  tt.backgroundColor = css("--ink"); tt.titleColor = css("--bg"); tt.bodyColor = css("--bg"); tt.footerColor = css("--bg");
  tt.titleFont = { family: css("--f-mono"), size: 12 }; tt.bodyFont = { family: css("--f-mono"), size: 12 };
  tt.footerFont = { family: css("--f-body"), size: 11, weight: "400" }; tt.footerMarginTop = 8;
  Chart.defaults.elements.line.borderWidth = 2;
  Chart.defaults.elements.point.radius = 0;
  Chart.defaults.elements.point.hoverRadius = 4;
}
function setEmpty(id, msg) {
  const el = $(id); if (!el) return;
  const box = el.parentElement; let o = box.querySelector(".nodata");
  if (msg) { if (charts[id]) { charts[id].destroy(); delete charts[id]; } if (!o) { o = document.createElement("div"); o.className = "nodata"; box.appendChild(o); } o.textContent = msg; el.style.visibility = "hidden"; }
  else { if (o) o.remove(); el.style.visibility = ""; }
}
function mk(id, cfg) {
  if (charts[id]) charts[id].destroy();
  const el = $(id); if (!el) return null;
  setEmpty(id, null);
  charts[id] = new Chart(el, cfg);
  return charts[id];
}
const grid = () => ({ color: css("--rule"), drawTicks: false });
function axis(title, extra = {}) {
  return Object.assign({ grid: grid(), border: { display: false }, ticks: { padding: 6, color: css("--ink-3") },
    title: title ? { display: true, text: title, color: css("--ink-3") } : undefined }, extra);
}
const hoverIdx = { mode: "index", intersect: false };
const hoverNear = { mode: "nearest", intersect: false, axis: "x" };
/* footer del tooltip: definizione breve del metodo (dataset._m) e della metrica */
function footerFor(metricId) {
  return (items) => {
    const ids = []; const seen = new Set();
    items.forEach((it) => { const m = it.dataset._m; if (m && !seen.has(m)) { seen.add(m); ids.push(["m", m]); } });
    const lines = [];
    ids.slice(0, 2).forEach(([, m]) => { const g = mInfo(m).g; if (g && GLOSS[g]) lines.push(...wrap(`${GLOSS[g].name}: ${shortDef(g)}`), ""); });
    if (metricId && GLOSS[metricId]) lines.push(...wrap(`${GLOSS[metricId].name}: ${shortDef(metricId)}`));
    return lines.filter((l, i, a) => !(l === "" && i === a.length - 1));
  };
}
function legend(elId, items) {
  const el = $(elId); if (!el) return;
  el.innerHTML = items.map((it) => `<span><i class="${it.dash ? "dash" : ""}" style="background:${it.color};color:${it.color}"></i>${it.term ? tm(it.term, esc(it.label)) : esc(it.label)}</span>`).join("");
}
const methodLegend = (keys) => keys.map((k) => ({ label: mLabel(k), color: mColor(k), dash: mInfo(k).dash, term: mInfo(k).g }));
function ema(arr, a) {
  if (!a) return arr; let s = null;
  return arr.map((v) => (v === null || v === undefined ? null : (s = s === null ? v : a * s + (1 - a) * v)));
}
const shadePlugin = {
  id: "shade",
  beforeDatasetsDraw(chart, _a, opt) {
    if (opt.x1 === undefined) return;
    const { ctx, chartArea: ca, scales: { x } } = chart;
    const a = Math.max(ca.left, x.getPixelForValue(opt.x0)), b = Math.min(ca.right, x.getPixelForValue(opt.x1));
    ctx.save(); ctx.fillStyle = css("--accent-soft"); ctx.fillRect(a, ca.top, b - a, ca.bottom - ca.top);
    ctx.strokeStyle = css("--ink-3"); ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(b, ca.top); ctx.lineTo(b, ca.bottom); ctx.stroke();
    ctx.fillStyle = css("--ink-3"); ctx.font = `11px ${css("--f-body")}`; ctx.fillText("contesto", a + 6, ca.top + 14);
    ctx.fillText("previsione →", b + 6, ca.top + 14); ctx.restore();
  },
};
Chart.register(shadePlugin);

/* ---------------------------------------------------------------- accesso ai dati */
const runs = () => D.runs || [];
const curRun = () => runs()[+$("runSel").value || 0];
function mergedMethods(sel) {
  const out = {}; const sets = D.results || [];
  const use = sel === "all" || sel === undefined ? sets : [sets[+sel]].filter(Boolean);
  use.forEach((rs) => Object.entries(rs.methods || {}).forEach(([k, v]) => { const key = out[k] ? `${rs.name}/${k}` : k; out[key] = v; }));
  return out;
}
const curMethods = () => mergedMethods($("resSel").value);
const methodOrder = (ms) => Object.keys(ms).sort((a, b) => {
  const o = ["model(A)", "model(B)", "hl_oracle", "hl_fair_o0", "hl_fair_fit", "dmd", "dmd4", "persistence", "oracle"];
  return (o.indexOf(base(a)) + 99 * !o.includes(base(a))) - (o.indexOf(base(b)) + 99 * !o.includes(base(b)));
});
const Ts = (ms) => [...new Set(Object.values(ms).flatMap((m) => Object.keys(m).map(Number)))].sort((a, b) => a - b);
const meanK = (arr, K = 16) => { const a = (arr || []).slice(0, K); return a.length ? a.reduce((s, v) => s + v, 0) / a.length : NaN; };
const modelKey = (ms) => Object.keys(ms).find((k) => base(k) === "model(A)") || Object.keys(ms).find(isModel);
const useS = (r) => !(r && r.config && r.config.model && r.config.model.use_S === false);

/* ---------------------------------------------------------------- banner */
function renderBanner() {
  const b = $("sourceBanner"), e = SRC.entry || {};
  const when = (D.generated_at || e.generated_at || "").replace("T", " ");
  const commit = (D.git_commit || e.git_commit || "?").slice(0, 10);
  const notes = D.notes || e.notes || "";
  const meta = `${when ? ` · ${esc(when)}` : ""} · commit <span class="mono">${esc(commit)}</span>${notes ? ` · ${esc(notes)}` : ""}`;
  if (SRC.kind === "demo") b.innerHTML = `<span class="pill warn">Dati demo</span><span>${esc(D.notes || "Run dimostrativa.")}${SRC.note ? ` <i>(${esc(SRC.note)})</i>` : ""}</span>`;
  else if (SRC.kind === "file") b.innerHTML = `<span class="pill good">File caricato</span><span><b>${esc(SRC.name)}</b>${meta}</span>`;
  else if (SRC.kind === "export") b.innerHTML = `<span class="pill good">Export</span><span><b>${esc(e.label || SRC.file)}</b>${meta}</span>`;
  else b.innerHTML = `<span class="pill bad">Nessun dato</span><span>Nessun export caricato.</span>`;
}
function showFatal(msg) {
  const f = $("fatal"); f.hidden = false;
  f.innerHTML = `<h3>Impossibile caricare i dati</h3><p>${esc(msg)}</p><p class="cap">Se hai aperto <span class="mono">index.html</span> direttamente dal disco (file://) il browser blocca il caricamento dei file JSON. Servi la cartella con <span class="mono">python -m http.server</span> dentro <span class="mono">docs/</span> e apri <span class="mono">http://localhost:8000</span>, oppure usa «Carica dashboard_data.json».</p>`;
}

/* ---------------------------------------------------------------- 00 panoramica */
function renderOverview() {
  const r = curRun(), ms = mergedMethods("all"), mk_ = modelKey(ms), T = Ts(ms);
  renderBanner();
  $("ovEyebrow").textContent = `Run dashboard · ${runs().length} run · ${(D.results || []).length} set di risultati`;
  const last = r && r.train.length ? r.train[r.train.length - 1] : null;
  const steps = r ? r.config.train.steps : null;
  const Tmax = T[T.length - 1];
  const mT = mk_ && ms[mk_][Tmax];
  const tiles = [
    { e: tm("params", "Parametri"), v: r && r.params ? (r.params / 1e6).toFixed(2) + "M" : "–", d: r ? `${tm("preset", "preset")} ${esc(r.config.preset)} · ${tm("headA", "head")} ${esc(r.config.model.head)}` : "nessuna run" },
    { e: tm("step", "Step di training"), v: last ? last.step.toLocaleString("it-IT") : "–", d: steps ? `su ${steps.toLocaleString("it-IT")} configurati` : "" },
    { e: `${tm("kstar", "k*")} a ${tm("T", "T")} = ${Tmax ?? "–"}`, v: mT ? mT.k_star : "–", d: mT ? `passi prevedibili con RMSE ≤ 0.05 (t = ${(mT.k_star * 0.1).toFixed(1)})` : "nessun risultato del modello" },
    { e: `${tm("fidmean", "Fidelity media F̄₂")} a ${tm("T", "T")} = ${Tmax ?? "–"}`, v: mT && mT.fidelity2 ? meanK(mT.fidelity2).toFixed(3) : "–", d: mT && mT.fidelity2 ? "1 = stato previsto identico al vero (k = 1..16)" : "non presente in questo export" },
    { e: `${tm("auroc", "AUROC")} grafo (${tm("gate", "gate")}) a ${tm("T", "T")} = ${Tmax ?? "–"}`, v: mT && mT.auroc_gate != null ? mT.auroc_gate.toFixed(3) : "–", d: "0.5 = caso, 1 = grafo recuperato; atteso > 0.9" },
  ];
  $("kpis").innerHTML = tiles.map((t) => `<div class="panel tile"><span class="eyebrow">${t.e}</span><span class="v">${t.v}</span><span class="d">${t.d}</span></div>`).join("");
  renderChecks(ms, mk_, T, r);
  const keys = methodOrder(ms);
  legend("kstarLegendOv", methodLegend(keys));
  if (keys.length) mk("kstarOv", kstarCfg(ms, keys)); else setEmpty("kstarOv", "Nessun risultato nei dati caricati");
}
function kstarCfg(ms, keys) {
  const T = Ts(ms);
  return { type: "line", data: { labels: T.map(String), datasets: keys.map((k) => ({
    label: mLabel(k), _m: k, data: T.map((t) => (ms[k][t] ? ms[k][t].k_star : null)), borderColor: mColor(k), backgroundColor: mColor(k),
    borderDash: mInfo(k).dash ? [5, 4] : [], pointRadius: 4, pointBorderColor: css("--surface"), pointBorderWidth: 2, spanGaps: true })) },
    options: { interaction: hoverIdx, plugins: { tooltip: { callbacks: { footer: footerFor("kstar") } } },
      scales: { x: axis("contesto T (passi)"), y: axis("k*", { min: 0, max: 30 }) } } };
}
function renderChecks(ms, mk_, T, r) {
  const rows = [];
  const pill = (cls, txt) => `<span class="pill ${cls}">${txt}</span>`;
  const add = (t, p, w) => rows.push(`<div class="check"><span class="t">${t}</span>${p}<span class="w">${w}</span></div>`);
  const dk = ["dmd", "dmd4"].filter((k) => ms[k]);
  const tDMD = tm("dmd1", "DMD ridge");
  if (mk_ && dk.length) {
    const Tl = T.filter((t) => t <= 32 && ms[mk_][t]);
    const best = (t) => Math.min(...dk.map((k) => ms[k][t] ? meanK(ms[k][t].rmse_n) : Infinity));
    const wins = Tl.filter((t) => meanK(ms[mk_][t].rmse_n) < best(t));
    const det = Tl.map((t) => `T=${t}: ${fmt(meanK(ms[mk_][t].rmse_n), 4)} vs ${fmt(best(t), 4)}`).join(" · ");
    add(`Il modello batte la ${tDMD} per ${tm("T", "T")} ≤ 32`, wins.length === Tl.length && Tl.length ? pill("good", "superato") : wins.length ? pill("warn", `${wins.length}/${Tl.length} T`) : pill("bad", "non superato"),
      `${tm("rmse_n", "RMSE<sub>n</sub>")} medio su k = 1..16, modello vs migliore DMD. ${det}. Se fallisce, la spec considera il design falsificato.`);
  } else add(`Il modello batte la ${tDMD} per ${tm("T", "T")} ≤ 32`, pill("na", "dati mancanti"), "Servono risultati del modello e della baseline DMD sullo stesso test set.");
  if (mk_) {
    const Tm = T.filter((t) => ms[mk_][t]); const ks = Tm.map((t) => ms[mk_][t].k_star);
    const mono = ks.every((v, i) => i === 0 || v >= ks[i - 1]); const up = ks.length > 1 && ks[ks.length - 1] > ks[0];
    add(`${tm("kstar", "k*")} cresce con il contesto ${tm("T", "T")}`, up && mono ? pill("good", "superato") : up ? pill("warn", "non monotono") : pill("bad", "non cresce"),
      `k*(T) del modello: ${Tm.map((t, i) => `T=${t} → ${ks[i]}`).join(", ")}. Se non cresce, il contesto non viene usato per identificare la dinamica.`);
    const Tl = Tm[Tm.length - 1], res = ms[mk_][Tl]; const a = res.auroc_gate;
    add(`Grafo recuperato: ${tm("auroc", "AUROC")} del ${tm("gate", "gate")} > 0.9`, a == null ? pill("na", "n/d") : a > 0.9 ? pill("good", fmt(a)) : a > 0.7 ? pill("warn", fmt(a)) : pill("bad", fmt(a)),
      `AUROC di g<sub>ij</sub> contro l'adiacenza vera a T = ${Tl}; attenzione spaziale ${tm("alpha", "ᾱ")}: ${fmt(res.auroc_attn)}.`);
    if (res.gen_auroc_adj != null) {
      const ga = res.gen_auroc_adj;
      add(`Lettura a posteriori da G: ${tm("gen_auroc_adj", "AUROC")} di ‖G<sub>ij</sub>‖ contro l'adiacenza`, ga > 0.9 ? pill("good", fmt(ga)) : ga > 0.7 ? pill("warn", fmt(ga)) : pill("bad", fmt(ga)),
        `A T = ${Tl}: ${tm("gen_corr_J", "corr")} con J sugli archi = ${fmt(res.gen_corr_J)}, ${tm("gen_corr_h", "corr")} di ‖G<sub>ii</sub>‖ con h = ${fmt(res.gen_corr_h)}. Informativo, non un criterio di successo: l'obiettivo è prevedere bene gli stati (fidelity e osservabili), G può essere lontano dall'Hamiltoniana vera.`);
    }
    if (res.probe_r2_h != null && res.probe_r2_J != null) {
      const gap = res.probe_r2_h - res.probe_r2_J;
      add(`Informazione locale e globale: ${tm("r2", "R²")} di ${tm("probe_h", "ĥ")} vs ${tm("probe_J", "Ĵ")}`, gap > 0.3 ? pill("warn", `Δ = ${fmt(gap, 2)}`) : pill("good", `Δ = ${fmt(gap, 2)}`),
        `R² ĥ = ${fmt(res.probe_r2_h)}, R² Ĵ = ${fmt(res.probe_r2_J)} a T = ${Tl}. Un divario grande (campi letti bene, accoppiamenti no) indica che le feature di coppia non contengono J: vedi sezione «Fisica e modello» → tracciatore.`);
    }
    if (useS(r)) {
      const sf = res.S_F, sf0 = r && r.train.length ? r.train[0].S_F : null;
      add(`Gauge di dissipazione ${tm("SF", "‖S‖<sub>F</sub>")} → 0 (sistema chiuso)`, sf == null ? pill("na", "n/d") : sf < 0.1 ? pill("good", fmt(sf)) : pill("warn", fmt(sf)),
        `Valore medio sugli ${tm("anchor", "anchor")} di test${sf0 != null ? `; all'inizio del training era ${fmt(sf0)}` : ""}. Sui dati chiusi il training dovrebbe spingerlo verso 0.`);
    } else add(`Dissipazione ${tm("SF", "‖S‖<sub>F</sub>")}`, pill("good", "disattivata"), "Run con model.use_S = false: G è antisimmetrica e S = 0 per costruzione.");
  } else add("Metriche del modello", pill("na", "dati mancanti"), "Nessun risultato del modello nei file caricati.");
  const s = D.sample;
  if (s && s.G_eig) add(`Generatore stabile: max Re ${tm("spectrum", "λ(G<sub>t</sub>)")} ≤ 10<sup>−5</sup>`, s.G_eig.max_re <= 1e-5 ? pill("good", fmt(s.G_eig.max_re)) : pill("bad", fmt(s.G_eig.max_re)),
    `Sulla traiettoria di esempio (${s.G_eig.re.length} autovalori). Garantito per costruzione da G = (A − Aᵀ) − SSᵀ.`);
  if (ms["oracle"] && T.includes(16)) {
    const rank = methodOrder(ms).filter((k) => ms[k][16]).map((k) => [k, meanK(ms[k][16].rmse_n)]).sort((a, b) => a[1] - b[1]);
    const pos = (k) => rank.findIndex((x) => base(x[0]) === k);
    const ok = mk_ && pos(base(mk_)) >= 0 && ["dmd", "dmd4", "persistence"].every((k) => pos(k) < 0 || pos(base(mk_)) < pos(k));
    add(`Ordine atteso a T = 16: ${tm("oracle", "oracolo")} &lt; ${tm("hl", "HL")} ≈ modello &lt; ${tm("dmd1", "DMD")} &lt; ${tm("persistence", "persistenza")}`, mk_ ? (ok ? pill("good", "rispettato") : pill("warn", "diverso")) : pill("na", "senza modello"),
      "Ordine osservato (RMSE<sub>n</sub> medio k = 1..16): " + rank.map(([k, v]) => `${esc(mLabel(k))} ${fmt(v, 4)}`).join(" &lt; "));
  }
  $("checks").innerHTML = rows.join("");
}

/* ---------------------------------------------------------------- 01 valutazione */
let evalT = null, evalNE = "rmse_n";
function renderEval() {
  const ms = curMethods(), keys = methodOrder(ms), T = Ts(ms);
  if (!keys.length) {
    ["cRmse", "cKstar", "cFid"].forEach((c) => setEmpty(c, "Nessun results.json nei dati caricati"));
    $("evalTable").innerHTML = `<tr><td class="muted">Nessun results.json nei dati caricati.</td></tr>`; $("Tseg").innerHTML = ""; $("lgMethods").innerHTML = ""; renderDiag(ms, keys, T); return;
  }
  if (!T.includes(evalT)) evalT = T.includes(32) ? 32 : T[Math.floor(T.length / 2)];
  $("Tseg").innerHTML = T.map((t) => `<button type="button" data-v="${t}" aria-pressed="${t === evalT}">${t}</button>`).join("");
  legend("lgMethods", [...methodLegend(keys), { label: "ε = 0.05", color: css("--ink-3"), dash: true, term: "eps" }]);
  const rid = evalNE === "rmse_n" ? "rmse_n" : "rmse_e";
  $("rmseTitle").innerHTML = `${tm(rid, evalNE === "rmse_n" ? "RMSE<sub>n</sub>(k)" : "RMSE<sub>e</sub>(k)")} a ${tm("T", "T")} = ${evalT}`;
  $("rmseAxisY").innerHTML = tm(rid, evalNE === "rmse_n" ? "RMSE<sub>n</sub>" : "RMSE<sub>e</sub>");
  const H = Math.max(...keys.map((k) => (ms[k][evalT] ? ms[k][evalT][evalNE].length : 0)));
  const floor = 1e-6;
  const ds = keys.filter((k) => ms[k][evalT]).map((k) => ({ label: mLabel(k), _m: k, data: ms[k][evalT][evalNE].map((v, i) => ({ x: i + 1, y: Math.max(v, floor) })),
    borderColor: mColor(k), backgroundColor: mColor(k), borderDash: mInfo(k).dash ? [5, 4] : [] }));
  ds.push({ label: "ε", data: [{ x: 1, y: 0.05 }, { x: H, y: 0.05 }], borderColor: css("--ink-3"), borderDash: [3, 3], borderWidth: 1 });
  mk("cRmse", { type: "line", data: { datasets: ds }, options: { parsing: false, interaction: hoverNear,
    plugins: { tooltip: { filter: (it) => it.dataset.label !== "ε", callbacks: { title: (it) => `k = ${it[0].parsed.x}  (t = ${(it[0].parsed.x * 0.1).toFixed(1)})`, label: (it) => ` ${it.dataset.label}: ${it.parsed.y <= floor ? "≈0" : it.parsed.y.toFixed(4)}`, footer: footerFor(rid) } } },
    scales: { x: axis("passi nel futuro k", { type: "linear", min: 1, max: H }), y: axis("RMSE", { type: "logarithmic", min: floor }) } } });
  mk("cKstar", kstarCfg(ms, keys));
  // fidelity
  const withF = keys.filter((k) => ms[k][evalT] && ms[k][evalT].fidelity2);
  $("fidTitle").innerHTML = `${tm("fid2", "Fidelity F<sub>2</sub>(k)")} a ${tm("T", "T")} = ${evalT}`;
  if (!withF.length) setEmpty("cFid", "Fidelity non presente in questo export (rigenera con la versione attuale di eval.evaluate)");
  else {
    const Hf = Math.max(...withF.map((k) => ms[k][evalT].fidelity2.length));
    mk("cFid", { type: "line", data: { datasets: withF.map((k) => ({ label: mLabel(k), _m: k, data: ms[k][evalT].fidelity2.map((v, i) => ({ x: i + 1, y: v })),
      borderColor: mColor(k), backgroundColor: mColor(k), borderDash: mInfo(k).dash ? [5, 4] : [] })) },
      options: { parsing: false, interaction: hoverNear, plugins: { tooltip: { callbacks: { title: (it) => `k = ${it[0].parsed.x}`, label: (it) => ` ${it.dataset.label}: ${it.parsed.y.toFixed(4)}`, footer: footerFor("fid2") } } },
        scales: { x: axis("passi nel futuro k", { type: "linear", min: 1, max: Hf }), y: axis("F₂", { min: 0.8, max: 1 }) } } });
  }
  // tabella
  const nonOr = keys.filter((k) => base(k) !== "oracle");
  const best = {}, bestF = {};
  T.forEach((t) => { best[t] = Math.min(...nonOr.filter((k) => ms[k][t]).map((k) => meanK(ms[k][t].rmse_n)));
    bestF[t] = Math.max(...nonOr.filter((k) => ms[k][t] && ms[k][t].fidelity2).map((k) => meanK(ms[k][t].fidelity2))); });
  $("evalTable").innerHTML = `<tr><th>metodo</th>${T.map((t) => `<th>T=${t} ${tm("rmse_n", "RMSE<sub>n</sub>")}</th><th>${tm("fidmean", "F̄₂")}</th><th>${tm("kstar", "k*")}</th>`).join("")}<th>${tm("sec_step", "s / passo")} (T=${T[T.length - 1]})</th></tr>` +
    keys.map((k) => `<tr><td>${mTerm(k)}</td>${T.map((t) => { const r = ms[k][t]; if (!r) return "<td>–</td><td>–</td><td>–</td>"; const v = meanK(r.rmse_n), f = r.fidelity2 ? meanK(r.fidelity2) : null;
      return `<td class="mono ${base(k) !== "oracle" && v === best[t] ? "best" : ""}">${fmt(v, 4)}</td><td class="mono ${f != null && base(k) !== "oracle" && f === bestF[t] ? "best" : ""}">${f == null ? "–" : fmt(f, 3)}</td><td class="mono">${r.k_star}</td>`; }).join("")}<td class="mono">${fmt((ms[k][T[T.length - 1]] || {}).sec_per_step)}</td></tr>`).join("");
  renderDiag(ms, keys, T);
}
function renderDiag(ms, keys, T) {
  const mods = keys.filter(isModel);
  $("diagTable").innerHTML = mods.length ? `<tr><th>modello</th><th>${tm("T", "T")}</th><th>${tm("auroc", "AUROC")} ${tm("gate", "gate")}</th><th>AUROC ${tm("alpha", "attenzione")}</th><th>AUROC ${tm("probe_a", "probe â")}</th><th>${tm("r2", "R²")} ${tm("probe_J", "Ĵ")} (archi)</th><th>R² ${tm("probe_h", "ĥ")}</th><th>${tm("SF", "‖S‖<sub>F</sub>")}</th><th>${tm("gen_auroc_adj", "AUROC")} G</th><th>${tm("gen_corr_J", "corr")} ‖G<sub>ij</sub>‖, J</th><th>${tm("gen_corr_h", "corr")} ‖G<sub>ii</sub>‖, h</th><th>${tm("ntraj", "traiettorie")}</th></tr>` +
    mods.flatMap((k) => T.filter((t) => ms[k][t]).map((t) => { const r = ms[k][t];
      return `<tr><td>${mTerm(k)}</td><td class="mono">${t}</td><td class="mono">${fmt(r.auroc_gate)}</td><td class="mono">${fmt(r.auroc_attn)}</td><td class="mono">${fmt(r.probe_auroc_adj)}</td><td class="mono">${fmt(r.probe_r2_J)}</td><td class="mono">${fmt(r.probe_r2_h)}</td><td class="mono">${fmt(r.S_F)}</td><td class="mono">${fmt(r.gen_auroc_adj)}</td><td class="mono">${fmt(r.gen_corr_J)}</td><td class="mono">${fmt(r.gen_corr_h)}</td><td class="mono">${r.ntraj}</td></tr>`; })).join("")
    : `<tr><td class="muted">Nessun risultato del modello in questo set.</td></tr>`;
}

/* ---------------------------------------------------------------- 02 esempio */
const CH = ["x", "y", "z"], ECH = { xx: "xx", yz: "yz", zz: "zz" };
let kind = "node";
const sampleKeys = () => (D.sample ? methodOrder(D.sample.preds) : []);
function renderSampleControls() {
  const s = D.sample; if (!s) return;
  const opts = (sel, n, skip) => { const v = sel.value; sel.innerHTML = [...Array(n).keys()].map((i) => `<option value="${i}">${i}</option>`).join(""); sel.value = v && +v < n ? v : (skip ? "1" : "0"); };
  opts($("siteI"), s.n); opts($("siteJ"), s.n, true);
  const cs = kind === "node" ? CH.map((c, i) => [i, `σ${sup(c)}  (${c})`]) : Object.keys(ECH).map((c) => [c, `${c[0]}${c[1]}  (σ${c[0]}ᵢ σ${c[1]}ⱼ)`]);
  const cv = $("chanSel").value; $("chanSel").innerHTML = cs.map(([v, l]) => `<option value="${v}">${l}</option>`).join("");
  if ([...$("chanSel").options].some((o) => o.value === cv)) $("chanSel").value = cv; else $("chanSel").value = kind === "node" ? "2" : "zz";
  $("siteJwrap").hidden = kind === "node";
  const sm = $("stripMethod").value; $("stripMethod").innerHTML = sampleKeys().map((k) => `<option value="${esc(k)}">${esc(mLabel(k))}</option>`).join("");
  if (sm && s.preds[sm]) $("stripMethod").value = sm;
}
const SUB = "₀₁₂₃₄₅₆₇₈₉", sub = (i) => String(i).split("").map((d) => SUB[d]).join("");
const SUPC = { x: "ˣ", y: "ʸ", z: "ᶻ" }, sup = (c) => SUPC[c];
function renderSample() {
  const s = D.sample;
  const ids = ["cTraj", "cSampleFid", "cSampleErr", "cEnergy", "cLam"];
  if (!s) { $("sampleLede").textContent = "Nessuna traiettoria di esempio nei dati caricati (esporta con --data)."; ids.forEach((c) => setEmpty(c, "Nessuna traiettoria di esempio")); ["stTrue", "stPred", "stErr"].forEach((c) => { const cv = $(c); cv.width = 1; cv.height = 1; }); $("lgSample").innerHTML = ""; return; }
  const nE = s.adj.flat().reduce((a, v) => a + v, 0) / 2;
  $("sampleLede").innerHTML = `${tm("traj", "Traiettoria")} n. ${s.index} di <span class="mono">${esc(s.dataset)}</span> (${tm("traj", "seed")} ${s.seed}, famiglia ${esc(s.family)}): ${tm("n", "n")} = ${s.n} qubit, grafo con ${nE} archi. Il modello riceve i passi 0…${s.T} (t = 0…${(s.T * s.dt).toFixed(1)}) e prevede i ${s.H} passi successivi.`;
  const keys = sampleKeys(), T = s.T, H = s.H, dt = s.dt, useT = $("xUnit").value === "time";
  const X = (t) => (useT ? +(t * dt).toFixed(3) : t);
  let i = +$("siteI").value, j = +$("siteJ").value, ch = $("chanSel").value;
  if (kind === "edge" && i === j) { j = (i + 1) % s.n; $("siteJ").value = j; }
  const tv = (t) => (kind === "node" ? s.node[t][i][+ch] : s.edge_sel[ch][t][i][j]);
  const pv = (p, k) => (kind === "node" ? p.node[k][i][+ch] : p.edge_sel[ch][k][i][j]);
  const ink = css("--ink");
  const ds = [
    { label: "contesto (input)", data: [...Array(T + 1).keys()].map((t) => ({ x: X(t), y: tv(t) })), borderColor: ink, backgroundColor: ink, borderWidth: 2 },
    { label: "verità futura", data: [...Array(H + 1).keys()].map((k) => ({ x: X(T + k), y: tv(T + k) })), borderColor: ink, backgroundColor: ink, borderDash: [4, 3], borderWidth: 1.5 },
    ...keys.map((k) => { const p = s.preds[k]; const K = p.node.length;
      return { label: mLabel(k), _m: k, data: [{ x: X(T), y: tv(T) }, ...[...Array(K).keys()].map((q) => ({ x: X(T + 1 + q), y: pv(p, q) }))], borderColor: mColor(k), backgroundColor: mColor(k), borderDash: mInfo(k).dash ? [5, 4] : [] }; }),
  ];
  const lab = kind === "node" ? `⟨σ${sup(CH[+ch])}${sub(i)}⟩` : `⟨σ${sup(ch[0])}${sub(i)} σ${sup(ch[1])}${sub(j)}⟩`;
  $("trajTitle").textContent = `Traiettoria di ${lab}`;
  legend("lgSample", [{ label: "contesto (input)", color: ink }, { label: "verità futura", color: ink, dash: true }, ...methodLegend(keys)]);
  mk("cTraj", { type: "line", data: { datasets: ds }, options: { parsing: false, interaction: hoverNear,
    plugins: { shade: { x0: X(0), x1: X(T) }, tooltip: { callbacks: { title: (it) => (useT ? `t = ${it[0].parsed.x}` : `passo ${it[0].parsed.x}`), label: (it) => ` ${it.dataset.label}: ${it.parsed.y.toFixed(4)}`, footer: footerFor(kind === "node" ? "Nt" : "Et") } } },
    scales: { x: axis(useT ? "tempo t·Δt (1/J)" : "passo t", { type: "linear", min: X(0), max: X(T + H) }), y: axis(lab, { min: -1, max: 1 }) } } });
  const floor = 1e-6;
  const withF = keys.filter((k) => s.preds[k].fid2);
  if (!withF.length) setEmpty("cSampleFid", "Fidelity non presente in questo export (rigenera con tools.export_dashboard aggiornato)");
  else mk("cSampleFid", { type: "line", data: { datasets: withF.map((k) => ({ label: mLabel(k), _m: k, data: s.preds[k].fid2.map((v, q) => ({ x: q + 1, y: v })), borderColor: mColor(k), backgroundColor: mColor(k), borderDash: mInfo(k).dash ? [5, 4] : [] })) },
    options: { parsing: false, interaction: hoverNear, plugins: { tooltip: { callbacks: { title: (it) => `k = ${it[0].parsed.x}`, label: (it) => ` ${it.dataset.label}: ${it.parsed.y.toFixed(4)}`, footer: footerFor("fidtraj") } } },
      scales: { x: axis("k", { type: "linear", min: 1, max: H }), y: axis("F₂", { min: 0.8, max: 1 }) } } });
  mk("cSampleErr", { type: "line", data: { datasets: keys.map((k) => ({ label: mLabel(k), _m: k, data: s.preds[k].rmse_n.map((v, q) => ({ x: q + 1, y: Math.max(v, floor) })), borderColor: mColor(k), backgroundColor: mColor(k), borderDash: mInfo(k).dash ? [5, 4] : [] })) },
    options: { parsing: false, interaction: hoverNear, plugins: { tooltip: { callbacks: { title: (it) => `k = ${it[0].parsed.x}`, label: (it) => ` ${it.dataset.label}: ${it.parsed.y.toFixed(4)}`, footer: footerFor("rmse_n") } } },
      scales: { x: axis("k", { type: "linear", min: 1, max: H }), y: axis(null, { type: "logarithmic", min: floor }) } } });
  if (s.energy_true) {
    mk("cEnergy", { type: "line", data: { datasets: [{ label: "verità", data: s.energy_true.map((v, t) => ({ x: X(t), y: v })), borderColor: ink, borderWidth: 1.5 },
      ...keys.filter((k) => s.preds[k].energy).map((k) => ({ label: mLabel(k), _m: k, data: s.preds[k].energy.map((v, q) => ({ x: X(T + 1 + q), y: v })), borderColor: mColor(k), borderDash: mInfo(k).dash ? [5, 4] : [] }))] },
      options: { parsing: false, interaction: hoverNear, plugins: { shade: { x0: X(0), x1: X(T) }, tooltip: { callbacks: { label: (it) => ` ${it.dataset.label}: ${it.parsed.y.toFixed(4)}`, footer: footerFor("energy") } } },
        scales: { x: axis(null, { type: "linear", min: X(0), max: X(T + H) }), y: axis("⟨H⟩", (() => { const e = s.energy_true, lo = Math.min(...e), hi = Math.max(...e), pad = Math.max(0.5, 2 * (hi - lo)); return { min: +(lo - pad).toFixed(2), max: +(hi + pad).toFixed(2) }; })()) } } });
  } else setEmpty("cEnergy", "Energia disponibile solo per le famiglie tfim e lr_ising");
  mk("cLam", { type: "line", data: { datasets: [{ label: "verità", data: s.lam_min_true.map((v, t) => ({ x: X(t), y: v })), borderColor: ink, borderWidth: 1.5 },
    ...keys.map((k) => ({ label: mLabel(k), _m: k, data: s.preds[k].lam_min.map((v, q) => ({ x: X(T + 1 + q), y: v })), borderColor: mColor(k), borderDash: mInfo(k).dash ? [5, 4] : [] })),
    { label: "0", data: [{ x: X(0), y: 0 }, { x: X(T + H), y: 0 }], borderColor: css("--bad"), borderWidth: 1, borderDash: [2, 3] }] },
    options: { parsing: false, interaction: hoverNear, plugins: { shade: { x0: X(0), x1: X(T) }, tooltip: { filter: (it) => it.dataset.label !== "0", callbacks: { label: (it) => ` ${it.dataset.label}: ${it.parsed.y.toFixed(4)}`, footer: footerFor("lammin") } } },
      scales: { x: axis(null, { type: "linear", min: X(0), max: X(T + H) }), y: axis("λ_min") } } });
  renderStrips();
}
function hex2rgb(h) { h = h.replace("#", ""); if (h.length === 3) h = h.split("").map((c) => c + c).join(""); const n = parseInt(h, 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
const mix = (a, b, t) => a.map((v, i) => Math.round(v + (b[i] - v) * t));
function renderStrips() {
  const s = D.sample; if (!s) return;
  const k = $("stripMethod").value, p = s.preds[k], c = +$("stripChan").value; if (!p) return;
  $("stPredLbl").textContent = `Previsione · ${mLabel(k)}`;
  const n = s.n, T = s.T, K = p.node.length, W = T + 1 + K, cw = 4, rh = 12;
  const neg = hex2rgb(css("--s1")), pos = hex2rgb(css("--s2")), mid = hex2rgb(css("--surface-2")), hot = hex2rgb(css("--ink"));
  const div = (v) => (v < 0 ? mix(mid, neg, Math.min(1, -v)) : mix(mid, pos, Math.min(1, v)));
  const draw = (id, val) => { const cv = $(id); cv.width = W * cw; cv.height = n * rh; const ctx = cv.getContext("2d");
    for (let t = 0; t < W; t++) for (let i = 0; i < n; i++) { const col = val(t, i); ctx.fillStyle = col ? `rgb(${col})` : css("--heat-0"); ctx.fillRect(t * cw, i * rh, cw, rh - 1); }
    ctx.fillStyle = css("--ink"); ctx.fillRect((T + 1) * cw - 1, 0, 2, n * rh);
    cv.onmousemove = (e) => { const r = cv.getBoundingClientRect(); const t = Math.floor((e.clientX - r.left) / r.width * W), i = Math.floor((e.clientY - r.top) / r.height * n);
      if (!s.node[t] || i < 0 || i >= n) return;
      const tr = s.node[t][i][c], pr = t > T ? p.node[t - T - 1][i][c] : tr;
      tip(e, `sito ${i} · passo ${t}<br>vero ${fmt(tr, 3)}${t > T ? ` · prev ${fmt(pr, 3)}` : ""}`); };
    cv.onmouseleave = () => ($("tip").hidden = true); };
  draw("stTrue", (t, i) => div(s.node[t][i][c]));
  draw("stPred", (t, i) => div(t <= T ? s.node[t][i][c] : p.node[t - T - 1][i][c]));
  draw("stErr", (t, i) => (t <= T ? null : mix(mid, hot, Math.min(1, Math.abs(p.node[t - T - 1][i][c] - s.node[t][i][c]) / 0.5))));
}
function tip(e, html) { const el = $("tip"); el.innerHTML = html; el.hidden = false; el.style.left = Math.min(e.clientX + 12, innerWidth - 180) + "px"; el.style.top = e.clientY + 12 + "px"; }

/* ---------------------------------------------------------------- 03 grafo inferito */
function auroc(scores, labels) {
  let pos = 0, neg = 0, s = 0;
  scores.forEach((a, i) => scores.forEach((b, j) => { if (labels[i] && !labels[j]) { s += a > b ? 1 : a === b ? 0.5 : 0; } }));
  labels.forEach((l) => (l ? pos++ : neg++));
  return pos && neg ? s / (pos * neg) : NaN;
}
function renderGraph() {
  const s = D.sample;
  if (!s) { $("heats").innerHTML = `<div class="empty">Nessuna traiettoria di esempio.</div>`; ["cProbeJ", "cEig"].forEach((c) => setEmpty(c, "Nessuna traiettoria di esempio")); renderRel(); return; }
  const n = s.n, pairs = []; for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) pairs.push([i, j]);
  const lab = pairs.map(([i, j]) => !!s.adj[i][j]);
  const Jmax = Math.max(...s.J.flat());
  const mats = [
    { t: `Grafo vero: ${tm("Jij", "J<sub>ij</sub>")}`, m: s.J, max: Jmax, d: `${lab.filter(Boolean).length} archi su ${pairs.length} coppie` },
    s.gate && { t: `Gate del modello ${tm("gate", "g<sub>ij</sub>")}`, m: s.gate, max: 1, au: true },
    s.attn && { t: `Attenzione spaziale ${tm("alpha", "ᾱ<sub>ij</sub>")}`, m: s.attn, max: Math.max(...pairs.map(([i, j]) => s.attn[i][j])), au: true },
    s.probe && { t: `Probe ${tm("probe_a", "â<sub>ij</sub>")} = σ(·)`, m: s.probe.adj, max: 1, au: true },
    s.G_norm && { t: `Generatore ${tm("gen_auroc_adj", "‖G<sub>ij</sub>‖<sub>F</sub>")}`, m: s.G_norm, max: Math.max(...pairs.map(([i, j]) => s.G_norm[i][j])), au: true },
  ].filter(Boolean);
  const c0 = hex2rgb(css("--heat-0")), c1 = hex2rgb(css("--heat-1"));
  $("heats").innerHTML = mats.map((M, mi) => {
    const sc = pairs.map(([i, j]) => M.m[i][j]);
    const d = M.au ? `${tm("auroc", "AUROC")} su questa traiettoria: <b class="mono">${fmt(auroc(sc, lab))}</b>` : M.d;
    const cells = [];
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const v = i === j ? null : M.m[i][j];
      const col = v === null ? css("--surface") : `rgb(${mix(c0, c1, Math.max(0, Math.min(1, v / (M.max || 1))))})`;
      const ring = i !== j && s.adj[i][j] && mi > 0 ? `box-shadow: inset 0 0 0 1.5px ${css("--s2")}` : "";
      cells.push(`<div data-i="${i}" data-j="${j}" data-v="${v === null ? "" : v}" style="background:${col};${ring}"></div>`);
    }
    return `<div class="panel heatwrap"><h3>${M.t}</h3><div class="heat" style="grid-template-columns:repeat(${n},1fr)" data-m="${mi}">${cells.join("")}</div><span class="cap">${d}${mi > 0 ? " · contorno arancio = arco vero" : ""}</span></div>`;
  }).join("");
  $("heats").querySelectorAll(".heat div").forEach((el) => {
    el.addEventListener("mousemove", (e) => { const i = el.dataset.i, j = el.dataset.j; if (i === j) return;
      tip(e, `(${i}, ${j}) · ${s.adj[i][j] ? "arco vero" : "nessun arco"}<br>valore ${fmt(+el.dataset.v, 3)} · J = ${fmt(s.J[i][j], 3)}`); });
    el.addEventListener("mouseleave", () => ($("tip").hidden = true));
  });
  if (s.probe) {
    const on = pairs.filter(([i, j]) => s.adj[i][j]).map(([i, j]) => ({ x: s.J[i][j], y: s.probe.J[i][j] }));
    const off = pairs.filter(([i, j]) => !s.adj[i][j]).map(([i, j]) => ({ x: s.J[i][j], y: s.probe.J[i][j] }));
    const all = [...on, ...off].flatMap((p) => [p.x, p.y]); const lo = Math.min(0, ...all) - 0.1, hi = Math.max(1.5, ...all) + 0.1;
    mk("cProbeJ", { type: "scatter", data: { datasets: [
      { label: "y = x", type: "line", data: [{ x: lo, y: lo }, { x: hi, y: hi }], borderColor: css("--ink-3"), borderDash: [3, 3], borderWidth: 1, pointRadius: 0 },
      { label: "archi veri", data: on, backgroundColor: css("--s1"), borderColor: css("--surface"), borderWidth: 1.5, pointRadius: 5 },
      { label: "coppie senza arco", data: off, backgroundColor: css("--ink-3"), borderColor: css("--surface"), borderWidth: 1.5, pointRadius: 4 } ] },
      options: { plugins: { legend: { display: true, position: "bottom", labels: { boxWidth: 10, filter: (it) => it.text !== "y = x" } }, tooltip: { filter: (it) => it.dataset.label !== "y = x", callbacks: { label: (it) => ` ${it.dataset.label}: J = ${it.parsed.x.toFixed(3)}, Ĵ = ${it.parsed.y.toFixed(3)}`, footer: footerFor("probe_J") } } },
        scales: { x: axis("J vero", { min: lo, max: hi }), y: axis("Ĵ probe", { min: lo, max: hi }) } } });
  } else if (s.G_norm) {
    $("gJTitle").innerHTML = `Generatore: ${tm("gen_corr_J", "‖G<sub>ij</sub>‖")} contro ${tm("Jij", "J<sub>ij</sub>")}`;
    $("gJAxes").innerHTML = `x: ${tm("Jij", "J vero")} · y: ${tm("gen_corr_J", "‖G<sub>ij</sub>‖<sub>F</sub>")} (norma del blocco di G)`;
    $("gJHow").innerHTML = "<b>Come leggerlo.</b> Lettura a posteriori: J non è mai dato al modello. Una relazione crescente sugli archi veri indica che l'intensità del generatore segue l'accoppiamento; una nuvola piatta indica che G descrive la dinamica in modo diverso da J, il che non è un errore se le previsioni sono accurate.";
    const pt = (f) => pairs.filter(([i, j]) => f(i, j)).map(([i, j]) => ({ x: s.J[i][j], y: s.G_norm[i][j] }));
    const on = pt((i, j) => s.adj[i][j]), off = pt((i, j) => !s.adj[i][j]);
    mk("cProbeJ", { type: "scatter", data: { datasets: [
      { label: "archi veri", data: on, backgroundColor: css("--s1"), borderColor: css("--surface"), borderWidth: 1.5, pointRadius: 5 },
      { label: "coppie senza arco", data: off, backgroundColor: css("--ink-3"), borderColor: css("--surface"), borderWidth: 1.5, pointRadius: 4 } ] },
      options: { plugins: { legend: { display: true, position: "bottom", labels: { boxWidth: 10 } }, tooltip: { callbacks: { label: (it) => ` ${it.dataset.label}: J = ${it.parsed.x.toFixed(3)}, ‖G‖ = ${it.parsed.y.toFixed(3)}`, footer: footerFor("gen_corr_J") } } },
        scales: { x: axis("J vero"), y: axis("‖G_ij‖_F", { beginAtZero: true }) } } });
  } else setEmpty("cProbeJ", "Né probe né generatore in questo export");
  if (s.G_eig) {
    const pts = s.G_eig.re.map((r, q) => ({ x: r, y: s.G_eig.im[q] }));
    $("eigCap").innerHTML = `${pts.length} autovalori (nm) · max Re λ = ${fmt(s.G_eig.max_re)} · ${tm("SF", "‖S‖<sub>F</sub>")} = ${fmt(s.G_eig.S_F)}`;
    const ymax = Math.max(...pts.map((p) => Math.abs(p.y))) * 1.1 || 1, xmin = Math.min(...pts.map((p) => p.x), -0.01);
    mk("cEig", { type: "scatter", data: { datasets: [
      { label: "Re = 0", type: "line", data: [{ x: 0, y: -ymax }, { x: 0, y: ymax }], borderColor: css("--bad"), borderDash: [3, 3], borderWidth: 1, pointRadius: 0 },
      { label: "λ(G)", data: pts, backgroundColor: css("--s1"), pointRadius: 3 } ] },
      options: { plugins: { tooltip: { filter: (it) => it.dataset.label !== "Re = 0", callbacks: { label: (it) => ` λ = ${it.parsed.x.toExponential(2)} ${it.parsed.y >= 0 ? "+" : "−"} ${Math.abs(it.parsed.y).toFixed(3)} i`, footer: footerFor("spectrum") } } },
        scales: { x: axis("Re λ  (≤ 0: contrazione)", { min: xmin * 1.1, max: Math.max(0.02, -xmin * 0.1) }), y: axis("Im λ  (frequenze)", { min: -ymax, max: ymax }) } } });
  } else setEmpty("cEig", "Spettro non presente (Head B)");
  renderRel();
}
/* errore relativo del probe: MSE / varianza del bersaglio. J è valutato su tutte le coppie i != j (anche J = 0). */
function targetVariances() {
  const ds = (D.datasets || [])[0], r = curRun();
  const n = (ds && ds.attrs && ds.attrs.n) || (r && r.config.train.n) || 8;
  const p = ds && ds.mean_degree != null ? ds.mean_degree / (n - 1) : 3 / (n - 1);
  const mu = ds && ds.J_on_edges ? ds.J_on_edges[0] : 1, sd = ds && ds.J_on_edges ? ds.J_on_edges[1] : Math.sqrt(1 / 12);
  const varJ = p * (sd * sd + mu * mu) - (p * mu) ** 2;
  const varH = ds && ds.h_stats ? ds.h_stats[1] ** 2 : 1 / 12;
  return { varH, varJ, n, p };
}
function renderRel() {
  const r = curRun();
  if (!r || !r.train.length) { setEmpty("cRel", "Nessun log di training nei dati caricati"); $("lgRel").innerHTML = ""; return; }
  if (!r.train.some((d) => d.probe_h != null)) { setEmpty("cRel", "Probe disattivato in questa run (nessun bersaglio h, J): vedi la lettura a posteriori da G"); $("lgRel").innerHTML = ""; return; }
  const v = targetVariances(), a = +$("smoothSel").value, x = r.train.map((d) => d.step);
  const line = (key, scale, color, label) => ({ label, data: ema(r.train.map((d) => (d[key] == null ? null : d[key] / scale)), a).map((y, i) => ({ x: x[i], y })), borderColor: color, backgroundColor: color });
  legend("lgRel", [{ label: "campi ĥ (locale)", color: css("--s1"), term: "probe_h" }, { label: "accoppiamenti Ĵ (globale)", color: css("--s2"), term: "probe_J" }, { label: "predittore costante = 1", color: css("--ink-3"), dash: true }]);
  mk("cRel", { type: "line", data: { datasets: [line("probe_h", v.varH, css("--s1"), "ĥ / Var(h)"), line("probe_J", v.varJ, css("--s2"), "Ĵ / Var(J)"),
    { label: "costante", data: [{ x: x[0], y: 1 }, { x: x[x.length - 1], y: 1 }], borderColor: css("--ink-3"), borderDash: [4, 3], borderWidth: 1 }] },
    options: { parsing: false, interaction: hoverIdx, plugins: { tooltip: { filter: (it) => it.dataset.label !== "costante", callbacks: { footer: () => [...wrap(`Var(h) = ${v.varH.toFixed(3)}, Var(J) su tutte le coppie = ${v.varJ.toFixed(3)} (p = ${v.p.toFixed(2)}).`)] } } },
      scales: { x: axis("step", { type: "linear", min: 0 }), y: axis("MSE / varianza", { type: "logarithmic" }) } } });
}

/* ---------------------------------------------------------------- 04 fisica e modello */
function archFlags(m) {
  const v = $("traceArch").value, base = { pauli_A: !!m.pauli_A, probe: m.probe !== false };  // probe absent in old configs = on
  if (v === "new") return { ...base, pair_attn: true, triangle: true, pair_latent: 8 };
  if (v === "old") return { ...base, pair_attn: false, triangle: false, pair_latent: 0 };
  return { ...base, pair_attn: !!m.pair_attn, triangle: !!m.triangle, pair_latent: m.pair_latent || 0 };
}
function dims() {
  const r = curRun(), s = D.sample;
  const m = (r && r.config.model) || { d: 128, de: 32, nh: 4, L: 3, m: 32, K: 32 }, t = (r && r.config.train) || { n: 8, H: 16 };
  return { n: s ? s.n : t.n, d: m.d, de: m.de, nh: m.nh, L: m.L, m: m.m, K: m.K, T: s ? s.T : 32, H: s ? s.H : t.H,
    ...archFlags(m) };
}
function renderModelInfo() {
  const dm = dims();
  $("Mval").textContent = `M = ${3 * dm.n + 9 * dm.n * (dm.n - 1) / 2} per n = ${dm.n}`;
  const r = curRun(); const l = r && r.config.loss;
  $("lamVals").textContent = l ? `${l.lambda_phys} / ${l.lambda_probe} / ${l.lambda_theta}` : "0.1 / 1 / 1e-4";
}
let traceKind = "node";
function renderEquations() {
  $("eqGrid").innerHTML = MI.EQS.map((q) => `<div class="panel eqcard"><div class="ph"><h3>${q.title}</h3><span class="tag">${q.tag}</span></div>
    <div class="math">\\[ ${q.tex} \\]</div><p class="cap">${q.note}</p>
    <ul class="where">${q.where.map(([id, sym, txt]) => `<li>${tm(id, `\\(${sym}\\)`)} <span>${txt}</span></li>`).join("")}</ul></div>`).join("");
}
function renderLossCards() {
  $("lossGrid").innerHTML = MI.LOSS_CARDS.map((c) => `<div class="panel losscard"><div class="ph"><h3>${tm(c.id, GLOSS[c.id] ? pretty(GLOSS[c.id].name) : c.id)}</h3>
    <span class="pill ${c.grad ? "good" : "warn"}">${c.grad ? "gradiente al corpo" : "nessun gradiente al corpo"}</span></div>
    <div class="math">\\[ ${c.tex} \\]</div><ol class="steps">${c.steps.map((s) => `<li>${esc(s)}</li>`).join("")}</ol>
    <dl class="kv"><dt>codice</dt><dd>${esc(c.code)}</dd><dt>valore tipico</dt><dd>${esc(c.typical)}</dd></dl></div>`).join("");
}
function renderContrib() {
  const r = curRun();
  if (!r || !r.train.length) { $("contribTable").innerHTML = `<tr><td class="muted">Nessun log di training.</td></tr>`; $("contribNote").textContent = ""; return; }
  const l = r.config.loss, last = r.train[r.train.length - 1];
  const terms = [["Lpred", "L_pred", last.pred, 1, true], ["Lphys", "L_phys", last.phys, l.lambda_phys, true], ["Lprobe", l.lambda_probe ? "L_probe" : "L_probe (disattivato)", last.probe || 0, l.lambda_probe, false], ["Ltheta", "‖θ‖²", last.theta, l.lambda_theta, true]];
  const tot = terms.reduce((a, t) => a + (t[2] || 0) * t[3], 0);
  $("contribNote").textContent = `step ${last.step.toLocaleString("it-IT")} · loss totale registrata ${fmt(last.loss, 4)}`;
  $("contribTable").innerHTML = `<tr><th>termine</th><th>valore</th><th>peso ${tm("lam", "λ")}</th><th>contributo</th><th>% della loss</th><th>gradiente al corpo</th></tr>` +
    terms.map(([id, nm, v, w, g]) => `<tr><td>${tm(id, nm)}</td><td class="mono">${fmt(v, 4)}</td><td class="mono">${w}</td><td class="mono">${fmt(v * w, 4)}</td><td class="mono">${tot ? (100 * v * w / tot).toFixed(1) : "–"}%</td><td>${g ? "sì" : "no"}</td></tr>`).join("");
}
function renderTrace() {
  const dm = dims(), r = curRun(), pg = r && r.param_groups;
  const selI = $("traceI"), selJ = $("traceJ");
  const fillSel = (sel, def) => { const v = sel.value; sel.innerHTML = [...Array(dm.n).keys()].map((i) => `<option value="${i}">${i}</option>`).join(""); sel.value = v !== "" && +v < dm.n ? v : String(def); };
  fillSel(selI, 0); fillSel(selJ, Math.min(1, dm.n - 1));
  $("traceJwrap").hidden = traceKind === "node";
  let i = +selI.value, j = +selJ.value;
  if (traceKind === "pair" && i === j) { j = (i + 1) % dm.n; selJ.value = j; }
  const stages = MI.traceStages(dm, { kind: traceKind, i, j }, pg);
  const roleLabel = { node: "percorso nodo", pair: "percorso coppia", both: "scambio nodo ↔ coppia" };
  $("traceFlow").innerHTML = `<div class="trace-head">${traceKind === "node" ? `Segui il sito <b>${i}</b> (token \\(h_{${i},t}\\))` : `Segui la coppia <b>(${i}, ${j})</b> (feature \\(e_{${i}${j},t}\\))`}</div>` + stages.map((st) => `
    <div class="stage ${st.role}"><div class="stage-h"><span class="sn">${st.n}</span><span class="st">${st.t}</span><span class="sr">${roleLabel[st.role]}</span></div>
      <dl class="stage-b">${st.rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("")}</dl>
      ${st.cross && st.cross.length ? `<div class="cross">${st.cross.map((c) => `<span>${esc(c)}</span>`).join("")}</div>` : ""}</div>`).join(`<div class="arrow"></div>`);
  renderTraceSide(dm, i, j, pg);
  scheduleTypeset();
}
const PG_LABEL = {
  embed_node: ["Embedding · nodi", "node"], embed_pair: ["Embedding · coppie", "pair"],
  spatial_node: ["Spaziale · attenzione e FFN nodi", "node"], spatial_pair_to_node: ["Spaziale · bias e valori dalle coppie", "pair"], spatial_pair_update: ["Spaziale · aggiornamento coppie", "pair"],
  temporal_node: ["Temporale · MHA e FFN nodi", "node"], temporal_pair: ["Temporale · convoluzione coppie", "pair"],
  final_ln_node: ["LayerNorm finale nodi", "node"], final_ln_pair: ["LayerNorm finale coppie", "pair"],
  head_node_and_generator: ["Head · latenti, θ, basi, decoder", "node"], head_pair: ["Head · gate e W_A (coppie)", "pair"], probe: ["Probe", "other"],
};
function renderTraceSide(dm, i, j, pg) {
  const s = D.sample, mk_ = s && sampleKeys().find((k) => isModel(k));
  let vals = "";
  if (s) {
    const p = mk_ && s.preds[mk_], T = s.T, Kp = p ? p.node.length : 0;
    const f = (v) => fmt(v, 3);
    if (traceKind === "node") {
      const rows = [["Ingresso a t = T", `⟨X⟩ ${f(s.node[T][i][0])} · ⟨Y⟩ ${f(s.node[T][i][1])} · ⟨Z⟩ ${f(s.node[T][i][2])}`],
        ["Verità a k = 1", s.node[T + 1][i].map(f).join(" · ")], ["Verità a k = " + Kp, Kp ? s.node[T + Kp][i].map(f).join(" · ") : "–"]];
      if (p) rows.push(["Previsione a k = 1", p.node[0][i].map(f).join(" · ")], ["Previsione a k = " + Kp, p.node[Kp - 1][i].map(f).join(" · ")]);
      if (s.probe) rows.push([`${tm("probe_h", "ĥ")}<sub>${i}</sub> contro ${tm("hi", "h")}<sub>${i}</sub>`, `${f(s.probe.h[i])} contro ${f(s.h[i])}`]);
      vals = rows;
    } else {
      const e = s.edge_sel;
      const rows = [["Ingresso a t = T (xx · yz · zz)", `${f(e.xx[T][i][j])} · ${f(e.yz[T][i][j])} · ${f(e.zz[T][i][j])}`],
        ["Verità ⟨ZZ⟩ a k = 1 e k = " + Kp, Kp ? `${f(e.zz[T + 1][i][j])} · ${f(e.zz[T + Kp][i][j])}` : "–"]];
      if (p) rows.push(["Previsione ⟨ZZ⟩ a k = 1 e k = " + Kp, `${f(p.edge_sel.zz[0][i][j])} · ${f(p.edge_sel.zz[Kp - 1][i][j])}`]);
      if (s.gate) rows.push([`${tm("gate", "gate g")}<sub>${i}${j}</sub> · adiacenza vera`, `${f(s.gate[i][j])} · ${s.adj[i][j]}`]);
      if (s.attn) rows.push([`${tm("alpha", "ᾱ")}<sub>${i}${j}</sub>`, f(s.attn[i][j])]);
      if (s.probe) rows.push([`${tm("probe_J", "Ĵ")}<sub>${i}${j}</sub> contro ${tm("Jij", "J")}<sub>${i}${j}</sub>`, `${f(s.probe.J[i][j])} contro ${f(s.J[i][j])}`], [`${tm("probe_a", "â")}<sub>${i}${j}</sub>`, f(s.probe.adj[i][j])]);
      vals = rows;
    }
  }
  const valsHTML = vals ? `<table class="vals">${vals.map(([k, v]) => `<tr><td>${k}</td><td class="mono">${v}</td></tr>`).join("")}</table>` : `<p class="cap">Nessuna traiettoria di esempio nei dati caricati.</p>`;
  const optPill = (opt) => { if (!opt) return ""; const on = !!dm[opt]; return ` <span class="pill ${on ? "good" : "na"}">${on ? "affrontato da" : "affrontabile con"} ${opt}${on ? " (attivo)" : ""}</span>`; };
  const asym = MI.ASYMMETRY.map(([t, d, src, opt]) => `<li><b>${esc(t)}.</b> ${esc(d)} <span class="mono src">${esc(src)}</span>${optPill(opt)}</li>`).join("");
  const cand = MI.CANDIDATES.map(([t, d, opt]) => `<li><b>${esc(t)}.</b> ${esc(d)}${opt ? ` <span class="pill ${dm[opt] ? "good" : "na"}">${dm[opt] ? "implementato, attivo" : "implementato, non attivo in questa run"}: ${opt}</span>` : ""}</li>`).join("");
  $("traceSide").innerHTML = `
    <h3>Dove finisce nell'esempio ${traceKind === "node" ? `il sito ${i}` : `la coppia (${i}, ${j})`}</h3>${valsHTML}
    <h3>Dove sta la capacità ${tm("pgroups", "(parametri per gruppo)")}</h3>
    <div class="chart" style="height:${pg ? 300 : 60}px"><canvas id="cParams"></canvas></div>
    <p class="cap" id="pgSummary"></p>
    <h3>Perché le coppie potrebbero imparare meno dei nodi</h3>
    <p class="cap">Fatti verificabili nel codice, non diagnosi sicure: spiegano perché l'architettura attuale dà più strumenti ai nodi che alle coppie.</p>
    <ul class="bul">${asym}</ul>
    <h3>Interventi candidati</h3><ul class="bul">${cand}</ul>`;
  if (!pg) { setEmpty("cParams", "Conteggio per gruppo non presente in questo export (serve il checkpoint)"); $("pgSummary").textContent = ""; return; }
  const ent = Object.entries(pg).map(([k, v]) => ({ k, v, l: (PG_LABEL[k] || [k, "other"])[0], role: (PG_LABEL[k] || [k, "other"])[1] })).sort((a, b) => b.v - a.v);
  const col = (role) => (role === "node" ? css("--s1") : role === "pair" ? css("--s2") : css("--ink-3"));
  mk("cParams", { type: "bar", data: { labels: ent.map((e) => e.l), datasets: [{ data: ent.map((e) => e.v), backgroundColor: ent.map((e) => col(e.role)), borderRadius: 3 }] },
    options: { indexAxis: "y", plugins: { tooltip: { callbacks: { label: (it) => ` ${it.parsed.x.toLocaleString("it-IT")} parametri`, footer: () => wrap(shortDef("pgroups")) } } },
      scales: { x: axis("parametri", { type: "logarithmic" }), y: axis(null, { grid: { display: false }, ticks: { font: { size: 11 } } }) } } });
  const sum = (role) => ent.filter((e) => e.role === role).reduce((a, e) => a + e.v, 0);
  $("pgSummary").innerHTML = `<span class="sw" style="background:${css("--s1")}"></span>percorso nodo ${sum("node").toLocaleString("it-IT")} · <span class="sw" style="background:${css("--s2")}"></span>percorso coppia ${sum("pair").toLocaleString("it-IT")} parametri. Scala logaritmica: la convoluzione temporale delle coppie è di ordini di grandezza più piccola dell'attenzione temporale dei nodi.`;
}

/* ---------------------------------------------------------------- 05 training */
function renderTraining() {
  const r = curRun();
  if (!r) { ["cLoss", "cContrib", "cProbe", "cPhys", "cSF", "cLr", "cGn"].forEach((c) => setEmpty(c, "Nessuna run con log.jsonl nei dati caricati")); ["lgLoss", "lgContrib", "lgProbe"].forEach((l) => ($(l).innerHTML = "")); $("valTable").innerHTML = ""; renderContrib(); return; }
  const a = +$("smoothSel").value, tr = r.train, x = tr.map((d) => d.step), lc = r.config.loss;
  const line = (key, color, label, sm = true, scale = 1) => ({ label, data: (sm ? ema(tr.map((d) => (d[key] == null ? null : d[key] * scale)), a) : tr.map((d) => (d[key] == null ? null : d[key] * scale))).map((v, i) => ({ x: x[i], y: v })), borderColor: color, backgroundColor: color });
  const sx = axis("step", { type: "linear", min: 0 });
  const opts = (ylog, ytitle, foot) => ({ interaction: hoverIdx, parsing: false, plugins: foot ? { tooltip: { callbacks: { footer: () => wrap(`${GLOSS[foot].name}: ${shortDef(foot)}`) } } } : {}, scales: { x: sx, y: axis(ytitle, ylog ? { type: "logarithmic" } : { beginAtZero: true }) } });
  legend("lgLoss", [{ label: "loss totale", color: css("--s1"), term: "Ltot" }, { label: "L_pred", color: css("--s2"), term: "Lpred" }]);
  mk("cLoss", { type: "line", data: { datasets: [line("loss", css("--s1"), "loss totale"), line("pred", css("--s2"), "L_pred")] }, options: opts(true, null, "Lpred") });
  legend("lgContrib", [{ label: "L_pred", color: css("--s2"), term: "Lpred" }, { label: "λ_phys·L_phys", color: css("--s3"), term: "Lphys" }, { label: "λ_probe·L_probe", color: css("--s5"), term: "Lprobe" }, { label: "λ_θ·‖θ‖²", color: css("--s7"), term: "Ltheta" }]);
  mk("cContrib", { type: "line", data: { datasets: [line("pred", css("--s2"), "L_pred"), line("phys", css("--s3"), "λ_phys·L_phys", true, lc.lambda_phys), line("probe", css("--s5"), "λ_probe·L_probe", true, lc.lambda_probe), line("theta", css("--s7"), "λ_θ·‖θ‖²", true, lc.lambda_theta)] }, options: opts(true, null, "lam") });
  legend("lgProbe", [{ label: "MSE h", color: css("--s1"), term: "probe_h" }, { label: "MSE J", color: css("--s2"), term: "probe_J" }, { label: "BCE adiacenza", color: css("--s3"), term: "probe_a" }]);
  const hasProbe = tr.some((d) => d.probe_h != null);
  if (!hasProbe) { $("lgProbe").innerHTML = ""; setEmpty("cProbe", "Probe disattivato in questa run (λ_probe = 0, model.probe = false)"); }
  else mk("cProbe", { type: "line", data: { datasets: [line("probe_h", css("--s1"), "MSE h"), line("probe_J", css("--s2"), "MSE J"), line("probe_adj", css("--s3"), "BCE adiacenza")] }, options: opts(true, null, "probe") });
  mk("cPhys", { type: "line", data: { datasets: [line("phys", css("--s1"), "L_phys")] }, options: opts(false, null, "Lphys") });
  mk("cSF", { type: "line", data: { datasets: [line("S_F", css("--s1"), "‖S‖_F")] }, options: opts(false, null, "SF") });
  $("sfCap").textContent = useS(r) ? "gauge di dissipazione, atteso → 0 su sistema chiuso" : "dissipazione disattivata (use_S = false): vale 0 per costruzione";
  mk("cLr", { type: "line", data: { datasets: [line("lr", css("--s1"), "lr", false)] }, options: opts(false, null, "lr") });
  mk("cGn", { type: "line", data: { datasets: [line("grad_norm", css("--s1"), "‖∇‖")] }, options: opts(true, null, "gradnorm") });
  $("evalEvery").textContent = r.config.train.eval_every;
  renderConvergence(r);
  const rowsV = [];
  (r.val || []).forEach((v) => {
    const Tset = [...new Set(Object.keys(v).filter((k) => /^T\d+\//.test(k)).map((k) => +k.slice(1, k.indexOf("/"))))].sort((p, q) => p - q);
    Tset.forEach((T) => { const g = (k) => v[`T${T}/${k}`]; rowsV.push([v.step, T, g("rmse_n_k1"), g("rmse_n_kH"), g("rmse_e_kH"), g("k_star"), g("auroc_gate"), g("auroc_attn"), g("probe_r2_J"), g("probe_r2_h"), g("S_F"), g("fid_mean16"), g("gen_auroc_adj"), g("gen_corr_J"), g("gen_corr_h")]); });
  });
  $("valTable").innerHTML = rowsV.length ? `<tr><th>${tm("step", "step")}</th><th>${tm("T", "T")}</th><th>${tm("rmse_n", "RMSE<sub>n</sub>(1)")}</th><th>${tm("rmse_n", "RMSE<sub>n</sub>(H)")}</th><th>${tm("rmse_e", "RMSE<sub>e</sub>(H)")}</th><th>${tm("kstar", "k*")}</th><th>${tm("auroc", "AUROC")} ${tm("gate", "gate")}</th><th>AUROC ${tm("alpha", "attn")}</th><th>${tm("r2", "R²")} ${tm("probe_J", "Ĵ")}</th><th>R² ${tm("probe_h", "ĥ")}</th><th>${tm("SF", "‖S‖<sub>F</sub>")}</th><th>${tm("fidmean", "F̄₂")}</th><th>${tm("gen_auroc_adj", "AUROC")} G</th><th>${tm("gen_corr_J", "corr J")}</th><th>${tm("gen_corr_h", "corr h")}</th></tr>` +
    rowsV.map((rw) => `<tr>${rw.map((v, i) => `<td class="mono">${i < 2 || i === 5 ? v : fmt(v, i <= 4 ? 4 : 3)}</td>`).join("")}</tr>`).join("") : `<tr><td class="muted">Nessuna validazione registrata (eval_every non raggiunto).</td></tr>`;
  renderContrib();
}

function renderConvergence(r) {
  const val = r.val || [];
  const Ts = [...new Set(val.flatMap((v) => Object.keys(v).filter((k) => /^T\d+\//.test(k)).map((k) => +k.slice(1, k.indexOf("/")))))].sort((p, q) => p - q);
  const cols = [css("--s1"), css("--s2"), css("--s3"), css("--s5"), css("--s7")];
  const draw = (id, key, ytitle, extra = {}) => {
    const ds = Ts.map((T, ti) => ({ label: `T = ${T}`, data: val.filter((v) => v[`T${T}/${key}`] != null).map((v) => ({ x: v.step, y: v[`T${T}/${key}`] })), borderColor: cols[ti % cols.length], backgroundColor: cols[ti % cols.length], pointRadius: 3, borderWidth: 1.6 })).filter((d) => d.data.length);
    if (!ds.length) { setEmpty(id, key === "fid_mean16" ? "Non registrata in questa run (validazione con le sole metriche vecchie)" : "Nessuna validazione registrata"); return; }
    mk(id, { type: "line", data: { datasets: ds }, options: { parsing: false, interaction: hoverIdx, scales: { x: axis("step", { type: "linear", min: 0 }), y: axis(ytitle, extra) } } });
  };
  legend("lgConvF", Ts.map((T, ti) => ({ label: `T = ${T}`, color: cols[ti % cols.length], term: "T" })));
  draw("cConvF", "fid_mean16", "F̄₂", { min: 0, max: 1 });
  draw("cConvR", "rmse_n_k1", "RMSE_n(1)", { type: "logarithmic" });
  draw("cConvK", "k_star", "k*", { beginAtZero: true });
  $("convCap").textContent = val.length ? `${val.length} validazioni · ultimo step ${val[val.length - 1].step.toLocaleString("it-IT")}` : "nessuna validazione registrata";
}

/* ---------------------------------------------------------------- 06 dettagli */
function renderDetails() {
  const r = curRun();
  if (!r) { $("hpTable").innerHTML = `<tr><td class="muted">Nessuna run caricata.</td></tr>`; $("paramNote").textContent = ""; }
  else {
    const c = r.config, m = c.model, t = c.train, o = c.optim, l = c.loss;
    $("paramNote").textContent = r.params ? `${r.params.toLocaleString("it-IT")} parametri (spec: ≈1.2M local, ≈9M HPC)` : "";
    const rows = [
      ["n", "n", "qubit nel training", t.n, 8, 10], ["dtok", "d", "larghezza token di sito", m.d, 128, 256], ["de", "d_e", "larghezza feature di coppia", m.de, 32, 64],
      ["nh", "n_h", "teste di attenzione", m.nh, 4, 8], ["Lbl", "L", "blocchi", m.L, 3, 6], ["m", "m", "dimensione latente per sito", m.m, 32, 64], ["K", "K", "coefficienti del generatore θ", m.K, 32, 64],
      ["T", "T", "contesto (campionato)", `U{${t.T_min}…${t.T_max}}`, "U{8…64}", "U{16…128}"], ["H", "H", "orizzonte supervisionato", t.H, 16, 32], ["anchors", "anchors", "anchor per traiettoria", t.anchors, 2, 4],
      ["batch", "batch", "traiettorie per step", t.batch_size, 8, 32], ["step", "steps", "step di ottimizzazione", t.steps, "~20k", "150k"], ["lr", "lr", "picco / minimo", `${o.lr} / ${o.lr_min}`, "3e-4 / 3e-5", "3e-4 / 3e-5"],
      ["warmup", "warm-up", "step di warm-up", o.warmup, 2000, 2000], ["lam", "λ", "phys / probe / θ", `${l.lambda_phys} / ${l.lambda_probe} / ${l.lambda_theta}`, "0.1 / 1 / 1e-4", "0.1 / 1 / 1e-4"],
      ["headA", "head", "A = generatore, B = ablation", m.head, "A", "A"], ["SF", "use_S", "dissipazione S attiva", m.use_S === undefined ? "sì (config senza flag)" : m.use_S ? "sì" : "no", "sì", "sì"],
    ];
    $("hpTable").innerHTML = `<tr><th>simbolo</th><th>significato</th><th>questa run</th><th>spec local</th><th>spec HPC</th></tr>` +
      rows.map((x) => `<tr><td class="mono">${tm(x[0], esc(x[1]))}</td><td style="text-align:left">${x[2]}</td><td class="mono">${esc(x[3])}</td><td class="muted mono">${x[4]}</td><td class="muted mono">${x[5]}</td></tr>`).join("");
  }
  const ds = D.datasets || [];
  $("dsTable").innerHTML = ds.length ? `<tr><th>file</th><th>famiglia</th><th>${tm("n", "n")}</th><th>traiettorie</th><th>L<sub>traj</sub></th><th>${tm("dt", "Δt")}</th><th>grado medio</th><th>${tm("Jij", "J")} archi (μ ± σ)</th><th>${tm("hi", "h")} (μ ± σ)</th><th>${tm("traj", "seed")}</th></tr>` +
    ds.map((d) => `<tr><td class="mono">${esc(d.path)}</td><td>${esc(d.attrs.family)}</td><td class="mono">${d.attrs.n}</td><td class="mono">${d.attrs.ntraj}</td><td class="mono">${d.attrs.L_traj}</td><td class="mono">${d.attrs.dt}</td><td class="mono">${fmt(d.mean_degree, 2)}</td><td class="mono">${fmt(d.J_on_edges[0], 2)} ± ${fmt(d.J_on_edges[1], 2)}</td><td class="mono">${fmt(d.h_stats[0], 2)} ± ${fmt(d.h_stats[1], 2)}</td><td class="mono">${d.attrs.base_seed}</td></tr>`).join("")
    : `<tr><td class="muted">Nessun dataset esportato.</td></tr>`;
  const e = SRC.entry || {};
  const kv = [["export", e.label || SRC.name || (SRC.kind === "demo" ? "demo" : "")], ["generato", D.generated_at], ["commit (export)", D.git_commit], ["run", r && r.name], ["commit (run)", r && r.config.git_commit], ["preset", r && r.config.preset],
    ["train / val / test", r && `${r.config.data.train} · ${r.config.data.val} · ${r.config.data.test}`], ["seed", r && r.config.train.seed], ["set di risultati", (D.results || []).map((x) => x.path).join(", ")], ["note", D.notes]];
  $("prov").innerHTML = kv.filter((x) => x[1] !== undefined && x[1] !== null && x[1] !== "").map(([k, v]) => `<dt>${k}</dt><dd>${esc(v)}</dd>`).join("");
  $("cfgJson").textContent = r ? JSON.stringify(r.config, null, 2) : "—";
  $("foot").textContent = `Schema ${D.schema || "?"} · Implementazione di propagator_spec.pdf v0.1 · shot noise (§2.5) e stadio Lindblad (§5.3.4) non implementati in v0.1.`;
}

/* ---------------------------------------------------------------- 07 glossario */
function renderGlossary() {
  const ids = Object.keys(GLOSS).sort((a, b) => GLOSS[a].name.localeCompare(GLOSS[b].name, "it", { sensitivity: "base" }));
  $("glossCount").textContent = `${ids.length} voci`;
  $("glossList").innerHTML = ids.map((id) => { const g = GLOSS[id];
    return `<article class="gl" id="g-${id}" data-name="${esc((g.name + " " + id).toLowerCase())}"><h3>${pretty(g.name)}</h3><p>${esc(g.def)}</p>${g.formula ? `<div class="math">\\[ ${g.formula} \\]</div>` : ""}
      <dl class="kv">${g.range ? `<dt>valori</dt><dd>${esc(g.range)}</dd>` : ""}${g.good ? `<dt>atteso</dt><dd>${esc(g.good)}</dd>` : ""}<dt>fonte</dt><dd>${esc(g.source || "")}</dd></dl></article>`; }).join("");
}
$("glossFilter").addEventListener("input", (e) => { const q = e.target.value.trim().toLowerCase(); document.querySelectorAll("#glossList .gl").forEach((el) => { el.hidden = q && !el.dataset.name.includes(q); }); });

/* ---------------------------------------------------------------- cablaggio e caricamento */
function fillSelectors() {
  $("runSel").innerHTML = runs().map((r, i) => `<option value="${i}">${esc(r.name)}${r.config ? ` · ${r.config.preset} · head ${r.config.model.head}` : ""}</option>`).join("") || `<option value="0">—</option>`;
  const rs = D.results || [];
  $("resSel").innerHTML = (rs.length > 1 ? `<option value="all">tutti (unione)</option>` : "") + rs.map((r, i) => `<option value="${i}">${esc(r.name)}</option>`).join("") || `<option value="all">—</option>`;
  if (rs.length > 1) $("resSel").value = "all";
}
function renderAll() {
  applyDefaults();
  renderOverview(); renderEval(); renderSampleControls(); renderSample(); renderGraph(); renderModelInfo(); renderTrace(); renderContrib(); renderTraining(); renderDetails();
  scheduleTypeset(); checkTerms();
}
function setData(json, src) {
  D = json; SRC = src; evalT = null; $("fatal").hidden = true; fillSelectors(); renderAll();
  const sel = $("exportSel");
  if (src.kind === "export") sel.value = src.file; else if (src.kind === "demo") sel.value = "__demo__"; else if (src.kind === "file") sel.value = "__file__";
}
function segClick(id, fn) {
  $(id).addEventListener("click", (e) => { const b = e.target.closest("button"); if (!b) return;
    $(id).querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", x === b)); fn(b.dataset.v); });
}
segClick("Tseg", (v) => { evalT = +v; renderEval(); scheduleTypeset(); });
segClick("NEseg", (v) => { evalNE = v; renderEval(); scheduleTypeset(); });
segClick("kindSeg", (v) => { kind = v; renderSampleControls(); renderSample(); });
segClick("traceKind", (v) => { traceKind = v; renderTrace(); });
["traceI", "traceJ", "traceArch"].forEach((id) => $(id).addEventListener("change", renderTrace));
["siteI", "siteJ", "chanSel", "xUnit"].forEach((id) => $(id).addEventListener("change", renderSample));
["stripMethod", "stripChan"].forEach((id) => $(id).addEventListener("change", renderStrips));
$("runSel").addEventListener("change", () => { renderOverview(); renderGraph(); renderModelInfo(); renderTrace(); renderTraining(); renderDetails(); scheduleTypeset(); checkTerms(); });
$("smoothSel").addEventListener("change", () => { renderTraining(); renderRel(); });
$("resSel").addEventListener("change", () => { renderEval(); scheduleTypeset(); });
document.querySelectorAll("details").forEach((d) => d.addEventListener("toggle", () => { Object.values(charts).forEach((c) => c.resize()); scheduleTypeset(); }));

async function fetchJSON(url) {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return r.json();
}
function checkSchema(j) { if (!j || !String(j.schema || "").startsWith("hbicp-dashboard")) throw new Error("il file non è un export di tools/export_dashboard.py (manca il campo schema)"); return j; }
async function loadExport(file) {
  const entry = (MANIFEST.exports || []).find((e) => e.file === file) || { file, label: file };
  const j = exportCache[file] || checkSchema(await fetchJSON("data/" + file));
  exportCache[file] = j;
  setData(j, { kind: "export", file, entry });
}
async function loadDemo(note) {
  const j = checkSchema(await fetchJSON("data/demo.json"));
  setData(j, { kind: "demo", note });
}
function fillExportMenu() {
  const ex = (MANIFEST && MANIFEST.exports) || [];
  $("exportSel").innerHTML = ex.map((e) => `<option value="${esc(e.file)}">${esc(e.label || e.file)}${e.generated_at ? " · " + esc(e.generated_at.slice(0, 16).replace("T", " ")) : ""}</option>`).join("") +
    `<option value="__demo__">Dati demo</option><option value="__file__" hidden>File caricato</option>`;
}
$("exportSel").addEventListener("change", async (e) => {
  const v = e.target.value;
  try { if (v === "__demo__") await loadDemo(); else if (v !== "__file__") await loadExport(v); }
  catch (err) { $("sourceBanner").innerHTML = `<span class="pill bad">Errore</span><span>${esc(err.message)}</span>`; }
});
$("loadBtn").addEventListener("click", () => $("fileIn").click());
$("fileIn").addEventListener("change", (e) => {
  const f = e.target.files[0]; if (!f) return;
  const rd = new FileReader();
  rd.onload = () => {
    try { const j = checkSchema(JSON.parse(rd.result)); $("exportSel").querySelector('option[value="__file__"]').hidden = false; setData(j, { kind: "file", name: f.name }); }
    catch (err) { $("sourceBanner").innerHTML = `<span class="pill bad">Errore</span><span>Impossibile leggere ${esc(f.name)}: ${esc(err.message)}. Rigenera il file con <span class="mono">python -m tools.export_dashboard</span>.</span>`; }
    e.target.value = "";
  };
  rd.readAsText(f);
});

/* tema: auto -> chiaro -> scuro */
function themeLabel() { const t = document.documentElement.dataset.theme; $("themeBtn").textContent = "Tema: " + (t === "light" ? "chiaro" : t === "dark" ? "scuro" : "auto"); }
try { const t = localStorage.getItem("hbicp-theme"); if (t === "light" || t === "dark") document.documentElement.dataset.theme = t; } catch (_) { /* storage non disponibile */ }
themeLabel();
$("themeBtn").addEventListener("click", () => {
  const cur = document.documentElement.dataset.theme, nxt = cur === undefined ? "light" : cur === "light" ? "dark" : null;
  if (nxt) document.documentElement.dataset.theme = nxt; else document.documentElement.removeAttribute("data-theme");
  try { nxt ? localStorage.setItem("hbicp-theme", nxt) : localStorage.removeItem("hbicp-theme"); } catch (_) { /* ignorato */ }
  themeLabel();
});
let rt; const rerender = () => { clearTimeout(rt); rt = setTimeout(() => { renderAll(); }, 50); };
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", rerender);
new MutationObserver(rerender).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });

(async function boot() {
  renderEquations(); renderLossCards(); renderGlossary();
  let indexErr = null;
  try {
    MANIFEST = await fetchJSON("data/index.json?v=" + Date.now());
    fillExportMenu();
    if (!MANIFEST.latest) { await loadDemo(); waitMathJax(); return; }   // nessun export ancora pubblicato
    await loadExport(MANIFEST.latest);
  } catch (err) {
    indexErr = err.message; MANIFEST = MANIFEST || { exports: [] };
    if (!$("exportSel").options.length) fillExportMenu();
    try { await loadDemo(`export pubblicato non disponibile: ${indexErr}`); }
    catch (err2) { renderAll(); $("sourceBanner").innerHTML = `<span class="pill bad">Errore</span><span>Nessun dato disponibile.</span>`; showFatal(`${indexErr}; demo: ${err2.message}`); }
  }
  waitMathJax();
})();
})();
