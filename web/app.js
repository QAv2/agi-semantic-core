// Oracle web app — Pyodide diagnosis + LLM reflection chat.
// The geometry runs in your browser; the Reflective Principle (LLM)
// runs on a Cloudflare Worker. No query leaves your device for the
// diagnosis itself — only the resulting reading and your follow-up
// messages are sent to the worker.

const PYODIDE_VERSION = '0.27.2';
const PYODIDE_INDEX_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
const BUNDLE_URL = 'assets/oracle-bundle.tar.gz';

// Worker URL — auto-detect dev vs prod
const ORACLE_API_URL = (() => {
  const host = location.hostname;
  if (host === 'localhost' || host === '127.0.0.1' || host === '0.0.0.0') {
    return 'http://127.0.0.1:8787/';
  }
  return 'https://oracle-api.qav2.workers.dev/';
})();

const SESSION_URL = ORACLE_API_URL.replace(/\/+$/, '') + '/session';

// Cloudflare Turnstile site key — public value, paired with TURNSTILE_SECRET
// in the worker. Issued by dash.cloudflare.com → Turnstile, scoped to
// qav2-oracle.netlify.app, Managed mode.
const TURNSTILE_SITE_KEY = '0x4AAAAAADD35pS7dnkiQTWs';

const TRIGRAM_SYMBOLS = {
  QIAN: '☰', KUN: '☷', ZHEN: '☳', KAN: '☵',
  GEN: '☶', XUN: '☴', LI: '☲', DUI: '☱',
};

const STORAGE_KEY = 'oracle:sessions';
const PREFS_KEY = 'oracle:prefs';

const $ = (id) => document.getElementById(id);

const state = {
  pyodide: null,
  oracleReady: false,
  busy: false,
  diagnosis: null,           // current Oracle reading
  history: [],               // chat messages [{role, content}] sent to worker
  sessionId: null,
  authSession: null,         // { sid, expires_at } — HMAC-signed auth session, in-memory only
  // Layer filter: each lit system contributes a reading to the LLM
  // context AND opens by default in the panel. Max MAX_ACTIVE_LAYERS
  // (3) at once — protects against payload bloat as more systems get
  // added. Default: all current systems active (lit).
  layers: { kingwen: true, toltec: true, tarot: true },
  theme: 'twilight',         // 'twilight' | 'mineral' | 'parchment'
  customColors: null,        // { '--bg': '#xxx', '--surface': '#xxx', '--text': '#xxx', '--gold': '#xxx' } | null
  panelMode: 'closed',       // 'dock-right' | 'dock-left' | 'closed'
};

// ─── Boot ─────────────────────────────────────────────────────

function setStep(step, status) {
  document.querySelectorAll(`[data-step="${step}"]`).forEach((el) => {
    el.dataset.state = status;
  });
}
function setBootLine(text) { $('boot-line').textContent = text; }
function setBootProgress(pct) { $('boot-bar').style.width = `${pct}%`; }

async function bootstrap() {
  try {
    // Pre-mint auth session in parallel with engine load — Turnstile +
    // /session round-trip overlaps Pyodide bootstrap, so the first user
    // message doesn't wait. Failure is non-fatal here; askMirror will
    // re-attempt at send time.
    mintAuthSession().catch((err) => {
      console.warn('[oracle] preempt auth session failed:', err.message || err);
    });

    setBootLine('loading runtime…');
    setBootProgress(5);
    setStep('pyodide', 'active');
    state.pyodide = await loadPyodide({ indexURL: PYODIDE_INDEX_URL });
    setStep('pyodide', 'done');

    setBootLine('loading numpy + sqlite…');
    setBootProgress(35);
    setStep('numpy', 'active');
    await state.pyodide.loadPackage(['numpy', 'sqlite3']);
    setStep('numpy', 'done');

    setBootLine('fetching dictionary…');
    setBootProgress(60);
    setStep('bundle', 'active');
    const res = await fetch(BUNDLE_URL);
    if (!res.ok) throw new Error(`bundle fetch failed: ${res.status}`);
    const bytes = await res.arrayBuffer();
    setStep('bundle', 'done');

    setBootLine('unpacking…');
    setBootProgress(75);
    setStep('unpack', 'active');
    state.pyodide.unpackArchive(bytes, 'gztar');
    state.pyodide.runPython("import sys; sys.path.insert(0, '/home/pyodide')");
    setStep('unpack', 'done');

    setBootLine('indexing concepts…');
    setBootProgress(92);
    setStep('engine', 'active');
    const warmupJson = await state.pyodide.runPythonAsync(
      'from oracle.web_api import warmup; warmup()'
    );
    const warmup = JSON.parse(warmupJson);
    if (!warmup.ready) throw new Error('engine failed to warm up');
    setStep('engine', 'done');

    setBootProgress(100);
    setBootLine(`oracle ready · ${warmup.concepts} concepts`);
    state.oracleReady = true;
    enableComposer();

    setTimeout(() => {
      $('status').classList.add('fade-out');
      setTimeout(() => $('status').classList.add('hidden'), 600);
    }, 700);
  } catch (err) {
    showError(`bootstrap failed: ${err.message || err}`);
    console.error(err);
  }
}

function enableComposer() {
  $('composer-input').disabled = false;
  $('send-btn').disabled = false;
  $('chips').hidden = false;
  $('composer-input').focus();
}

// ─── Diagnosis (local Pyodide call) ───────────────────────────

async function runDiagnosis(text) {
  if (!state.oracleReady) throw new Error('oracle not ready');
  state.pyodide.globals.set('_query', text);
  const json = await state.pyodide.runPythonAsync(
    'from oracle.web_api import diagnose; diagnose(_query)'
  );
  return JSON.parse(json);
}

// ─── Turnstile + signed auth session ──────────────────────────
// Turnstile is rendered in 'interaction-only' appearance — invisible
// for trusted visitors, brief challenge for flagged ones. On a valid
// token, /session mints an HMAC-signed sid we send as X-Oracle-Session
// on every /reflect call. 30-min TTL, in-memory only (no localStorage).

let _turnstileWidgetId = null;

async function _waitForTurnstile() {
  if (typeof window.turnstile !== 'undefined') return;
  if (window.__turnstileReady) await window.__turnstileReady;
  if (typeof window.turnstile === 'undefined') {
    throw new Error('Turnstile failed to load');
  }
}

function _getToken() {
  return new Promise((resolve, reject) => {
    const mount = $('turnstile-mount');
    if (!mount) return reject(new Error('Turnstile mount missing'));
    if (_turnstileWidgetId !== null) {
      try { window.turnstile.remove(_turnstileWidgetId); } catch {}
      _turnstileWidgetId = null;
    }
    _turnstileWidgetId = window.turnstile.render(mount, {
      sitekey: TURNSTILE_SITE_KEY,
      appearance: 'interaction-only',
      retry: 'never',
      callback: (token) => resolve(token),
      'error-callback': () => reject(new Error('Turnstile error')),
      'expired-callback': () => reject(new Error('Turnstile expired')),
      'timeout-callback': () => reject(new Error('Turnstile timeout')),
    });
  });
}

async function mintAuthSession() {
  await _waitForTurnstile();
  const token = await _getToken();
  const res = await fetch(SESSION_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ turnstile_token: token }),
  });
  if (!res.ok) {
    let msg = `session_mint_failed: ${res.status}`;
    try { const e = await res.json(); if (e.error) msg = e.error; } catch {}
    throw new Error(msg);
  }
  const data = await res.json();
  state.authSession = { sid: data.session_id, expires_at: data.expires_at };
  return state.authSession;
}

async function getAuthSession() {
  const a = state.authSession;
  // 30s skew — re-mint if we're within 30s of expiry
  if (a && a.expires_at && a.expires_at > Math.floor(Date.now() / 1000) + 30) {
    return a;
  }
  return mintAuthSession();
}

// ─── Coffer meter ─────────────────────────────────────────────

const BALANCE_URL = ORACLE_API_URL.replace(/\/+$/, '') + '/balance';

async function fetchCofferState() {
  try {
    const res = await fetch(BALANCE_URL);
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

function renderCofferMeter(state) {
  if (!state) {
    $('coffer-meter').hidden = true;
    return;
  }
  const meter = $('coffer-meter');
  const balance = (state.balance_cents / 100).toFixed(2);
  $('cm-balance').textContent = balance;

  const modeEl = $('cm-mode');
  if (state.mode === 'claude') {
    modeEl.textContent = 'Claude active';
    modeEl.className = 'coffer-mode mode-claude';
  } else {
    let reason = '';
    if (state.reason === 'pot-empty') reason = 'pot dry';
    else if (state.reason === 'daily-cap') reason = 'daily cap reached';
    else if (state.reason === 'no-key') reason = 'no API key';
    else if (state.reason === 'claude-error') reason = 'Claude unavailable';
    modeEl.textContent = reason ? `Llama (free) · ${reason}` : 'Llama (free)';
    modeEl.className = 'coffer-mode mode-llama';
  }
  meter.hidden = false;
}

async function refreshCofferMeter() {
  const state = await fetchCofferState();
  renderCofferMeter(state);
}

// ─── LLM call (Cloudflare Worker) ─────────────────────────────

async function askMirror(message) {
  if (!state.diagnosis) throw new Error('no diagnosis to reflect on');
  const body = JSON.stringify({
    message,
    history: state.history,
    diagnosis: filteredDiagnosis(),
  });

  async function send(sid) {
    return fetch(ORACLE_API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Oracle-Session': sid },
      body,
    });
  }

  let auth = await getAuthSession();
  let res = await send(auth.sid);

  if (res.status === 401) {
    // Session expired or rotated — silently re-challenge once and retry.
    state.authSession = null;
    auth = await mintAuthSession();
    res = await send(auth.sid);
  }

  if (!res.ok) {
    let msg = `worker error: ${res.status}`;
    try {
      const err = await res.json();
      if (err.error) msg = err.error;
    } catch {}
    throw new Error(msg);
  }
  const data = await res.json();
  return data.response;
}

// ─── Submission flow ──────────────────────────────────────────

async function handleSubmit(text) {
  text = text.trim();
  if (!text || state.busy || !state.oracleReady) return;
  state.busy = true;
  setBusy(true);

  document.body.classList.add('has-messages');

  // Render the user message immediately
  appendUserMsg(text);
  const composer = $('composer-input');
  composer.value = '';
  resizeComposer();

  try {
    let messageForLLM = text;
    if (!state.diagnosis) {
      // First message of session: run Oracle diagnosis, populate the
      // sticky reading bar (compact summary), prep the panel content
      // for on-demand expansion. No diag card injected into chat stream.
      const diagnosis = await runDiagnosis(text);
      state.diagnosis = diagnosis;
      state.sessionId = `s_${Date.now().toString(36)}`;
      showReadingBar(diagnosis);
      populatePanel(diagnosis);
      updateSidebar();
      saveSession();
      messageForLLM = `(opening turn — the user just said: "${text}". The reading has been generated and you have it in your system context. Speak to them like a thoughtful person who has just read what they're sitting with. Don't recite the structure. Say something real about what the reading shows, in plain language, and let the conversation begin. Skip greetings and preambles.)`;
    }

    // Add user message to LLM history
    state.history.push({ role: 'user', content: messageForLLM });

    // Show thinking indicator
    const mirrorEl = appendMirrorMsg('', { thinking: true });
    let response;
    try {
      response = await askMirror(messageForLLM);
    } catch (err) {
      mirrorEl.remove();
      throw err;
    }

    state.history.push({ role: 'assistant', content: response });
    saveSession();
    renderMirrorMsg(mirrorEl, response);
    scrollToBottom();
    // Refresh the coffer meter — balance just changed (or we just fell back)
    refreshCofferMeter();
  } catch (err) {
    showError(err.message || String(err));
    console.error(err);
  } finally {
    state.busy = false;
    setBusy(false);
    composer.focus();
  }
}

function setBusy(busy) {
  $('send-btn').disabled = busy || !state.oracleReady;
  $('composer-input').disabled = busy || !state.oracleReady;
}

// ─── Message rendering ────────────────────────────────────────

function appendUserMsg(text) {
  const el = document.createElement('div');
  el.className = 'msg msg-user';
  el.textContent = text;
  $('messages').appendChild(el);
  scrollToBottom();
  return el;
}

function appendMirrorMsg(text, opts = {}) {
  const el = document.createElement('div');
  el.className = 'msg msg-mirror' + (opts.thinking ? ' thinking' : '');
  const inner = document.createElement('div');
  inner.className = 'msg-mirror-text';
  inner.textContent = text;
  el.appendChild(inner);
  $('messages').appendChild(el);
  scrollToBottom();
  return el;
}

function renderMirrorMsg(el, text) {
  el.classList.remove('thinking');
  const inner = el.querySelector('.msg-mirror-text');
  inner.innerHTML = '';
  // Split by double newline to make paragraphs
  const paragraphs = text.split(/\n\n+/).filter(Boolean);
  paragraphs.forEach((para) => {
    const p = document.createElement('p');
    p.textContent = para.trim();
    inner.appendChild(p);
  });
  scrollToBottom();
}

// ─── Reading bar + panel ──────────────────────────────────────

function showReadingBar(d) {
  const bar = $('reading-bar');
  $('rb-p-symbol').textContent = TRIGRAM_SYMBOLS[d.presenting.trigram] || '·';
  $('rb-p-name').textContent = `${d.presenting.trigram} · ${d.presenting.element}`;
  $('rb-m-symbol').textContent = TRIGRAM_SYMBOLS[d.medicine.trigram] || '·';
  $('rb-medicine').textContent = d.medicine.concept;
  $('rb-hex').textContent = `#${d.hexagram.number}`;
  $('rb-hex-name').textContent = d.hexagram.name;
  const w = $('rb-witness');
  w.textContent = d.witness_state;
  w.className = `rb-witness-state state-${d.witness_state}`;
  bar.hidden = false;
}

function hideReadingBar() {
  $('reading-bar').hidden = true;
  closePanel();
}

function populatePanel(d) {
  $('panel-body').innerHTML = renderDiagHtml(d);
}

const PANEL_MODES = ['dock-right', 'dock-left', 'closed'];

function applyPanelMode() {
  const panel = $('reading-panel');
  panel.classList.remove('reading-panel-dock-right', 'reading-panel-dock-left');

  if (state.panelMode === 'closed') {
    panel.hidden = true;
    return;
  }
  panel.hidden = false;
  panel.classList.add(`reading-panel-${state.panelMode}`);
}

function setPanelMode(mode) {
  if (!PANEL_MODES.includes(mode)) return;
  state.panelMode = mode;
  applyPanelMode();
  savePrefs();
}

function openPanel() {
  if (!state.diagnosis) return;
  if (state.panelMode === 'closed') {
    setPanelMode(state._lastOpenMode || 'dock-right');
  } else {
    applyPanelMode();
  }
}

function closePanel() {
  if (state.panelMode !== 'closed') state._lastOpenMode = state.panelMode;
  setPanelMode('closed');
}

function togglePanel() {
  if (state.panelMode === 'closed') openPanel();
  else closePanel();
}

function flipPanelSide() {
  if (state.panelMode === 'dock-right') setPanelMode('dock-left');
  else if (state.panelMode === 'dock-left') setPanelMode('dock-right');
}


const MAX_ACTIVE_LAYERS = 3;

function activeLayerCount() {
  return Object.values(state.layers).filter(Boolean).length;
}

function isLayerOpen(layer) {
  // Active layers open by default in the panel; inactive collapsed.
  // (Inactive layers still render in the panel — user can manually expand.)
  return !!state.layers[layer];
}

function renderDiagHtml(d) {
  const tokens = d.tokens.map((t) => `<span class="diag-token">${escapeHtml(t)}</span>`).join('');
  const missed = d.missed.map((t) => `<span class="diag-token missed">${escapeHtml(t)}</span>`).join('');
  const tokensBlock = (tokens || missed)
    ? `<div class="diag-tokens">${tokens}${missed}</div>`
    : '<span class="diag-tokens">no recognized concepts</span>';

  const domainRows = ['spatial', 'temporal', 'relational', 'personal'].map((k) => {
    const v = d.presenting.domain[k] ?? 0;
    const isDom = k === d.presenting.dominant;
    const sign = v >= 0 ? '+' : '';
    return `
      <div class="domain-row ${isDom ? 'dominant' : ''}">
        <span>${k}</span>
        <span>${sign}${v.toFixed(2)}</span>
        <div class="domain-track"><div class="domain-fill" style="width:${Math.min(100, Math.abs(v) * 100)}%"></div></div>
      </div>`;
  }).join('');

  const witnessPct = Math.min(100, (d.witness / 1.6) * 100);

  const toltecBlock = d.toltec ? `
    <details class="diag-layer" ${isLayerOpen('toltec') ? 'open' : ''}>
      <summary>
        <span class="diag-layer-icon">☷</span>
        <span>Toltec correspondence</span>
        <span class="diag-layer-pointer">#${d.toltec.number}: ${escapeHtml(d.toltec.name)}</span>
      </summary>
      <div class="diag-layer-body">
        <h4>Image</h4><p>${escapeHtml(d.toltec.image || '—')}</p>
        <h4>Interpretation</h4><p>${escapeHtml(d.toltec.interp || '—')}</p>
        <h4>Action</h4><p>${escapeHtml(d.toltec.action || '—')}</p>
        <h4>Intent</h4><p>${escapeHtml(d.toltec.intent || '—')}</p>
        <h4>Summary</h4><p>${escapeHtml(d.toltec.summary || '—')}</p>
      </div>
    </details>` : '';

  const tarotBlock = d.tarot ? `
    <details class="diag-layer" ${isLayerOpen('tarot') ? 'open' : ''}>
      <summary>
        <span class="diag-layer-icon">✦</span>
        <span>Tarot correspondence</span>
        <span class="diag-layer-pointer">${d.tarot.presenting.numeral}. ${escapeHtml(d.tarot.presenting.name)} ⟶ ${d.tarot.medicine.numeral}. ${escapeHtml(d.tarot.medicine.name)}</span>
      </summary>
      <div class="diag-layer-body">
        <div class="tarot-pair">
          ${tarotMiniHtml(d.tarot.presenting, 'PRESENTING')}
          ${tarotMiniHtml(d.tarot.medicine, 'MEDICINE')}
        </div>
      </div>
    </details>` : '';

  const judgmentBlock = d.hexagram.judgment ? `
    <details class="diag-layer" ${isLayerOpen('kingwen') ? 'open' : ''}>
      <summary>
        <span class="diag-layer-icon">☰</span>
        <span>King Wen interpretation</span>
        <span class="diag-layer-pointer">judgment · image · counsel</span>
      </summary>
      <div class="diag-layer-body">
        <h4>Judgment</h4><p>${escapeHtml(d.hexagram.judgment)}</p>
        <h4>Image</h4><p>${escapeHtml(d.hexagram.image || '—')}</p>
        <h4>Counsel</h4><p>${escapeHtml(d.hexagram.counsel || '—')}</p>
      </div>
    </details>` : '';

  return `
    <div class="diag-head">
      <span>oracle reading</span>
      ${tokensBlock}
    </div>

    <div class="diag-axes">
      <div class="diag-axis">
        <div class="diag-axis-label">Presenting</div>
        <div>
          <span class="diag-axis-symbol">${TRIGRAM_SYMBOLS[d.presenting.trigram] || '?'}</span>
          <span class="diag-axis-name">${escapeHtml(d.presenting.trigram)} · ${escapeHtml(d.presenting.element)}</span>
        </div>
        <div class="diag-axis-meaning">${escapeHtml(d.presenting.meaning)}</div>
      </div>
      <div class="diag-arrow">⟶</div>
      <div class="diag-axis">
        <div class="diag-axis-label">Medicine</div>
        <div>
          <span class="diag-axis-symbol">${TRIGRAM_SYMBOLS[d.medicine.trigram] || '?'}</span>
          <span class="diag-axis-name">${escapeHtml(d.medicine.trigram)} · ${escapeHtml(d.medicine.element)}</span>
        </div>
        <div class="diag-medicine">${escapeHtml(d.medicine.concept)}</div>
      </div>
    </div>

    <div class="diag-domain">${domainRows}</div>

    <div class="diag-section">
      <div class="diag-section-label">Hexagram</div>
      <div class="diag-hex-line">
        <span class="diag-hex-num">#${String(d.hexagram.number).padStart(2, '0')}</span>
        <span class="diag-hex-name">${escapeHtml(d.hexagram.name)}</span>
        <span class="diag-hex-title">${escapeHtml(d.hexagram.title)}</span>
      </div>
      <div class="diag-hex-trigrams">${escapeHtml(d.hexagram.upper)} above · ${escapeHtml(d.hexagram.lower)} below</div>
      <div class="diag-hex-reading">${escapeHtml(d.hexagram.reading || '')}</div>
    </div>

    <div class="diag-section">
      <div class="diag-section-label">Witness</div>
      <div class="diag-witness">
        <span class="diag-witness-value">${d.witness.toFixed(3)}</span>
        <span class="diag-witness-state state-${d.witness_state}">${d.witness_state}</span>
        <div class="diag-witness-bar-track">
          <div class="diag-witness-bar state-${d.witness_state}" style="width:${witnessPct}%"></div>
        </div>
      </div>
    </div>

    ${judgmentBlock}
    ${toltecBlock}
    ${tarotBlock}
  `;
}

function tarotMiniHtml(card, role) {
  return `
    <div class="tarot-card-mini">
      <div class="tarot-mini-role">${role}</div>
      <div class="tarot-mini-numeral">${escapeHtml(card.numeral)}</div>
      <div class="tarot-mini-name">${escapeHtml(card.name)}</div>
      <div class="tarot-mini-angle">nearest at ${card.angle.toFixed(1)}°</div>
      <h4>Image</h4><p>${escapeHtml(card.image)}</p>
      <h4>Upright</h4><p>${escapeHtml(card.upright)}</p>
      <h4>Counsel</h4><p>${escapeHtml(card.counsel)}</p>
    </div>
  `;
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text == null ? '' : String(text);
  return div.innerHTML;
}

function scrollToBottom() {
  const chat = $('chat');
  requestAnimationFrame(() => { chat.scrollTop = chat.scrollHeight; });
}

// ─── Sidebar ──────────────────────────────────────────────────

function updateSidebar() {
  const summary = $('current-summary');
  if (!state.diagnosis) {
    summary.innerHTML = '<p class="empty">No reading yet.</p>';
    return;
  }
  const d = state.diagnosis;
  summary.innerHTML = `
    <div class="summary-input">"${escapeHtml(d.input)}"</div>
    <div class="summary-line"><span class="lbl">presenting</span><span class="val">${escapeHtml(d.presenting.trigram)}</span></div>
    <div class="summary-line"><span class="lbl">medicine</span><span class="val">${escapeHtml(d.medicine.concept)}</span></div>
    <div class="summary-line"><span class="lbl">hexagram</span><span class="val">#${d.hexagram.number} ${escapeHtml(d.hexagram.name)}</span></div>
    <div class="summary-line"><span class="lbl">witness</span><span class="val">${d.witness.toFixed(2)} (${escapeHtml(d.witness_state)})</span></div>
  `;
  refreshHistoryList();
}

function refreshHistoryList() {
  const sessions = loadAllSessions();
  const list = $('history-list');
  list.innerHTML = '';
  if (!sessions.length) {
    list.innerHTML = '<li class="empty">none yet</li>';
    return;
  }
  sessions.forEach((s) => {
    const li = document.createElement('li');
    if (s.id === state.sessionId) li.classList.add('active');
    li.dataset.id = s.id;
    const date = new Date(parseInt(s.id.slice(2), 36));
    li.innerHTML = `
      <div class="history-item-input">"${escapeHtml(s.diagnosis.input)}"</div>
      <div class="history-item-meta">
        ${escapeHtml(s.diagnosis.presenting.trigram)} → ${escapeHtml(s.diagnosis.medicine.concept)}
        · ${date.toLocaleDateString()}
      </div>
    `;
    li.addEventListener('click', () => restoreSession(s.id));
    list.appendChild(li);
  });
}

// ─── Persistence ──────────────────────────────────────────────

function loadAllSessions() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch { return []; }
}

function saveSession() {
  if (!state.sessionId || !state.diagnosis) return;
  const sessions = loadAllSessions();
  const idx = sessions.findIndex((s) => s.id === state.sessionId);
  const entry = {
    id: state.sessionId,
    diagnosis: state.diagnosis,
    history: state.history,
  };
  if (idx >= 0) sessions[idx] = entry;
  else sessions.unshift(entry);
  // cap to last 30 sessions
  const capped = sessions.slice(0, 30);
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(capped)); } catch {}
}

function restoreSession(id) {
  const sessions = loadAllSessions();
  const s = sessions.find((x) => x.id === id);
  if (!s) return;
  state.diagnosis = s.diagnosis;
  state.history = s.history || [];
  state.sessionId = s.id;
  $('messages').innerHTML = '';
  document.body.classList.add('has-messages');
  showReadingBar(s.diagnosis);
  populatePanel(s.diagnosis);
  // Replay: just the chat messages (reading lives in the bar/panel now)
  appendUserMsg(s.diagnosis.input);
  s.history.forEach((m, i) => {
    if (m.role === 'user' && i === 0) return; // already shown as the input
    if (m.role === 'user') appendUserMsg(m.content);
    if (m.role === 'assistant') {
      const el = appendMirrorMsg('');
      renderMirrorMsg(el, m.content);
    }
  });
  updateSidebar();
  closeSidebar();
}

function newReading() {
  state.diagnosis = null;
  state.history = [];
  state.sessionId = null;
  $('messages').innerHTML = '';
  document.body.classList.remove('has-messages');
  hideReadingBar();
  $('composer-input').value = '';
  resizeComposer();
  updateSidebar();
  closeSidebar();
  $('composer-input').focus();
}

// ─── Layer prefs ──────────────────────────────────────────────

const THEMES = ['twilight', 'mineral', 'parchment'];
const CUSTOM_VAR_KEYS = ['--bg', '--surface', '--text', '--gold'];

function isHexColor(s) {
  return typeof s === 'string' && /^#[0-9a-f]{6}$/i.test(s);
}

function loadPrefs() {
  try {
    const raw = localStorage.getItem(PREFS_KEY);
    if (raw) {
      const p = JSON.parse(raw);
      Object.assign(state.layers, p.layers || {});
      if (PANEL_MODES.includes(p.panelMode)) {
        state.panelMode = p.panelMode === 'closed' ? 'closed' : p.panelMode;
        state._lastOpenMode = p.panelMode === 'closed' ? 'dock-right' : p.panelMode;
      }
      if (THEMES.includes(p.theme)) state.theme = p.theme;
      if (p.customColors && typeof p.customColors === 'object') {
        const valid = {};
        for (const k of CUSTOM_VAR_KEYS) {
          if (isHexColor(p.customColors[k])) valid[k] = p.customColors[k];
        }
        if (Object.keys(valid).length > 0) state.customColors = valid;
      }
    }
  } catch {}
}

function savePrefs() {
  try {
    localStorage.setItem(PREFS_KEY, JSON.stringify({
      layers: state.layers,
      panelMode: state.panelMode,
      theme: state.theme,
      customColors: state.customColors,
    }));
  } catch {}
}

function applyTheme() {
  document.documentElement.setAttribute('data-theme', state.theme);
  document.querySelectorAll('.theme-option').forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.theme === state.theme);
  });
  applyCustomColors();
  syncColorPickers();
}

// All CSS variables that get derived from the four user-picked primaries.
// Listed here so we know what to clear when reverting.
const DERIVED_VAR_KEYS = [
  '--bg', '--bg-deep', '--bg-translucent', '--bg-translucent-strong',
  '--surface', '--surface-2', '--surface-3',
  '--surface-translucent', '--surface-translucent-soft',
  '--panel-header-bg-1', '--panel-header-bg-2',
  '--text', '--text-dim', '--text-faint',
  '--border', '--border-soft',
  '--gold', '--gold-soft', '--gold-deep',
];

function hexToRgb(hex) {
  const m = /^#([0-9a-f]{6})$/i.exec(hex);
  if (!m) return null;
  const n = parseInt(m[1], 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function rgbToHex([r, g, b]) {
  const c = (x) => Math.max(0, Math.min(255, Math.round(x))).toString(16).padStart(2, '0');
  return '#' + c(r) + c(g) + c(b);
}

function mix(c1, c2, t) {
  return [
    c1[0] + (c2[0] - c1[0]) * t,
    c1[1] + (c2[1] - c1[1]) * t,
    c1[2] + (c2[2] - c1[2]) * t,
  ];
}

function rgbaStr([r, g, b], a) {
  return `rgba(${Math.round(r)}, ${Math.round(g)}, ${Math.round(b)}, ${a})`;
}

function deriveCustomVars(customs) {
  const out = {};
  const bg = customs['--bg'] && hexToRgb(customs['--bg']);
  const surface = customs['--surface'] && hexToRgb(customs['--surface']);
  const text = customs['--text'] && hexToRgb(customs['--text']);
  const gold = customs['--gold'] && hexToRgb(customs['--gold']);
  const black = [0, 0, 0];

  if (bg) {
    out['--bg'] = rgbToHex(bg);
    out['--bg-deep'] = rgbToHex(mix(bg, black, 0.4));
    out['--bg-translucent'] = rgbaStr(bg, 0.88);
    out['--bg-translucent-strong'] = rgbaStr(bg, 0.94);
  }
  if (surface) {
    out['--surface'] = rgbToHex(surface);
    out['--surface-translucent'] = rgbaStr(surface, 0.95);
    out['--surface-translucent-soft'] = rgbaStr(surface, 0.88);
    out['--panel-header-bg-2'] = rgbaStr(surface, 0.92);
    if (text) {
      out['--surface-2'] = rgbToHex(mix(surface, text, 0.10));
      out['--surface-3'] = rgbToHex(mix(surface, text, 0.18));
      out['--panel-header-bg-1'] = rgbaStr(mix(surface, text, 0.08), 0.92);
    }
  }
  if (text) {
    out['--text'] = rgbToHex(text);
    if (bg) {
      out['--text-dim'] = rgbToHex(mix(text, bg, 0.40));
      out['--text-faint'] = rgbToHex(mix(text, bg, 0.65));
      out['--border'] = rgbToHex(mix(bg, text, 0.18));
      out['--border-soft'] = rgbToHex(mix(bg, text, 0.10));
    }
  }
  if (gold) {
    out['--gold'] = rgbToHex(gold);
    out['--gold-soft'] = rgbToHex(mix(gold, black, 0.20));
    out['--gold-deep'] = rgbToHex(mix(gold, black, 0.40));
  }
  return out;
}

function applyCustomColors() {
  const root = document.documentElement;
  // Clear all derived vars first so revert/theme-switch works clean
  for (const key of DERIVED_VAR_KEYS) {
    root.style.removeProperty(key);
  }
  if (state.customColors) {
    const derived = deriveCustomVars(state.customColors);
    for (const [key, val] of Object.entries(derived)) {
      root.style.setProperty(key, val);
    }
  }
}

function setTheme(name) {
  if (!THEMES.includes(name)) return;
  state.theme = name;
  applyTheme();
  savePrefs();
}

function setCustomColor(varKey, hex) {
  if (!CUSTOM_VAR_KEYS.includes(varKey) || !isHexColor(hex)) return;
  if (!state.customColors) state.customColors = {};
  state.customColors[varKey] = hex;
  applyCustomColors();
  savePrefs();
}

function resetCustomColors() {
  state.customColors = null;
  applyCustomColors();
  syncColorPickers();
  savePrefs();
}

// Sync color picker inputs with the currently-effective values (after
// any active theme + custom overrides resolve through CSS).
function syncColorPickers() {
  const cs = getComputedStyle(document.documentElement);
  document.querySelectorAll('.color-row input[type="color"]').forEach((input) => {
    const target = input.dataset.target;
    if (!target) return;
    const v = cs.getPropertyValue(target).trim();
    const hex = colorToHex(v);
    if (hex) input.value = hex;
  });
}

// Convert "rgb(r, g, b)" or "#xxx" or "#xxxxxx" to "#xxxxxx" for color inputs.
function colorToHex(c) {
  if (!c) return null;
  c = c.trim();
  if (/^#[0-9a-f]{6}$/i.test(c)) return c.toLowerCase();
  if (/^#[0-9a-f]{3}$/i.test(c)) {
    return ('#' + c.slice(1).split('').map((x) => x + x).join('')).toLowerCase();
  }
  const m = c.match(/^rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)/);
  if (m) {
    return '#' + [m[1], m[2], m[3]].map((n) => parseInt(n, 10).toString(16).padStart(2, '0')).join('');
  }
  return null;
}

function applyLayerToggleStates() {
  document.querySelectorAll('.layer-toggle').forEach((btn) => {
    const layer = btn.dataset.layer;
    const on = !!state.layers[layer];
    btn.classList.toggle('active', on);
    btn.setAttribute('aria-pressed', on ? 'true' : 'false');
  });
}

function toggleLayer(layer) {
  const turningOn = !state.layers[layer];
  if (turningOn && activeLayerCount() >= MAX_ACTIVE_LAYERS) {
    flashLayerCap(layer);
    return;
  }
  state.layers[layer] = turningOn;
  applyLayerToggleStates();
  savePrefs();
  const iconMap = { kingwen: '☰', toltec: '☷', tarot: '✦' };
  document.querySelectorAll('.panel-body .diag-layer').forEach((det) => {
    const icon = det.querySelector('.diag-layer-icon')?.textContent;
    const layerName = Object.keys(iconMap).find((k) => iconMap[k] === icon);
    if (layerName) det.open = isLayerOpen(layerName);
  });
}

// Brief visual feedback when a click would exceed the cap.
function flashLayerCap(layer) {
  const btn = document.querySelector(`.layer-toggle[data-layer="${layer}"]`);
  if (!btn) return;
  btn.classList.add('layer-cap-flash');
  setTimeout(() => btn.classList.remove('layer-cap-flash'), 600);
  showToast(`max ${MAX_ACTIVE_LAYERS} systems active — deselect one first`, 'info');
}

// Filter the diagnosis JSON before sending to the worker. Drops layers
// the user has turned off — saves tokens, especially as more systems are
// added. Always preserves presenting/medicine/hexagram core.
function filteredDiagnosis() {
  if (!state.diagnosis) return null;
  const d = { ...state.diagnosis };
  if (!state.layers.kingwen) {
    d.hexagram = { ...d.hexagram, judgment: '', image: '', counsel: '' };
  }
  if (!state.layers.toltec) d.toltec = null;
  if (!state.layers.tarot) d.tarot = null;
  return d;
}

// ─── Composer ─────────────────────────────────────────────────

function resizeComposer() {
  const ta = $('composer-input');
  ta.style.height = 'auto';
  ta.style.height = Math.min(ta.scrollHeight, 200) + 'px';
}

// ─── Sidebar helpers ──────────────────────────────────────────

function openSidebar() { document.body.classList.add('sidebar-open'); }
function closeSidebar() { document.body.classList.remove('sidebar-open'); }
function toggleSidebar() { document.body.classList.toggle('sidebar-open'); }

// ─── About dialog ─────────────────────────────────────────────

function openAbout() {
  const dlg = $('about-dialog');
  if (typeof dlg.showModal === 'function') {
    dlg.showModal();
  } else {
    dlg.setAttribute('open', '');
  }
}
function closeAbout() {
  const dlg = $('about-dialog');
  if (typeof dlg.close === 'function') {
    dlg.close();
  } else {
    dlg.removeAttribute('open');
  }
}

// ─── Errors ───────────────────────────────────────────────────

function showError(message) {
  const existing = document.querySelector('.error-toast');
  if (existing) existing.remove();
  const toast = document.createElement('div');
  toast.className = 'error-toast';
  toast.innerHTML = `<div class="error-title">ERROR</div><div>${escapeHtml(message)}</div>`;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 6000);
}

function showToast(message, kind = 'info') {
  const existing = document.querySelector('.error-toast');
  if (existing) existing.remove();
  const toast = document.createElement('div');
  toast.className = `error-toast toast-${kind}`;
  toast.innerHTML = `<div>${escapeHtml(message)}</div>`;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 3500);
}

// ─── Init ─────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  loadPrefs();
  applyTheme();
  applyLayerToggleStates();
  refreshHistoryList();
  refreshCofferMeter();

  // Theme switcher
  document.querySelectorAll('.theme-option').forEach((btn) => {
    btn.addEventListener('click', () => setTheme(btn.dataset.theme));
  });

  // Custom color picker
  const customizerToggle = $('theme-customize-toggle');
  const customizer = $('theme-customizer');
  customizerToggle.addEventListener('click', () => {
    const open = customizer.hidden;
    customizer.hidden = !open;
    customizerToggle.setAttribute('aria-expanded', open ? 'true' : 'false');
    if (open) syncColorPickers();
  });
  document.querySelectorAll('.color-row input[type="color"]').forEach((input) => {
    input.addEventListener('input', (e) => setCustomColor(input.dataset.target, e.target.value));
  });
  $('theme-reset').addEventListener('click', resetCustomColors);

  // Composer events
  const composer = $('composer-input');
  $('composer-form').addEventListener('submit', (e) => {
    e.preventDefault();
    handleSubmit(composer.value);
  });
  composer.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(composer.value);
    }
  });
  composer.addEventListener('input', resizeComposer);

  // Chips
  document.querySelectorAll('.chip').forEach((chip) => {
    chip.addEventListener('click', () => {
      composer.value = chip.textContent;
      resizeComposer();
      handleSubmit(chip.textContent);
    });
  });

  // Layer toggles
  document.querySelectorAll('.layer-toggle').forEach((btn) => {
    btn.addEventListener('click', () => toggleLayer(btn.dataset.layer));
  });

  // Sidebar
  $('hamburger').addEventListener('click', toggleSidebar);
  $('sidebar-close').addEventListener('click', closeSidebar);
  $('overlay').addEventListener('click', closeSidebar);
  $('new-reading-btn').addEventListener('click', newReading);

  // Reset
  $('reset-btn').addEventListener('click', () => {
    if (state.diagnosis && !confirm('Start a new reading?')) return;
    newReading();
  });

  // About
  $('about-btn').addEventListener('click', () => { openAbout(); closeSidebar(); });
  $('welcome-about').addEventListener('click', openAbout);
  $('about-close').addEventListener('click', closeAbout);
  $('about-dialog').addEventListener('click', (e) => {
    if (e.target === $('about-dialog')) closeAbout();
  });

  // Reading bar / panel
  $('reading-bar-toggle').addEventListener('click', togglePanel);
  $('panel-close').addEventListener('click', closePanel);
  $('panel-flip').addEventListener('click', flipPanelSide);
  $('panel-minimize').addEventListener('click', closePanel);
  // Esc closes panel
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && state.panelMode !== 'closed') closePanel();
  });

  // Service worker — skip in local dev (avoids stale cache during iteration).
  // Unregister any existing SW + clear caches so a hot reload is clean.
  const isDev = ['localhost', '127.0.0.1', '0.0.0.0'].includes(location.hostname);
  if ('serviceWorker' in navigator) {
    if (isDev) {
      navigator.serviceWorker.getRegistrations().then((regs) => {
        regs.forEach((r) => r.unregister());
      });
      if (typeof caches !== 'undefined') {
        caches.keys().then((keys) => keys.forEach((k) => caches.delete(k)));
      }
    } else {
      navigator.serviceWorker.register('service-worker.js').catch((err) => {
        console.warn('service worker registration failed:', err);
      });
    }
  }

  bootstrap();
});
