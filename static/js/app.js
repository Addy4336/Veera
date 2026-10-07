/**
 * Vera Voice AI Agent - Web Client & Real-time Telemetry Dashboard
 * Features:
 * - Daily.co WebRTC & WebSocket Voice Streaming
 * - Live Latency & Pipeline Metrics HUD
 * - Canvas Audio Waveform Visualizer
 * - Real-time Interruption (Barge-in) Handling
 * - Dynamic Flow State Machine Progression
 */

// State Variables
let isCallActive = false;
let wsMetrics = null;
let wsAudio = null;
let audioContext = null;
let captureContext = null;
let captureProcessor = null;
let mediaStream = null;
let audioAnalyser = null;
let animationFrameId = null;
let dailyCallFrame = null;

// Audio playback scheduling (server TTS PCM @16kHz s16le mono)
const SERVER_SAMPLE_RATE = 16000;
let playbackContext = null;
let nextPlayTime = 0;
let isBotSpeaking = false;

// DOM Elements
const btnStartCall = document.getElementById('btnStartCall');
const btnInterrupt = document.getElementById('btnInterrupt');
const avatarContainer = document.getElementById('avatarContainer');
const avatarStateLabel = document.getElementById('avatarStateLabel');
const canvas = document.getElementById('waveCanvas');
const canvasCtx = canvas.getContext('2d');
const transcriptFeed = document.getElementById('transcriptFeed');

// Metrics Cards
const metricStt = document.getElementById('valStt');
const metricTtft = document.getElementById('valTtft');
const metricTtfa = document.getElementById('valTtfa');
const metricTotal = document.getElementById('valTotal');
const badgeBargeIns = document.getElementById('valBargeIns');

// Provider Badges
const pillGroqStt = document.getElementById('pillGroqStt');
const pillGroqLlm = document.getElementById('pillGroqLlm');
const pillCartesiaTts = document.getElementById('pillCartesiaTts');
const pillWebRtc = document.getElementById('pillWebRtc');

// Stepper Nodes
const flowNodes = ['Greeting', 'IntentCapture', 'OrderStatus', 'ScheduleCallback', 'GeneralFAQ', 'Resolution', 'Close'];

// ============================================================================
// Initialization
// ============================================================================

window.addEventListener('DOMContentLoaded', () => {
  setupCanvas();
  connectMetricsWebSocket();
  checkSystemStatus();
  attachEventListeners();
  drawIdleVisualizer();
});

function attachEventListeners() {
  btnStartCall.addEventListener('click', toggleCall);
  btnInterrupt.addEventListener('click', triggerBargeIn);

  // Quick Demo Buttons
  document.querySelectorAll('.chip-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      const text = e.target.getAttribute('data-prompt');
      if (text) sendSimulatedSpeech(text);
    });
  });
}

// ============================================================================
// System Status & Setup
// ============================================================================

async function checkSystemStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    console.log('[System Status]', data);

    if (data.providers.cartesia_tts.includes('Fallback')) {
      pillCartesiaTts.classList.add('warning');
      pillCartesiaTts.querySelector('.name').innerText = 'TTS: Kokoro Fallback';
    }
    updateActiveNode(data.current_flow_node || 'Greeting');
  } catch (err) {
    console.warn('Could not fetch status:', err);
  }
}

function setupCanvas() {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  canvasCtx.scale(dpr, dpr);
}

// ============================================================================
// Telemetry WebSocket
// ============================================================================

function connectMetricsWebSocket() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws/metrics`;

  wsMetrics = new WebSocket(wsUrl);

  wsMetrics.onopen = () => {
    console.log('⚡ Connected to Vera Telemetry Stream');
  };

  wsMetrics.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      handleTelemetryEvent(msg);
    } catch (e) {
      console.error('Error parsing metrics frame:', e);
    }
  };

  wsMetrics.onclose = () => {
    setTimeout(connectMetricsWebSocket, 2500);
  };
}

function handleTelemetryEvent(event) {
  const { type, data } = event;

  switch (type) {
    case 'latency_summary':
      updateLatencyHUD(data);
      break;

    case 'flow_node_change':
      updateActiveNode(data.node);
      break;

    case 'tts_provider_change':
      handleTTSProviderChange(data.provider);
      break;

    case 'barge_in':
      handleBargeInAlert(data);
      break;

    case 'tool_call':
      renderToolCallBadge(data);
      break;

    case 'transcription':
      renderChatMessage('user', data.text);
      setAgentState('thinking', 'Thinking...');
      break;

    case 'llm_first_token':
      setAgentState('speaking', 'Speaking');
      break;

    case 'user_speech_start':
      setAgentState('listening', 'Listening');
      break;
  }
}

// ============================================================================
// Visual State & HUD Updates
// ============================================================================

function setAgentState(state, label) {
  avatarContainer.className = `agent-avatar ${state}`;
  avatarStateLabel.innerText = label;

  const icons = { idle: '🎧', listening: '👂', thinking: '🧠', speaking: '🗣️', interrupted: '⚡' };
  const iconEl = document.getElementById('avatarIcon');
  if (iconEl && icons[state]) iconEl.innerText = icons[state];

  if (state === 'speaking') {
    startSpeakingWave();
  } else if (state === 'listening') {
    startListeningWave();
  } else if (state === 'interrupted') {
    avatarContainer.classList.add('interrupted');
    setTimeout(() => {
      if (isCallActive) setAgentState('listening', 'Listening');
    }, 1200);
  }
}

function updateLatencyHUD(metrics) {
  setGauge('arcStt', 'gaugeStt', metrics.stt_latency_ms, 250, 500);
  setGauge('arcTtft', 'gaugeTtft', metrics.llm_ttft_ms, 300, 700);
  setGauge('arcTtfa', 'gaugeTtfa', metrics.tts_ttfa_ms, 150, 400);
  setGauge('arcTotal', 'gaugeTotal', metrics.total_latency_ms, 600, 1200);
}

// Radial gauge driver: sets arc fill, center value, and SLA color state
const GAUGE_CIRCUMFERENCE = 263.9; // 2*PI*r, r=42
const GAUGE_MAX_MS = 1500;

function setGauge(arcId, valueId, ms, goodThreshold, warnThreshold) {
  if (ms === null || ms === undefined) return;
  const arc = document.getElementById(arcId);
  const val = document.getElementById(valueId);
  const gaugeEl = arc ? arc.closest('.gauge') : null;
  if (!arc || !val) return;

  const clamped = Math.min(ms, GAUGE_MAX_MS);
  arc.style.strokeDashoffset = GAUGE_CIRCUMFERENCE * (1 - clamped / GAUGE_MAX_MS);
  val.innerText = Math.round(ms);

  if (gaugeEl) {
    gaugeEl.classList.remove('sla-good', 'sla-warn', 'sla-alert');
    if (ms <= goodThreshold) gaugeEl.classList.add('sla-good');
    else if (ms <= warnThreshold) gaugeEl.classList.add('sla-warn');
    else gaugeEl.classList.add('sla-alert');
  }
}

function applySlaClass(element, value, goodThreshold, warnThreshold) {
  // Legacy hook kept for compatibility; gauges handle SLA coloring now.
}

const FLOW_ORDER = ['Greeting', 'IntentCapture', 'OrderStatus', 'ScheduleCallback', 'GeneralFAQ', 'Resolution', 'Close'];

function updateActiveNode(nodeName) {
  const activeIdx = FLOW_ORDER.indexOf(nodeName);
  document.querySelectorAll('.flow-node-step').forEach(el => {
    const name = el.getAttribute('data-node');
    const idx = FLOW_ORDER.indexOf(name);
    el.classList.remove('active', 'done');
    if (name === nodeName) {
      el.classList.add('active');
    } else if (activeIdx !== -1 && idx !== -1 && idx < activeIdx) {
      el.classList.add('done');
    }
  });
}

function handleTTSProviderChange(provider) {
  if (provider.includes('fallback') || provider.includes('kokoro')) {
    pillCartesiaTts.classList.add('warning');
    pillCartesiaTts.querySelector('.name').innerText = 'TTS: Kokoro Fallback';
  } else {
    pillCartesiaTts.classList.remove('warning');
    pillCartesiaTts.querySelector('.name').innerText = 'TTS: Cartesia Sonic';
  }
}

function handleBargeInAlert(data) {
  badgeBargeIns.innerText = data.total_barge_ins;
  setAgentState('interrupted', 'Barge-in Interrupt');

  const alertDiv = document.createElement('div');
  alertDiv.className = 'barge-in-alert';
  alertDiv.innerHTML = `⚡ <strong>Barge-in Interruption:</strong> Cancelled in-flight synthesis (Count: ${data.total_barge_ins})`;
  transcriptFeed.appendChild(alertDiv);
  transcriptFeed.scrollTop = transcriptFeed.scrollHeight;
}

// ============================================================================
// Transcript & Messages
// ============================================================================

function renderChatMessage(sender, text) {
  const msgDiv = document.createElement('div');
  msgDiv.className = `chat-msg ${sender}`;

  const tagDiv = document.createElement('div');
  tagDiv.className = 'sender-tag';
  tagDiv.innerText = sender === 'user' ? 'You' : 'Vera';

  const bubbleDiv = document.createElement('div');
  bubbleDiv.className = 'chat-bubble';
  bubbleDiv.innerText = text;

  msgDiv.appendChild(tagDiv);
  msgDiv.appendChild(bubbleDiv);
  transcriptFeed.appendChild(msgDiv);
  transcriptFeed.scrollTop = transcriptFeed.scrollHeight;
}

function renderToolCallBadge(toolData) {
  const badge = document.createElement('div');
  badge.className = 'tool-telemetry-badge';
  badge.innerHTML = `
    <div class="tool-header">🔧 Tool Executed: <strong>${toolData.name}</strong></div>
    <div>Args: <code>${JSON.stringify(toolData.args)}</code></div>
    <div>Status: <span style="color: #4ade80;">Success</span></div>
  `;
  transcriptFeed.appendChild(badge);
  transcriptFeed.scrollTop = transcriptFeed.scrollHeight;
}

// ============================================================================
// Call Lifecycle & Audio Stream
// ============================================================================

async function toggleCall() {
  if (!isCallActive) {
    await startVoiceCall();
  } else {
    endVoiceCall();
  }
}

async function startVoiceCall() {
  try {
    btnStartCall.disabled = true;
    btnStartCall.innerHTML = 'Connecting...';

    // Step 1: Request microphone permissions
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });

    // Step 2: Connect the realtime audio pipeline WebSocket
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    wsAudio = new WebSocket(`${protocol}//${window.location.host}/ws/audio`);
    wsAudio.binaryType = 'arraybuffer';
    wsAudio.onmessage = handleAudioWsMessage;
    wsAudio.onclose = () => { if (isCallActive) endVoiceCall(); };
    await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error('Pipeline WebSocket handshake timed out after 8s')), 8000);
      wsAudio.onopen = () => { clearTimeout(timeout); resolve(); };
      wsAudio.onerror = () => {
        clearTimeout(timeout);
        reject(new Error(`WebSocket connection to ${wsAudio.url} was rejected (check server console)`));
      };
    });

    // Step 3: Mic capture pipeline -> 16kHz mono s16le PCM frames -> wsAudio
    captureContext = new (window.AudioContext || window.webkitAudioContext)();
    // Edge/Safari start AudioContexts suspended until explicitly resumed
    if (captureContext.state === 'suspended') await captureContext.resume();

    const source = captureContext.createMediaStreamSource(mediaStream);
    audioAnalyser = captureContext.createAnalyser();
    audioAnalyser.fftSize = 64;
    source.connect(audioAnalyser);

    captureProcessor = captureContext.createScriptProcessor(4096, 1, 1);
    captureProcessor.onaudioprocess = (e) => {
      if (!isCallActive || wsAudio?.readyState !== WebSocket.OPEN) return;
      const float32 = e.inputBuffer.getChannelData(0);
      const downsampled = downsampleTo16k(float32, captureContext.sampleRate);
      const pcm16 = floatTo16BitPCM(downsampled);
      wsAudio.send(pcm16.buffer);
    };
    source.connect(captureProcessor);
    captureProcessor.connect(captureContext.destination); // keeps processor alive

    // Step 4: Playback context at the server's 16kHz rate
    playbackContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SERVER_SAMPLE_RATE });
    if (playbackContext.state === 'suspended') await playbackContext.resume();

    isCallActive = true;
    btnStartCall.disabled = false;
    btnStartCall.innerHTML = '🔴 End Call';
    btnStartCall.classList.add('active');
    btnInterrupt.disabled = false;

    setAgentState('listening', 'Connected & Listening');
    startSessionTimer();
    showToast('Voice call connected — start talking!', 'success');
    startAudioVisualizer();

  } catch (err) {
    console.error('Failed to start call:', err);
    alert('Microphone access or pipeline connection error: ' + err.message);
    endVoiceCall();
  }
}

// ============================================================================
// Realtime Audio Pipeline Helpers
// ============================================================================

function handleAudioWsMessage(event) {
  if (event.data instanceof ArrayBuffer) {
    playPcmChunk(event.data);
    return;
  }
  try {
    const msg = JSON.parse(event.data);
    switch (msg.type) {
      case 'session_ready':
        console.log('[Audio Pipeline] Session ready:', msg.session_id);
        break;
      case 'transcript':
        renderChatMessage('user', msg.text);
        setAgentState('thinking', 'Thinking...');
        break;
      case 'bot_token':
        setAgentState('speaking', 'Speaking');
        break;
      case 'bot_turn_end':
        if (msg.reply) renderAgentMessageTyping(msg.reply);
        incrementTurnCounter();
        break;
      case 'stt_empty':
        setAgentState('listening', 'Listening');
        break;
      case 'interrupt_ack':
        stopPlayback();
        setAgentState('interrupted', 'Interrupted');
        showToast('Barge-in — Vera stopped listening mid-response', 'info', 2500);
        break;
      case 'error':
        console.error('[Audio Pipeline]', msg.message);
        break;
    }
  } catch (e) {
    console.warn('Bad WS message:', e);
  }
}

function playPcmChunk(arrayBuffer) {
  if (!playbackContext) return;
  const int16 = new Int16Array(arrayBuffer);
  const float32 = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) {
    float32[i] = int16[i] / 32768;
  }
  const audioBuffer = playbackContext.createBuffer(1, float32.length, SERVER_SAMPLE_RATE);
  audioBuffer.copyToChannel(float32, 0);

  const src = playbackContext.createBufferSource();
  src.buffer = audioBuffer;
  src.connect(playbackContext.destination);

  const now = playbackContext.currentTime;
  if (nextPlayTime < now) {
    nextPlayTime = now + 0.05;
    isBotSpeaking = true;
  }
  src.start(nextPlayTime);
  nextPlayTime += audioBuffer.duration;

  src.onended = () => {
    if (nextPlayTime <= playbackContext.currentTime + 0.02) {
      isBotSpeaking = false;
      if (isCallActive) setAgentState('listening', 'Listening');
    }
  };
}

function stopPlayback() {
  if (!playbackContext) return;
  nextPlayTime = playbackContext.currentTime;
  isBotSpeaking = false;
}

function downsampleTo16k(float32Array, inputRate) {
  if (inputRate === SERVER_SAMPLE_RATE) return float32Array;
  const ratio = inputRate / SERVER_SAMPLE_RATE;
  const outLength = Math.floor(float32Array.length / ratio);
  const result = new Float32Array(outLength);
  for (let i = 0; i < outLength; i++) {
    const pos = i * ratio;
    const idx = Math.floor(pos);
    const frac = pos - idx;
    const a = float32Array[idx] || 0;
    const b = float32Array[idx + 1] || a;
    result[i] = a + (b - a) * frac;
  }
  return result;
}

function floatTo16BitPCM(float32Array) {
  const out = new Int16Array(float32Array.length);
  for (let i = 0; i < float32Array.length; i++) {
    const s = Math.max(-1, Math.min(1, float32Array[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}

function endVoiceCall() {
  isCallActive = false;
  btnStartCall.disabled = false;
  btnStartCall.innerHTML = '🎙️ Start Voice Call';
  btnStartCall.classList.remove('active');
  btnInterrupt.disabled = true;

  // Tell the server to end this session
  if (wsAudio && wsAudio.readyState === WebSocket.OPEN) {
    wsAudio.send(JSON.stringify({ type: 'end' }));
  }
  wsAudio = null;
  stopPlayback();
  stopSessionTimer();
  showToast('Call ended', 'info', 2500);

  if (captureProcessor) {
    captureProcessor.disconnect();
    captureProcessor = null;
  }
  if (captureContext) {
    captureContext.close();
    captureContext = null;
  }
  if (playbackContext) {
    playbackContext.close();
    playbackContext = null;
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach(track => track.stop());
    mediaStream = null;
  }
  if (audioContext) {
    audioContext.close();
    audioContext = null;
  }
  if (animationFrameId) {
    cancelAnimationFrame(animationFrameId);
  }

  setAgentState('idle', 'Ready');
  drawIdleVisualizer();
}

function triggerBargeIn() {
  if (!isCallActive) return;
  stopPlayback();
  if (wsAudio && wsAudio.readyState === WebSocket.OPEN) {
    wsAudio.send(JSON.stringify({ type: 'interrupt' }));
  }
  handleBargeInAlert({ total_barge_ins: parseInt(badgeBargeIns.innerText || '0') + 1 });
}

async function sendSimulatedSpeech(text) {
  renderChatMessage('user', text);
  setAgentState('thinking', 'Processing...');

  try {
    const res = await fetch('/api/chat-turn', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();
    setAgentState('speaking', 'Speaking');
    renderChatMessage('agent', data.response_text);
    updateLatencyHUD({
      stt_latency_ms: data.latencies.stt_ms,
      llm_ttft_ms: data.latencies.llm_ttft_ms,
      tts_ttfa_ms: data.latencies.tts_ttfa_ms,
      total_latency_ms: data.latencies.total_ms,
    });
    setTimeout(() => {
      setAgentState('listening', 'Listening');
    }, 2000);
  } catch (e) {
    console.error('Turn error:', e);
  }
}

// ============================================================================
// Canvas Audio Waveform Visualizer
// ============================================================================

function drawIdleVisualizer() {
  const width = canvas.width;
  const height = canvas.height;
  canvasCtx.clearRect(0, 0, width, height);

  canvasCtx.beginPath();
  canvasCtx.arc(width / 2, height / 2, 75, 0, 2 * Math.PI);
  canvasCtx.strokeStyle = 'rgba(99, 102, 241, 0.25)';
  canvasCtx.lineWidth = 2;
  canvasCtx.stroke();
}

function startAudioVisualizer() {
  // Superseded by the always-on drawOrbVisualizer ring (audio-reactive in all states).
}

function startSpeakingWave() {
  // Boost pulse visual when bot speaks
}

function startListeningWave() {
  // Mic input visual
}

// ============================================================================
// Cinematic Visual Modules (v2 UI) - part 1
// ============================================================================

// ---- 1. Particle Starfield Background ----
const starCanvas = document.getElementById('bgStars');
const starCtx = starCanvas ? starCanvas.getContext('2d') : null;
let stars = [];

function initStarfield() {
  if (!starCtx) return;
  starCanvas.width = window.innerWidth;
  starCanvas.height = window.innerHeight;
  stars = [];
  const count = Math.floor((window.innerWidth * window.innerHeight) / 9000);
  for (let i = 0; i < count; i++) {
    stars.push({
      x: Math.random() * starCanvas.width,
      y: Math.random() * starCanvas.height,
      r: Math.random() * 1.4 + 0.3,
      speed: Math.random() * 0.15 + 0.03,
      tw: Math.random() * Math.PI * 2,
    });
  }
}

function drawStarfield() {
  if (!starCtx) return;
  starCtx.clearRect(0, 0, starCanvas.width, starCanvas.height);
  for (const s of stars) {
    s.y -= s.speed;
    if (s.y < -2) { s.y = starCanvas.height + 2; s.x = Math.random() * starCanvas.width; }
    s.tw += 0.03;
    const alpha = 0.25 + Math.abs(Math.sin(s.tw)) * 0.55;
    starCtx.beginPath();
    starCtx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
    starCtx.fillStyle = `rgba(199, 210, 254, ${alpha})`;
    starCtx.fill();
  }
  requestAnimationFrame(drawStarfield);
}

// ---- 2. Session Timer ----
let sessionStartTime = null;
let sessionTimerInterval = null;

function startSessionTimer() {
  sessionStartTime = Date.now();
  if (sessionTimerInterval) clearInterval(sessionTimerInterval);
  sessionTimerInterval = setInterval(() => {
    const el = document.getElementById('sessionTimer');
    if (!el) return;
    const s = Math.floor((Date.now() - sessionStartTime) / 1000);
    el.innerText = `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
  }, 1000);
}

function stopSessionTimer() {
  if (sessionTimerInterval) clearInterval(sessionTimerInterval);
  sessionTimerInterval = null;
  const el = document.getElementById('sessionTimer');
  if (el) el.innerText = '00:00';
}

// ---- 3. Toast Notifications ----
function showToast(message, type = 'info', durationMs = 3500) {
  const container = document.getElementById('toastContainer');
  if (!container) return;
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  const icons = { success: '✅', error: '❌', info: '💡' };
  toast.innerHTML = `<span>${icons[type] || '💡'}</span><span>${message}</span>`;
  container.appendChild(toast);
  setTimeout(() => {
    toast.classList.add('leaving');
    setTimeout(() => toast.remove(), 320);
  }, durationMs);
}

// ---- 4. Audio-Reactive Orb Ring Visualizer (canvas behind the orb) ----
let orbLevel = 0;
let micLevelSmoothed = 0;
let orbAnimStarted = false;
let lastLoudTime = null;
let silenceWarned = false;

function drawOrbVisualizer() {
  if (!canvasCtx) return;
  const rect = canvas.getBoundingClientRect();
  const w = rect.width, h = rect.height;
  canvasCtx.clearRect(0, 0, w, h);

  const centerX = w / 2, centerY = h / 2;
  const baseRadius = 78;

  let level;
  if (audioAnalyser && isCallActive) {
    const bufferLength = audioAnalyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    audioAnalyser.getByteFrequencyData(dataArray);
    let sum = 0;
    for (let i = 0; i < bufferLength; i++) sum += dataArray[i];
    level = Math.min(1, (sum / bufferLength) / 90);
  } else {
    level = 0.12 + Math.sin(Date.now() * 0.0018) * 0.06;
  }
  orbLevel += (level - orbLevel) * 0.15;
  updateMicMeter(level);

  // Silence watchdog: warn once if the mic delivers ~silence for 5s during a call
  if (isCallActive) {
    const now = Date.now();
    if (level >= 0.02) {
      lastLoudTime = now;
      silenceWarned = false;
    } else if (!silenceWarned && lastLoudTime && now - lastLoudTime > 5000) {
      silenceWarned = true;
      showToast('Your microphone appears silent. Check the mic device/mute in Windows sound settings.', 'error', 6000);
    }
  }

  const bars = 48;
  const time = Date.now() * 0.002;
  for (let i = 0; i < bars; i++) {
    const angle = (i * 2 * Math.PI) / bars + time * 0.25;
    const wobble = Math.sin(time * 3 + i * 0.7) * 4;
    const radius = baseRadius + wobble + orbLevel * 42;
    const len = 6 + orbLevel * 30 + Math.abs(Math.sin(time * 2 + i)) * 10;

    const x1 = centerX + Math.cos(angle) * radius;
    const y1 = centerY + Math.sin(angle) * radius;
    const x2 = centerX + Math.cos(angle) * (radius + len);
    const y2 = centerY + Math.sin(angle) * (radius + len);

    const hue = 210 + Math.sin(time + i * 0.3) * 60;
    canvasCtx.beginPath();
    canvasCtx.moveTo(x1, y1);
    canvasCtx.lineTo(x2, y2);
    canvasCtx.strokeStyle = `hsla(${hue}, 90%, 68%, ${0.35 + orbLevel * 0.5})`;
    canvasCtx.lineWidth = 2.4;
    canvasCtx.lineCap = 'round';
    canvasCtx.stroke();
  }

  const halo = canvasCtx.createRadialGradient(centerX, centerY, baseRadius * 0.6, centerX, centerY, baseRadius + 90);
  halo.addColorStop(0, `rgba(129, 140, 248, ${0.10 + orbLevel * 0.25})`);
  halo.addColorStop(1, 'rgba(129, 140, 248, 0)');
  canvasCtx.fillStyle = halo;
  canvasCtx.fillRect(0, 0, w, h);

  requestAnimationFrame(drawOrbVisualizer);
}

function updateMicMeter(level) {
  micLevelSmoothed += (level - micLevelSmoothed) * 0.2;
  const pct = Math.min(100, Math.round(micLevelSmoothed * 140));
  const fill = document.getElementById('micLevelFill');
  const val = document.getElementById('micLevelValue');
  if (fill) fill.style.width = pct + '%';
  if (val) val.innerText = pct + '%';
}

// ---- 5. Typing Effect for Agent Messages ----
function renderAgentMessageTyping(text) {
  const msgDiv = document.createElement('div');
  msgDiv.className = 'chat-msg agent';
  msgDiv.innerHTML = `
    <div class="sender-tag">Vera</div>
    <div class="chat-bubble"><span class="typing-text"></span><span class="typing-cursor"></span></div>`;
  transcriptFeed.appendChild(msgDiv);
  transcriptFeed.scrollTop = transcriptFeed.scrollHeight;

  const target = msgDiv.querySelector('.typing-text');
  const cursor = msgDiv.querySelector('.typing-cursor');
  let i = 0;
  const speed = Math.max(8, Math.min(28, 1200 / Math.max(text.length, 1)));
  const timer = setInterval(() => {
    target.textContent = text.slice(0, ++i);
    transcriptFeed.scrollTop = transcriptFeed.scrollHeight;
    if (i >= text.length) {
      clearInterval(timer);
      if (cursor) cursor.remove();
    }
  }, speed);
}

// ---- 6. Turn Counter ----
let turnCount = 0;
function incrementTurnCounter() {
  turnCount++;
  const el = document.getElementById('valTurns');
  if (el) el.innerText = turnCount;
}

// ---- 7. Boot the cinematic modules ----
window.addEventListener('load', () => {
  initStarfield();
  drawStarfield();
  drawOrbVisualizer();
});
window.addEventListener('resize', initStarfield);


