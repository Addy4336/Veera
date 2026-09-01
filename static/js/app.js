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
  if (metrics.stt_latency_ms !== null) {
    metricStt.innerText = Math.round(metrics.stt_latency_ms);
    applySlaClass(metricStt.parentElement, metrics.stt_latency_ms, 250, 500);
  }
  if (metrics.llm_ttft_ms !== null) {
    metricTtft.innerText = Math.round(metrics.llm_ttft_ms);
    applySlaClass(metricTtft.parentElement, metrics.llm_ttft_ms, 300, 700);
  }
  if (metrics.tts_ttfa_ms !== null) {
    metricTtfa.innerText = Math.round(metrics.tts_ttfa_ms);
    applySlaClass(metricTtfa.parentElement, metrics.tts_ttfa_ms, 150, 400);
  }
  if (metrics.total_latency_ms !== null) {
    metricTotal.innerText = Math.round(metrics.total_latency_ms);
    applySlaClass(metricTotal.parentElement, metrics.total_latency_ms, 600, 1200);
  }
}

function applySlaClass(element, value, goodThreshold, warnThreshold) {
  element.classList.remove('sla-good', 'sla-warn', 'sla-alert');
  if (value <= goodThreshold) {
    element.classList.add('sla-good');
  } else if (value <= warnThreshold) {
    element.classList.add('sla-warn');
  } else {
    element.classList.add('sla-alert');
  }
}

function updateActiveNode(nodeName) {
  document.querySelectorAll('.flow-node-step').forEach(el => {
    el.classList.remove('active');
    const name = el.getAttribute('data-node');
    if (name === nodeName) {
      el.classList.add('active');
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
      wsAudio.onopen = resolve;
      wsAudio.onerror = reject;
      setTimeout(reject, 8000);
    });

    // Step 3: Mic capture pipeline -> 16kHz mono s16le PCM frames -> wsAudio
    captureContext = new (window.AudioContext || window.webkitAudioContext)();
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

    isCallActive = true;
    btnStartCall.disabled = false;
    btnStartCall.innerHTML = '🔴 End Call';
    btnStartCall.classList.add('active');
    btnInterrupt.disabled = false;

    setAgentState('listening', 'Connected & Listening');
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
        renderChatMessage('agent', 'Hi! Thanks for calling Vera Support. How can I help you today?');
        break;
      case 'transcript':
        renderChatMessage('user', msg.text);
        setAgentState('thinking', 'Thinking...');
        break;
      case 'bot_token':
        setAgentState('speaking', 'Speaking');
        break;
      case 'bot_turn_end':
        renderChatMessage('agent', msg.reply || '(no response)');
        if (msg.reply) isBotSpeaking = true;
        break;
      case 'stt_empty':
        setAgentState('listening', 'Listening');
        break;
      case 'interrupt_ack':
        stopPlayback();
        setAgentState('interrupted', 'Interrupted');
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
  const bufferLength = audioAnalyser ? audioAnalyser.frequencyBinCount : 32;
  const dataArray = new Uint8Array(bufferLength);

  function render() {
    if (!isCallActive) return;
    animationFrameId = requestAnimationFrame(render);

    if (audioAnalyser) {
      audioAnalyser.getByteFrequencyData(dataArray);
    }

    const width = canvas.width;
    const height = canvas.height;
    const centerX = width / 2;
    const centerY = height / 2;
    const radius = 65;

    canvasCtx.clearRect(0, 0, width, height);

    const bars = 36;
    for (let i = 0; i < bars; i++) {
      const rad = (i * 2 * Math.PI) / bars;
      const value = dataArray[i % bufferLength] || (15 + Math.sin(Date.now() * 0.005 + i) * 10);
      const barHeight = (value / 255) * 35 + 4;

      const x1 = centerX + Math.cos(rad) * radius;
      const y1 = centerY + Math.sin(rad) * radius;
      const x2 = centerX + Math.cos(rad) * (radius + barHeight);
      const y2 = centerY + Math.sin(rad) * (radius + barHeight);

      canvasCtx.beginPath();
      canvasCtx.moveTo(x1, y1);
      canvasCtx.lineTo(x2, y2);
      canvasCtx.strokeStyle = `hsl(${(i * 10 + Date.now() * 0.05) % 360}, 85%, 65%)`;
      canvasCtx.lineWidth = 3;
      canvasCtx.lineCap = 'round';
      canvasCtx.stroke();
    }
  }

  render();
}

function startSpeakingWave() {
  // Boost pulse visual when bot speaks
}

function startListeningWave() {
  // Mic input visual
}
