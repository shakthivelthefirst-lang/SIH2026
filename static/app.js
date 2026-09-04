// Causal DPCRN Web Application Frontend Engine

let selectedNoiseType = 'rotor_blades';
let selectedFile = null;
let currentEngineMode = 'streaming';
let visualizerSource = 'enhanced'; // 'noisy' or 'enhanced'

// Web Audio API Visualizer Contexts
let audioCtx = null;
let analyserNode = null;
let visualizerAnimationId = null;

// Live Microphone WebSocket Stream
let micStream = null;
let micAudioCtx = null;
let micWorklet = null;
let ws = null;
let micFramesCount = 0;

// Tab Switcher
function switchTab(tabId) {
  document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
  document.querySelectorAll('.tab-pane').forEach(pane => pane.classList.remove('active'));

  document.getElementById(`tab-btn-${tabId}`).classList.add('active');
  document.getElementById(`pane-${tabId}`).classList.add('active');
}

// SNR Slider Label
function updateSnrLabel(val) {
  const num = parseFloat(val);
  const sign = num > 0 ? '+' : '';
  document.getElementById('snr-display').textContent = `${sign}${num.toFixed(1)} dB`;
}

// Noise Selector
function selectNoiseType(type, element) {
  selectedNoiseType = type;
  document.querySelectorAll('.noise-opt').forEach(opt => opt.classList.remove('active'));
  element.classList.add('active');
}

// Visualizer Source Toggle
function setVisualizerSource(source) {
  visualizerSource = source;
  document.getElementById('btn-vis-noisy').classList.toggle('active', source === 'noisy');
  document.getElementById('btn-vis-enh').classList.toggle('active', source === 'enhanced');

  const audioEl = source === 'noisy' 
    ? document.getElementById('audio-noisy') 
    : document.getElementById('audio-enhanced');
  
  if (audioEl && audioEl.src) {
    hookAudioVisualizer(audioEl);
  }
}

// Engine Mode Toggle (Batch vs Streaming)
function setEngineMode(mode) {
  currentEngineMode = mode;
  document.getElementById('seg-stream').classList.toggle('active', mode === 'streaming');
  document.getElementById('seg-batch').classList.toggle('active', mode === 'batch');
}

// ==========================================
// 1. Defence Noise Simulator & Synthesizer
// ==========================================
async function runSimulation() {
  const btn = document.getElementById('btn-run-sim');
  const snr = document.getElementById('snr-slider').value;

  btn.disabled = true;
  btn.innerHTML = `<span class="radar-dot" style="display:inline-block"></span> Synthesizing & Denoising...`;

  try {
    const formData = new FormData();
    formData.append('noise_type', selectedNoiseType);
    formData.append('snr_db', snr);
    formData.append('duration_s', '3.0');

    const response = await fetch('/api/generate_defence_sample', {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();
    if (data.success) {
      // Update Telemetry Metrics
      document.getElementById('m-input-sisnr').textContent = `${data.metrics.input_sisnr_db} dB`;
      document.getElementById('m-enh-sisnr').textContent = `${data.metrics.enhanced_sisnr_db} dB`;
      
      const delta = data.metrics.delta_sisnr_db;
      const sign = delta >= 0 ? '+' : '';
      document.getElementById('m-delta-sisnr').textContent = `${sign}${delta} dB`;
      document.getElementById('m-latency').textContent = `${data.metrics.avg_latency_ms} ms`;

      // Update Audio Elements
      const cleanAudio = document.getElementById('audio-clean');
      const noisyAudio = document.getElementById('audio-noisy');
      const enhAudio = document.getElementById('audio-enhanced');

      cleanAudio.src = data.clean_url;
      noisyAudio.src = data.noisy_url;
      enhAudio.src = data.enhanced_url;

      // Autoplay enhanced output
      enhAudio.play();
      hookAudioVisualizer(enhAudio);
    }
  } catch (err) {
    console.error('Simulation error:', err);
    alert('Failed to synthesize audio. Check server console.');
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg> Synthesize & Denoise Stream`;
  }
}

// ==========================================
// 2. Audio File Denoiser
// ==========================================
function handleFileSelect(file) {
  if (!file) return;
  selectedFile = file;

  const infoBadge = document.getElementById('file-info');
  infoBadge.style.display = 'inline-block';
  infoBadge.textContent = `${file.name} (${(file.size / (1024*1024)).toFixed(2)} MB)`;

  document.getElementById('btn-denoise-file').disabled = false;
}

// Drag and drop handlers
const dropZone = document.getElementById('drop-zone');
if (dropZone) {
  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
  });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    if (e.dataTransfer.files.length > 0) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  });
}

async function denoiseUploadedFile() {
  if (!selectedFile) return;

  const btn = document.getElementById('btn-denoise-file');
  btn.disabled = true;
  btn.innerHTML = `Processing Audio...`;

  try {
    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('mode', currentEngineMode);

    const response = await fetch('/api/denoise_uploaded_file', {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();
    if (data.success) {
      document.getElementById('file-proc-time').textContent = `${data.metrics.proc_time_s} s`;
      document.getElementById('file-rtf').textContent = `${data.metrics.rtf}x`;
      document.getElementById('file-frames').textContent = `${data.metrics.frames_processed}`;
      document.getElementById('file-lat').textContent = `${data.metrics.avg_latency_ms} ms`;

      const rawAudio = document.getElementById('audio-file-noisy');
      const enhAudio = document.getElementById('audio-file-enhanced');

      rawAudio.src = data.noisy_url;
      enhAudio.src = data.enhanced_url;

      enhAudio.play();
    }
  } catch (err) {
    console.error('File denoising failed:', err);
    alert('File processing error.');
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg> Execute Noise Cancellation`;
  }
}

// ==========================================
// 3. Live Microphone WebSocket Stream
// ==========================================
async function startMicStreaming() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, sampleRate: 16000 } });
    micStream = stream;

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${window.location.host}/ws/stream`);
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
      document.getElementById('mic-conn-status').textContent = 'Connected (8 ms)';
      document.getElementById('mic-conn-status').className = 'm-val text-emerald';
    };

    ws.onclose = () => {
      document.getElementById('mic-conn-status').textContent = 'Disconnected';
      document.getElementById('mic-conn-status').className = 'm-val text-amber';
    };

    micAudioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
    const source = micAudioCtx.createMediaStreamSource(stream);
    
    // Use ScriptProcessor / AudioWorklet to extract 128 sample chunks
    const processorNode = micAudioCtx.createScriptProcessor(256, 1, 1);
    
    processorNode.onaudioprocess = (e) => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        const inputData = e.inputBuffer.getChannelData(0);
        // Send first 128 samples
        const chunk128 = inputData.slice(0, 128);
        ws.send(chunk128.buffer);

        micFramesCount++;
        document.getElementById('mic-frames-count').textContent = micFramesCount;
      }
    };

    source.connect(processorNode);
    processorNode.connect(micAudioCtx.destination);

    document.getElementById('mic-indicator').classList.add('recording');
    document.getElementById('mic-status-title').textContent = 'Live ANC Active';
    document.getElementById('mic-status-desc').textContent = 'Streaming microphone input via WebSocket into Causal DPCRN model in 8 ms chunks.';
    document.getElementById('btn-mic-start').disabled = true;
    document.getElementById('btn-mic-stop').disabled = false;

    renderLiveMicCanvas(source);

  } catch (err) {
    console.error('Microphone access denied:', err);
    alert('Microphone access required for real-time streaming.');
  }
}

function stopMicStreaming() {
  if (micStream) {
    micStream.getTracks().forEach(track => track.stop());
    micStream = null;
  }
  if (ws) {
    ws.close();
    ws = null;
  }
  if (micAudioCtx) {
    micAudioCtx.close();
    micAudioCtx = null;
  }

  document.getElementById('mic-indicator').classList.remove('recording');
  document.getElementById('mic-status-title').textContent = 'Microphone Idle';
  document.getElementById('mic-status-desc').textContent = 'Stream stopped. Click Start above to resume.';
  document.getElementById('btn-mic-start').disabled = false;
  document.getElementById('btn-mic-stop').disabled = true;
}

// ==========================================
// 4. Web Audio API Canvas Visualizer
// ==========================================
function hookAudioVisualizer(audioElement) {
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }

  if (!audioElement._hasVisualizer) {
    const source = audioCtx.createMediaElementSource(audioElement);
    analyserNode = audioCtx.createAnalyser();
    analyserNode.fftSize = 512;
    source.connect(analyserNode);
    analyserNode.connect(audioCtx.destination);
    audioElement._hasVisualizer = true;
  }

  startVisualizerLoop();
}

function startVisualizerLoop() {
  if (visualizerAnimationId) cancelAnimationFrame(visualizerAnimationId);

  const canvas = document.getElementById('waveform-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const bufferLength = analyserNode.frequencyBinCount;
  const dataArray = new Uint8Array(bufferLength);
  const timeArray = new Uint8Array(bufferLength);

  function draw() {
    visualizerAnimationId = requestAnimationFrame(draw);

    analyserNode.getByteFrequencyData(dataArray);
    analyserNode.getByteTimeDomainData(timeArray);

    ctx.fillStyle = '#040711';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // 1. Draw Frequency Spectrum Bars
    const barWidth = (canvas.width / bufferLength) * 2.2;
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {
      const barHeight = (dataArray[i] / 255) * canvas.height * 0.75;
      
      const grad = ctx.createLinearGradient(0, canvas.height, 0, 0);
      grad.addColorStop(0, '#00f0ff');
      grad.addColorStop(0.5, '#10b981');
      grad.addColorStop(1, '#f59e0b');

      ctx.fillStyle = grad;
      ctx.fillRect(x, canvas.height - barHeight, barWidth - 1, barHeight);
      x += barWidth;
    }

    // 2. Draw Oscilloscope Waveform Overlay
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#ffffff';
    ctx.beginPath();

    const sliceWidth = canvas.width / bufferLength;
    let waveX = 0;

    for (let i = 0; i < bufferLength; i++) {
      const v = timeArray[i] / 128.0;
      const y = (v * canvas.height) / 2;

      if (i === 0) ctx.moveTo(waveX, y);
      else ctx.lineTo(waveX, y);

      waveX += sliceWidth;
    }
    ctx.stroke();
  }

  draw();
}

function renderLiveMicCanvas(sourceNode) {
  const canvas = document.getElementById('live-mic-canvas');
  if (!canvas || !micAudioCtx) return;
  const ctx = canvas.getContext('2d');

  const analyser = micAudioCtx.createAnalyser();
  analyser.fftSize = 256;
  sourceNode.connect(analyser);

  const bufferLength = analyser.frequencyBinCount;
  const dataArray = new Uint8Array(bufferLength);

  function draw() {
    if (!micStream) return;
    requestAnimationFrame(draw);

    analyser.getByteTimeDomainData(dataArray);
    ctx.fillStyle = '#040711';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.lineWidth = 2;
    ctx.strokeStyle = '#10b981';
    ctx.beginPath();

    const sliceWidth = canvas.width / bufferLength;
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {
      const v = dataArray[i] / 128.0;
      const y = (v * canvas.height) / 2;

      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);

      x += sliceWidth;
    }
    ctx.stroke();
  }

  draw();
}
