const startBtn = document.getElementById("start-btn");
const stopBtn = document.getElementById("stop-btn");
const confidenceInput = document.getElementById("confidence");
const confidenceRange = document.getElementById("confidence-range");
const statusBox = document.getElementById("status");
const video = document.getElementById("video");
const placeholder = document.getElementById("placeholder");
const classList = document.getElementById("class-list");

const MODEL_ERROR = "Model not found. Please put best.pt in model folder.";
const CAMERA_ERROR = "Camera not available. Please check USB camera connection.";

let streaming = false;
let restartTimer = 0;
let healthTimer = 0;

function setStatus(message, isError) {
  statusBox.textContent = message;
  statusBox.classList.toggle("error", Boolean(isError));
}

function currentConfidence() {
  const value = Number(confidenceInput.value);
  if (!Number.isFinite(value)) {
    return 0.25;
  }
  return Math.min(0.99, Math.max(0.01, value));
}

function syncConfidence(source) {
  const value = source === confidenceRange
    ? Number(confidenceRange.value)
    : currentConfidence();
  const text = value.toFixed(2);
  confidenceInput.value = text;
  confidenceRange.value = text;
}

function startStream() {
  const conf = currentConfidence();
  confidenceInput.value = conf.toFixed(2);
  streaming = true;
  placeholder.hidden = true;
  video.hidden = false;
  video.src = `/video_feed?conf=${conf.toFixed(2)}&t=${Date.now()}`;
  startBtn.disabled = true;
  stopBtn.disabled = false;
  setStatus("正在開啟攝像頭...");
  clearInterval(healthTimer);
  healthTimer = setInterval(refreshHealth, 1000);
}

function stopStream() {
  streaming = false;
  clearTimeout(restartTimer);
  video.src = "";
  video.removeAttribute("src");
  video.hidden = true;
  placeholder.hidden = false;
  placeholder.textContent = "按「開始攝像頭」";
  startBtn.disabled = false;
  stopBtn.disabled = true;
  clearInterval(healthTimer);
  setStatus("已停止");
  fetch("/video_feed?stop=1").catch(() => {});
}

async function refreshHealth() {
  try {
    const response = await fetch("/health");
    if (!response.ok) {
      throw new Error("health failed");
    }
    const data = await response.json();
    if (Array.isArray(data.classes) && data.classes.length) {
      classList.textContent = data.classes.join(" · ");
    }
    if (data.model_message) {
      setStatus(data.model_message, true);
      placeholder.textContent = MODEL_ERROR;
      startBtn.disabled = true;
      return data;
    }
    if (streaming) {
      if (data.camera_message) {
        setStatus(data.camera_message, true);
        placeholder.hidden = true;
      } else if (data.camera_open) {
        const summary = data.summary || {};
        const index = data.camera_index == null ? "-" : data.camera_index;
        const poles = summary.poles ?? 0;
        const danger = summary.dangerous_poles ?? 0;
        const critical = summary.critical_vegetation ?? 0;
        const dangerText = danger > 0 || critical > 0 ? "有危險" : "未見危急植被";
        setStatus(
          `檢測中 · 攝像頭 ${index} · ${dangerText} · 電線桿 ${poles} · 危險電線桿 ${danger} · 危急植被 ${critical}`
        );
        statusBox.classList.toggle("error", danger > 0 || critical > 0);
      }
    }
    return data;
  } catch (_error) {
    setStatus("偵測伺服器已停止，請重新整理頁面。", true);
    return null;
  }
}

startBtn.addEventListener("click", startStream);
stopBtn.addEventListener("click", stopStream);

function pushConfidence() {
  const conf = currentConfidence();
  fetch(`/confidence?conf=${conf.toFixed(2)}`).catch(() => {});
}

confidenceInput.addEventListener("change", () => {
  syncConfidence(confidenceInput);
  if (!streaming) {
    return;
  }
  clearTimeout(restartTimer);
  restartTimer = setTimeout(pushConfidence, 150);
});

confidenceRange.addEventListener("input", () => {
  syncConfidence(confidenceRange);
  if (!streaming) {
    return;
  }
  clearTimeout(restartTimer);
  restartTimer = setTimeout(pushConfidence, 150);
});

video.addEventListener("load", () => {
  if (streaming) {
    refreshHealth();
  }
});

video.addEventListener("error", () => {
  if (!streaming) {
    return;
  }
  setStatus(CAMERA_ERROR, true);
});

refreshHealth();
