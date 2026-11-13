const startBtn = document.getElementById("start-btn");
const stopBtn = document.getElementById("stop-btn");
const confidenceInput = document.getElementById("confidence");
const confidenceRange = document.getElementById("confidence-range");
const statusBox = document.getElementById("status");
const video = document.getElementById("video");
const placeholder = document.getElementById("placeholder");
const sourcePc = document.getElementById("source-pc");
const sourceXiao = document.getElementById("source-xiao");
const classList = document.getElementById("class-list");
const xiaoLinkText = document.getElementById("xiao-link");
const wifiForm = document.getElementById("xiao-wifi");
const wifiSsid = document.getElementById("wifi-ssid");
const wifiPass = document.getElementById("wifi-pass");
const wifiBtn = document.getElementById("wifi-btn");
const wifiNote = document.getElementById("wifi-note");
const linkUsb = document.getElementById("link-usb");
const linkWifi = document.getElementById("link-wifi");

const MODEL_ERROR = "Model not found. Please put best.pt in model folder.";
const CAMERA_ERROR = "電腦鏡頭打不開。請確認鏡頭已接上。";
const XIAO_ERROR = "看不到 XIAO 畫面。請確認 USB 已插好，然後再選一次 XIAO 鏡頭。";

let streaming = false;
let source = "pc";
let xiaoLink = "usb";
let linkPinned = false;
let restartTimer = 0;
let healthTimer = 0;

function linkSentence(link) {
  if (link === "wifi") {
    return "XIAO 鏡頭：Wi-Fi 連接。";
  }
  if (link === "usb") {
    return "XIAO 鏡頭：USB 連接。";
  }
  if (link === "plugged") {
    return "XIAO 鏡頭：USB 已插上，尚未開啟。";
  }
  return "XIAO 鏡頭：尚未連接。";
}

function paintLink(next) {
  xiaoLink = next === "wifi" ? "wifi" : "usb";
  linkUsb.classList.toggle("active", xiaoLink === "usb");
  linkWifi.classList.toggle("active", xiaoLink === "wifi");
  wifiForm.hidden = xiaoLink !== "wifi";
}

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
  const link = source === "xiao" ? `&link=${xiaoLink}` : "";
  video.src = `/video_feed?conf=${conf.toFixed(2)}&source=${source}${link}&t=${Date.now()}`;
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
    if (data.camera_link) {
      xiaoLinkText.textContent = linkSentence(data.camera_link);
      if (!linkPinned && streaming && data.camera_source === "xiao" && (data.camera_link === "usb" || data.camera_link === "wifi")) {
        paintLink(data.camera_link);
      }
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
        const name = data.camera_label || (source === "xiao" ? "XIAO" : "電腦");
        const poles = summary.poles ?? 0;
        const danger = summary.dangerous_poles ?? 0;
        const critical = summary.critical_vegetation ?? 0;
        const dangerText = danger > 0 || critical > 0 ? "有危險" : "未見危急植被";
        setStatus(
          `檢測中 · ${name}鏡頭 · ${dangerText} · 電線桿 ${poles} · 危險電線桿 ${danger} · 危急植被 ${critical}`
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

function selectSource(next) {
  source = next === "xiao" ? "xiao" : "pc";
  sourcePc.classList.toggle("active", source === "pc");
  sourceXiao.classList.toggle("active", source === "xiao");
  if (streaming) {
    startStream();
  }
}

startBtn.addEventListener("click", startStream);
stopBtn.addEventListener("click", stopStream);
sourcePc.addEventListener("click", () => selectSource("pc"));
sourceXiao.addEventListener("click", () => selectSource("xiao"));
linkUsb.addEventListener("click", () => {
  linkPinned = false;
  paintLink("usb");
  source = "xiao";
  sourcePc.classList.toggle("active", false);
  sourceXiao.classList.toggle("active", true);
  startStream();
});
linkWifi.addEventListener("click", () => {
  linkPinned = true;
  paintLink("wifi");
  source = "xiao";
  sourcePc.classList.toggle("active", false);
  sourceXiao.classList.toggle("active", true);
  wifiSsid.focus();
});

wifiForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const ssid = wifiSsid.value.trim();
  if (!ssid) {
    wifiNote.textContent = "請填 2.4GHz 的 Wi-Fi 名稱。";
    return;
  }
  wifiBtn.disabled = true;
  wifiNote.textContent = "正在幫 XIAO 加入 Wi-Fi…";
  try {
    const response = await fetch("/xiao/wifi", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ssid, password: wifiPass.value }),
    });
    const data = await response.json();
    wifiNote.textContent = data.message || "XIAO 沒有回應。";
    if (data.ok) {
      linkPinned = false;
      paintLink("wifi");
      source = "xiao";
      sourcePc.classList.toggle("active", false);
      sourceXiao.classList.toggle("active", true);
      startStream();
    } else if (data.link) {
      xiaoLinkText.textContent = linkSentence(data.link);
    }
  } catch (_error) {
    wifiNote.textContent = "送不出 Wi-Fi 設定。請確認智桿視界還開著。";
  } finally {
    wifiBtn.disabled = false;
  }
});

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
  setStatus(source === "xiao" ? XIAO_ERROR : CAMERA_ERROR, true);
});

refreshHealth();
