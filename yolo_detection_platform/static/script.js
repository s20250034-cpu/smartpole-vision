const fileInput = document.getElementById("file-input");
const dropzone = document.getElementById("dropzone");
const dropCopy = document.getElementById("drop-copy");
const localPreview = document.getElementById("local-preview");
const fileMeta = document.getElementById("file-meta");
const detectBtn = document.getElementById("detect-btn");
const confidenceInput = document.getElementById("confidence");
const confidenceValue = document.getElementById("confidence-value");
const statusBox = document.getElementById("status");
const classList = document.getElementById("class-list");
const totalCount = document.getElementById("total-count");
const awaitState = document.getElementById("await-state");
const resultView = document.getElementById("result-view");
const emptyMessage = document.getElementById("empty-message");
const originalImage = document.getElementById("original-image");
const resultImage = document.getElementById("result-image");
const originalNote = document.getElementById("original-note");
const tableWrap = document.getElementById("table-wrap");
const detectionBody = document.getElementById("detection-body");

const ALLOWED = ["image/jpeg", "image/png", "image/bmp", "image/webp"];
let selectedFile = null;
let previewUrl = "";

function setStatus(message, isError) {
  statusBox.textContent = message || "";
  statusBox.classList.toggle("error", Boolean(isError));
}

function formatSize(bytes) {
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function clearResult() {
  totalCount.textContent = "—";
  awaitState.hidden = false;
  resultView.hidden = true;
  emptyMessage.hidden = true;
  detectionBody.replaceChildren();
  originalImage.removeAttribute("src");
  resultImage.removeAttribute("src");
}

function setFile(file) {
  if (!file) {
    return;
  }

  const extension = file.name.split(".").pop().toLowerCase();
  const allowedExt = ["jpg", "jpeg", "png", "bmp", "webp"];
  if (!allowedExt.includes(extension) && !ALLOWED.includes(file.type)) {
    setStatus("只接受 JPG、PNG、BMP、WEBP 圖片。", true);
    return;
  }
  if (file.size > 20 * 1024 * 1024) {
    setStatus("圖片超過 20MB。", true);
    return;
  }

  selectedFile = file;
  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
  }
  previewUrl = URL.createObjectURL(file);
  localPreview.src = previewUrl;
  localPreview.hidden = false;
  dropCopy.hidden = true;
  fileMeta.textContent = `${file.name} · ${formatSize(file.size)}`;
  detectBtn.disabled = false;
  setStatus("");
  clearResult();
}

function errorDetail(payload) {
  if (!payload || payload.detail == null) {
    return "檢測失敗，請稍後再試。";
  }
  if (typeof payload.detail === "string") {
    return payload.detail;
  }
  if (Array.isArray(payload.detail)) {
    return payload.detail.map((item) => item.msg || "請求格式不正確").join("；");
  }
  return "檢測失敗，請稍後再試。";
}

function staticUrl(url) {
  if (typeof url !== "string" || !url.startsWith("/static/")) {
    throw new Error("回應的圖片路徑不正確");
  }
  return `${url}?t=${Date.now()}`;
}

function tagClass(name) {
  const key = String(name || "").toLowerCase();
  if (key === "critical" || key === "cautious" || key === "low") {
    return key;
  }
  return "";
}

function renderDetections(detections) {
  detectionBody.replaceChildren();
  detections.forEach((item, index) => {
    const row = document.createElement("tr");

    const indexCell = document.createElement("td");
    indexCell.textContent = String(index + 1);

    const classCell = document.createElement("td");
    const tag = document.createElement("span");
    tag.className = `tag ${tagClass(item.class_name)}`.trim();
    tag.textContent = item.class_name;
    classCell.appendChild(tag);

    const confCell = document.createElement("td");
    confCell.className = "conf";
    const score = Number(item.confidence);
    confCell.append(document.createTextNode(score.toFixed(4)));
    const bar = document.createElement("span");
    bar.className = "conf-bar";
    const fill = document.createElement("span");
    fill.style.width = `${Math.max(0, Math.min(100, score * 100))}%`;
    bar.appendChild(fill);
    confCell.appendChild(bar);

    const bboxCell = document.createElement("td");
    bboxCell.className = "bbox";
    const bbox = Array.isArray(item.bbox) ? item.bbox : [];
    const [x1, y1, x2, y2] = bbox;
    bboxCell.textContent = `x1 ${x1}, y1 ${y1}, x2 ${x2}, y2 ${y2}`;

    row.append(indexCell, classCell, confCell, bboxCell);
    detectionBody.appendChild(row);
  });
}

async function loadModelInfo() {
  try {
    const response = await fetch("/model-info");
    if (!response.ok) {
      throw new Error("model info failed");
    }
    const data = await response.json();
    const classes = Array.isArray(data.classes) ? data.classes : [];
    classList.textContent = classes.length ? classes.join("、") : "best.pt";
  } catch (_error) {
    classList.textContent = "best.pt";
  }
}

async function detect() {
  if (!selectedFile || detectBtn.disabled) {
    return;
  }

  detectBtn.disabled = true;
  detectBtn.classList.add("loading");
  detectBtn.textContent = "檢測中…";
  setStatus("正在使用 best.pt 檢測，請稍候。");

  const form = new FormData();
  form.append("file", selectedFile);
  form.append("confidence", confidenceInput.value);

  try {
    const response = await fetch("/detect", {
      method: "POST",
      body: form,
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      throw new Error(errorDetail(payload));
    }

    const detections = Array.isArray(payload.detections) ? payload.detections : [];
    totalCount.textContent = String(payload.count ?? detections.length);
    originalImage.src = staticUrl(payload.original_url);
    originalImage.alt = `原始圖片 ${payload.filename || ""}`.trim();
    resultImage.src = staticUrl(payload.result_url);
    originalNote.textContent = `${payload.filename || ""} · ${payload.image_width} × ${payload.image_height}`;

    const found = Number(payload.count) > 0 && detections.length > 0;
    emptyMessage.hidden = found;
    emptyMessage.textContent = payload.message || "沒有檢測到目標";
    tableWrap.hidden = !found;
    if (found) {
      renderDetections(detections);
    } else {
      detectionBody.replaceChildren();
    }

    awaitState.hidden = true;
    resultView.hidden = false;
    setStatus("檢測完成。");
  } catch (error) {
    setStatus(error.message || "檢測失敗，請稍後再試。", true);
  } finally {
    detectBtn.classList.remove("loading");
    detectBtn.textContent = "開始檢測";
    detectBtn.disabled = !selectedFile;
  }
}

fileInput.addEventListener("change", () => {
  setFile(fileInput.files && fileInput.files[0]);
});

["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragover");
  });
});

dropzone.addEventListener("drop", (event) => {
  const file = event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files[0];
  setFile(file);
});

confidenceInput.addEventListener("input", () => {
  confidenceValue.textContent = Number(confidenceInput.value).toFixed(2);
});

detectBtn.addEventListener("click", detect);
loadModelInfo();
