"""YOLO real-time USB camera detection for Windows."""

from __future__ import annotations

import os

# Torch 與 OpenCV 各自帶一份 OpenMP，Windows 上同時載入會把行程弄崩。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "best.pt"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

MODEL_ERROR = "Model not found. Please put best.pt in model folder."
CAMERA_ERROR = "Camera not available. Please check USB camera connection."

app = FastAPI(title="Smartpole vision 智桿視界")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

model = YOLO(str(MODEL_PATH)) if MODEL_PATH.is_file() else None
DEVICE = 0 if torch.cuda.is_available() else "cpu"
predict_lock = threading.Lock()
summary_lock = threading.Lock()

POLE = "utility poles"
CRITICAL = "critical vegetation"
NONCRITICAL = "non-critical vegetation"
CLASS_COLOR = {
    POLE: (0, 255, 0),
    CRITICAL: (0, 0, 255),
    NONCRITICAL: (0, 255, 255),
}
DRAW_ORDER = {
    NONCRITICAL: 0,
    CRITICAL: 1,
    POLE: 2,
}
CLASS_LABEL = {
    POLE: "電線桿",
    CRITICAL: "危急植被",
    NONCRITICAL: "非危急植被",
}
# 畫面上的藤蔓照片大約 0.13 就有框。0.70 會把這些框全部拿掉。
# 鮮豔花田用顏色去掉，不靠把門檻拉高。
CRITICAL_FLOOR = 0.12
NONCRITICAL_FLOOR = 0.70
latest_summary = {
    "poles": 0,
    "dangerous_poles": 0,
    "critical_vegetation": 0,
    "noncritical_vegetation": 0,
    "dropped_vegetation": 0,
}
_fonts: dict[int, ImageFont.ImageFont] = {}


def clamp_confidence(value: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.25
    return min(0.99, max(0.01, number))


class CameraHub:
    """Keep one USB camera open while a browser is watching the stream."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cap: cv2.VideoCapture | None = None
        self._clients = 0
        self._generation = 0
        self.index: int | None = None
        self.error: str | None = None
        self.confidence = 0.25

    def is_open(self) -> bool:
        with self._lock:
            return self._cap is not None and self._cap.isOpened()

    def set_confidence(self, value: float) -> float:
        self.confidence = clamp_confidence(value)
        return self.confidence

    def next_token(self) -> int:
        """新的影像串流取代舊的，避免兩條串流同時做 GPU 推論。"""
        stabilizer.reset()
        with self._lock:
            self._generation += 1
            return self._generation

    def request_stop(self) -> None:
        stabilizer.reset()
        with self._lock:
            self._generation += 1
            cap = self._cap
            self._cap = None
            self.index = None
        if cap is not None:
            cap.release()

    def is_current(self, token: int) -> bool:
        with self._lock:
            return token == self._generation

    def acquire(self) -> bool:
        with self._lock:
            self._clients += 1
            if self._cap is not None and self._cap.isOpened():
                self.error = None
                return True
            try:
                opened = self._open_unlocked()
            except Exception as exc:
                print(f"開啟攝像頭失敗：{exc}")
                self.error = CAMERA_ERROR
                opened = False
            if not opened:
                self._clients = max(0, self._clients - 1)
            return opened

    def release(self) -> None:
        with self._lock:
            self._clients = max(0, self._clients - 1)
            clients = self._clients
        if clients > 0:
            return
        # Slider changes reconnect the stream. Wait so the camera stays open.
        time.sleep(0.8)
        with self._lock:
            if self._clients == 0 and self._cap is not None:
                self._cap.release()
                self._cap = None
                self.index = None

    def read(self) -> tuple[bool, np.ndarray | None]:
        with self._lock:
            cap = self._cap
            if cap is None or not cap.isOpened():
                return False, None
            ok, frame = cap.read()
            if not ok or frame is None:
                return False, None
            return True, frame.copy()

    def _open_unlocked(self) -> bool:
        self.error = None
        for index in (0, 1, 2):
            cap = self._open_index(index)
            if cap is None:
                continue
            self._prepare(cap)
            frame = self._warmup(cap)
            if frame is not None:
                self._cap = cap
                self.index = index
                self.error = None
                print(f"攝像頭已開啟：index {index}")
                return True
            cap.release()
        self._cap = None
        self.index = None
        self.error = CAMERA_ERROR
        print(CAMERA_ERROR)
        return False

    @staticmethod
    def _open_index(index: int) -> cv2.VideoCapture | None:
        cap = cv2.VideoCapture()
        if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 2500)
        # DirectShow is the reliable backend for Logitech USB cameras on Windows.
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        if not cap.open(index, backend):
            cap.release()
            if os.name == "nt":
                cap = cv2.VideoCapture(index)
                if cap.isOpened():
                    return cap
                cap.release()
            return None
        return cap

    @staticmethod
    def _prepare(cap: cv2.VideoCapture) -> None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    @staticmethod
    def _warmup(cap: cv2.VideoCapture) -> np.ndarray | None:
        for _ in range(10):
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                return frame
            time.sleep(0.05)
        return None


camera = CameraHub()


def class_names() -> list[str]:
    if model is None:
        return []
    names = model.names
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names)]
    return [str(name) for name in names]


def draw_message(text: str) -> np.ndarray:
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    image[:] = (18, 22, 18)
    cv2.rectangle(image, (36, 36), (1244, 684), (74, 255, 214), 1)
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = f"{current} {word}".strip()
        if len(trial) > 48:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    y = 360 - (len(lines) - 1) * 22
    for line in lines:
        size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        x = max(48, (image.shape[1] - size[0]) // 2)
        cv2.putText(
            image,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (74, 255, 214),
            2,
            cv2.LINE_AA,
        )
        y += 46
    return image


def encode_jpeg(frame: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        raise RuntimeError("無法編碼影像")
    return buffer.tobytes()


def mjpeg_part(payload: bytes) -> bytes:
    return b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + payload + b"\r\n"


def canonical_name(name: str) -> str:
    return " ".join(str(name).strip().lower().split())


def box_area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def intersection_area(a: list[float], b: list[float]) -> float:
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return width * height


def center_inside(inner: list[float], outer: list[float]) -> bool:
    cx = (inner[0] + inner[2]) / 2
    cy = (inner[1] + inner[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def boxes_overlap(a: list[float], b: list[float]) -> bool:
    shared = intersection_area(a, b)
    if shared <= 0:
        return False
    smaller = min(box_area(a), box_area(b))
    if smaller <= 0:
        return False
    return shared / smaller >= 0.15 or center_inside(a, b) or center_inside(b, a)


def clip_box_away(big: list[float], small: list[float]) -> list[float] | None:
    """Cut the large box back so a smaller vegetation box is not swallowed inside it."""
    x1, y1, x2, y2 = big
    sx1, sy1, sx2, sy2 = small
    width = x2 - x1
    height = y2 - y1
    if width <= 1 or height <= 1:
        return None
    candidates: list[list[float]] = []
    if sx1 > x1 + width * 0.18:
        candidates.append([x1, y1, min(x2, sx1), y2])
    if sx2 < x2 - width * 0.18:
        candidates.append([max(x1, sx2), y1, x2, y2])
    if sy1 > y1 + height * 0.18:
        candidates.append([x1, y1, x2, min(y2, sy1)])
    if sy2 < y2 - height * 0.18:
        candidates.append([x1, max(y1, sy2), x2, y2])
    best: list[float] | None = None
    best_area = 0.0
    original = box_area(big)
    for candidate in candidates:
        area = box_area(candidate)
        if area < original * 0.2 or area <= best_area:
            continue
        best = candidate
        best_area = area
    return best


def separate_vegetation_chunks(items: list[dict], confidence: float) -> list[dict]:
    """Split overlapping vegetation, and drop side-edge yellow false boxes."""
    plants = [item for item in items if item["name"] in {CRITICAL, NONCRITICAL}]
    plants.sort(key=lambda item: box_area(item["box"]), reverse=True)
    for index, big in enumerate(plants):
        for small in plants[index + 1 :]:
            small_area = box_area(small["box"])
            if small_area <= 0 or small_area >= box_area(big["box"]) * 0.72:
                continue
            if intersection_area(big["box"], small["box"]) / small_area < 0.4:
                continue
            clipped = clip_box_away(big["box"], small["box"])
            if clipped is not None:
                big["box"] = clipped
    return items


def flower_field(patch: np.ndarray) -> bool:
    """A bright multicolor flower photo, not the dull vines on a pole or wire."""
    if patch.size == 0:
        return False
    saturation = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[:, :, 1]
    return float(saturation.mean()) >= 100


def keep_stable_classes(frame: np.ndarray, items: list[dict], confidence: float) -> list[dict]:
    """Drop flower fields, dark edge boxes, and yellow boxes that are not on a pole."""
    height, width = frame.shape[:2]
    poles = [
        item
        for item in items
        if item["name"] == POLE and item["confidence"] >= confidence
    ]
    kept: list[dict] = []
    for item in items:
        x1, y1, x2, y2 = item["box"]
        ix1 = max(0, int(x1))
        iy1 = max(0, int(y1))
        ix2 = min(width, int(x2))
        iy2 = min(height, int(y2))
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        gray = cv2.cvtColor(frame[iy1:iy2, ix1:ix2], cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        on_side = x1 < width * 0.08 or x2 > width * 0.92
        thin_strip = (x2 - x1) < width * 0.3 and (y2 - y1) > height * 0.35
        if item["name"] == NONCRITICAL and on_side and thin_strip and brightness < 90:
            continue
        if item["name"] == POLE:
            if item["confidence"] >= confidence:
                kept.append(item)
            continue
        if item["name"] == CRITICAL:
            if item["confidence"] < min(confidence, CRITICAL_FLOOR) or brightness < 18:
                continue
            on_pole = any(boxes_overlap(item["box"], pole["box"]) for pole in poles)
            if not on_pole and flower_field(frame[iy1:iy2, ix1:ix2]):
                continue
            kept.append(item)
            continue
        if item["name"] != NONCRITICAL or item["confidence"] < max(confidence, NONCRITICAL_FLOOR):
            continue
        if brightness < 50 or not any(boxes_overlap(item["box"], pole["box"]) for pole in poles):
            continue
        kept.append(item)
    return kept


def box_iou(a: list[float], b: list[float]) -> float:
    shared = intersection_area(a, b)
    union = box_area(a) + box_area(b) - shared
    if union <= 0:
        return 0.0
    return shared / union


def same_box_score(a: list[float], b: list[float]) -> float:
    """Treat a jittering box as the same object without merging two separate ones."""
    iou = box_iou(a, b)
    ax = (a[0] + a[2]) * 0.5
    ay = (a[1] + a[3]) * 0.5
    bx = (b[0] + b[2]) * 0.5
    by = (b[1] + b[3]) * 0.5
    span = min(
        max(a[2] - a[0], a[3] - a[1], 1.0),
        max(b[2] - b[0], b[3] - b[1], 1.0),
    )
    dist = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
    if dist < span * 0.65:
        return max(iou, 0.3)
    return iou


class BoxStabilizer:
    """Draw a box on the first hit, and keep it while detection blinks."""

    def __init__(self) -> None:
        self._tracks: list[dict] = []

    def reset(self) -> None:
        self._tracks = []

    def update(self, items: list[dict], start_confidence: float = 0.15) -> list[dict]:
        used: set[int] = set()
        for track in self._tracks:
            best_index = None
            best_score = 0.12
            for index, item in enumerate(items):
                if index in used or item["name"] != track["name"]:
                    continue
                score = box_iou(track["box"], item["box"])
                if score > best_score:
                    best_score = score
                    best_index = index
            if best_index is None:
                track["misses"] += 1
                continue
            used.add(best_index)
            item = items[best_index]
            track["box"] = [
                track["box"][axis] * 0.62 + item["box"][axis] * 0.38 for axis in range(4)
            ]
            track["confidence"] = item["confidence"]
            track["seen"] += 1
            track["misses"] = 0
        for index, item in enumerate(items):
            if index in used:
                continue
            if item["confidence"] < start_confidence:
                continue
            self._tracks.append(
                {
                    "name": item["name"],
                    "box": list(item["box"]),
                    "confidence": item["confidence"],
                    "seen": 1,
                    "misses": 0,
                }
            )
        self._tracks = [track for track in self._tracks if track["misses"] <= 24]
        stable: list[dict] = []
        for track in self._tracks:
            if track["misses"] > 20:
                continue
            stable.append(
                {
                    "name": track["name"],
                    "confidence": track["confidence"],
                    "box": list(track["box"]),
                    "danger": False,
                    "caption": "",
                }
            )
        return stable


stabilizer = BoxStabilizer()


def empty_summary() -> dict[str, int]:
    return {
        "poles": 0,
        "dangerous_poles": 0,
        "critical_vegetation": 0,
        "noncritical_vegetation": 0,
        "dropped_vegetation": 0,
    }


def set_summary(summary: dict[str, int]) -> None:
    with summary_lock:
        latest_summary.update(summary)


def get_summary() -> dict[str, int]:
    with summary_lock:
        return dict(latest_summary)


def load_font(size: int) -> ImageFont.ImageFont:
    cached = _fonts.get(size)
    if cached is not None:
        return cached
    font = ImageFont.load_default()
    for path in (
        r"C:\Windows\Fonts\msjh.ttc",
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\mingliu.ttc",
    ):
        if Path(path).is_file():
            font = ImageFont.truetype(path, size)
            break
    _fonts[size] = font
    return font


def read_detections(result) -> list[dict]:
    items: list[dict] = []
    boxes = result.boxes
    if boxes is None:
        return items
    names = result.names or {}
    for box in boxes:
        class_id = int(box.cls[0])
        if isinstance(names, dict):
            raw_name = str(names.get(class_id, class_id))
        else:
            raw_name = str(names[class_id])
        items.append(
            {
                "name": canonical_name(raw_name),
                "confidence": float(box.conf[0]),
                "box": [float(value) for value in box.xyxy[0].tolist()],
                "danger": False,
                "caption": "",
            }
        )
    return items


def judge_poles(items: list[dict]) -> dict[str, int]:
    """畫出模型識別到的每一個框。電線桿與危急植被重疊時，標成危險電線桿。"""
    for item in items:
        item["keep"] = True
        item["linked"] = False
    poles = [item for item in items if item["name"] == POLE]
    plants = [item for item in items if item["name"] in {CRITICAL, NONCRITICAL}]
    dangerous = 0
    for index, pole in enumerate(poles, start=1):
        linked_critical = False
        linked_noncritical = False
        for plant in plants:
            if not boxes_overlap(pole["box"], plant["box"]):
                continue
            plant["linked"] = True
            if plant["name"] == CRITICAL:
                linked_critical = True
            else:
                linked_noncritical = True
        if linked_critical:
            pole["danger"] = True
            dangerous += 1
            title = f"危險電線桿 #{index}"
        elif linked_noncritical:
            title = f"電線桿 #{index} · 非危急植被"
        else:
            title = f"電線桿 #{index}"
        pole["caption"] = f"{title} {pole['confidence']:.2f}"
    for plant in plants:
        label = CLASS_LABEL.get(plant["name"], plant["name"])
        if plant["name"] == CRITICAL and plant["linked"]:
            label = "危急植被 · 危險"
        plant["caption"] = f"{label} {plant['confidence']:.2f}"
    return {
        "poles": len(poles),
        "dangerous_poles": dangerous,
        "critical_vegetation": sum(1 for plant in plants if plant["name"] == CRITICAL),
        "noncritical_vegetation": sum(1 for plant in plants if plant["name"] == NONCRITICAL),
        "dropped_vegetation": 0,
    }


def paint_labels(frame: np.ndarray, items: list[dict], summary: dict[str, int]) -> np.ndarray:
    height, width = frame.shape[:2]
    font_size = max(16, round(min(width, height) / 38))
    font = load_font(font_size)
    hud_font = load_font(max(18, font_size))
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    for item in items:
        x1, y1, x2, y2 = item["box"]
        color_bgr = CLASS_COLOR.get(item["name"], (0, 230, 255))
        color = (color_bgr[2], color_bgr[1], color_bgr[0])
        text = item["caption"] or item["name"]
        left = max(8, min(int(x1), width - 8))
        top = int(y1) - font_size - 10
        if top < 42:
            top = min(height - font_size - 8, max(42, int(y1) + 6))
        bbox = draw.textbbox((left, top), text, font=font)
        if bbox[2] > width - 8:
            left = max(8, width - 8 - (bbox[2] - bbox[0]))
            bbox = draw.textbbox((left, top), text, font=font)
        draw.rectangle(bbox, fill=(12, 16, 12))
        draw.text((left, top), text, font=font, fill=color)
    hud = (
        f"電線桿 {summary['poles']}    "
        f"危險電線桿 {summary['dangerous_poles']}    "
        f"危急植被 {summary['critical_vegetation']}    "
        f"非危急植被 {summary['noncritical_vegetation']}"
    )
    hud_box = draw.textbbox((12, 12), hud, font=hud_font)
    draw.rectangle(
        (hud_box[0] - 6, hud_box[1] - 4, min(width - 8, hud_box[2] + 8), hud_box[3] + 6),
        fill=(12, 16, 12),
    )
    hud_color = (255, 90, 70) if summary["dangerous_poles"] or summary["critical_vegetation"] else (214, 255, 74)
    draw.text((12, 12), hud, font=hud_font, fill=hud_color)
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def drop_side_yellow(items: list[dict], width: int, height: int) -> list[dict]:
    """Drop tall yellow boxes stuck to the left or right edge of the frame."""
    kept: list[dict] = []
    for item in items:
        if item["name"] != NONCRITICAL:
            kept.append(item)
            continue
        x1, y1, x2, y2 = item["box"]
        box_w = x2 - x1
        box_h = y2 - y1
        touches_side = x1 <= width * 0.08 or x2 >= width * 0.92
        tall_strip = box_h > height * 0.45 and box_w < width * 0.35
        if touches_side and tall_strip:
            continue
        kept.append(item)
    return kept


def drop_flower_fields(frame: np.ndarray, items: list[dict]) -> list[dict]:
    """Drop bright flower photos. Vine on a pole is much less saturated."""
    height, width = frame.shape[:2]
    kept: list[dict] = []
    for item in items:
        if item["name"] != CRITICAL:
            kept.append(item)
            continue
        x1, y1, x2, y2 = [int(value) for value in item["box"]]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        saturation = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)[:, :, 1]
        if float(saturation.mean()) >= 85:
            continue
        kept.append(item)
    return kept


def annotate(frame: np.ndarray, confidence: float) -> np.ndarray:
    assert model is not None
    with predict_lock:
        result = model.predict(
            source=frame,
            conf=confidence,
            imgsz=640,
            verbose=False,
            device=DEVICE,
        )[0]
        items = drop_side_yellow(read_detections(result), frame.shape[1], frame.shape[0])
        items = drop_flower_fields(frame, items)
        items = stabilizer.update(items, start_confidence=confidence)
        del result
    summary = judge_poles(items)
    set_summary(summary)
    visible = [item for item in items if item.get("keep")]
    height, width = frame.shape[:2]
    line_width = max(2, round(min(width, height) / 420))
    canvas = frame.copy()
    for item in sorted(visible, key=lambda det: DRAW_ORDER.get(det["name"], 1)):
        x1, y1, x2, y2 = [int(round(value)) for value in item["box"]]
        color = CLASS_COLOR.get(item["name"], (0, 255, 255))
        thickness = line_width + (2 if item["name"] == POLE else 0)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
    return paint_labels(canvas, visible, summary)


def stream_frames(token: int):
    if model is None:
        payload = encode_jpeg(draw_message(MODEL_ERROR))
        while camera.is_current(token):
            yield mjpeg_part(payload)
            time.sleep(0.4)
        return

    opened = camera.acquire()
    try:
        if not opened:
            payload = encode_jpeg(draw_message(camera.error or CAMERA_ERROR))
            while camera.is_current(token):
                yield mjpeg_part(payload)
                time.sleep(0.4)
            return
        misses = 0
        while camera.is_current(token):
            ok, frame = camera.read()
            if not ok or frame is None:
                misses += 1
                if misses >= 12:
                    camera.error = CAMERA_ERROR
                    yield mjpeg_part(encode_jpeg(draw_message(CAMERA_ERROR)))
                    time.sleep(0.2)
                continue
            misses = 0
            camera.error = None
            try:
                frame = annotate(frame, camera.confidence)
            except Exception as exc:
                print(f"檢測失敗：{exc}")
            yield mjpeg_part(encode_jpeg(frame))
    finally:
        if opened:
            camera.release()


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "classes": class_names(),
            "model_loaded": model is not None,
        },
    )


@app.get("/video_feed")
def video_feed(conf: float = 0.25, stop: int = 0):
    if stop:
        camera.request_stop()
        return {"stopped": True}
    camera.set_confidence(conf)
    return StreamingResponse(
        stream_frames(camera.next_token()),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/confidence")
def update_confidence(conf: float = 0.25):
    return {"confidence": camera.set_confidence(conf)}


@app.get("/health")
def health():
    model_loaded = model is not None
    return {
        "model_loaded": model_loaded,
        "model_message": None if model_loaded else MODEL_ERROR,
        "camera_open": camera.is_open(),
        "camera_index": camera.index,
        "camera_message": camera.error,
        "classes": class_names(),
        "confidence": camera.confidence,
        "summary": get_summary(),
    }


if __name__ == "__main__":
    import threading
    import webbrowser

    import uvicorn

    if model is None:
        print(MODEL_ERROR)
    else:
        print(f"已載入模型：{MODEL_PATH}")
        print("可檢測類別：" + "、".join(class_names()))
    print("請開啟 http://127.0.0.1:8000")
    if model is not None:
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        with predict_lock:
            model.predict(source=dummy, conf=0.25, imgsz=640, verbose=False, device=DEVICE)
        where = "GPU" if DEVICE == 0 else "CPU"
        print(f"模型預熱完成，推論裝置：{where}")
    threading.Timer(1.0, lambda: webbrowser.open("http://127.0.0.1:8000")).start()
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
