"""YOLO real-time USB camera detection for Windows."""

from __future__ import annotations

import os
import urllib.request

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
CAMERA_ERROR = "電腦鏡頭打不開。請確認鏡頭已接上。"
XIAO_ERROR = "看不到 XIAO 畫面。請確認 USB 已插好，然後再選一次 XIAO 鏡頭。"

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


class SerialJpegCapture:
    """從 XIAO 的 USB 序列埠讀 JPGS + 長度 + JPEG，並收下狀態列。"""

    def __init__(self, link) -> None:
        self._link = link
        self._buf = b""
        self._notes: list[str] = []
        self._io = threading.Lock()

    def isOpened(self) -> bool:
        return self._link is not None and getattr(self._link, "is_open", False)

    def _harvest(self) -> None:
        while True:
            magic = self._buf.find(b"JPGS")
            head = self._buf if magic < 0 else self._buf[:magic]
            newline = head.rfind(b"\n")
            if newline >= 0:
                text = head[:newline].decode("utf-8", "replace")
                for line in text.splitlines():
                    line = line.strip()
                    if line.startswith(("MODE ", "STREAM http://", "WIFI ", "XIAO ")):
                        self._notes.append(line)
                        print(line)
                if len(self._notes) > 30:
                    del self._notes[:-15]
            if magic >= 0:
                self._buf = self._buf[magic:]
                return
            if newline >= 0:
                self._buf = head[newline + 1 :]
            return

    def _pop_jpeg(self) -> bytes | None:
        magic = self._buf.find(b"JPGS")
        if magic < 0 or len(self._buf) < magic + 8:
            return None
        length = int.from_bytes(self._buf[magic + 4 : magic + 8], "little")
        if length < 100 or length > 250_000:
            self._buf = self._buf[magic + 4 :]
            return None
        need = magic + 8 + length
        if len(self._buf) < need:
            return None
        jpg = self._buf[magic + 8 : need]
        self._buf = self._buf[need:]
        return jpg

    def _pull(self) -> bool:
        try:
            chunk = self._link.read(8192)
        except Exception:
            return False
        if chunk:
            self._buf += chunk
            if len(self._buf) > 500_000:
                self._buf = self._buf[-100_000:]
        return True

    def read(self, timeout: float = 4.0) -> tuple[bool, np.ndarray | None]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            jpg = None
            with self._io:
                if self._link is None:
                    return False, None
                self._harvest()
                jpg = self._pop_jpeg()
                if jpg is None:
                    if not self._pull():
                        return False, None
                    self._harvest()
                    jpg = self._pop_jpeg()
            if jpg is not None and jpg.startswith(b"\xff\xd8"):
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    return True, frame
            else:
                time.sleep(0.01)
        return False, None

    def send(self, payload: bytes) -> bool:
        with self._io:
            if self._link is None:
                return False
            try:
                self._link.write(payload)
                self._link.flush()
                return True
            except Exception:
                return False

    def clear_notes(self) -> None:
        with self._io:
            self._notes.clear()

    def wait_note(self, prefixes: tuple[str, ...], timeout: float) -> str | None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._io:
                if self._link is None:
                    return None
                self._harvest()
                found = next((line for line in reversed(self._notes) if line.startswith(prefixes)), None)
                if found:
                    return found
                if not self._pull():
                    return None
                while self._pop_jpeg() is not None:
                    pass
            time.sleep(0.02)
        with self._io:
            self._harvest()
            return next((line for line in reversed(self._notes) if line.startswith(prefixes)), None)

    def release(self) -> None:
        # 序列埠要留著。關掉它會讓 XIAO 重開，剛連上的 Wi-Fi 會掉。
        with self._io:
            self._link = None


class MjpegCapture:
    """讀 ESP32 的 multipart JPEG。OpenCV 會把這種串流丢掉。"""

    def __init__(self, url: str, timeout: float = 8.0) -> None:
        self.url = url
        self._response = None
        self._buf = b""
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "smartpole"})
            self._response = urllib.request.urlopen(request, timeout=timeout)
            raw = getattr(getattr(self._response, "fp", None), "raw", None)
            sock = getattr(raw, "_sock", None) or getattr(raw, "sock", None)
            if sock is not None:
                sock.settimeout(3)
        except Exception as exc:
            print(f"XIAO 串流打不開：{exc}")
            self._response = None

    def isOpened(self) -> bool:
        return self._response is not None

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._response is None:
            return False, None
        deadline = time.time() + 4
        while time.time() < deadline:
            start = self._buf.find(b"\xff\xd8")
            end = self._buf.find(b"\xff\xd9", start + 2) if start >= 0 else -1
            if start >= 0 and end >= 0:
                jpg = self._buf[start : end + 2]
                self._buf = self._buf[end + 2 :]
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    return True, frame
                continue
            if start > 0:
                self._buf = self._buf[start:]
            elif start < 0 and len(self._buf) > 2:
                self._buf = self._buf[-2:]
            try:
                chunk = self._response.read(8192)
            except Exception:
                return False, None
            if not chunk:
                return False, None
            self._buf += chunk
            if len(self._buf) > 2_000_000:
                self._buf = self._buf[-200_000:]
        return False, None

    def release(self) -> None:
        response = self._response
        self._response = None
        if response is not None:
            response.close()


def _probe_http(url: str) -> bool:
    remote = MjpegCapture(url, timeout=5.0)
    if not remote.isOpened():
        return False
    ok, _frame = remote.read()
    remote.release()
    return ok


def _explain_wifi(line: str | None) -> str:
    if line and "not in scan" in line:
        return "XIAO 掃描不到這個 Wi-Fi。請填 2.4GHz 的名稱，5GHz 它連不上。"
    if line and line.startswith("WIFI fail"):
        return "XIAO 沒能加入 Wi-Fi。請核對密碼，並把板子靠近路由器。鏡頭仍可用 USB。"
    if line and line.startswith("STREAM "):
        return "XIAO 已拿到位址，但這台電腦還沒打開畫面。請確認電腦和 XIAO 在同一個路由器。"
    return "XIAO 沒有回應。請確認 USB 已插上，然後再試一次。"


_xiao_port = None


def hold_xiao_port():
    """打開一次 USB 序列埠並一直留著，避免每次開關都把板子重開。"""
    global _xiao_port
    if _xiao_port is not None and getattr(_xiao_port, "is_open", False):
        return _xiao_port
    port_name = find_xiao_port()
    if not port_name:
        return None
    try:
        import serial

        link = serial.Serial(port_name, 115200, timeout=0.2)
        link.dtr = False
        link.rts = False
    except Exception as exc:
        print(f"打不開 {port_name}：{exc}")
        return None
    _xiao_port = link
    return link


def find_xiao_port() -> str | None:
    chosen = os.environ.get("CAMERA_PORT", "").strip()
    try:
        from serial.tools import list_ports
    except ImportError:
        return chosen or None
    for port in list_ports.comports():
        if port.vid == 0x303A or (chosen and port.device == chosen):
            return port.device
    return chosen or None


def discover_stream_url(timeout: float = 8.0) -> str | None:
    fixed = os.environ.get("CAMERA_URL", "").strip() or os.environ.get("XIAO_URL", "").strip()
    if fixed:
        return fixed
    port_name = find_xiao_port()
    if not port_name:
        print("找不到 XIAO 的 USB 序列埠")
        return None
    try:
        import serial
    except ImportError:
        print("未安裝 pyserial")
        return None
    try:
        link = serial.Serial(port_name, 115200, timeout=0.3)
        link.dtr = False
        link.rts = False
    except Exception as exc:
        print(f"打不開 {port_name}：{exc}")
        return None
    deadline = time.time() + timeout
    buf = b""
    found = None
    try:
        while time.time() < deadline and found is None:
            buf += link.read(500)
            text = buf.decode("utf-8", "replace")
            for line in text.splitlines():
                marker = line.find("http://")
                if marker >= 0 and ":81/stream" in line:
                    found = line[marker:].strip().split()[0]
                    break
            if len(buf) > 8000:
                buf = buf[-4000:]
    finally:
        link.close()
    if found:
        print(f"XIAO 位址 {found}")
    return found


class CameraHub:
    """瀏覽器觀看時維持一個鏡頭：電腦 USB 或 XIAO 網路串流。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cap: cv2.VideoCapture | MjpegCapture | None = None
        self._clients = 0
        self._generation = 0
        self.source = "pc"
        self.index: int | None = None
        self.label = "電腦"
        self.stream_url: str | None = None
        self.error: str | None = None
        self.confidence = 0.25
        self.xiao_link = "idle"
        self.xiao_ip = ""
        self.wifi_url: str | None = None
        self.prefer_link = "usb"

    def is_open(self) -> bool:
        with self._lock:
            return self._cap is not None and self._cap.isOpened()

    def set_confidence(self, value: float) -> float:
        self.confidence = clamp_confidence(value)
        return self.confidence

    def use_link(self, link: str) -> None:
        if link not in ("usb", "wifi"):
            return
        self.prefer_link = link
        with self._lock:
            cap = self._cap
            if cap is None:
                return
            if link == "wifi" and isinstance(cap, MjpegCapture):
                return
            if link == "usb" and isinstance(cap, SerialJpegCapture):
                return
        self.request_stop()

    def set_source(self, source: str) -> str:
        source = "xiao" if source == "xiao" else "pc"
        with self._lock:
            changed = source != self.source
            self.source = source
            self.label = "XIAO" if source == "xiao" else "電腦"
            cap = self._cap if changed else None
            if changed:
                self._cap = None
                self.index = None
                self.stream_url = None
                self.error = None
        if cap is not None:
            cap.release()
        return source

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
                self.error = XIAO_ERROR if self.source == "xiao" else CAMERA_ERROR
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

    def _open_http(self, url: str) -> bool:
        remote = MjpegCapture(url)
        if not remote.isOpened():
            return False
        ok, _frame = remote.read()
        if not ok:
            remote.release()
            return False
        self._cap = remote
        self.index = None
        self.stream_url = url
        self.label = "XIAO"
        self.error = None
        self.xiao_link = "wifi"
        self.wifi_url = url
        host = url.split("/")[2] if "://" in url else ""
        self.xiao_ip = host.split(":")[0]
        print(f"XIAO 已由 Wi-Fi 開啟：{url}")
        return True

    def _open_xiao(self) -> bool:
        if self.prefer_link == "wifi" and self.wifi_url:
            if self._open_http(self.wifi_url):
                return True
            print("Wi-Fi 畫面打不開，改走 USB")
            self.prefer_link = "usb"
        port_name = find_xiao_port()
        link = hold_xiao_port()
        if link is None:
            self.error = XIAO_ERROR
            print("找不到 XIAO 的 USB 序列埠" if not port_name else XIAO_ERROR)
            return False
        cap = SerialJpegCapture(link)
        ok, _frame = cap.read(timeout=10.0)
        if ok:
            self._cap = cap
            self.index = None
            self.stream_url = f"usb://{port_name}"
            self.label = "XIAO"
            self.error = None
            self.xiao_link = "usb"
            print(f"XIAO 已由 USB 開啟：{port_name}")
            return True
        text = cap._buf.decode("utf-8", "replace")
        cap.release()
        url = None
        for line in text.splitlines():
            marker = line.find("http://")
            if marker >= 0 and ":81/stream" in line:
                url = line[marker:].split()[0]
                break
        if not url:
            self.error = XIAO_ERROR
            print(XIAO_ERROR)
            return False
        remote = MjpegCapture(url)
        if not remote.isOpened():
            self.error = XIAO_ERROR
            return False
        ok, _frame = remote.read()
        if not ok:
            remote.release()
            self.error = XIAO_ERROR
            print(XIAO_ERROR)
            return False
        self._cap = remote
        self.index = None
        self.stream_url = url
        self.label = "XIAO"
        self.error = None
        self.xiao_link = "wifi"
        self.wifi_url = url
        host = url.split("/")[2] if "://" in url else ""
        self.xiao_ip = host.split(":")[0]
        print(f"XIAO 已開啟：{url}")
        return True

    def connect_wifi(self, ssid: str, password: str) -> dict:
        command = f"WIFI\t{ssid}\t{password}\n".encode("utf-8")
        print(f"要求 XIAO 加入 Wi-Fi：{ssid}")
        with self._lock:
            cap = self._cap if isinstance(self._cap, SerialJpegCapture) and self._cap.isOpened() else None
        if cap is None:
            already_open = _xiao_port is not None and getattr(_xiao_port, "is_open", False)
            link = hold_xiao_port()
            if link is None:
                return {"ok": False, "message": "找不到 XIAO 的 USB。請先插上板子。", "link": "idle"}
            cap = SerialJpegCapture(link)
            if not already_open:
                cap.wait_note(("XIAO ",), timeout=4.0)
        cap.clear_notes()
        if not cap.send(command):
            return {"ok": False, "message": "沒能把 Wi-Fi 設定送到 XIAO。", "link": self.xiao_link}
        line = cap.wait_note(("STREAM http://", "WIFI fail"), timeout=24.0)
        url = line.split()[1] if line and line.startswith("STREAM http://") else ""
        probed = bool(url) and _probe_http(url)
        if not probed:
            return {"ok": False, "message": _explain_wifi(line), "link": "usb" if self.xiao_link == "usb" else self.xiao_link}
        self.wifi_url = url
        self.prefer_link = "wifi"
        self.xiao_link = "wifi"
        self.xiao_ip = url.split("/")[2].split(":")[0]
        return {
            "ok": True,
            "message": f"XIAO 已加入 {ssid}，正在用 Wi-Fi 開啟鏡頭。",
            "link": "wifi",
            "url": url,
        }

    def _open_unlocked(self) -> bool:
        self.error = None
        if self.source == "xiao":
            return self._open_xiao()
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
                self.label = "電腦"
                self.stream_url = None
                print(f"電腦鏡頭已開啟：index {index}")
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
                    camera.error = camera.error or (XIAO_ERROR if camera.source == "xiao" else CAMERA_ERROR)
                    yield mjpeg_part(encode_jpeg(draw_message(camera.error)))
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
def video_feed(conf: float = 0.25, stop: int = 0, source: str = "", link: str = ""):
    if stop:
        camera.request_stop()
        return {"stopped": True}
    if source:
        camera.set_source(source)
    if link in ("usb", "wifi"):
        camera.use_link(link)
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


def _clean_wifi_field(value: str, limit: int) -> str:
    return value.replace("\t", "").replace("\r", "").replace("\n", "").strip()[:limit]


def _xiao_link_state() -> str:
    url = camera.stream_url or ""
    if camera.source == "xiao" and camera.is_open():
        if url.startswith("http://"):
            return "wifi"
        if url.startswith("usb://"):
            return "usb"
    if camera.xiao_link == "wifi" and camera.wifi_url:
        return "wifi"
    if find_xiao_port():
        return "plugged"
    return "idle"


@app.post("/xiao/wifi")
async def xiao_wifi(request: Request):
    data = await request.json()
    ssid = _clean_wifi_field(str(data.get("ssid", "")), 32)
    password = _clean_wifi_field(str(data.get("password", "")), 63)
    if not ssid:
        return {"ok": False, "message": "請填 2.4GHz 的 Wi-Fi 名稱。", "link": _xiao_link_state()}
    return camera.connect_wifi(ssid, password)


@app.get("/health")
def health():
    model_loaded = model is not None
    return {
        "model_loaded": model_loaded,
        "model_message": None if model_loaded else MODEL_ERROR,
        "camera_open": camera.is_open(),
        "camera_index": camera.index,
        "camera_source": camera.source,
        "camera_label": camera.label,
        "camera_url": camera.stream_url,
        "camera_link": _xiao_link_state(),
        "camera_ip": camera.xiao_ip,
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
