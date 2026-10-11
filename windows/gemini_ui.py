"""Hand-pace GUI driver for the Gemini app in Firefox (W1 Amendment 2). Runs under the system
python3 (needs cv2, numpy, PIL; xdotool, xclip, ImageMagick `import`).

One chat at a time. The target window's focus is asserted before every keystroke (law:
feedback-gui-focus-guard); stray keys are never cleared with Escape. Screens are read with
template matching on binarized glyphs (white-on-dark icons), so an icon is found over any
background. Every finished turn leaves a screenshot.
"""
import io
import random
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

TPL_DIR = Path(__file__).resolve().parent / "ui_templates"
WIN_NAME = "Google Gemini — Mozilla Firefox"
NEW_CHAT = (26, 229)
TEMP_TOGGLE = (1336, 144)
NEUTRAL = (1250, 560)          # an empty spot for the mouse to rest (no hover artifacts)
DOWNLOADS = Path.home() / "Downloads"


class UIError(RuntimeError):
    pass


def sh(*args, input=None, check=True):
    r = subprocess.run(args, input=input, capture_output=True, check=False)
    if check and r.returncode != 0:
        raise UIError(f"{args[0]} failed: {r.stderr.decode()[:200]}")
    return r.stdout


def set_clipboard(data):
    """xclip forks a daemon that serves the selection and holds any captured pipe open, so its
    output must go to /dev/null or the call never returns."""
    subprocess.run(["xclip", "-selection", "clipboard"], input=data, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL, check=True, timeout=10)


def _bin(gray, thr=170):
    return (gray > thr).astype(np.uint8) * 255


class GeminiUI:
    def __init__(self, log=print, pace=(3.0, 8.0), seed=0, shots_dir=None):
        self.log, self.pace = log, pace
        self.rng = random.Random(seed)
        self.shots = Path(shots_dir) if shots_dir else None
        ids = sh("xdotool", "search", "--name", WIN_NAME).decode().split()
        if not ids:
            raise UIError("no Gemini window in Firefox")
        self.win = ids[0]
        self.tpl = {}
        for name in ("temp_header", "flash_label", "mic", "copy", "regen", "download"):
            g = cv2.cvtColor(cv2.imread(str(TPL_DIR / f"{name}.png")), cv2.COLOR_BGR2GRAY)
            self.tpl[name] = g

    # ---- basics -------------------------------------------------------------------------
    def guard(self):
        sh("xdotool", "windowactivate", "--sync", self.win)
        name = sh("xdotool", "getwindowname", sh("xdotool", "getactivewindow").decode().strip()).decode().strip()
        if name != WIN_NAME:
            raise UIError(f"focus guard: active window is {name!r}")

    def shot(self):
        png = sh("import", "-silent", "-window", "root", "png:-")
        return cv2.cvtColor(np.array(Image.open(io.BytesIO(png)).convert("RGB")), cv2.COLOR_RGB2GRAY)

    def save_shot(self, name):
        if self.shots:
            self.shots.mkdir(parents=True, exist_ok=True)
            sh("import", "-silent", "-window", "root", str(self.shots / f"{name}.png"))

    def move(self, x, y):
        sh("xdotool", "mousemove", str(x), str(y))

    def click(self, x, y):
        self.guard()
        sh("xdotool", "mousemove", str(x), str(y), "click", "1")

    def key(self, *keys):
        self.guard()
        sh("xdotool", "key", *keys)

    def wheel(self, x, y, notches, down=True):
        self.move(x, y)
        for _ in range(notches):
            sh("xdotool", "click", "5" if down else "4")
            time.sleep(0.05)

    def pause(self):
        time.sleep(self.rng.uniform(*self.pace))

    def find(self, name, img=None, region=None, thr=0.80, binary=True):
        """-> list of (cx, cy, score), best first, non-overlapping."""
        img = self.shot() if img is None else img
        x0 = y0 = 0
        if region:
            x0, y0, x1, y1 = region
            img = img[y0:y1, x0:x1]
        t = self.tpl[name]
        a, b = (_bin(img), _bin(t)) if binary else (img, t)
        if a.shape[0] < b.shape[0] or a.shape[1] < b.shape[1]:
            return []
        res = cv2.matchTemplate(a, b, cv2.TM_CCOEFF_NORMED)
        hits = []
        h, w = t.shape
        ys, xs = np.where(res >= thr)
        for y, x in sorted(zip(ys, xs), key=lambda p: -res[p[0], p[1]]):
            cx, cy = x0 + x + w // 2, y0 + y + h // 2
            if all(abs(cx - hx) > w // 2 or abs(cy - hy) > h // 2 for hx, hy, _ in hits):
                hits.append((int(cx), int(cy), float(res[y, x])))
        return hits

    # ---- chat operations ----------------------------------------------------------------
    def new_temp_chat(self):
        """New chat -> temporary chat; G1: the temp page and the 'Flash' (= Low) label."""
        self.click(*NEW_CHAT)
        time.sleep(2.0)
        self.click(*TEMP_TOGGLE)
        time.sleep(2.0)
        self.move(*NEUTRAL)
        time.sleep(0.4)
        img = self.shot()
        head = self.find("temp_header", img, binary=False, thr=0.85)
        label = self.find("flash_label", img, region=(820, 380, 1000, 720), binary=False, thr=0.85)
        if not head or not label:
            self.save_shot(f"G1_fail_{int(time.time())}")
            raise UIError(f"G1: temporary chat page {bool(head)}, 'Flash' label {bool(label)}")
        return {"temp_header": head[0][2], "flash_label": label[0][2]}

    def input_point(self, img=None):
        mic = self.find("mic", img, region=(950, 380, 1060, 740), binary=False, thr=0.80)
        if not mic:
            return None
        return 600, mic[0][1]

    def send(self, text):
        """Paste `text` into the input and press Enter."""
        pt = self.input_point()
        if not pt:
            raise UIError("input box not found (no mic icon)")
        self.click(*pt)
        time.sleep(0.3)
        set_clipboard(text.encode())
        self.key("ctrl+a")
        self.key("ctrl+v")
        time.sleep(0.5)
        self.key("Return")
        self.move(*NEUTRAL)

    def wait_done(self, timeout=240, settle=2):
        """Done when the input shows its mic again and the screen has been still for `settle`
        consecutive polls (1.5 s apart), with the page scrolled to the bottom."""
        t0 = time.time()
        time.sleep(3.0)
        prev, still = None, 0
        while time.time() - t0 < timeout:
            self.wheel(700, 400, 15)
            self.move(*NEUTRAL)
            time.sleep(1.5)
            img = self.shot()
            ready = bool(self.input_point(img))
            if prev is not None and ready and np.array_equal(img[120:650], prev[120:650]):
                still += 1
                if still >= settle:
                    return img, time.time() - t0
            else:
                still = 0
            prev = img
        raise UIError(f"reply not finished after {timeout} s")

    def action_bar(self, img):
        """The latest reply's action bar: the lowest copy icon in the reply column."""
        hits = [h for h in self.find("copy", img, region=(40, 120, 600, 660)) if h[2] > 0.8]
        return max(hits, key=lambda h: h[1]) if hits else None

    def copy_last_text(self, img):
        bar = self.action_bar(img)
        if not bar:
            return None
        set_clipboard(b"\x00")
        self.click(bar[0], bar[1])
        time.sleep(0.8)
        self.move(*NEUTRAL)
        txt = sh("xclip", "-selection", "clipboard", "-o", check=False).decode("utf-8", "replace")
        return None if txt in ("", "\x00") else txt.strip()

    def regenerate(self, img):
        bar = self.action_bar(img)
        if not bar:
            return False
        regen = [h for h in self.find("regen", img, region=(bar[0] - 60, bar[1] - 15, bar[0], bar[1] + 15))]
        if not regen:
            return False
        self.click(regen[0][0], regen[0][1])
        return True

    def download_last_image(self, img, dest, tries=8):
        """Hover inside the lowest image, click its download icon, move the file to `dest`."""
        bar = self.action_bar(img)
        if not bar:
            return None
        y = bar[1] - 90
        t0 = time.time() - 1
        for k in range(tries):
            self.move(300, max(140, y))
            time.sleep(0.9)
            shot = self.shot()
            hits = self.image_buttons(shot, below=bar[1])
            if hits:
                hx, hy, _ = max(hits, key=lambda h: h[1])
                self.click(hx, hy)
                f = self._wait_download(t0)
                self.move(*NEUTRAL)
                if f is None:
                    return None
                dest = Path(dest).with_suffix(f.suffix)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(f), str(dest))
                return dest
            self.wheel(700, 400, 2, down=False)       # the image top is above the view
            time.sleep(0.6)
            bar_now = self.action_bar(self.shot())
            y = (bar_now[1] - 90) if bar_now else 300
        self.move(*NEUTRAL)
        return None

    def image_buttons(self, img, below):
        """Download icons of images above y=`below`. The icon sits on a semi-transparent button
        over the picture, so neither match alone is reliable: a candidate needs the binarized
        glyph match >= 0.6 AND the grayscale match >= 0.75 at the same spot."""
        region = (40, 120, 1000, 660)
        dl = self.find("download", img, region=region, thr=0.6)
        gray = self.find("download", img, region=region, thr=0.75, binary=False)
        return [(x, y, sc) for x, y, sc in dl if y < below
                and any(abs(gx - x) <= 3 and abs(gy - y) <= 3 for gx, gy, _ in gray)]

    def _wait_download(self, t0, timeout=40):
        end = time.time() + timeout
        while time.time() < end:
            fs = [p for p in DOWNLOADS.glob("Gemini_Generated_Image_*") if p.stat().st_mtime >= t0
                  and not p.name.endswith(".part")]
            if fs and not list(DOWNLOADS.glob("*.part")):
                f = max(fs, key=lambda p: p.stat().st_mtime)
                s1 = f.stat().st_size
                time.sleep(0.6)
                if f.stat().st_size == s1 and s1 > 0:
                    return f
            time.sleep(0.5)
        return None

    # ---- one turn -----------------------------------------------------------------------
    def turn(self, text, want_image=False, img_dest=None, shot_name=None):
        """Send one message; -> dict(text, image path | None, latency, regenerated)."""
        self.send(text)
        img, lat = self.wait_done()
        out = {"text": self.copy_last_text(img), "image": None, "latency_s": round(lat, 2), "regenerated": False}
        if want_image:
            out["image"] = self.download_last_image(img, img_dest)
        bad = (not out["text"] and not want_image) or (want_image and not out["image"])
        if bad and self.regenerate(img):                       # G2: one retry = regenerate
            out["regenerated"] = True
            img, lat = self.wait_done()
            out["latency_s"] = round(lat, 2)
            out["text"] = self.copy_last_text(img)
            if want_image:
                out["image"] = self.download_last_image(img, img_dest)
        if shot_name:
            self.save_shot(shot_name)
        self.pause()
        return out
