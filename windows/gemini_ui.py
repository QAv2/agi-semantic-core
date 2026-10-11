"""Hand-pace GUI driver for the Gemini app in Firefox (W1 Amendments 2 and 3). Runs under the
system python3 (python3-gi with Atspi 2.0; xdotool, xclip, ImageMagick `import`).

Perception is Firefox's accessibility tree (AT-SPI): every control is found by its role and
accessible name, and its screen box comes from the tree, so nothing on the page is read from
pixels. Firefox must have been started with accessibility on (GNOME_ACCESSIBILITY=1). Input is
xdotool: real pointer clicks at a control's centre and real key presses, one chat at a time.
The target window's focus is asserted before every click and keystroke (law:
feedback-gui-focus-guard); stray keys are never cleared with Escape. A message counts as sent
only when its "You said" heading appears; a reply is done when the stop button is gone and the
reply's action bar is there and still. Every finished turn can leave a screenshot.
"""
import shutil
import subprocess
import time
from pathlib import Path
import random
import re

try:
    import gi
    gi.require_version("Atspi", "2.0")
    from gi.repository import Atspi, GLib
except (ImportError, ValueError):     # the analysis venv has no gi: only the rules below are used there
    Atspi = GLib = None

WIN_NAME = "Google Gemini — Mozilla Firefox"
DOC_NAME = "Google Gemini"
ENTRY = "Enter a prompt for Gemini"
MODE_PREFIX = "Open mode picker, currently "
W1_MODE = ("3.8 Flash All-around help", "Low Quick and efficient", "Flash")      # Amendment 2
JOE_MODE = ("3.1 Pro Advanced reasoning", "High Extra thorough", "Pro High")     # restored after a sitting
TEMP_HEADING = "Just stopping by?"
TOOL_ITEM = "Create image"
TOOL_CHIP = "Deselect Images"
STOP = "Stop response"
YOU_SAID = "You said"
GEMINI_SAID = "Gemini said"
NEUTRAL = (1250, 560)          # an empty margin: the mouse rests and wheels here
DOWNLOADS = Path.home() / "Downloads"


class UIError(RuntimeError):
    pass


class LimitStop(UIError):
    """A drawing turn ended, after its retry, with no image and a reply that reads like the app's
    usage-limit notice. The run stops without logging the turn (Amendment 3)."""

    def __init__(self, out):
        super().__init__(f"no image, and the reply reads like a usage limit: {out.get('text')!r}")
        self.out = out


LIMIT_WORDS = re.compile(r"\b(limit|limits|quota)\b|try again (later|tomorrow)|come back (later|tomorrow)", re.I)
IMAGE_WORDS = re.compile(r"\b(image|images|generat\w*|creat\w*)\b", re.I)


def reads_like_limit(text):
    return bool(text) and bool(LIMIT_WORDS.search(text)) and bool(IMAGE_WORDS.search(text))


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


def get_clipboard():
    return sh("xclip", "-selection", "clipboard", "-o", check=False)


def squash(s):
    """Text with all whitespace removed, for comparing what was pasted with what the page shows."""
    return "".join((s or "").replace("￼", "").split())


LIST_MARKS = str.maketrans("", "", "*•◦▪-")


def same_text(a, b):
    """Whether the input (or a posted message) holds `b`. The app's editor turns pasted lines
    that start with "* " or "- " into list items, which the tree shows with a bullet glyph, so
    list and emphasis marks are left out of the comparison (smoke 2, clarification 5)."""
    return squash(a).translate(LIST_MARKS) == squash(b).translate(LIST_MARKS)


# ---- the accessibility tree ---------------------------------------------------------------

def walk(o):
    """Pre-order (document order) over a subtree. Nodes that vanish mid-walk (the page is
    changing under us) are skipped."""
    stack = [o]
    while stack:
        x = stack.pop()
        if x is None:
            continue
        try:
            n = x.get_child_count()
        except GLib.Error:
            continue
        yield x
        for i in range(n - 1, -1, -1):
            try:
                stack.append(x.get_child_at_index(i))
            except GLib.Error:
                pass


def name(o):
    try:
        return o.get_name() or ""
    except GLib.Error:
        return ""


def role(o):
    try:
        return o.get_role_name() or ""
    except GLib.Error:
        return ""


def box(o):
    try:
        e = o.get_extents(Atspi.CoordType.SCREEN)
    except GLib.Error:
        return 0, 0, 0, 0
    return e.x, e.y, e.width, e.height


def _state(o, st):
    try:
        return o.get_state_set().contains(st)
    except GLib.Error:
        return False


def showing(o):
    return _state(o, Atspi.StateType.SHOWING)


def focused(o):
    return _state(o, Atspi.StateType.FOCUSED)


def text_of(o):
    """All the text under a node, in document order (embedded-object marks dropped)."""
    out = []
    for x in walk(o):
        if role(x) in ("paragraph", "heading", "list item", "static", "code", "block quote",
                                 "table cell", "label", "link"):
            try:
                t = Atspi.Text.get_text(x, 0, Atspi.Text.get_character_count(x))
            except Exception:
                continue
            t = t.replace("￼", "")
            if t.strip():
                out.append(t)
    return "\n".join(out)


class GeminiUI:
    def __init__(self, log=print, pace=(3.0, 8.0), seed=0, shots_dir=None):
        self.log, self.pace = log, pace
        self.rng = random.Random(seed)
        self.shots = Path(shots_dir) if shots_dir else None
        ids = sh("xdotool", "search", "--name", WIN_NAME).decode().split()
        if not ids:
            raise UIError("no Gemini window in Firefox")
        self.win = ids[0]
        self._doc = self._app = None
        self.doc()
        self.clip_saved = get_clipboard()

    # ---- the tree -----------------------------------------------------------------------
    def doc(self):
        """The Gemini page's document node (found again if the cached one has gone stale)."""
        try:
            if self._doc is not None and role(self._doc) == "document web" \
                    and name(self._doc).startswith(DOC_NAME):
                return self._doc
        except Exception:
            pass
        desk = Atspi.get_desktop(0)
        for i in range(desk.get_child_count()):
            app = desk.get_child_at_index(i)
            if app is None or "Firefox" not in name(app):
                continue
            for x in walk(app):
                if role(x) == "document web" and name(x).startswith(DOC_NAME) and showing(x.get_parent()):
                    self._doc, self._app = x, app
                    return x
        raise UIError("the Gemini page is not in the accessibility tree: start Firefox with "
                      "GNOME_ACCESSIBILITY=1 and keep the Gemini tab in front")

    def all(self, role_, exact=None, prefix=None, root=None):
        out = []
        for x in walk(root or self.doc()):
            if role(x) != role_:
                continue
            n = name(x)
            if (exact is not None and n != exact) or (prefix is not None and not n.startswith(prefix)):
                continue
            out.append(x)
        return out

    def one(self, role_, exact=None, prefix=None, root=None, timeout=10.0, what=None):
        end = time.time() + timeout
        while True:
            hits = self.all(role_, exact, prefix, root)
            if hits:
                return hits[-1]
            if time.time() > end:
                raise UIError(f"not found: {what or role_} {exact or prefix!r}")
            time.sleep(0.4)

    def mode_name(self):
        hits = self.all("button", prefix=MODE_PREFIX)
        return name(hits[0])[len(MODE_PREFIX):] if hits else None

    # ---- basics -------------------------------------------------------------------------
    def guard(self):
        sh("xdotool", "windowactivate", "--sync", self.win)
        name_ = sh("xdotool", "getwindowname", sh("xdotool", "getactivewindow").decode().strip()).decode().strip()
        if name_ != WIN_NAME:
            raise UIError(f"focus guard: active window is {name_!r}")

    def save_shot(self, label):
        if self.shots:
            self.shots.mkdir(parents=True, exist_ok=True)
            sh("import", "-silent", "-window", "root", str(self.shots / f"{label}.png"))

    def move(self, x, y):
        sh("xdotool", "mousemove", str(x), str(y))

    def key(self, *keys):
        self.guard()
        sh("xdotool", "key", *keys)

    def wheel(self, notches, down=True):
        self.move(*NEUTRAL)
        for _ in range(notches):
            sh("xdotool", "click", "5" if down else "4")
            time.sleep(0.05)

    def pause(self):
        time.sleep(self.rng.uniform(*self.pace))

    def safe_band(self):
        """The y range where a conversation control is visible and not under the top bar or
        the input panel."""
        ent = self.all("entry", exact=ENTRY)
        bottom = (box(ent[-1])[1] - 24) if ent else 700
        return 175, bottom

    def reveal(self, o, in_conversation=True):
        """Bring a control into the clickable band; -> its centre."""
        for k in range(8):
            x, y, w, h = box(o)
            cx, cy = x + w // 2, y + h // 2
            lo, hi = self.safe_band() if in_conversation else (113, 768)
            if showing(o) and w > 0 and lo <= cy <= hi and 0 <= cx < 1366:
                return cx, cy
            if k == 0:
                try:
                    Atspi.Component.scroll_to(o, Atspi.ScrollType.ANYWHERE)
                except Exception:
                    pass
            else:
                self.wheel(2, down=cy > hi)
            time.sleep(0.5)
        raise UIError(f"could not bring {role(o)} {name(o)!r} into view")

    def click(self, o, in_conversation=True, rest=True):
        cx, cy = self.reveal(o, in_conversation)
        self.guard()
        sh("xdotool", "mousemove", str(cx), str(cy), "click", "1")
        if rest:
            time.sleep(0.3)
            self.move(*NEUTRAL)

    # ---- the mode picker ----------------------------------------------------------------
    def set_mode(self, mode=W1_MODE):
        """Pick a model, then a thinking level, in the mode picker; check the picker's label.
        Picking an item that is already selected only closes the menu."""
        model, level, label = mode
        for item in (model, level):
            if self.mode_name() == label:
                break
            self.click(self.one("button", prefix=MODE_PREFIX), in_conversation=False)
            self.one("menu item", timeout=5, what="mode picker menu")
            hits = [x for x in self.all("menu item") if name(x).removeprefix("Selected ") == item]
            if not hits:
                raise UIError(f"mode picker has no item {item!r}")
            self.click(hits[0], in_conversation=False)
            time.sleep(1.0)
        if self.mode_name() != label:
            raise UIError(f"mode picker reads {self.mode_name()!r}, wanted {label!r}")
        return label

    # ---- chat operations ----------------------------------------------------------------
    def new_temp_chat(self):
        """New chat -> temporary chat. G1: the temporary-chat heading, the picker label, an
        empty chat, the image tool off."""
        self.click(self.one("link", exact="New chat"), in_conversation=False)
        end = time.time() + 15
        while self.all("heading", prefix=YOU_SAID) or not self.all("entry", exact=ENTRY):
            if time.time() > end:
                raise UIError("G1: the new chat did not open empty")
            time.sleep(0.5)
        end = time.time() + 3                 # the app may keep temporary mode across New chat
        while not self.all("heading", prefix=TEMP_HEADING) and time.time() < end:
            time.sleep(0.4)
        if not self.all("heading", prefix=TEMP_HEADING):
            self.click(self.one("button", exact="Temporary chat"), in_conversation=False)
            self.one("heading", prefix=TEMP_HEADING, timeout=10, what="temporary-chat heading")
        mode = self.mode_name()
        tool = bool(self.all("button", exact=TOOL_CHIP))
        if mode != W1_MODE[2] or tool:
            self.save_shot(f"G1_fail_{int(time.time())}")
            raise UIError(f"G1: picker {mode!r} (wanted {W1_MODE[2]!r}), image tool on: {tool}")
        return {"temp": True, "mode": mode}

    def image_tool(self):
        """Switch on the input's Create image tool (Amendment 3); -> True once its chip is shown."""
        self.click(self.one("button", exact="Upload & tools"), in_conversation=False)
        self.click(self.one("check menu item", exact=TOOL_ITEM, timeout=5, what="tools menu"),
                   in_conversation=False)
        self.one("button", exact=TOOL_CHIP, timeout=8, what="the Images chip")
        return True

    def entry_text(self):
        return text_of(self.one("entry", exact=ENTRY))

    def send(self, text):
        """Paste `text` into the input, check it landed, press Enter, and wait for the message's
        "You said" heading."""
        n0 = len(self.all("heading", prefix=YOU_SAID))
        ent = self.one("entry", exact=ENTRY)
        if not focused(ent):
            Atspi.Component.grab_focus(ent)
            time.sleep(0.3)
        if not focused(self.one("entry", exact=ENTRY)):
            raise UIError("the input would not take focus")
        set_clipboard(text.encode())
        self.key("ctrl+a")
        self.key("ctrl+v")
        time.sleep(0.6)
        got = self.entry_text()
        if not same_text(got, text):
            raise UIError(f"the input holds {got[:60]!r}, not the message")
        self.key("Return")
        self.move(*NEUTRAL)
        end = time.time() + 20
        while time.time() < end:
            hs = self.all("heading", prefix=YOU_SAID)
            if len(hs) == n0 + 1:
                posted = squash(name(hs[-1])[len(YOU_SAID):]).translate(LIST_MARKS)
                if not posted.startswith(squash(text).translate(LIST_MARKS)[:40]):
                    raise UIError(f"posted message reads {name(hs[-1])[:60]!r}")
                return
            time.sleep(0.4)
        raise UIError("the message did not post (no new 'You said' heading)")

    def last_reply(self):
        """The newest reply's block: the ancestor of the last 'Gemini said' heading just below the
        level that also holds a 'You said' heading. None until the reply has started."""
        heads = [x for x in walk(self.doc()) if role(x) == "heading"
                 and (name(x).startswith(YOU_SAID) or name(x) == GEMINI_SAID)]
        if not heads or name(heads[-1]) != GEMINI_SAID:
            return None
        node = heads[-1]
        while True:
            parent = node.get_parent()
            if parent is None or any(name(h).startswith(YOU_SAID) for h in self.all("heading", root=parent)):
                return node
            node = parent

    def reply_state(self):
        rep = self.last_reply()
        if rep is None:
            return None
        return {"node": rep,
                "done_bar": bool(self.all("button", exact="Copy", root=rep)),
                "images": len(self.all("button", exact="Download full size image", root=rep)),
                "text": text_of(rep)}

    def wait_done(self, timeout=240, settle=2):
        """Done when the stop button is gone and the reply's action bar is there, unchanged for
        `settle` polls (1 s apart)."""
        t0 = time.time()
        prev, still = None, 0
        while time.time() - t0 < timeout:
            time.sleep(1.0)
            st = self.reply_state()
            busy = bool(self.all("button", exact=STOP))
            if st and st["done_bar"] and not busy:
                sig = (st["images"], st["text"])
                still = still + 1 if sig == prev else 0
                prev = sig
                if still >= settle:
                    return st, time.time() - t0
            else:
                prev, still = None, 0
        raise UIError(f"reply not finished after {timeout} s")

    def copy_last_text(self, st):
        """The reply's text through its own Copy button (the app's markdown)."""
        for _ in range(2):
            set_clipboard(b"\x00")
            self.click(self.all("button", exact="Copy", root=st["node"])[-1])
            time.sleep(0.8)
            txt = get_clipboard().decode("utf-8", "replace")
            if txt not in ("", "\x00"):
                return txt.strip()
            if not squash(st["text"].replace(GEMINI_SAID, "")):
                return None                       # nothing to copy: an image-only reply
        return None

    def regenerate(self, st):
        """The reply's Redo button opens a menu; its 'Try again' replaces the reply."""
        redo = self.all("button", exact="Redo", root=st["node"])
        if not redo:
            return False
        self.click(redo[-1])
        self.click(self.one("menu item", exact="Try again", timeout=5, what="Redo menu"), in_conversation=False)
        return True

    def download_last_image(self, st, dest):
        """Click the lowest image's 'Download full size image' and move the file to `dest`."""
        dls = self.all("button", exact="Download full size image", root=st["node"])
        if not dls:
            return None
        t0 = time.time() - 1
        self.click(dls[-1])
        f = self._wait_download(t0)
        self.close_downloads_panel()
        if f is None:
            return None
        dest = Path(dest).with_suffix(f.suffix)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(f), str(dest))
        return dest

    def downloads_panel(self):
        self.doc()
        return [x for x in walk(self._app) if role(x) == "panel" and name(x) == "Downloads" and showing(x)]

    def close_downloads_panel(self):
        """Firefox opens its Downloads panel after each download, over the page; it would swallow
        the next click. A click on the page's empty margin closes it."""
        for _ in range(3):
            time.sleep(0.6)
            if not self.downloads_panel():
                return
            self.guard()
            sh("xdotool", "mousemove", str(NEUTRAL[0]), str(NEUTRAL[1]), "click", "1")
        if self.downloads_panel():
            raise UIError("Firefox's Downloads panel would not close")

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
    def turn(self, text, want_image=False, img_dest=None, shot_name=None, image_tool=False):
        """Send one message; -> dict(text, image path | None, latency, regenerated)."""
        if image_tool:
            self.image_tool()
        self.send(text)
        st, lat = self.wait_done()
        out = {"text": self.copy_last_text(st), "image": None, "latency_s": round(lat, 2), "regenerated": False}
        if want_image:
            out["image"] = self.download_last_image(st, img_dest)
        bad = (not out["text"] and not want_image) or (want_image and not out["image"])
        if bad and self.regenerate(st):                       # G2: one retry = regenerate
            out["regenerated"] = True
            st, lat = self.wait_done()
            out["latency_s"] = round(lat, 2)
            out["text"] = self.copy_last_text(st)
            if want_image:
                out["image"] = self.download_last_image(st, img_dest)
        if shot_name:
            self.save_shot(shot_name)
        if want_image and not out["image"] and reads_like_limit(out["text"]):
            raise LimitStop(out)
        self.pause()
        return out

    def close(self, mode=JOE_MODE):
        """End of a sitting: Joe's picker setting and clipboard back."""
        try:
            self.set_mode(mode)
        finally:
            set_clipboard(self.clip_saved)
