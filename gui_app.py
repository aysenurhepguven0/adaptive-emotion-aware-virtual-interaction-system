"""
gui_app.py - Adaptive Emotion-Aware Virtual Interaction System GUI
===================================================================
Three-panel interface matching the DESIGN.png mockup:

  Left:   Control Panel  (model selection, UDP config, sensitivity,
          webcam preview with Grad-CAM, START / STOP)
  Center: TouchDesigner visualization area (Spout receiver or
          placeholder when TD is not connected)
  Right:  Monitoring panel (emotion probability bars, dominant
          emotion, FPS / latency / UDP status)

Usage:
    python gui_app.py

The center panel can receive TouchDesigner output via Spout.
See the README section at the bottom of this file for TD setup.
"""

from __future__ import annotations

import ctypes
import gc
import platform as _platform
if _platform.system() == "Windows":
    from ctypes import wintypes
import json
import os
import socket
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
import customtkinter as ctk
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageTk
from torchvision import transforms

from data.dataset import build_transforms
from models import get_model_config, load_model_from_checkpoint
from utils.grad_cam import GradCAM, get_target_layer, overlay_heatmap
from utils.calibration import CalibrationManager
from utils.ensemble import EnsembleManager

# ── Paths ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
BEST_DIR = PROJECT_ROOT / "best_models_output"

# ── Emotion definitions ──────────────────────────────────────────
EMOTION_CLASSES: Tuple[str, ...] = (
    "angry", "happy", "neutral", "sad", "surprise",
)

# Hex colors for tkinter widgets
EMOTION_HEX: Dict[str, str] = {
    "happy":    "#E6B800",
    "neutral":  "#888888",
    "surprise": "#00B4D8",
    "sad":      "#4169E1",
    "angry":    "#CC2936",
}

# BGR colours for OpenCV overlays
EMOTION_BGR: Dict[str, Tuple[int, int, int]] = {
    "happy":    (0, 215, 255),
    "neutral":  (180, 180, 180),
    "surprise": (216, 180, 0),
    "sad":      (225, 105, 65),
    "angry":    (54, 41, 204),
}

# ── Model checkpoint discovery ───────────────────────────────────
MODEL_OPTIONS: Dict[str, Dict] = {
    "ResNet-18": {
        "name": "resnet18",
        "checkpoint": RESULTS_DIR / "resnet18" / "best_resnet18.pth",
        "input_size": 224,
        "grayscale": True,
    },
    "HSEmotion": {
        "name": "hsemotion",
        "checkpoint": RESULTS_DIR / "hsemotion" / "best_hsemotion.pth",
        "input_size": 224,
        "grayscale": True,
    },
    "EfficientNet-B0": {
        "name": "efficientnet_b0",
        "checkpoint": RESULTS_DIR / "efficientnet_b0" / "best_efficientnet_b0.pth",
        "input_size": 224,
        "grayscale": True,
    },
    "Mini-Xception": {
        "name": "mini_xception",
        "checkpoint": RESULTS_DIR / "mini_xception" / "best_mini_xception.pth",
        "input_size": 48,
        "grayscale": True,
    },
}

# ── Haar cascade constants ───────────────────────────────────────
FACE_PADDING_RATIO = 0.2
MIN_FACE_SIZE = (48, 48)
SCALE_FACTOR = 1.1
MIN_NEIGHBORS = 5

# ── UDP defaults ─────────────────────────────────────────────────
DEFAULT_UDP_IP = "127.0.0.1"
DEFAULT_UDP_PORT = 7000
SEND_INTERVAL = 0.05  # max 20 sends/s

# ── Frame pacing ─────────────────────────────────────────────────
# Capture/inference target rate. 12 FPS is plenty for emotion output
# (emotion rarely changes faster than human reaction time) and keeps
# CPU well below half of an unthrottled 30 FPS loop.
TARGET_FPS = 12
FRAME_INTERVAL = 1.0 / TARGET_FPS

# Single-model forward pass runs every Nth frame; cached probs fill
# the gaps. 3 → ~4 forwards/sec, which still feels real-time.
SINGLE_INFER_EVERY = 3

# Syphon sender name set in TouchDesigner (macOS only)
SYPHON_SENDER_NAME = "EmotionViz"

# Target display size for TouchDesigner Spout/Syphon frames
# Computed dynamically in EmotionGUI.__init__ from screen size.
SPOUT_DISPLAY_SIZE = (800, 800)  # fallback; overridden at runtime

# Preview size for webcam/test image panel (must match webcam_frame in UI)
WEBCAM_PREVIEW_SIZE = (240, 180)

# ── Platform font ────────────────────────────────────────────────
_SYS = _platform.system()
if _SYS == "Darwin":
    UI_FONT = "Helvetica Neue"
elif _SYS == "Windows":
    UI_FONT = "Segoe UI"
else:
    UI_FONT = "DejaVu Sans"

# ── GUI colour palette ───────────────────────────────────────────
BG_COLOR = "#1E1E1E"
PANEL_BG = "#2D2D2D"
ACCENT = "#3C3C3C"
TEXT_COLOR = "#D4D4D4"
HEADING_COLOR = "#FFFFFF"
BORDER_COLOR = "#6A6A6A"
BTN_START = "#1B9E3E"
BTN_START_HOVER = "#22C44E"
BTN_STOP = "#D32F2F"
BTN_STOP_HOVER = "#EF5350"
BTN_PRIMARY = "#1565C0"       # blue – Calibrate, Load Image
BTN_PRIMARY_HOVER = "#1E88E5"
BTN_SECONDARY = "#455A64"     # blue-grey – Load, Clear
BTN_SECONDARY_HOVER = "#607D8B"
BAR_BG = "#3C3C3C"


# ── Hidden OpenGL context for Spout (Windows-only) ──────────────
if _platform.system() == "Windows":
    class _PIXELFORMATDESCRIPTOR(ctypes.Structure):
        _fields_ = [
            ('nSize', wintypes.WORD), ('nVersion', wintypes.WORD),
            ('dwFlags', wintypes.DWORD), ('iPixelType', ctypes.c_byte),
            ('cColorBits', ctypes.c_byte),
            ('cRedBits', ctypes.c_byte), ('cRedShift', ctypes.c_byte),
            ('cGreenBits', ctypes.c_byte), ('cGreenShift', ctypes.c_byte),
            ('cBlueBits', ctypes.c_byte), ('cBlueShift', ctypes.c_byte),
            ('cAlphaBits', ctypes.c_byte), ('cAlphaShift', ctypes.c_byte),
            ('cAccumBits', ctypes.c_byte),
            ('cAccumRedBits', ctypes.c_byte), ('cAccumGreenBits', ctypes.c_byte),
            ('cAccumBlueBits', ctypes.c_byte), ('cAccumAlphaBits', ctypes.c_byte),
            ('cDepthBits', ctypes.c_byte), ('cStencilBits', ctypes.c_byte),
            ('cAuxBuffers', ctypes.c_byte), ('iLayerType', ctypes.c_byte),
            ('bReserved', ctypes.c_byte),
            ('dwLayerMask', wintypes.DWORD), ('dwVisibleMask', wintypes.DWORD),
            ('dwDamageMask', wintypes.DWORD),
        ]

    def _create_hidden_gl_context():
        """Create a hidden 1x1 window with an OpenGL context for Spout."""
        _PFD_SUPPORT_OPENGL = 0x20
        _PFD_DOUBLEBUFFER = 0x01

        hwnd = ctypes.windll.user32.CreateWindowExW(
            0, "STATIC", "SpoutGL", 0, 0, 0, 1, 1, 0, 0, 0, 0,
        )
        hdc = ctypes.windll.user32.GetDC(hwnd)

        pfd = _PIXELFORMATDESCRIPTOR()
        pfd.nSize = ctypes.sizeof(_PIXELFORMATDESCRIPTOR)
        pfd.nVersion = 1
        pfd.dwFlags = _PFD_SUPPORT_OPENGL | _PFD_DOUBLEBUFFER
        pfd.cColorBits = 32
        pfd.cDepthBits = 24

        fmt = ctypes.windll.gdi32.ChoosePixelFormat(hdc, ctypes.byref(pfd))
        ctypes.windll.gdi32.SetPixelFormat(hdc, fmt, ctypes.byref(pfd))

        hglrc = ctypes.windll.opengl32.wglCreateContext(hdc)
        ctypes.windll.opengl32.wglMakeCurrent(hdc, hglrc)
        return hwnd, hdc, hglrc

    def _destroy_hidden_gl_context(hwnd, hdc, hglrc):
        """Clean up the hidden OpenGL context."""
        ctypes.windll.opengl32.wglMakeCurrent(0, 0)
        ctypes.windll.opengl32.wglDeleteContext(hglrc)
        ctypes.windll.user32.ReleaseDC(hwnd, hdc)
        ctypes.windll.user32.DestroyWindow(hwnd)
else:
    def _create_hidden_gl_context():
        return None, None, None

    def _destroy_hidden_gl_context(hwnd, hdc, hglrc):
        pass


# ═════════════════════════════════════════════════════════════════
#  Main Application
# ═════════════════════════════════════════════════════════════════
class EmotionGUI:
    """Three-panel emotion recognition GUI."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(
            "TouchDesigner Interface \u2014 Adaptive Emotion Visualization"
        )

        # ── Dark colour theme for TTK widgets ───────────────────
        # ctk.set_appearance_mode / set_default_color_theme require
        # Tk 8.6+; we have Tk 8.5 on macOS, so call them only on
        # platforms where Tk 8.6+ is available (Windows / Linux).
        _tk_ver = tuple(int(x) for x in self.root.tk.eval(
            "info patchlevel"
        ).split("."))
        if _tk_ver >= (8, 6):
            ctk.set_appearance_mode("dark")
            ctk.set_default_color_theme("dark-blue")
        else:
            # Tk 8.5 (macOS system Python) — configure ttk style manually
            _s = ttk.Style()
            try:
                _s.theme_use("default")
            except tk.TclError:
                pass
            _s.configure("TCombobox",
                          background=ACCENT, foreground=TEXT_COLOR,
                          fieldbackground=ACCENT,
                          selectbackground=ACCENT,
                          selectforeground=TEXT_COLOR)
            _s.map("TCombobox",
                   fieldbackground=[("readonly", ACCENT)],
                   selectbackground=[("readonly", ACCENT)],
                   foreground=[("readonly", TEXT_COLOR)])
            _s.configure("TProgressbar",
                          troughcolor=ACCENT, background="#4CAF50")

        # ── Platform maximization ────────────────────────────────
        self.root.update_idletasks()
        scr_w = self.root.winfo_screenwidth()
        scr_h = self.root.winfo_screenheight()
        if _platform.system() == "Windows":
            self.root.state("zoomed")
        elif _platform.system() == "Darwin":
            usable_h = scr_h - 45 - 80
            self.root.geometry(f"{scr_w}x{usable_h}+0+45")
            self.root.update_idletasks()
        else:
            self.root.attributes("-zoomed", True)

        # ── Dynamic visualization canvas size ────────────────────
        # Keep the canvas square and small enough for the center panel.
        canvas_dim = min(int(scr_w * 0.48), int(scr_h * 0.72), 760)
        global SPOUT_DISPLAY_SIZE
        SPOUT_DISPLAY_SIZE = (canvas_dim, canvas_dim)

        # ── State variables ──────────────────────────────────────
        self.running = False
        self.cap: Optional[cv2.VideoCapture] = None
        self.model: Optional[torch.nn.Module] = None
        self.class_names: Tuple[str, ...] = EMOTION_CLASSES
        self.transform: Optional[transforms.Compose] = None
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        if self.device.type == "cpu":
            # Limit to 2 threads: 4+ threads pin every core to ~100% and
            # starve Tkinter/OpenCV. 2 leaves headroom for the rest of
            # the app and for TouchDesigner running alongside.
            torch.set_num_threads(2)
            torch.set_num_interop_threads(1)
        self.face_detector: Optional[cv2.CascadeClassifier] = None

        # Grad-CAM
        self.grad_cam_obj: Optional[GradCAM] = None
        # Off by default: the backward pass roughly doubles inference
        # cost on CPU. User can enable it from the control panel.
        self.grad_cam_enabled = False

        # Model cache: display_name -> {model, class_names, transform, grad_cam}
        # Keeps previously-loaded models in RAM so switching is instant.
        self._model_cache: Dict[str, Dict] = {}
        self._model_loading = False
        self._current_model_name: Optional[str] = None

        # UDP socket
        self.udp_sock = socket.socket(
            socket.AF_INET, socket.SOCK_DGRAM
        )
        self.udp_ip = DEFAULT_UDP_IP
        self.udp_port = DEFAULT_UDP_PORT
        self._last_udp_send = 0.0
        self.udp_status = "Idle"

        # Performance metrics
        self.current_fps = 0.0
        self.current_latency_ms = 0.0

        # Current emotion probabilities
        self.current_probs: Dict[str, float] = {
            e: 0.0 for e in EMOTION_CLASSES
        }
        self.dominant_emotion = ""
        self.dominant_confidence = 0.0

        # Sensitivity (confidence threshold)
        self.sensitivity = 0.3

        # Calibration
        self.calibration_mgr = CalibrationManager(EMOTION_CLASSES)

        # Ensemble
        self.ensemble_mode = False
        self.ensemble_mgr: Optional[EnsembleManager] = None
        self._ensemble_frame_interval = 3
        self._cached_ensemble_probs: Optional[Dict[str, float]] = None
        self._ensemble_winner = ""

        # Frame skipping for performance
        self._frame_count = 0
        self._cached_faces = ()
        self._cached_gradcam = None
        self._cached_single_probs: Optional[Dict[str, float]] = None

        # Test image mode
        self.use_test_image = False
        self.test_image_bgr: Optional[np.ndarray] = None
        self.test_image_path: Optional[str] = None

        # Spout receiver (optional)
        self._spout_available = False
        self._spout_connected = False
        self._latest_spout_frame = None
        self._spout_photo = None
        self._spout_placeholder_visible = True
        self._init_spout()

        # ── Build UI ─────────────────────────────────────────────
        print("[GUI] Building title bar...")
        self._build_title_bar()
        print("[GUI] Building panels...")
        self._build_panels()
        self.root.update_idletasks()
        print(f"[GUI] Window geometry after build: {self.root.winfo_width()}x{self.root.winfo_height()}")
        print(f"[GUI] Screen size: {self.root.winfo_screenwidth()}x{self.root.winfo_screenheight()}")
        print("[GUI] UI build complete. Loading default model...")
        self._load_default_model()

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Reset welcome flag so the dialog shows on every fresh launch.
        # "Don't show again" only suppresses it within the current session.
        self._WELCOME_FLAG.unlink(missing_ok=True)

        # Show KVKK/GDPR privacy notice, then welcome dialog on first launch
        self.root.after(600, self._show_privacy_notice)

    # ─────────────────────────────────────────────────────────────
    #  KVKK / GDPR Privacy Notice  (shown every launch)
    # ─────────────────────────────────────────────────────────────
    # Structured content: list of (tag, text) pairs
    _PRIVACY_CONTENT = [
        # ── KVKK ─────────────────────────────────────────────────
        ("kvkk_title",   "KİŞİSEL VERİLERİN KORUNMASI KANUNU (KVKK)"),
        ("kvkk_title",   "KAPSAMINDA AYDINLATMA METNİ"),
        ("spacer",       ""),
        ("kvkk_section", "Veri Sorumlusu"),
        ("body",         "Bu uygulamayı çalıştıran araştırmacı / kurum, 6698 sayılı KVKK "
                         "uyarınca veri sorumlusu sıfatını taşımaktadır."),
        ("spacer",       ""),
        ("kvkk_section", "İşlenen Kişisel Veriler"),
        ("body",         "Webcam aracılığıyla anlık yüz görüntünüz işlenmektedir. "
                         "Biyometrik nitelik taşıyan bu veri KVKK Madde 6 kapsamında "
                         "özel nitelikli kişisel veri sayılmaktadır."),
        ("spacer",       ""),
        ("kvkk_section", "İşleme Amacı"),
        ("bullet",       "Gerçek zamanlı yüz ifadesi / duygu analizi"),
        ("bullet",       "TouchDesigner görselleştirme sisteminin yönlendirilmesi"),
        ("bullet",       "Akademik / araştırma amaçlı model değerlendirmesi"),
        ("spacer",       ""),
        ("kvkk_section", "Hukuki Dayanak"),
        ("body",         "KVKK Madde 5/2-(f): Veri sorumlusunun meşru menfaatleri "
                         "için işlemenin zorunlu olması."),
        ("spacer",       ""),
        ("kvkk_section", "Saklama Süresi"),
        ("body",         "Görüntüler diske kaydedilmez; yalnızca anlık çerçeve işlemi "
                         "için RAM'e alınır ve işlem tamamlandıktan hemen sonra bellekten silinir."),
        ("spacer",       ""),
        ("kvkk_section", "Veri Aktarımı"),
        ("body",         "Kişisel veriler hiçbir üçüncü tarafla paylaşılmaz. "
                         "Tüm işlem yerel cihazınız üzerinde gerçekleşir."),
        ("spacer",       ""),
        ("kvkk_section", "Haklarınız  (KVKK Madde 11)"),
        ("bullet",       "Verilerinizin işlenip işlenmediğini öğrenme"),
        ("bullet",       "İşleme amacını ve amacına uygun kullanımı sorgulama"),
        ("bullet",       "Verilerin aktarıldığı üçüncü tarafları bilme"),
        ("bullet",       "Eksik / yanlış verilerin düzeltilmesini isteme"),
        ("bullet",       "Verilerinizin silinmesini / yok edilmesini talep etme"),
        ("bullet",       "İşlemeye ve otomatik kararlara itiraz etme"),
        # ── GDPR separator ───────────────────────────────────────
        ("separator",    ""),
        # ── GDPR ─────────────────────────────────────────────────
        ("gdpr_title",   "GDPR — GENERAL DATA PROTECTION REGULATION"),
        ("gdpr_title",   "PRIVACY NOTICE (FOR INTERNATIONAL USERS)"),
        ("spacer",       ""),
        ("gdpr_section", "Data Controller"),
        ("body",         "The researcher or institution operating this application "
                         "acts as the data controller under the GDPR."),
        ("spacer",       ""),
        ("gdpr_section", "Data Processed"),
        ("body",         "Your facial image is captured in real time via webcam. "
                         "Facial data constitutes biometric data and is classified as a "
                         "special category of personal data under GDPR Article 9."),
        ("spacer",       ""),
        ("gdpr_section", "Purpose & Legal Basis"),
        ("bullet",       "Real-time facial expression / emotion recognition"),
        ("bullet",       "Driving the TouchDesigner visualisation system"),
        ("bullet",       "Academic research and model evaluation"),
        ("body",         "Legal basis: Article 6(1)(f) GDPR — legitimate interests of "
                         "the controller, balanced against your fundamental rights."),
        ("spacer",       ""),
        ("gdpr_section", "Retention"),
        ("body",         "No images are written to disk. Processing is in-memory only; "
                         "each frame is discarded immediately after analysis."),
        ("spacer",       ""),
        ("gdpr_section", "Data Transfers"),
        ("body",         "No personal data is transferred to third parties or "
                         "transmitted outside your local device."),
        ("spacer",       ""),
        ("gdpr_section", "Your Rights  (GDPR Articles 15-22)"),
        ("bullet",       "Right of access to your personal data"),
        ("bullet",       "Right to rectification of inaccurate data"),
        ("bullet",       "Right to erasure  ('right to be forgotten')"),
        ("bullet",       "Right to restriction of processing"),
        ("bullet",       "Right to object to processing"),
        ("bullet",       "Right not to be subject to solely automated decisions"),
        ("spacer",       ""),
        ("body",         "To exercise your rights or raise a data-protection concern, "
                         "please contact the application operator directly."),
        ("spacer",       ""),
    ]

    def _show_privacy_notice(self) -> None:
        self._build_privacy_notice_window()

    def _on_privacy_reject(self, win: tk.Toplevel) -> None:
        win.destroy()
        self._on_close()

    def _build_privacy_notice_window(self) -> None:
        win = tk.Toplevel(self.root)
        win.title("Aydınlatma Metni / Privacy Notice  —  KVKK & GDPR")
        win.resizable(True, True)
        win.configure(bg="#1A1A1A")
        win.grab_set()
        win.protocol("WM_DELETE_WINDOW", lambda: self._on_privacy_reject(win))

        self.root.update_idletasks()
        px = self.root.winfo_x() + self.root.winfo_width() // 2
        py = self.root.winfo_y() + self.root.winfo_height() // 2
        win.geometry(f"720x640+{px - 360}+{py - 320}")

        BG       = "#1A1A1A"
        HDR_BG   = "#202025"
        TXT_BG   = "#1C1C1E"
        GOLD     = "#E6B800"
        BLUE_SEC = "#5BC8F5"
        GRN_SEC  = "#7EC8A0"
        BODY_FG  = "#C8C8C8"
        BULLET_FG= "#A0A0A0"
        DIM_FG   = "#686868"

        # ── Accent stripe ─────────────────────────────────────────
        tk.Frame(win, bg="#1C3A6A", height=4).pack(fill=tk.X)

        # ── Header panel ──────────────────────────────────────────
        hdr = tk.Frame(win, bg=HDR_BG, pady=14)
        hdr.pack(fill=tk.X)

        left_hdr = tk.Frame(hdr, bg=HDR_BG)
        left_hdr.pack(side=tk.LEFT, padx=22)

        tk.Label(
            left_hdr, text="🔒",
            font=(UI_FONT, 20), bg=HDR_BG, fg=GOLD,
        ).pack(side=tk.LEFT, padx=(0, 12), anchor="center")

        lbl_col = tk.Frame(left_hdr, bg=HDR_BG)
        lbl_col.pack(side=tk.LEFT)
        tk.Label(
            lbl_col, text="Aydınlatma Metni",
            font=(UI_FONT, 15, "bold"), bg=HDR_BG, fg=GOLD, anchor="w",
        ).pack(anchor="w")
        tk.Label(
            lbl_col, text="Privacy Notice",
            font=(UI_FONT, 10), bg=HDR_BG, fg=DIM_FG, anchor="w",
        ).pack(anchor="w")

        badge_frame = tk.Frame(hdr, bg=HDR_BG)
        badge_frame.pack(side=tk.RIGHT, padx=22, anchor="center")
        for label, bg_c, fg_c in [
            ("KVKK", "#1A3060", "#5BC8F5"),
            ("GDPR", "#103828", "#7EC8A0"),
        ]:
            tk.Label(
                badge_frame, text=label,
                font=(UI_FONT, 9, "bold"),
                bg=bg_c, fg=fg_c,
                padx=10, pady=4,
                relief=tk.FLAT,
            ).pack(side=tk.LEFT, padx=4)

        tk.Frame(win, bg="#2A2A2E", height=1).pack(fill=tk.X)

        # ── Scrollable text area ──────────────────────────────────
        txt_outer = tk.Frame(win, bg=TXT_BG)
        txt_outer.pack(fill=tk.BOTH, expand=True)

        vsb = ttk.Scrollbar(txt_outer)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        txt = tk.Text(
            txt_outer,
            yscrollcommand=vsb.set,
            bg=TXT_BG, fg=BODY_FG,
            font=(UI_FONT, 10),
            wrap=tk.WORD,
            relief=tk.FLAT,
            padx=26, pady=18,
            cursor="arrow",
            bd=0,
            highlightthickness=0,
            selectbackground="#3A3A40",
            insertbackground=TXT_BG,
        )
        txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.config(command=txt.yview)

        # Configure tags
        txt.tag_config(
            "kvkk_title",
            font=(UI_FONT, 12, "bold"), foreground=GOLD,
            spacing1=2, spacing3=2,
        )
        txt.tag_config(
            "gdpr_title",
            font=(UI_FONT, 12, "bold"), foreground=GRN_SEC,
            spacing1=2, spacing3=2,
        )
        txt.tag_config(
            "kvkk_section",
            font=(UI_FONT, 10, "bold"), foreground=BLUE_SEC,
            spacing1=10, spacing3=2,
        )
        txt.tag_config(
            "gdpr_section",
            font=(UI_FONT, 10, "bold"), foreground=GRN_SEC,
            spacing1=10, spacing3=2,
        )
        txt.tag_config(
            "body",
            font=(UI_FONT, 10), foreground=BODY_FG,
            spacing3=4,
        )
        txt.tag_config(
            "bullet",
            font=(UI_FONT, 10), foreground=BULLET_FG,
            lmargin1=18, lmargin2=28,
            spacing1=1, spacing3=1,
        )
        txt.tag_config(
            "separator",
            font=(UI_FONT, 7), foreground="#2E2E32",
            spacing1=14, spacing3=14,
        )
        txt.tag_config("spacer", font=(UI_FONT, 4), spacing1=0)

        for tag, text in self._PRIVACY_CONTENT:
            if tag == "separator":
                txt.insert(tk.END, "─" * 72 + "\n", tag)
            elif tag == "bullet":
                txt.insert(tk.END, f"  ›  {text}\n", tag)
            elif tag == "spacer":
                txt.insert(tk.END, "\n", tag)
            else:
                txt.insert(tk.END, text + "\n", tag)

        txt.config(state=tk.DISABLED)

        # ── Bottom bar ────────────────────────────────────────────
        tk.Frame(win, bg="#2A2A2E", height=1).pack(fill=tk.X)

        bottom = tk.Frame(win, bg=BG, pady=12)
        bottom.pack(fill=tk.X, padx=22)

        tk.Label(
            bottom,
            text="Devam etmek için metni okuyup onaylamanız gerekmektedir.",
            font=(UI_FONT, 9), bg=BG, fg=DIM_FG,
            anchor="w",
        ).pack(side=tk.LEFT, anchor="w")

        def _accept() -> None:
            win.destroy()
            self._show_welcome_dialog()

        ctk.CTkButton(
            bottom,
            text="Kapat / Exit",
            font=(UI_FONT, 10),
            fg_color="#3A1A1A", hover_color="#5A2A2A",
            text_color="#CC7777",
            corner_radius=8, height=34, width=130,
            command=lambda: self._on_privacy_reject(win),
        ).pack(side=tk.RIGHT, padx=(8, 0))

        ctk.CTkButton(
            bottom,
            text="Okudum, Anladım  ✓",
            font=(UI_FONT, 11, "bold"),
            fg_color="#163A22", hover_color="#1F5230",
            text_color="#7EC8A0",
            corner_radius=8, height=34, width=190,
            command=_accept,
        ).pack(side=tk.RIGHT)

        win.bind("<Return>", lambda _e: _accept())
        win.bind("<Escape>", lambda _e: self._on_privacy_reject(win))

    # ─────────────────────────────────────────────────────────────
    #  Welcome / Quick-Start Tutorial
    # ─────────────────────────────────────────────────────────────
    _WELCOME_FLAG = Path.home() / ".emotion_gui_welcome_shown"

    def _show_welcome_dialog(self) -> None:
        if self._WELCOME_FLAG.exists():
            return
        self._build_welcome_window()

    def _build_welcome_window(self) -> None:
        win = tk.Toplevel(self.root)
        win.title("Quick Start — How to Use")
        win.resizable(False, False)
        win.configure(bg="#1A1A1A")
        win.grab_set()

        # Centre on parent
        self.root.update_idletasks()
        px = self.root.winfo_x() + self.root.winfo_width() // 2
        py = self.root.winfo_y() + self.root.winfo_height() // 2
        win.geometry(f"520x530+{px - 260}+{py - 265}")

        TITLE_FG = "#E6B800"
        BODY_FG  = "#DDDDDD"
        STEP_FG  = "#AAAAAA"
        BG       = "#1A1A1A"

        tk.Label(
            win, text="Welcome — Adaptive Emotion Visualization",
            font=(UI_FONT, 14, "bold"), bg=BG, fg=TITLE_FG,
            wraplength=480,
        ).pack(pady=(22, 4), padx=20)

        tk.Label(
            win,
            text="This app recognises facial expressions in real time and "
                 "drives the TouchDesigner visualisation. Follow the steps below:",
            font=(UI_FONT, 10), bg=BG, fg=BODY_FG,
            wraplength=480, justify="left",
        ).pack(padx=24, pady=(0, 12), anchor="w")

        steps = [
            ("1", "Select a Model",
             "Choose a model from the dropdown at the top of the left panel.\n"
             "(Default: ResNet-18 — recommended starting point)"),
            ("2", "Press START  →  Webcam opens",
             "Click the green START button at the bottom-left.\n"
             "The camera opens and a live preview appears in the panel above it."),
            ("3", "Face the camera",
             "Sit ~50 cm away, look straight ahead — good lighting gives best results."),
            ("4", "Watch the emotion labels",
             "The right panel shows the detected emotion and confidence score.\n"
             "The centre panel reacts visually if TouchDesigner is connected."),
            ("5", "Test with a saved image  (optional)",
             "Tick 'Use test image', click Load and pick any photo from disk.\n"
             "Press START — the model runs on the image instead of the webcam."),
            ("6", "Press STOP when done",
             "Click the red STOP button to close the camera and end the session."),
        ]

        for num, title, body in steps:
            row = tk.Frame(win, bg=BG)
            row.pack(padx=22, pady=3, fill=tk.X, anchor="w")

            tk.Label(
                row, text=num, width=2,
                font=(UI_FONT, 11, "bold"), bg="#E6B800", fg="#000000",
            ).pack(side=tk.LEFT, anchor="n", padx=(0, 10), pady=2)

            col = tk.Frame(row, bg=BG)
            col.pack(side=tk.LEFT, fill=tk.X, expand=True)

            tk.Label(
                col, text=title,
                font=(UI_FONT, 11, "bold"), bg=BG, fg=BODY_FG,
                anchor="w",
            ).pack(anchor="w")
            tk.Label(
                col, text=body,
                font=(UI_FONT, 9), bg=BG, fg=STEP_FG,
                anchor="w", justify="left", wraplength=390,
            ).pack(anchor="w")

        tk.Frame(win, bg="#333333", height=1).pack(fill=tk.X, padx=20, pady=(12, 0))

        bottom = tk.Frame(win, bg=BG)
        bottom.pack(fill=tk.X, padx=20, pady=10)

        dont_show_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            bottom, text="Don't show again",
            variable=dont_show_var,
            bg=BG, fg=STEP_FG, selectcolor="#333333",
            activebackground=BG, activeforeground=STEP_FG,
            font=(UI_FONT, 10),
        ).pack(side=tk.LEFT)

        def _close():
            if dont_show_var.get():
                self._WELCOME_FLAG.touch()
            win.destroy()

        def _tour():
            _close()
            self.root.after(200, lambda: TutorialOverlay(self))

        ctk.CTkButton(
            bottom, text="Take a Tour →",
            font=(UI_FONT, 11),
            fg_color="#2A4080", hover_color="#3A50A0",
            text_color="white", corner_radius=6, height=32, width=130,
            command=_tour,
        ).pack(side=tk.RIGHT, padx=(6, 0))

        ctk.CTkButton(
            bottom, text="Got it, Let's Start!",
            font=(UI_FONT, 12, "bold"),
            fg_color=BTN_START, hover_color=BTN_START_HOVER,
            text_color="white", corner_radius=6, height=32, width=160,
            command=_close,
        ).pack(side=tk.RIGHT)

        win.bind("<Return>", lambda _e: _close())
        win.bind("<Escape>", lambda _e: _close())

    # ─────────────────────────────────────────────────────────────
    #  Optional: Spout Receiver for TouchDesigner
    # ─────────────────────────────────────────────────────────────
    def _init_spout(self) -> None:
        """Check whether SpoutGL (Windows) or Syphon (macOS) is available."""
        self._spout_protocol = None
        if _platform.system() == "Windows":
            try:
                import SpoutGL                      # noqa: F401
                from OpenGL import GL as _GL        # noqa: F401
                self._spout_available = True
                self._spout_protocol = "spout"
                print("[INFO] SpoutGL available – will receive TD frames.")
            except ImportError:
                self._spout_available = False
                print("[INFO] SpoutGL not found – Spout disabled.")
        elif _platform.system() == "Darwin":
            try:
                import syphon                       # noqa: F401
                self._spout_available = True
                self._spout_protocol = "syphon"
                print("[INFO] Syphon available – will receive TD frames.")
            except ImportError:
                self._spout_available = False
                print("[INFO] syphon-python not found – Syphon disabled.")
                print("[INFO] UDP → TouchDesigner still fully active.")
            except Exception as _e:
                self._spout_available = False
                print(f"[INFO] Syphon import failed ({type(_e).__name__}: {_e}) – Syphon disabled.")
                print("[INFO] UDP → TouchDesigner still fully active.")
        else:
            self._spout_available = False
            print("[INFO] Texture sharing not supported on this platform.")

    def _syphon_receive_loop(self) -> None:
        """Background thread: receive TouchDesigner frames via Syphon (macOS)."""
        from syphon.server_directory import SyphonServerDirectory
        from syphon.client import SyphonMetalClient
        from syphon.utils.numpy import copy_mtl_texture_to_image

        directory = SyphonServerDirectory()
        client = None

        try:
            while self.running:
                # Re-connect if needed
                if client is None or not client.is_valid:
                    # Try the named sender first, fall back to any sender
                    servers = directory.servers_matching_name(name=SYPHON_SENDER_NAME)
                    if not servers:
                        servers = directory.servers
                    if servers:
                        try:
                            client = SyphonMetalClient(servers[0])
                            self._spout_connected = True
                            print(
                                f"[INFO] Syphon connected: "
                                f"{servers[0].name} ({servers[0].app_name})"
                            )
                        except Exception as e:
                            print(f"[WARNING] Syphon connect error: {e}")
                            client = None
                            self._spout_connected = False
                    else:
                        self._spout_connected = False
                    time.sleep(1.0)
                    continue

                if client.has_new_frame:
                    try:
                        texture = client.new_frame_image
                        if texture is not None:
                            bgra = copy_mtl_texture_to_image(texture)
                            # Metal textures are BGRA; convert to RGB
                            # for correct colour display in PIL/tkinter.
                            self._latest_spout_frame = cv2.cvtColor(
                                bgra, cv2.COLOR_BGRA2RGB,
                            )
                    except Exception as e:
                        print(f"[WARNING] Syphon frame error: {e}")

                time.sleep(0.016)
        except Exception as exc:
            print(f"[WARNING] Syphon receive error: {exc}")
        finally:
            if client is not None:
                try:
                    client.stop()
                except Exception:
                    pass
            self._spout_connected = False

    def _spout_receive_loop(self) -> None:
        """Background thread: receive TouchDesigner frames via Spout (Windows)."""
        import SpoutGL
        from OpenGL import GL

        hwnd, hdc, hglrc = _create_hidden_gl_context()
        receiver = SpoutGL.SpoutReceiver()

        # Start with a tiny buffer; resize on first sender connection
        width, height = 16, 16
        buf = (ctypes.c_ubyte * (width * height * 4))()

        try:
            while self.running:
                result = receiver.receiveImage(buf, GL.GL_RGBA, False, 0)

                if receiver.isUpdated():
                    width = receiver.getSenderWidth()
                    height = receiver.getSenderHeight()
                    if width > 0 and height > 0:
                        buf = (ctypes.c_ubyte * (width * height * 4))()
                        self._spout_connected = True
                        print(
                            f"[INFO] Spout sender connected: "
                            f"{width}x{height}"
                        )

                if self._spout_connected and width > 0 and height > 0:
                    frame = np.frombuffer(buf, dtype=np.uint8).reshape(
                        height, width, 4,
                    )
                    # RGBA → RGB, flip Y (OpenGL origin is bottom-left)
                    self._latest_spout_frame = np.flipud(
                        frame[:, :, :3]
                    ).copy()

                time.sleep(0.016)  # ~60 fps
        except Exception as exc:
            print(f"[WARNING] Spout receive error: {exc}")
        finally:
            receiver.releaseReceiver()
            _destroy_hidden_gl_context(hwnd, hdc, hglrc)
            self._spout_connected = False

    # ─────────────────────────────────────────────────────────────
    #  UI Construction
    # ─────────────────────────────────────────────────────────────
    def _build_title_bar(self) -> None:
        print("[GUI]   _build_title_bar: creating title frame")
        title_frame = tk.Frame(self.root, bg="#252526", pady=0)
        title_frame.pack(side=tk.TOP, fill=tk.X)
        tk.Frame(title_frame, bg="#007ACC", height=3).pack(fill=tk.X)

    def _build_panels(self) -> None:
        print("[GUI]   _build_panels: creating container frame")
        container = tk.Frame(self.root, bg=BG_COLOR)
        container.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))
        print("[GUI]   _build_panels: container packed")

        # Use pack layout instead of grid — grid is unreliable on
        # macOS Tk 8.5 (columns collapse to zero width).
        # Pack order: LEFT panel first, RIGHT panel second, CENTER
        # last so it fills the remaining space.
        print("[GUI]   _build_panels: building control panel (LEFT)...")
        self._build_control_panel(container)
        print("[GUI]   _build_panels: building monitoring panel (RIGHT)...")
        self._build_monitoring_panel(container)
        print("[GUI]   _build_panels: building visualization panel (CENTER)...")
        self._build_visualization_panel(container)
        print("[GUI]   _build_panels: all panels built")

    # ── LEFT: Control Panel ──────────────────────────────────────
    def _build_control_panel(self, parent: tk.Frame) -> None:
        print("[GUI]     _build_control_panel: start")
        panel = tk.Frame(
            parent, bg=PANEL_BG,
            highlightbackground=BORDER_COLOR, highlightthickness=1,
        )
        panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 4))
        panel.pack_propagate(False)
        panel.configure(width=260)
        print("[GUI]     _build_control_panel: panel packed (side=LEFT, width=260)")

        self._section_label(panel, "Control Panel")
        print("[GUI]     _build_control_panel: section label done")

        # Model Selection
        self._heading(panel, "Model Selection")
        print("[GUI]     _build_control_panel: Model Selection heading done")
        self.model_var = tk.StringVar(value="ResNet-18")
        model_menu = ttk.Combobox(
            panel, textvariable=self.model_var,
            values=list(MODEL_OPTIONS.keys()) + ["Ensemble"],
            state="readonly", font=(UI_FONT, 11),
        )
        model_menu.pack(padx=15, pady=(0, 8), fill=tk.X)
        model_menu.bind("<<ComboboxSelected>>", self._on_model_change)
        self.model_menu = model_menu
        print("[GUI]     _build_control_panel: model combobox packed")

        # Ensemble settings — scrollable container (hidden by default)
        # Outer frame with a max height; packs only when Ensemble is
        # selected.  Inner canvas + scrollbar give real scrolling.
        self.ensemble_panel = tk.Frame(panel, bg=PANEL_BG)

        _ens_canvas = tk.Canvas(
            self.ensemble_panel, bg=PANEL_BG,
            highlightthickness=0, bd=0, height=180,
        )
        _ens_sb = tk.Scrollbar(
            self.ensemble_panel, orient=tk.VERTICAL,
            command=_ens_canvas.yview, width=8,
        )
        _ens_canvas.configure(yscrollcommand=_ens_sb.set)
        _ens_sb.pack(side=tk.RIGHT, fill=tk.Y)
        _ens_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        _ens_inner = tk.Frame(_ens_canvas, bg=PANEL_BG)
        _ens_win = _ens_canvas.create_window(
            (0, 0), window=_ens_inner, anchor="nw",
        )

        def _ens_cfg(_e=None):
            _ens_canvas.configure(scrollregion=_ens_canvas.bbox("all"))
        def _ens_canvas_cfg(e):
            _ens_canvas.itemconfig(_ens_win, width=e.width)
        _ens_inner.bind("<Configure>", _ens_cfg)
        _ens_canvas.bind("<Configure>", _ens_canvas_cfg)

        # Mouse-wheel scrolling only inside ensemble area
        def _ens_wheel(evt):
            if _platform.system() == "Darwin":
                _ens_canvas.yview_scroll(-evt.delta, "units")
            else:
                _ens_canvas.yview_scroll(
                    -1 if evt.delta > 0 else 1, "units",
                )
        def _ens_enter(_e):
            _ens_canvas.bind_all("<MouseWheel>", _ens_wheel)
        def _ens_leave(_e):
            _ens_canvas.unbind_all("<MouseWheel>")
        _ens_canvas.bind("<Enter>", _ens_enter)
        _ens_canvas.bind("<Leave>", _ens_leave)

        # ── Ensemble widgets (inside scrollable inner frame) ─────
        self.ensemble_hint_label = tk.Label(
            _ens_inner, text="",
            font=(UI_FONT, 9, "italic"),
            bg=PANEL_BG, fg="#E6B800",
            height=1, anchor="w",
        )
        self.ensemble_hint_label.pack(
            padx=15, fill=tk.X, pady=(2, 2),
        )
        self._ensemble_hint_after: Optional[str] = None

        strat_frame = tk.Frame(_ens_inner, bg=PANEL_BG)
        strat_frame.pack(padx=15, fill=tk.X, pady=(0, 6))
        tk.Label(
            strat_frame, text="Strategy:",
            font=(UI_FONT, 10), bg=PANEL_BG, fg=TEXT_COLOR,
        ).pack(side=tk.LEFT)
        self.ensemble_strategy_var = tk.StringVar(
            value="weighted_avg",
        )
        strat_menu = ttk.Combobox(
            strat_frame,
            textvariable=self.ensemble_strategy_var,
            values=["weighted_avg", "max_confidence"],
            state="readonly", font=(UI_FONT, 10), width=16,
        )
        strat_menu.pack(side=tk.LEFT, padx=(6, 0))

        tk.Label(
            _ens_inner, text="Include models:",
            font=(UI_FONT, 10), bg=PANEL_BG, fg=TEXT_COLOR,
        ).pack(padx=15, anchor="w")

        self.ensemble_model_vars: Dict[str, tk.BooleanVar] = {}
        for model_name in MODEL_OPTIONS:
            var = tk.BooleanVar(value=(model_name != "Mini-Xception"))
            chk = tk.Checkbutton(
                _ens_inner, text=model_name,
                variable=var, bg=PANEL_BG,
                fg=TEXT_COLOR, selectcolor=ACCENT,
                activebackground=PANEL_BG,
                activeforeground=TEXT_COLOR,
                font=(UI_FONT, 10),
                command=lambda n=model_name: self._on_ensemble_toggle(n),
            )
            chk.pack(padx=25, anchor="w")
            self.ensemble_model_vars[model_name] = var

        # UDP Settings
        print("[GUI]     _build_control_panel: UDP Settings heading...")
        self._heading(panel, "UDP Settings")
        udp_frame = tk.Frame(panel, bg=PANEL_BG)
        udp_frame.pack(padx=15, fill=tk.X)

        tk.Label(
            udp_frame, text="Port:", bg=PANEL_BG, fg=TEXT_COLOR,
            font=(UI_FONT, 10),
        ).grid(row=0, column=0, sticky="w")
        self.port_var = tk.StringVar(value=str(DEFAULT_UDP_PORT))
        port_entry = tk.Entry(
            udp_frame, textvariable=self.port_var, width=8,
            font=(UI_FONT, 10),
        )
        port_entry.grid(row=0, column=1, padx=(5, 15))

        tk.Label(
            udp_frame, text="Host:", bg=PANEL_BG, fg=TEXT_COLOR,
            font=(UI_FONT, 10),
        ).grid(row=0, column=2, sticky="w")
        self.host_var = tk.StringVar(value=DEFAULT_UDP_IP)
        host_entry = tk.Entry(
            udp_frame, textvariable=self.host_var, width=12,
            font=(UI_FONT, 10),
        )
        host_entry.grid(row=0, column=3)

        # Sensitivity
        print("[GUI]     _build_control_panel: Sensitivity heading...")
        self._heading(panel, "Sensitivity")
        self.sensitivity_var = tk.DoubleVar(value=0.3)
        sens_slider = tk.Scale(
            panel, from_=0.0, to=1.0, resolution=0.05,
            orient=tk.HORIZONTAL, variable=self.sensitivity_var,
            bg=PANEL_BG, fg=TEXT_COLOR, troughcolor=ACCENT,
            highlightthickness=0, font=(UI_FONT, 9),
            command=self._on_sensitivity_change,
        )
        sens_slider.pack(padx=15, fill=tk.X)
        print("[GUI]     _build_control_panel: sensitivity slider packed")

        # Grad-CAM toggle — off by default (CPU-heavy backward pass).
        self.gradcam_var = tk.BooleanVar(value=False)
        gradcam_chk = tk.Checkbutton(
            panel, text="Show Grad-CAM overlay  (CPU intensive)",
            variable=self.gradcam_var, bg=PANEL_BG,
            fg=TEXT_COLOR, selectcolor=ACCENT,
            activebackground=PANEL_BG, activeforeground=TEXT_COLOR,
            font=(UI_FONT, 10),
            command=self._on_gradcam_toggle,
        )
        gradcam_chk.pack(padx=15, anchor="w", pady=(4, 0))
        print("[GUI]     _build_control_panel: grad-cam checkbox packed")

        # Calibration
        print("[GUI]     _build_control_panel: Calibration heading...")
        self._heading(panel, "Calibration")
        self.cal_status_label = tk.Label(
            panel, text="Inactive",
            font=(UI_FONT, 10), fg="#888888", bg=PANEL_BG,
        )
        self.cal_status_label.pack(padx=15, anchor="w")

        cal_btn_frame = tk.Frame(panel, bg=PANEL_BG)
        cal_btn_frame.pack(padx=15, fill=tk.X, pady=(4, 0))

        self.calibrate_btn = ctk.CTkButton(
            cal_btn_frame, text="Calibrate",
            font=(UI_FONT, 11, "bold"),
            fg_color=BTN_PRIMARY, hover_color=BTN_PRIMARY_HOVER,
            text_color="white", corner_radius=6, height=28, width=0,
            command=self._open_calibration_wizard,
        )
        self.calibrate_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 3))

        self.load_cal_btn = ctk.CTkButton(
            cal_btn_frame, text="Load",
            font=(UI_FONT, 11),
            fg_color=BTN_SECONDARY, hover_color=BTN_SECONDARY_HOVER,
            text_color="white", corner_radius=6, height=28, width=0,
            command=self._load_calibration_profile,
        )
        self.load_cal_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 3))

        self.clear_cal_btn = ctk.CTkButton(
            cal_btn_frame, text="Clear",
            font=(UI_FONT, 11),
            fg_color=BTN_SECONDARY, hover_color=BTN_SECONDARY_HOVER,
            text_color="white", corner_radius=6, height=28, width=0,
            command=self._clear_calibration,
        )
        self.clear_cal_btn.pack(side=tk.LEFT, expand=True, fill=tk.X)
        print("[GUI]     _build_control_panel: calibration buttons packed")

        # Webcam Preview
        print("[GUI]     _build_control_panel: Webcam Preview heading...")
        self._heading(panel, "Webcam Preview")
        tk.Label(
            panel,
            text="▶  Press START to open the camera",
            font=(UI_FONT, 9, "italic"),
            bg=PANEL_BG, fg="#888888",
        ).pack(anchor="w", padx=15, pady=(0, 4))
        webcam_frame = tk.Frame(
            panel,
            bg="#1A1A1A",
            width=240,
            height=180,
        )
        webcam_frame.pack(padx=10, pady=(0, 6))
        webcam_frame.pack_propagate(False)
        print("[GUI]     _build_control_panel: webcam frame packed")

        self.webcam_canvas = tk.Label(
            webcam_frame, bg="#1A1A1A",
            text="Camera Off", fg="#666666",
            font=(UI_FONT, 11),
        )
        self.webcam_canvas.pack(fill=tk.BOTH, expand=True)

        # Test Image
        print("[GUI]     _build_control_panel: Test Image heading...")
        self._heading(panel, "Test Image")
        test_frame = tk.Frame(panel, bg=PANEL_BG)
        test_frame.pack(padx=15, pady=(0, 8), fill=tk.X)

        self.use_test_image_var = tk.BooleanVar(value=False)
        test_toggle = tk.Checkbutton(
            test_frame,
            text="Use test image",
            variable=self.use_test_image_var,
            bg=PANEL_BG,
            fg=TEXT_COLOR,
            selectcolor=ACCENT,
            activebackground=PANEL_BG,
            activeforeground=TEXT_COLOR,
            font=(UI_FONT, 10),
            command=self._on_test_image_toggle,
        )
        test_toggle.pack(side=tk.LEFT)

        self.load_image_btn = ctk.CTkButton(
            test_frame, text="Load",
            font=(UI_FONT, 11),
            fg_color=BTN_PRIMARY, hover_color=BTN_PRIMARY_HOVER,
            text_color="white", corner_radius=6,
            width=70, height=28,
            command=self._load_test_image,
        )
        self.load_image_btn.pack(side=tk.RIGHT)

        # START / STOP
        btn_frame = tk.Frame(panel, bg=PANEL_BG)
        btn_frame.pack(padx=15, pady=(5, 4), fill=tk.X)

        self.start_btn = ctk.CTkButton(
            btn_frame, text="START",
            font=(UI_FONT, 14, "bold"),
            fg_color=BTN_START, hover_color=BTN_START_HOVER,
            text_color="white", corner_radius=6, height=32, width=0,
            command=self._start,
        )
        self.start_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))

        self.stop_btn = ctk.CTkButton(
            btn_frame, text="STOP",
            font=(UI_FONT, 14, "bold"),
            fg_color=BTN_STOP, hover_color=BTN_STOP_HOVER,
            text_color="white", corner_radius=6, height=32, width=0,
            command=self._stop, state=tk.DISABLED,
        )
        self.stop_btn.pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(5, 0))

        print("[GUI]     _build_control_panel: START/STOP buttons packed — DONE")

    # ── CENTER: Visualization ────────────────────────────────────
    def _build_visualization_panel(self, parent: tk.Frame) -> None:
        print("[GUI]     _build_visualization_panel: start")
        panel = tk.Frame(
            parent, bg="#0A0A0A",
            highlightbackground=BORDER_COLOR, highlightthickness=1,
        )
        panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        print("[GUI]     _build_visualization_panel: panel packed (fill=BOTH, expand=True)")

        header = tk.Label(
            panel,
            text="Particle Visualization Area",
            font=(UI_FONT, 12, "italic"),
            bg="#0A0A0A", fg="#888888",
        )
        header.pack(pady=(6, 0))
        print("[GUI]     _build_visualization_panel: header label packed")

        viz_frame = tk.Frame(panel, bg="#0A0A0A")
        viz_frame.pack(expand=True)
        print("[GUI]     _build_visualization_panel: viz_frame packed")

        print(f"[GUI]     _build_visualization_panel: canvas size = {SPOUT_DISPLAY_SIZE}")
        self.viz_canvas = tk.Canvas(
            viz_frame,
            width=SPOUT_DISPLAY_SIZE[0],
            height=SPOUT_DISPLAY_SIZE[1],
            bg="#0A0A0A",
            highlightthickness=0,
        )
        self.viz_canvas.pack(padx=4, pady=4)
        self._spout_canvas_image = None
        print("[GUI]     _build_visualization_panel: canvas packed")

        # Dominant emotion overlay (big text on the center panel)
        # Hidden by default; shown only when Spout is NOT connected
        self.viz_emotion_label = tk.Label(
            self.viz_canvas,
            text="",
            font=(UI_FONT, 48, "bold"),
            bg="#0A0A0A", fg="#FFD700",
        )
        self._viz_emotion_visible = False

        # Placeholder text when TD is not connected
        if _platform.system() == "Darwin":
            _placeholder_text = (
                "TouchDesigner'a UDP ile bağlanıldı\n"
                "Texture paylaşımı için Syphon gereklidir\n"
                "(pip install syphon-python)\n\n"
                "UDP verisi gönderiliyor:\n"
                f"{DEFAULT_UDP_IP}:{DEFAULT_UDP_PORT}"
            )
        else:
            _placeholder_text = (
                "Connect TouchDesigner via Spout\n"
                "or run TD alongside this application.\n\n"
                "UDP data is being sent to\n"
                f"{DEFAULT_UDP_IP}:{DEFAULT_UDP_PORT}"
            )
        self.viz_placeholder = tk.Label(
            self.viz_canvas,
            text=_placeholder_text,
            font=(UI_FONT, 11),
            bg="#0A0A0A", fg="#555555",
            justify=tk.CENTER,
        )
        self.viz_placeholder.place(relx=0.5, rely=0.7, anchor="center")
        print("[GUI]     _build_visualization_panel: placeholder label placed")

        footer = tk.Label(
            panel,
            text="Real-time adaptive particle rendering",
            font=(UI_FONT, 9, "italic"),
            bg="#0A0A0A", fg="#555555",
        )
        footer.pack(side=tk.BOTTOM, pady=(0, 6))
        print("[GUI]     _build_visualization_panel: footer packed — DONE")

    # ── RIGHT: Monitoring ────────────────────────────────────────
    def _build_monitoring_panel(self, parent: tk.Frame) -> None:
        print("[GUI]     _build_monitoring_panel: start")
        panel = tk.Frame(
            parent, bg=PANEL_BG,
            highlightbackground=BORDER_COLOR, highlightthickness=1,
        )
        panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(4, 0))
        panel.pack_propagate(False)
        panel.configure(width=260)
        print("[GUI]     _build_monitoring_panel: panel packed (side=RIGHT, width=260)")

        self._section_label(panel, "Monitoring")
        print("[GUI]     _build_monitoring_panel: section label done")

        # Emotion Probabilities
        print("[GUI]     _build_monitoring_panel: Emotion Probabilities heading...")
        self._heading(panel, "Emotion Probabilities")
        self.prob_bars: Dict[str, Dict] = {}
        # Display order: happy, neutral, surprise, sad, angry
        display_order = ["happy", "neutral", "surprise", "sad", "angry"]
        for emotion in display_order:
            row = tk.Frame(panel, bg=PANEL_BG)
            row.pack(padx=15, fill=tk.X, pady=2)

            color = EMOTION_HEX[emotion]
            label = tk.Label(
                row, text=emotion.capitalize(),
                font=(UI_FONT, 10, "bold"),
                fg=color, bg=PANEL_BG, width=8, anchor="e",
            )
            label.pack(side=tk.LEFT)

            bar_container = tk.Frame(
                row, bg=BAR_BG, height=16,
                highlightbackground="#666666", highlightthickness=1,
            )
            bar_container.pack(
                side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 4),
            )
            bar_container.pack_propagate(False)

            bar_fill = tk.Frame(bar_container, bg=color)
            bar_fill.place(x=0, y=0, relheight=1.0, relwidth=0.0)

            pct_label = tk.Label(
                row, text="0%", font=(UI_FONT, 9),
                fg=TEXT_COLOR, bg=PANEL_BG, width=5, anchor="w",
            )
            pct_label.pack(side=tk.LEFT)

            self.prob_bars[emotion] = {
                "container": bar_container,
                "fill": bar_fill,
                "pct": pct_label,
            }
        print("[GUI]     _build_monitoring_panel: emotion probability bars packed")

        # Separator
        self._separator(panel)

        # Dominant Emotion
        self._heading(panel, "Dominant Emotion")
        dominant_frame = tk.Frame(panel, bg=PANEL_BG, height=40)
        dominant_frame.pack(padx=15, fill=tk.X, pady=(0, 4))
        dominant_frame.pack_propagate(False)

        self.dominant_label = tk.Label(
            dominant_frame, text="---",
            font=(UI_FONT, 22, "bold"),
            bg=PANEL_BG, fg="#FFD700",
        )
        self.dominant_label.pack(expand=True)

        self.dominant_tag_label = tk.Label(
            panel, text="",
            font=(UI_FONT, 9),
            bg=PANEL_BG, fg="#AAAAAA",
        )
        self.dominant_tag_label.pack(pady=(0, 4))

        # Separator
        self._separator(panel)

        # Performance
        self._heading(panel, "Performance")
        perf_frame = tk.Frame(panel, bg=PANEL_BG)
        perf_frame.pack(padx=15, fill=tk.X)

        self.fps_label = tk.Label(
            perf_frame, text="FPS: --",
            font=(UI_FONT, 11), fg=TEXT_COLOR, bg=PANEL_BG,
            anchor="w",
        )
        self.fps_label.pack(fill=tk.X)

        self.latency_label = tk.Label(
            perf_frame, text="Latency: -- ms",
            font=(UI_FONT, 11), fg=TEXT_COLOR, bg=PANEL_BG,
            anchor="w",
        )
        self.latency_label.pack(fill=tk.X)

        self.udp_label = tk.Label(
            perf_frame, text="UDP Recv: Idle",
            font=(UI_FONT, 11), fg=TEXT_COLOR, bg=PANEL_BG,
            anchor="w",
        )
        self.udp_label.pack(fill=tk.X)

        self.device_label = tk.Label(
            perf_frame,
            text=f"Device: {self.device}",
            font=(UI_FONT, 10), fg="#888888", bg=PANEL_BG,
            anchor="w",
        )
        self.device_label.pack(fill=tk.X, pady=(8, 0))

        # "? How to use" — bottom-right of monitoring panel
        tk.Button(
            panel, text="?  How to use",
            font=(UI_FONT, 9), bg=PANEL_BG, fg="#666666",
            relief=tk.FLAT, cursor="hand2", bd=0,
            activebackground=PANEL_BG, activeforeground="#AAAAAA",
            command=self._build_welcome_window,
        ).pack(side=tk.BOTTOM, anchor="e", padx=12, pady=(0, 8))
        print("[GUI]     _build_monitoring_panel: performance labels packed — DONE")

    # ── UI helpers ───────────────────────────────────────────────
    def _section_label(self, parent: tk.Frame, text: str) -> None:
        tk.Label(
            parent, text=text,
            font=(UI_FONT, 14, "bold"),
            bg=PANEL_BG, fg=HEADING_COLOR,
        ).pack(pady=(10, 4))

    def _heading(self, parent: tk.Frame, text: str) -> None:
        frame = tk.Frame(parent, bg=PANEL_BG)
        frame.pack(fill=tk.X, padx=15, pady=(8, 2))
        tk.Frame(frame, bg=BORDER_COLOR, height=1).pack(
            side=tk.LEFT, fill=tk.X, expand=True, pady=8,
        )
        tk.Label(
            frame, text=f"  {text}  ",
            font=(UI_FONT, 10), bg=PANEL_BG, fg="#AAAAAA",
        ).pack(side=tk.LEFT)
        tk.Frame(frame, bg=BORDER_COLOR, height=1).pack(
            side=tk.LEFT, fill=tk.X, expand=True, pady=8,
        )

    def _separator(self, parent: tk.Frame) -> None:
        tk.Frame(
            parent, bg=BORDER_COLOR, height=1,
        ).pack(fill=tk.X, padx=15, pady=8)

    # ─────────────────────────────────────────────────────────────
    #  Model Loading
    # ─────────────────────────────────────────────────────────────
    def _load_default_model(self) -> None:
        self._load_model("ResNet-18")

    def _load_model(self, display_name: str) -> None:
        cached = self._model_cache.get(display_name)
        if cached is not None:
            self._apply_model_bundle(display_name, cached)
            print(f"[INFO] {display_name} loaded from cache.")
            return

        cfg = MODEL_OPTIONS[display_name]
        checkpoint = cfg["checkpoint"]

        if not checkpoint.exists():
            # Try best_models_output as fallback
            alt = BEST_DIR / f"best_{cfg['name']}.pth"
            if alt.exists():
                checkpoint = alt
            else:
                print(f"[WARNING] Checkpoint not found: {checkpoint}")
                return

        print(f"[INFO] Loading {display_name} from {checkpoint} ...")
        model, class_names = load_model_from_checkpoint(
            cfg["name"], checkpoint, self.device,
            class_names=EMOTION_CLASSES,
        )

        transform = build_transforms(
            cfg["input_size"], grayscale=cfg["grayscale"],
        )["eval"]

        grad_cam = None
        try:
            target_layer = get_target_layer(model, cfg["name"])
            grad_cam = GradCAM(model, target_layer)
        except Exception as e:
            print(f"[WARNING] Grad-CAM init failed: {e}")

        bundle = {
            "model": model,
            "class_names": class_names,
            "transform": transform,
            "grad_cam": grad_cam,
        }
        self._model_cache[display_name] = bundle
        self._apply_model_bundle(display_name, bundle)
        print(f"[INFO] {display_name} loaded successfully.")

    def _apply_model_bundle(
        self, display_name: str, bundle: Dict,
    ) -> None:
        """Swap the active model/transform/grad-cam to a cached bundle.

        Cached Grad-CAM objects stay alive between switches — we only
        rebind self.grad_cam_obj rather than releasing hooks.
        """
        self.model = bundle["model"]
        self.class_names = bundle["class_names"]
        self.transform = bundle["transform"]
        self.grad_cam_obj = bundle.get("grad_cam")
        self._current_model_name = display_name

    # ─────────────────────────────────────────────────────────────
    #  Event Handlers
    # ─────────────────────────────────────────────────────────────
    def _on_model_change(self, event=None) -> None:
        if self._model_loading:
            # Revert dropdown to the currently-loaded model to avoid UI drift
            if self._current_model_name is not None:
                self.model_var.set(self._current_model_name)
            return

        name = self.model_var.get()
        was_running = self.running
        if was_running:
            self._stop()

        # Ensemble path
        if name == "Ensemble":
            self.ensemble_mode = True
            self.ensemble_panel.pack(
                padx=15, fill=tk.X, pady=(0, 8),
            )
            # Ensemble init can also touch the disk — run async.
            self._begin_async_load(
                self._init_ensemble,
                lambda: self._after_model_change(was_running),
            )
            return

        self.ensemble_mode = False
        self.ensemble_panel.pack_forget()

        # Cached models swap instantly on the UI thread.
        if name in self._model_cache:
            self._load_model(name)
            self._after_model_change(was_running)
            return

        # Cold load: do disk I/O + model construction off the UI thread.
        self._begin_async_load(
            lambda: self._load_model(name),
            lambda: self._after_model_change(was_running),
        )

    def _begin_async_load(self, target, on_done) -> None:
        """Run `target` in a worker thread; invoke `on_done` on UI thread."""
        self._set_loading(True)

        def worker() -> None:
            try:
                target()
            except Exception as exc:
                print(f"[ERROR] Model load failed: {exc}")
            finally:
                self.root.after(0, on_done)

        threading.Thread(target=worker, daemon=True).start()

    def _after_model_change(self, was_running: bool) -> None:
        """UI-thread cleanup after a model / ensemble swap completes."""
        self.calibration_mgr.set_active_profile(None)
        self._update_calibration_status()
        self._cached_ensemble_probs = None
        self._set_loading(False)
        if was_running:
            self._start()

    def _set_loading(self, loading: bool) -> None:
        self._model_loading = loading
        if hasattr(self, "model_menu"):
            self.model_menu.config(
                state="disabled" if loading else "readonly",
            )
        if hasattr(self, "device_label"):
            suffix = "  (loading\u2026)" if loading else ""
            self.device_label.config(
                text=f"Device: {self.device}{suffix}",
            )

    def _on_sensitivity_change(self, value=None) -> None:
        self.sensitivity = self.sensitivity_var.get()

    def _on_gradcam_toggle(self) -> None:
        self.grad_cam_enabled = self.gradcam_var.get()

    # ── Ensemble handlers ───────────────────────────────────────
    def _init_ensemble(self) -> None:
        """Load all available models for ensemble inference."""
        if self.ensemble_mgr is not None:
            self.ensemble_mgr.release()
        self.ensemble_mgr = EnsembleManager(
            self._device_for_ensemble(), EMOTION_CLASSES,
        )
        loaded = self.ensemble_mgr.load_models(MODEL_OPTIONS)
        if loaded:
            self._apply_ensemble_checkboxes()
            # Use first loaded model as fallback for single-model state
            first = loaded[0]
            mt = self.ensemble_mgr.get_model_and_transform(first)
            if mt is not None:
                self.model, self.transform = mt[0], mt[1]
                self.class_names = EMOTION_CLASSES
                self._init_grad_cam_from_ensemble(first)
            print(
                f"[INFO] Ensemble ready: {len(loaded)} models loaded."
            )
        else:
            print("[WARNING] No models loaded for ensemble.")

    def _device_for_ensemble(self) -> torch.device:
        """Return the device to use for ensemble models."""
        return self.device

    def _apply_ensemble_checkboxes(self) -> None:
        """Sync the checkbox states to EnsembleManager."""
        if self.ensemble_mgr is None:
            return
        active = [
            name for name, var in self.ensemble_model_vars.items()
            if var.get()
        ]
        self.ensemble_mgr.set_active_models(active)

    def _on_ensemble_models_change(self) -> None:
        self._apply_ensemble_checkboxes()
        self._cached_ensemble_probs = None

    def _on_ensemble_toggle(self, model_name: str) -> None:
        """Checkbox handler that enforces a minimum of 2 active models."""
        active = [
            n for n, v in self.ensemble_model_vars.items() if v.get()
        ]
        if len(active) < 2:
            # Revert the just-unchecked box; ensemble needs >= 2 models.
            self.ensemble_model_vars[model_name].set(True)
            self._flash_ensemble_hint(
                "En az 2 model seçili olmalı.",
            )
            return
        self._on_ensemble_models_change()

    def _flash_ensemble_hint(self, text: str, duration_ms: int = 2500) -> None:
        if not hasattr(self, "ensemble_hint_label"):
            return
        self.ensemble_hint_label.config(text=text)
        if self._ensemble_hint_after is not None:
            try:
                self.root.after_cancel(self._ensemble_hint_after)
            except Exception:
                pass
        self._ensemble_hint_after = self.root.after(
            duration_ms,
            lambda: self.ensemble_hint_label.config(text=""),
        )

    def _init_grad_cam_from_ensemble(
        self, display_name: str,
    ) -> None:
        """Set Grad-CAM to point at a specific ensemble model."""
        if self.grad_cam_obj is not None:
            self.grad_cam_obj.release()
            self.grad_cam_obj = None
        if self.ensemble_mgr is not None:
            self.grad_cam_obj = (
                self.ensemble_mgr.get_grad_cam_for_model(display_name)
            )

    # ── Calibration handlers ────────────────────────────────────
    def _update_calibration_status(self) -> None:
        """Update the calibration status label in the control panel."""
        profile = self.calibration_mgr.get_active_profile()
        if profile is not None:
            self.cal_status_label.configure(
                text=f"Active: {profile.user_name}",
                fg="#4CAF50",
            )
        else:
            self.cal_status_label.configure(
                text="Inactive", fg="#888888",
            )

    def _open_calibration_wizard(self) -> None:
        if not self.running:
            self._start()
            if not self.running:
                messagebox.showerror(
                    "Camera Error",
                    "Could not start the camera. Please check your camera and try again.",
                )
                return
        if self.model is None:
            messagebox.showwarning(
                "Model Required",
                "Please select a model before calibrating.",
            )
            return
        CalibrationWizard(self)

    def _load_calibration_profile(self) -> None:
        path = filedialog.askopenfilename(
            title="Load calibration profile",
            initialdir=str(
                self.calibration_mgr._profiles_dir
            ),
            filetypes=[
                ("JSON files", "*.json"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            profile = self.calibration_mgr.load_profile(Path(path))
            current_model = self.model_var.get()
            if profile.model_name != current_model:
                print(
                    f"[WARNING] Profile model ({profile.model_name}) "
                    f"does not match current model ({current_model}). "
                    f"Loading anyway."
                )
            self.calibration_mgr.set_active_profile(profile)
            self._update_calibration_status()
            print(f"[INFO] Calibration profile loaded: {path}")
        except Exception as e:
            print(f"[ERROR] Failed to load profile: {e}")

    def _clear_calibration(self) -> None:
        self.calibration_mgr.set_active_profile(None)
        self._update_calibration_status()
        print("[INFO] Calibration cleared.")

    def _on_test_image_toggle(self) -> None:
        self.use_test_image = self.use_test_image_var.get()
        if self.use_test_image and self.test_image_bgr is None:
            self.use_test_image = False
            self.use_test_image_var.set(False)
            print("[WARNING] No test image loaded.")
            return
        if self.use_test_image and not self.running:
            self._process_test_image()

    def _load_test_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Select test image",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("JPEG files", "*.jpeg"),
                ("BMP files", "*.bmp"),
                ("All files", "*"),
            ],
        )
        if not path:
            return

        try:
            data = np.fromfile(path, dtype=np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception:
            image = None
        if image is None:
            print(f"[WARNING] Failed to load image: {path}")
            return

        self.test_image_bgr = image
        self.test_image_path = path
        self.use_test_image = True
        self.use_test_image_var.set(True)
        self._process_test_image()

    # ─────────────────────────────────────────────────────────────
    #  Test Image Processing (face detection + Grad-CAM)
    # ─────────────────────────────────────────────────────────────
    def _process_test_image(self) -> None:
        """Run face detection, inference, and Grad-CAM on the test image
        and update the preview and monitoring panels immediately."""
        if self.test_image_bgr is None or self.model is None:
            # Fallback: show raw image if model isn't ready
            self._latest_frame = self._cv2_to_tk(
                self.test_image_bgr, WEBCAM_PREVIEW_SIZE[0],
                WEBCAM_PREVIEW_SIZE[1],
            )
            self.webcam_canvas.configure(image=self._latest_frame, text="")
            self.webcam_canvas.image = self._latest_frame
            return

        # Ensure face detector is ready
        if self.face_detector is None:
            cascade_path = (
                cv2.data.haarcascades
                + "haarcascade_frontalface_default.xml"
            )
            self.face_detector = cv2.CascadeClassifier(cascade_path)

        frame = self.test_image_bgr.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=SCALE_FACTOR,
            minNeighbors=MIN_NEIGHBORS,
            minSize=MIN_FACE_SIZE,
        )

        display_frame = frame.copy()
        probs_dict: Dict[str, float] = {e: 0.0 for e in EMOTION_CLASSES}
        best_label = ""
        best_conf = 0.0

        if len(faces) > 0:
            # Use the largest face
            areas = [w * h for (_, _, w, h) in faces]
            idx = int(np.argmax(areas))
            x, y, w, h = faces[idx]

            # Crop with padding
            fh, fw = frame.shape[:2]
            pad_w = int(w * FACE_PADDING_RATIO)
            pad_h = int(h * FACE_PADDING_RATIO)
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(fw, x + w + pad_w)
            y2 = min(fh, y + h + pad_h)
            face_crop = frame[y1:y2, x1:x2]

            if face_crop.shape[0] >= 10 and face_crop.shape[1] >= 10:
                # Preprocess
                face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                face_pil = Image.fromarray(face_rgb)

                # Inference — ensemble or single model
                if (
                    self.ensemble_mode
                    and self.ensemble_mgr is not None
                ):
                    strategy = self.ensemble_strategy_var.get()
                    (
                        probs_dict, winner, _,
                    ) = self.ensemble_mgr.predict(
                        face_pil, strategy,
                    )
                    self._ensemble_winner = winner
                else:
                    tensor = self.transform(
                        face_pil,
                    ).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        output = self.model(tensor)
                        prob_tensor = torch.softmax(
                            output, dim=1,
                        )[0]
                    for i, cls in enumerate(self.class_names):
                        probs_dict[cls] = float(prob_tensor[i])

                # Apply calibration correction if active
                if self.calibration_mgr.has_active_profile():
                    probs_dict = self.calibration_mgr.apply_correction(
                        probs_dict,
                    )

                best_label = max(probs_dict, key=probs_dict.get)
                best_conf = probs_dict[best_label]
                best_idx = self.class_names.index(best_label)

                # Grad-CAM overlay on the face region
                gc_obj = self.grad_cam_obj
                if (
                    self.ensemble_mode
                    and self.ensemble_mgr is not None
                ):
                    mt = self.ensemble_mgr.get_model_and_transform(
                        self._ensemble_winner,
                    )
                    if mt is not None:
                        tensor = mt[1](face_pil).unsqueeze(0).to(
                            self.device,
                        )
                    gc_obj = (
                        self.ensemble_mgr.get_grad_cam_for_model(
                            self._ensemble_winner,
                        )
                    )
                elif 'tensor' not in locals():
                    tensor = self.transform(
                        face_pil,
                    ).unsqueeze(0).to(self.device)

                if (
                    self.grad_cam_enabled
                    and gc_obj is not None
                    and best_conf >= self.sensitivity
                ):
                    try:
                        tensor_gc = tensor.clone().requires_grad_(True)
                        heatmap = gc_obj.generate(
                            tensor_gc, target_class=best_idx,
                        )
                        emotion_color = EMOTION_BGR.get(
                            best_label, (0, 255, 0)
                        )
                        grad_cam_overlay = overlay_heatmap(
                            face_crop, heatmap, alpha=0.45,
                            emotion_color=emotion_color,
                        )
                        # Small Grad-CAM thumbnail next to face
                        gc_h_target = min(h // 2, 80)
                        gc_w_target = int(
                            gc_h_target
                            * grad_cam_overlay.shape[1]
                            / grad_cam_overlay.shape[0]
                        )
                        gc_resized = cv2.resize(
                            grad_cam_overlay,
                            (gc_w_target, gc_h_target),
                        )
                        gy1 = min(y + h - gc_h_target, fh - gc_h_target)
                        gx1 = min(x + w + 4, fw - gc_w_target)
                        gy1 = max(0, gy1)
                        gx1 = max(0, gx1)
                        gy2 = gy1 + gc_h_target
                        gx2 = gx1 + gc_w_target
                        if gy2 <= fh and gx2 <= fw:
                            display_frame[gy1:gy2, gx1:gx2] = gc_resized
                            cv2.putText(
                                display_frame, "Grad-CAM",
                                (gx1, gy1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                                (200, 200, 200), 1,
                            )
                    except Exception:
                        pass

                # Draw stylized face overlay
                self._draw_face_overlay(
                    display_frame, x, y, w, h,
                    best_label, best_conf,
                    *WEBCAM_PREVIEW_SIZE,
                )

        # Update preview
        self._latest_frame = self._cv2_to_tk(
            display_frame, WEBCAM_PREVIEW_SIZE[0], WEBCAM_PREVIEW_SIZE[1],
        )
        self.webcam_canvas.configure(image=self._latest_frame, text="")
        self.webcam_canvas.image = self._latest_frame

        # Update monitoring panel
        self.current_probs = probs_dict
        self.dominant_emotion = best_label if best_conf >= self.sensitivity else ""
        self.dominant_confidence = best_conf if best_conf >= self.sensitivity else 0.0

        for emotion, widgets in self.prob_bars.items():
            prob = probs_dict.get(emotion, 0.0)
            pct = prob * 100
            widgets["pct"].configure(text=f"{pct:.0f}%")
            widgets["fill"].place_configure(relwidth=prob)

        if self.dominant_emotion:
            color_hex = EMOTION_HEX.get(self.dominant_emotion, "#FFFFFF")
            self.dominant_label.configure(
                text=f"{self.dominant_emotion.upper()} ({self.dominant_confidence:.1%})",
                fg=color_hex,
            )
        else:
            self.dominant_label.configure(text="---", fg="#888888")

    # ─────────────────────────────────────────────────────────────
    #  Start / Stop
    # ─────────────────────────────────────────────────────────────
    def _send_udp_reset(self) -> None:
        """Send an all-zero emotion payload so TouchDesigner CHOPs
        don't hold stale values from a previous session."""
        try:
            zero_dict = {cls: 0.0 for cls in EMOTION_CLASSES}
            payload = json.dumps(zero_dict) + "\n"
            self.udp_sock.sendto(
                payload.encode("utf-8"),
                (self.udp_ip, self.udp_port),
            )
        except Exception as exc:
            print(f"[WARNING] UDP reset send failed: {exc}")

    def _start(self) -> None:
        if self.running:
            return
        if self.model is None or self.transform is None:
            messagebox.showerror(
                "Model Not Loaded",
                "No model is loaded. Please select a valid model checkpoint before starting.",
            )
            return

        # Read UDP settings
        try:
            self.udp_port = int(self.port_var.get())
        except ValueError:
            self.udp_port = DEFAULT_UDP_PORT
        self.udp_ip = self.host_var.get() or DEFAULT_UDP_IP

        # Reset TD values before streaming starts.
        self._send_udp_reset()

        # Init face detector
        cascade_path = (
            cv2.data.haarcascades
            + "haarcascade_frontalface_default.xml"
        )
        self.face_detector = cv2.CascadeClassifier(cascade_path)

        self.use_test_image = self.use_test_image_var.get()
        if self.use_test_image and self.test_image_bgr is None:
            self.use_test_image = False
            self.use_test_image_var.set(False)

        # Open webcam (only if not using test image)
        if not self.use_test_image:
            self.cap = self._open_camera()
            if self.cap is None:
                self.face_detector = None
                if _platform.system() == "Darwin":
                    _cam_hint = (
                        "Webcam açılamadı.\n\n"
                        "Kontrol edin:\n"
                        "  • Kamera başka bir uygulama (FaceTime, Zoom, "
                        "tarayıcı) tarafından kullanılıyor olabilir\n"
                        "  • Sistem Ayarları → Gizlilik ve Güvenlik → "
                        "Kamera → Terminal (veya bu uygulamayı) etkinleştirin\n"
                        "  • Uygulamayı kapatıp tekrar açın"
                    )
                else:
                    _cam_hint = (
                        "Webcam açılamadı.\n\n"
                        "Kontrol edin:\n"
                        "  • Kamera başka bir uygulama (Zoom, Teams, "
                        "tarayıcı) tarafından kullanılıyor olabilir\n"
                        "  • Windows Ayarlar → Gizlilik → Kamera "
                        "izninin açık olduğundan emin olun\n"
                        "  • Kamerayı çıkarıp tekrar takmayı deneyin"
                    )
                messagebox.showerror("Camera Error", _cam_hint)
                return
        else:
            self.cap = None

        self.running = True
        self.start_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        self.udp_status = "Sending"

        # Start capture thread
        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True,
        )
        self._capture_thread.start()

        # Start Spout/Syphon receiver thread
        if self._spout_available:
            _loop = (
                self._syphon_receive_loop
                if self._spout_protocol == "syphon"
                else self._spout_receive_loop
            )
            self._spout_thread = threading.Thread(
                target=_loop, daemon=True,
            )
            self._spout_thread.start()

        # Start GUI refresh
        self._refresh_gui()

    def _open_camera(self) -> Optional[cv2.VideoCapture]:
        # Windows MSMF (default) often hangs 5-10s opening the camera
        # on Win11 and silently fails on some webcams. DirectShow opens
        # immediately and is the reliable choice on Windows. On other
        # platforms we let OpenCV pick the backend.
        if sys.platform == "win32":
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        else:
            backends = [cv2.CAP_ANY]

        cap = None
        for backend in backends:
            cap = cv2.VideoCapture(0, backend)
            if cap.isOpened():
                # Some backends report opened but fail on first read.
                ret, _ = cap.read()
                if ret:
                    break
            cap.release()
            cap = None

        if cap is None:
            return None

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_PREVIEW_SIZE[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_PREVIEW_SIZE[1])
        cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap

    def _stop(self) -> None:
        self.running = False

        # Wait for background threads to exit before touching shared
        # state (self.model, self.grad_cam_obj, …). Otherwise a model
        # switch while inference is in flight races with _load_model
        # and crashes the process silently in the PyTorch C++ layer.
        cap_thread = getattr(self, "_capture_thread", None)
        if cap_thread is not None and cap_thread.is_alive():
            cap_thread.join(timeout=2.0)
        spout_thread = getattr(self, "_spout_thread", None)
        if spout_thread is not None and spout_thread.is_alive():
            spout_thread.join(timeout=2.0)

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.start_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        self.udp_status = "Idle"
        self.webcam_canvas.configure(image="", text="Camera Off")
        self._latest_spout_frame = None
        self._spout_connected = False

        # Reset monitoring panel
        self.current_probs = {e: 0.0 for e in EMOTION_CLASSES}
        self.dominant_emotion = ""
        self.dominant_confidence = 0.0
        self._cached_ensemble_probs = None

        # Zero-out TD CHOPs so they don't linger on the last emotion.
        self._send_udp_reset()
        for emotion, widgets in self.prob_bars.items():
            widgets["pct"].configure(text="0%")
            widgets["fill"].place_configure(relwidth=0.0)
        self.dominant_label.configure(text="---", fg="#888888")
        self.dominant_tag_label.configure(text="")
        self.viz_emotion_label.configure(text="")
        self.fps_label.configure(text="FPS: --")
        self.latency_label.configure(text="Latency: -- ms")
        self.udp_label.configure(text="UDP Send: Idle")

    # ─────────────────────────────────────────────────────────────
    #  Capture & Inference Loop (background thread)
    # ─────────────────────────────────────────────────────────────
    def _capture_loop(self) -> None:
        """Background thread: capture frames, detect, infer."""
        prev_time = time.time()
        self._frame_count = 0
        self._cached_faces = ()
        self._cached_gradcam = None
        self._cached_single_probs = None
        consecutive_read_failures = 0

        while self.running:
            if self.use_test_image and self.test_image_bgr is not None:
                frame = self.test_image_bgr.copy()
            else:
                if self.cap is None:
                    break
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    # Webcams occasionally drop a frame — only give up
                    # after several consecutive failures (~1s at 12 FPS)
                    # so a single hiccup doesn't kill the camera feed.
                    consecutive_read_failures += 1
                    if consecutive_read_failures >= 15:
                        print("[ERROR] Webcam read failed repeatedly.")
                        break
                    time.sleep(FRAME_INTERVAL)
                    continue
                consecutive_read_failures = 0
                frame = cv2.flip(frame, 1)
            t0 = time.time()
            self._frame_count += 1

            # Face detection: every 2nd frame, reuse cached otherwise
            if self._frame_count % 2 == 1 or len(self._cached_faces) == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_detector.detectMultiScale(
                    gray,
                    scaleFactor=SCALE_FACTOR,
                    minNeighbors=MIN_NEIGHBORS,
                    minSize=MIN_FACE_SIZE,
                )
                self._cached_faces = faces
            else:
                faces = self._cached_faces

            display_frame = frame.copy()
            probs_dict: Dict[str, float] = {
                e: 0.0 for e in EMOTION_CLASSES
            }
            best_label = ""
            best_conf = 0.0
            grad_cam_overlay = None

            if len(faces) > 0:
                # Use the largest face
                areas = [w * h for (_, _, w, h) in faces]
                idx = int(np.argmax(areas))
                x, y, w, h = faces[idx]

                # Crop with padding
                fh, fw = frame.shape[:2]
                pad_w = int(w * FACE_PADDING_RATIO)
                pad_h = int(h * FACE_PADDING_RATIO)
                x1 = max(0, x - pad_w)
                y1 = max(0, y - pad_h)
                x2 = min(fw, x + w + pad_w)
                y2 = min(fh, y + h + pad_h)
                face_crop = frame[y1:y2, x1:x2]

                if face_crop.shape[0] >= 10 and face_crop.shape[1] >= 10:
                    # Preprocess
                    face_rgb = cv2.cvtColor(
                        face_crop, cv2.COLOR_BGR2RGB
                    )
                    face_pil = Image.fromarray(face_rgb)

                    # Inference — ensemble or single model
                    if (
                        self.ensemble_mode
                        and self.ensemble_mgr is not None
                    ):
                        run_ensemble = (
                            self._frame_count
                            % self._ensemble_frame_interval == 1
                            or self._cached_ensemble_probs is None
                        )
                        if run_ensemble:
                            strategy = self.ensemble_strategy_var.get()
                            (
                                probs_dict,
                                winner,
                                winner_lm,
                            ) = self.ensemble_mgr.predict(
                                face_pil, strategy,
                            )
                            self._cached_ensemble_probs = probs_dict
                            self._ensemble_winner = winner
                        else:
                            probs_dict = dict(
                                self._cached_ensemble_probs,
                            )
                    else:
                        if self.transform is None:
                            continue
                        # Tensor is always built (also needed for Grad-CAM),
                        # but the forward pass only runs every 2nd frame —
                        # cached probs fill the gap so FPS stays smooth.
                        tensor = self.transform(
                            face_pil,
                        ).unsqueeze(0).to(self.device)
                        run_single = (
                            self._frame_count % SINGLE_INFER_EVERY == 1
                            or self._cached_single_probs is None
                        )
                        if run_single:
                            with torch.no_grad():
                                output = self.model(tensor)
                                prob_tensor = torch.softmax(
                                    output, dim=1,
                                )[0]
                            for i, cls in enumerate(self.class_names):
                                probs_dict[cls] = float(prob_tensor[i])
                            self._cached_single_probs = dict(probs_dict)
                        else:
                            probs_dict = dict(self._cached_single_probs)

                    # Apply calibration correction if active
                    if self.calibration_mgr.has_active_profile():
                        probs_dict = self.calibration_mgr.apply_correction(
                            probs_dict,
                        )

                    best_label = max(probs_dict, key=probs_dict.get)
                    best_conf = probs_dict[best_label]
                    best_idx = self.class_names.index(best_label)

                    # Grad-CAM: every 3rd frame, reuse cached otherwise
                    # In ensemble mode, get tensor from winner model
                    gc_obj = self.grad_cam_obj
                    if (
                        self.ensemble_mode
                        and self.ensemble_mgr is not None
                    ):
                        mt = self.ensemble_mgr.get_model_and_transform(
                            self._ensemble_winner,
                        )
                        if mt is not None:
                            tensor = mt[1](face_pil).unsqueeze(0).to(
                                self.device,
                            )
                        gc_obj = (
                            self.ensemble_mgr.get_grad_cam_for_model(
                                self._ensemble_winner,
                            )
                        )
                    else:
                        # Single-model mode: tensor already computed above
                        pass

                    if (
                        self.grad_cam_enabled
                        and gc_obj is not None
                        and best_conf >= self.sensitivity
                    ):
                        if self._frame_count % 3 == 1 or self._cached_gradcam is None:
                            try:
                                tensor_gc = tensor.clone().requires_grad_(True)
                                heatmap = gc_obj.generate(
                                    tensor_gc, target_class=best_idx,
                                )
                                emotion_color = EMOTION_BGR.get(
                                    best_label, (0, 255, 0)
                                )
                                grad_cam_overlay = overlay_heatmap(
                                    face_crop, heatmap, alpha=0.45,
                                    emotion_color=emotion_color,
                                )
                                self._cached_gradcam = grad_cam_overlay
                            except Exception:
                                grad_cam_overlay = self._cached_gradcam
                        else:
                            grad_cam_overlay = self._cached_gradcam

                    # Draw stylized face overlay
                    self._draw_face_overlay(
                        display_frame, x, y, w, h,
                        best_label, best_conf,
                        *WEBCAM_PREVIEW_SIZE,
                    )

                    # Grad-CAM overlay — small thumbnail next to face
                    if grad_cam_overlay is not None:
                        gc_h_target = min(h // 2, 80)
                        gc_w_target = int(
                            gc_h_target
                            * grad_cam_overlay.shape[1]
                            / grad_cam_overlay.shape[0]
                        )
                        gc_resized = cv2.resize(
                            grad_cam_overlay,
                            (gc_w_target, gc_h_target),
                        )
                        gy1 = min(
                            y + h - gc_h_target, fh - gc_h_target
                        )
                        gx1 = min(x + w + 4, fw - gc_w_target)
                        gy1 = max(0, gy1)
                        gx1 = max(0, gx1)
                        gy2 = gy1 + gc_h_target
                        gx2 = gx1 + gc_w_target
                        if gy2 <= fh and gx2 <= fw:
                            display_frame[
                                gy1:gy2, gx1:gx2
                            ] = gc_resized
                            cv2.putText(
                                display_frame, "Grad-CAM",
                                (gx1, gy1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                                (200, 200, 200), 1,
                            )

                    # Send to TouchDesigner via UDP
                    now = time.time()
                    if now - self._last_udp_send >= SEND_INTERVAL:
                        emotion_dict = {
                            cls: round(probs_dict[cls], 4)
                            for cls in self.class_names
                        }
                        payload = json.dumps(emotion_dict) + "\n"
                        try:
                            self.udp_sock.sendto(
                                payload.encode("utf-8"),
                                (self.udp_ip, self.udp_port),
                            )
                            self.udp_status = "OK"
                        except Exception:
                            self.udp_status = "Error"
                        self._last_udp_send = now

            else:
                now = time.time()
                if now - self._last_udp_send >= SEND_INTERVAL:
                    self._send_udp_reset()
                    self._last_udp_send = now

            # FPS / latency
            t1 = time.time()
            self.current_latency_ms = (t1 - t0) * 1000
            self.current_fps = 1.0 / max(t1 - prev_time, 1e-6)
            prev_time = t1

            # Store data for GUI thread
            self.current_probs = probs_dict
            self.dominant_emotion = best_label if best_conf >= self.sensitivity else ""
            self.dominant_confidence = best_conf if best_conf >= self.sensitivity else 0.0

            # Convert frame for tkinter
            preview_w, preview_h = WEBCAM_PREVIEW_SIZE
            self._latest_frame = self._cv2_to_tk(
                display_frame, preview_w, preview_h
            )

            # Periodic GC: long-running loops accumulate small PIL /
            # numpy / tensor allocations that the cyclic collector is
            # slow to reclaim on CPython. Every ~25 s here.
            if self._frame_count % 300 == 0:
                gc.collect()

            # Wall-clock pacer: cap effective loop rate at TARGET_FPS.
            # Covers test-image mode and webcams that ignore CAP_PROP_FPS.
            elapsed = time.time() - t0
            remaining = FRAME_INTERVAL - elapsed
            if remaining > 0:
                time.sleep(remaining)

    def _draw_face_overlay(
        self,
        frame: np.ndarray,
        x: int, y: int, w: int, h: int,
        label: str, conf: float,
        preview_w: int = 240, preview_h: int = 180,
    ) -> None:
        """Minimalist face overlay: thin rectangle + small floating label."""
        fh, fw = frame.shape[:2]
        sf = max(1.0, max(fw / preview_w, fh / preview_h))
        color = EMOTION_BGR.get(label, (0, 255, 0))

        # Thin rectangle (1px in preview space)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, max(1, int(sf)))

        # Small label above top-left corner
        text = f"{label.upper()}  {conf:.0%}"
        font_scale = max(0.32, 0.40 * sf)
        thickness = max(1, int(sf))
        (tw, th), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        tx = x
        ty = max(th + int(2 * sf), y - int(4 * sf))
        cv2.putText(
            frame, text, (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
            color, thickness, cv2.LINE_AA,
        )

    def _cv2_to_tk(
        self, frame: np.ndarray, w: int, h: int,
    ) -> ImageTk.PhotoImage:
        """Convert an OpenCV BGR frame to a tkinter PhotoImage."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        src_w, src_h = pil_img.size
        scale = min(w / src_w, h / src_h, 1.0)
        if scale < 1.0:
            new_size = (int(src_w * scale), int(src_h * scale))
            pil_img = pil_img.resize(new_size, Image.LANCZOS)

        canvas = Image.new("RGB", (w, h), (0, 0, 0))
        offset = (
            (w - pil_img.size[0]) // 2,
            (h - pil_img.size[1]) // 2,
        )
        canvas.paste(pil_img, offset)
        return ImageTk.PhotoImage(canvas)

    def _spout_to_tk(self, frame: np.ndarray, w: int, h: int) -> ImageTk.PhotoImage:
        """Convert a Spout RGB frame to a tkinter PhotoImage."""
        pil_img = Image.fromarray(frame).resize((w, h), Image.LANCZOS)
        return ImageTk.PhotoImage(pil_img)

    # ─────────────────────────────────────────────────────────────
    #  GUI Refresh (main thread, ~30 fps)
    # ─────────────────────────────────────────────────────────────
    def _refresh_gui(self) -> None:
        if not self.running:
            return

        # Update webcam preview
        if hasattr(self, "_latest_frame"):
            self.webcam_canvas.configure(
                image=self._latest_frame, text="",
            )
            self.webcam_canvas.image = self._latest_frame

        # Update probability bars
        for emotion, widgets in self.prob_bars.items():
            prob = self.current_probs.get(emotion, 0.0)
            pct = prob * 100
            widgets["pct"].configure(text=f"{pct:.0f}%")
            widgets["fill"].place_configure(relwidth=prob)

        # Update dominant emotion
        if self.dominant_emotion:
            color = EMOTION_HEX.get(self.dominant_emotion, "#FFFFFF")
            self.dominant_label.configure(
                text=f"{self.dominant_emotion.upper()} "
                     f"({self.dominant_confidence:.1%})",
                fg=color,
            )
            # Show ensemble/calibration tags on separate line
            tags = []
            if self.calibration_mgr.has_active_profile():
                tags.append("CAL")
            if self.ensemble_mode and self._ensemble_winner:
                tags.append(f"ENS: {self._ensemble_winner}")
            self.dominant_tag_label.configure(
                text="  ".join(f"[{t}]" for t in tags) if tags else "",
            )
            # Also update center panel overlay (only when Spout is not active)
            if not (self._spout_available and self._latest_spout_frame is not None):
                self.viz_emotion_label.configure(
                    text=self.dominant_emotion.upper(),
                    fg=color,
                )
                if not self._viz_emotion_visible:
                    self.viz_emotion_label.place(relx=0.5, rely=0.5, anchor="center")
                    self._viz_emotion_visible = True
            else:
                if self._viz_emotion_visible:
                    self.viz_emotion_label.place_forget()
                    self._viz_emotion_visible = False
        else:
            self.dominant_label.configure(text="---", fg="#888888")
            self.dominant_tag_label.configure(text="")
            if self._viz_emotion_visible:
                self.viz_emotion_label.place_forget()
                self._viz_emotion_visible = False

        # Update center panel with Spout/Syphon frames (if available)
        if self._spout_available and self._latest_spout_frame is not None:
            canvas_w = self.viz_canvas.winfo_width() or SPOUT_DISPLAY_SIZE[0]
            canvas_h = self.viz_canvas.winfo_height() or SPOUT_DISPLAY_SIZE[1]
            self._spout_photo = self._spout_to_tk(
                self._latest_spout_frame, canvas_w, canvas_h,
            )
            if self._spout_canvas_image is None:
                self._spout_canvas_image = self.viz_canvas.create_image(
                    0, 0, anchor="nw", image=self._spout_photo,
                )
            else:
                self.viz_canvas.itemconfig(
                    self._spout_canvas_image, image=self._spout_photo,
                )
            if self._spout_placeholder_visible:
                self.viz_placeholder.place_forget()
                self._spout_placeholder_visible = False
        elif self._spout_placeholder_visible is False:
            self.viz_placeholder.place(relx=0.5, rely=0.7, anchor="center")
            self._spout_placeholder_visible = True

        # Update performance
        self.fps_label.configure(
            text=f"FPS: {self.current_fps:.0f} FPS",
        )
        self.latency_label.configure(
            text=f"Latency: {self.current_latency_ms:.0f} ms",
        )
        self.udp_label.configure(
            text=f"UDP Send: {self.udp_status}",
        )

        # Match UI refresh to capture rate. Running the UI at 30 fps
        # while capture emits new frames at 12 fps just redraws the
        # same data and wastes the main thread.
        self.root.after(
            int(1000 * FRAME_INTERVAL), self._refresh_gui,
        )

    # ─────────────────────────────────────────────────────────────
    #  Cleanup
    # ─────────────────────────────────────────────────────────────
    def _on_close(self) -> None:
        self.running = False
        if self.cap is not None:
            self.cap.release()
        if self.ensemble_mgr is not None:
            self.ensemble_mgr.release()
        # Active grad_cam_obj is a reference into _model_cache, so release
        # every cached Grad-CAM rather than just the current one.
        for bundle in self._model_cache.values():
            gc = bundle.get("grad_cam")
            if gc is not None:
                try:
                    gc.release()
                except Exception:
                    pass
        self._model_cache.clear()
        self.grad_cam_obj = None
        self.udp_sock.close()
        self.root.destroy()


# ═════════════════════════════════════════════════════════════════
#  Interactive Tutorial Overlay
# ═════════════════════════════════════════════════════════════════
class TutorialOverlay:
    """Step-by-step arrow tour that highlights key GUI widgets.

    Shows a dim overlay over the main window with a dashed spotlight
    border around each target widget, plus a callout box with navigation.
    Launch via: TutorialOverlay(gui_instance)
    """

    STEPS = [
        dict(
            attr="model_menu",
            side="below",
            title="Step 1 — Select a Model",
            body=(
                "Pick a model from this dropdown.\n"
                "ResNet-18 is recommended for first-time use.\n"
                "It loads automatically on selection."
            ),
        ),
        dict(
            attr="start_btn",
            side="above",
            title="Step 2 — Start the Webcam",
            body=(
                "Click the green START button.\n"
                "The built-in camera opens and a live\n"
                "preview appears in the panel above."
            ),
        ),
        dict(
            attr="webcam_canvas",
            side="right",
            title="Step 3 — Live Preview",
            body=(
                "Your face appears here once the\n"
                "webcam is running.\n"
                "Aim for ~50 cm distance, good lighting."
            ),
        ),
        dict(
            attr="dominant_label",
            side="left",
            title="Step 4 — Emotion Output",
            body=(
                "The detected emotion and confidence\n"
                "score appear here in real time.\n"
                "The bar chart above shows all emotions."
            ),
        ),
        dict(
            attr="load_image_btn",
            side="above",
            title="Step 5 — Test with an Image",
            body=(
                "Tick 'Use test image' then click Load\n"
                "to pick a photo from disk.\n"
                "Press START — the model runs on the image\n"
                "instead of the live webcam."
            ),
        ),
    ]

    _CW  = 268   # callout width
    _CH  = 158   # callout height
    _GAP = 14    # px gap between widget edge and callout

    def __init__(self, gui: "EmotionGUI") -> None:
        self._gui = gui
        self._step = 0
        self._overlay: Optional[tk.Toplevel] = None
        self._callout: Optional[tk.Toplevel] = None
        self._ov_canvas: Optional[tk.Canvas] = None
        self._build_overlay()
        self._render()

    # ── dim overlay (covers full main window) ────────────────────
    def _build_overlay(self) -> None:
        root = self._gui.root
        root.update_idletasks()
        self._overlay = tk.Toplevel(root)
        self._overlay.overrideredirect(True)
        self._overlay.wm_attributes("-topmost", True)
        self._overlay.wm_attributes("-alpha", 0.52)
        self._overlay.configure(bg="#000000")
        self._overlay.geometry(
            f"{root.winfo_width()}x{root.winfo_height()}"
            f"+{root.winfo_rootx()}+{root.winfo_rooty()}"
        )
        self._ov_canvas = tk.Canvas(
            self._overlay, bg="#000000",
            highlightthickness=0, cursor="arrow",
        )
        self._ov_canvas.pack(fill=tk.BOTH, expand=True)

    # ── render current step ──────────────────────────────────────
    def _render(self) -> None:
        step = self.STEPS[self._step]
        root  = self._gui.root
        target = getattr(self._gui, step["attr"])

        root.update_idletasks()
        target.update_idletasks()

        tx = target.winfo_rootx()
        ty = target.winfo_rooty()
        tw = max(target.winfo_width(), 1)
        th = max(target.winfo_height(), 1)

        ox = self._overlay.winfo_rootx()
        oy = self._overlay.winfo_rooty()
        lx, ly = tx - ox, ty - oy

        c = self._ov_canvas
        c.delete("all")

        # Dashed spotlight border around target
        pad = 7
        c.create_rectangle(
            lx - pad, ly - pad,
            lx + tw + pad, ly + th + pad,
            outline="#E6B800", width=3, dash=(10, 5),
        )

        # Numbered badge at top-left corner of spotlight
        bx, by = lx - pad, ly - pad
        r = 13
        c.create_oval(bx - r, by - r, bx + r, by + r,
                      fill="#E6B800", outline="")
        c.create_text(bx, by, text=str(self._step + 1),
                      font=(UI_FONT, 10, "bold"), fill="#000000")

        self._place_callout(step, tx, ty, tw, th)

    # ── callout box ──────────────────────────────────────────────
    def _place_callout(self, step, tx, ty, tw, th) -> None:
        if self._callout and self._callout.winfo_exists():
            self._callout.destroy()

        side = step["side"]
        cw, ch, gap = self._CW, self._CH, self._GAP
        n = len(self.STEPS)
        is_last = self._step == n - 1

        if side == "right":
            win_x = tx + tw + gap
            win_y = ty + th // 2 - ch // 2
        elif side == "left":
            win_x = tx - cw - gap
            win_y = ty + th // 2 - ch // 2
        elif side == "below":
            win_x = tx + tw // 2 - cw // 2
            win_y = ty + th + gap
        else:  # above
            win_x = tx + tw // 2 - cw // 2
            win_y = ty - ch - gap

        sw = self._gui.root.winfo_screenwidth()
        sh = self._gui.root.winfo_screenheight()
        win_x = max(4, min(win_x, sw - cw - 4))
        win_y = max(4, min(win_y, sh - ch - 4))

        win = tk.Toplevel(self._gui.root)
        win.overrideredirect(True)
        win.wm_attributes("-topmost", True)
        win.geometry(f"{cw}x{ch}+{win_x}+{win_y}")
        win.configure(bg="#1C1C1C")
        self._callout = win

        # Arrow indicator pointing toward the target widget
        ARROW = {"right": " ←", "left": " →", "below": " ↑", "above": " ↓"}

        c = tk.Canvas(win, width=cw, height=ch,
                      bg="#1C1C1C", highlightthickness=0)
        c.pack(fill=tk.BOTH, expand=True)

        c.create_rectangle(1, 1, cw - 1, ch - 1, outline="#E6B800", width=2)

        # Title with directional arrow
        c.create_text(
            cw // 2, 22,
            text=step["title"] + ARROW.get(side, ""),
            font=(UI_FONT, 11, "bold"), fill="#E6B800",
            anchor="center", width=cw - 20,
        )

        # Horizontal divider
        c.create_line(16, 38, cw - 16, 38, fill="#333333")

        # Body text
        c.create_text(
            cw // 2, 78,
            text=step["body"],
            font=(UI_FONT, 10), fill="#CCCCCC",
            anchor="center", width=cw - 24, justify="center",
        )

        # Step counter
        btn_y = ch - 30
        c.create_text(
            cw // 2, btn_y + 11,
            text=f"{self._step + 1} / {n}",
            font=(UI_FONT, 9), fill="#555555", anchor="center",
        )

        # Skip button
        ctk.CTkButton(
            win, text="Skip",
            font=(UI_FONT, 10),
            fg_color="#262626", hover_color="#363636",
            text_color="#777777", corner_radius=4, height=22, width=48,
            command=self.close,
        ).place(x=8, y=btn_y)

        # Back button (not on first step)
        if self._step > 0:
            ctk.CTkButton(
                win, text="← Back",
                font=(UI_FONT, 10),
                fg_color="#262626", hover_color="#363636",
                text_color="#AAAAAA", corner_radius=4, height=22, width=62,
                command=self._prev,
            ).place(x=cw - 136, y=btn_y)

        # Next / Finish button
        ctk.CTkButton(
            win,
            text="Finish ✓" if is_last else "Next →",
            font=(UI_FONT, 10, "bold"),
            fg_color=BTN_START if is_last else "#1A3A70",
            hover_color=BTN_START_HOVER if is_last else "#2A4A90",
            text_color="white", corner_radius=4, height=22, width=68,
            command=self.close if is_last else self._next,
        ).place(x=cw - 76, y=btn_y)

    def _next(self) -> None:
        self._step += 1
        self._render()

    def _prev(self) -> None:
        self._step -= 1
        self._render()

    def close(self) -> None:
        for w in (self._callout, self._overlay):
            if w and w.winfo_exists():
                w.destroy()


# ═════════════════════════════════════════════════════════════════
#  Calibration Wizard
# ═════════════════════════════════════════════════════════════════
class CalibrationWizard(tk.Toplevel):
    """Modal dialog that guides the user through per-emotion calibration."""

    PREP_SECONDS = 3
    RECORD_SECONDS = 3
    TICK_MS = 100

    EMOTION_HINTS: Dict[str, str] = {
        "angry":    "Frown, clench your jaw",
        "happy":    "Smile naturally",
        "neutral":  "Relax your face",
        "sad":      "Let your face droop slightly",
        "surprise": "Raise eyebrows, open mouth",
    }


    def __init__(self, parent: EmotionGUI) -> None:
        super().__init__(parent.root)
        self.parent = parent
        self.mgr = parent.calibration_mgr

        self.title("Emotion Calibration")
        self.configure(bg=BG_COLOR)
        self.geometry("560x560")
        self.resizable(False, False)
        self.transient(parent.root)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)

        self._emotion_order = list(EMOTION_CLASSES)
        self._current_idx = 0
        self._phase = "setup"  # setup | prep | record | done
        self._countdown = 0.0
        self._frames_recorded = 0

        self._build_ui()

    # ── UI ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # ── Header ───────────────────────────────────────────────
        tk.Label(
            self, text="Emotion Calibration",
            font=(UI_FONT, 17, "bold"),
            bg=BG_COLOR, fg=HEADING_COLOR,
        ).pack(pady=(20, 2))

        tk.Label(
            self,
            text="Show each expression and hold it for a few seconds.",
            font=(UI_FONT, 10), bg=BG_COLOR, fg="#888888",
        ).pack()

        # ── Name input ───────────────────────────────────────────
        name_frame = tk.Frame(self, bg=BG_COLOR)
        name_frame.pack(pady=(14, 0))
        tk.Label(
            name_frame, text="Your name:",
            font=(UI_FONT, 10), bg=BG_COLOR, fg=TEXT_COLOR,
        ).pack(side=tk.LEFT)
        self.name_var = tk.StringVar(value="user")
        self.name_entry = ctk.CTkEntry(
            name_frame, textvariable=self.name_var,
            font=(UI_FONT, 11), width=160,
            fg_color=ACCENT, text_color=TEXT_COLOR,
            border_color="#555555", border_width=1,
        )
        self.name_entry.pack(side=tk.LEFT, padx=(8, 0))

        # ── Stepper dots ─────────────────────────────────────────
        sep = tk.Frame(self, bg="#3A3A3A", height=1)
        sep.pack(fill=tk.X, padx=30, pady=(14, 0))

        stepper_frame = tk.Frame(self, bg=BG_COLOR)
        stepper_frame.pack(pady=(10, 0))
        self._stepper_dots:  Dict[str, tk.Label] = {}
        self._stepper_names: Dict[str, tk.Label] = {}
        for emotion in self._emotion_order:
            col = tk.Frame(stepper_frame, bg=BG_COLOR)
            col.pack(side=tk.LEFT, padx=14)
            dot = tk.Label(col, text="○", font=(UI_FONT, 15),
                           bg=BG_COLOR, fg="#444444")
            dot.pack()
            name_lbl = tk.Label(col, text=emotion,
                                font=(UI_FONT, 8), bg=BG_COLOR, fg="#444444")
            name_lbl.pack()
            self._stepper_dots[emotion]  = dot
            self._stepper_names[emotion] = name_lbl

        sep2 = tk.Frame(self, bg="#3A3A3A", height=1)
        sep2.pack(fill=tk.X, padx=30, pady=(10, 0))

        # ── Emotion prompt ───────────────────────────────────────
        self.prompt_label = tk.Label(
            self, text="Press Start to begin",
            font=(UI_FONT, 28, "bold"),
            bg=BG_COLOR, fg="#FFD700",
        )
        self.prompt_label.pack(pady=(18, 2))

        self.hint_label = tk.Label(
            self, text="",
            font=(UI_FONT, 11, "italic"),
            bg=BG_COLOR, fg="#888888",
        )
        self.hint_label.pack()

        # ── Status / countdown ───────────────────────────────────
        self.status_label = tk.Label(
            self, text="",
            font=(UI_FONT, 13),
            bg=BG_COLOR, fg=TEXT_COLOR,
        )
        self.status_label.pack(pady=(10, 2))

        # ── Live detection feedback ──────────────────────────────
        self.live_label = tk.Label(
            self, text="",
            font=(UI_FONT, 10, "italic"),
            bg=BG_COLOR, fg="#666666",
        )
        self.live_label.pack()

        # ── Per-emotion recording bar ────────────────────────────
        style = ttk.Style()
        style.configure("Rec.Horizontal.TProgressbar",
                        troughcolor=ACCENT, background="#4CAF50")
        self.record_var = tk.DoubleVar(value=0.0)
        self.record_bar = ttk.Progressbar(
            self, variable=self.record_var,
            maximum=1.0, length=420,
            style="Rec.Horizontal.TProgressbar",
        )
        # record_bar and text start hidden; shown when recording begins
        self.record_bar.pack_forget()

        self.record_text = tk.Label(
            self, text="",
            font=(UI_FONT, 9), bg=BG_COLOR, fg="#666666",
        )

        # Overall progress — stepper dots already show this visually;
        # keep a simple text counter, no extra bar needed
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_text = tk.Label(
            self, text="",
            font=(UI_FONT, 9), bg=BG_COLOR, fg="#666666",
        )
        self.progress_text.pack(pady=(4, 0))

        # ── Buttons ──────────────────────────────────────────────
        btn_frame = tk.Frame(self, bg=BG_COLOR)
        btn_frame.pack(pady=(14, 18))

        self.start_btn = ctk.CTkButton(
            btn_frame, text="Start",
            font=(UI_FONT, 12, "bold"),
            fg_color=BTN_START, hover_color=BTN_START_HOVER,
            text_color="white", width=130,
            command=self._on_start,
        )
        self.start_btn.pack(side=tk.LEFT, padx=8)

        self.cancel_btn = ctk.CTkButton(
            btn_frame, text="Cancel",
            font=(UI_FONT, 12),
            fg_color="#3A3A3A", hover_color="#4A4A4A",
            text_color="#AAAAAA", width=130,
            command=self._on_cancel,
        )
        self.cancel_btn.pack(side=tk.LEFT, padx=8)

    def _update_stepper(self, current_idx: int) -> None:
        for i, emotion in enumerate(self._emotion_order):
            if i < current_idx:
                color = "#4CAF50"
                dot   = "●"
            elif i == current_idx:
                color = EMOTION_HEX.get(emotion, "#FFD700")
                dot   = "●"
            else:
                color = "#444444"
                dot   = "○"
            self._stepper_dots[emotion].configure(text=dot, fg=color)
            self._stepper_names[emotion].configure(fg=color)

    # ── Wizard flow ──────────────────────────────────────────────

    def _on_start(self) -> None:
        user_name = self.name_var.get().strip()
        if not user_name:
            self.status_label.configure(
                text="Please enter your name.", fg="#FF5555",
            )
            return

        model_name = self.parent.model_var.get()
        fps = max(self.parent.current_fps, 10.0)
        frames_per_emotion = int(self.RECORD_SECONDS * fps)
        frames_per_emotion = max(frames_per_emotion, 15)

        self.mgr.start_session(user_name, model_name, frames_per_emotion)
        self._current_idx = 0
        self._phase = "prep"
        self._update_stepper(0)

        self.start_btn.configure(state="disabled")
        self.name_entry.configure(state=tk.DISABLED)

        self._begin_prep()

    def _begin_prep(self) -> None:
        """Start the countdown phase for the current emotion."""
        emotion = self._emotion_order[self._current_idx]
        color   = EMOTION_HEX.get(emotion, "#FFD700")
        hint    = self.EMOTION_HINTS.get(emotion, "")

        self._update_stepper(self._current_idx)
        self.prompt_label.configure(text=emotion.upper(), fg=color)
        self.hint_label.configure(text=hint)
        self._countdown = self.PREP_SECONDS
        self._phase = "prep"
        self.record_var.set(0.0)
        self.record_bar.pack_forget()
        self.record_text.configure(text="")
        self.live_label.configure(text="")
        self.status_label.configure(
            text=f"Get ready...  {self.PREP_SECONDS}", fg="#FFD700",
        )
        self._tick()

    def _begin_record(self) -> None:
        """Switch from prep countdown to frame recording."""
        self._phase = "record"
        self._frames_recorded = 0
        self.record_var.set(0.0)
        self.record_bar.pack(pady=(8, 2))
        self.record_text.pack()
        self.status_label.configure(text="● Recording", fg="#4CAF50")
        self._tick()

    def _tick(self) -> None:
        """Timer callback running every TICK_MS."""
        if self._phase == "prep":
            self._countdown -= self.TICK_MS / 1000.0
            if self._countdown <= 0:
                self._begin_record()
                return
            self.status_label.configure(
                text=f"Get ready... {int(self._countdown) + 1}",
            )
            self.after(self.TICK_MS, self._tick)

        elif self._phase == "record":
            # Read the latest probabilities from the parent's capture loop
            probs = dict(self.parent.current_probs)
            if any(v > 0 for v in probs.values()):
                emotion, recorded, needed = self.mgr.record_frame(probs)
                self._frames_recorded = recorded
                self.record_var.set(recorded / max(needed, 1))
                self.record_text.configure(
                    text=f"{recorded} / {needed} frames captured",
                    fg="#4CAF50",
                )
                # Live detection feedback
                live_emotion = max(probs, key=probs.get)
                live_conf    = probs[live_emotion]
                self.live_label.configure(
                    text=f"Model sees: {live_emotion} ({live_conf:.0%})",
                    fg=EMOTION_HEX.get(live_emotion, "#888888"),
                )

            if self.mgr.get_current_emotion() is None or (
                self.mgr.get_current_emotion()
                != self._emotion_order[self._current_idx]
            ):
                # Current emotion done — advance to next
                self._current_idx += 1
                self.progress_var.set(self._current_idx)
                self.progress_text.configure(
                    text=(
                        f"{self._current_idx} / "
                        f"{len(self._emotion_order)} emotions"
                    ),
                )

                if self.mgr.is_session_complete():
                    self._finish()
                    return
                self._begin_prep()
                return

            self.after(self.TICK_MS, self._tick)

    def _finish(self) -> None:
        """Calibration complete — show summary and save."""
        self._phase = "done"
        profile = self.mgr.finish_session()
        scores = self.mgr.get_diagonal_scores(profile)

        # Build summary text
        lines = ["Calibration complete!\n"]
        warnings = []
        for emotion in self._emotion_order:
            score = scores[emotion]
            pct = f"{score:.0%}"
            mark = "  OK" if score >= 0.2 else "  LOW"
            lines.append(f"  {emotion.capitalize():>10}  {pct:>5} {mark}")
            if score < 0.2:
                warnings.append(emotion)

        self._update_stepper(len(self._emotion_order))
        self.record_var.set(0.0)
        self.record_text.configure(text="")
        self.live_label.configure(text="")
        self.prompt_label.configure(text="Done!", fg="#4CAF50")
        self.hint_label.configure(text="")

        summary = "\n".join(lines)
        if warnings:
            summary += (
                f"\n\nWarning: {', '.join(warnings)} scored low.\n"
                "Consider re-calibrating those emotions."
            )
        self.status_label.configure(
            text=summary, fg=TEXT_COLOR, justify="left",
            font=("Consolas", 10),
        )

        # Save profile
        path = self.mgr.save_profile(profile)
        print(f"[INFO] Calibration profile saved: {path}")

        # Activate profile
        self.mgr.set_active_profile(profile)
        self.parent._update_calibration_status()

        # Replace buttons with Close
        self.start_btn.configure(
            text="Close", state="normal",
            command=self.destroy,
            fg_color=ACCENT, hover_color=ACCENT,
        )
        self.cancel_btn.pack_forget()

    def _on_cancel(self) -> None:
        if self.mgr.is_recording():
            self.mgr.cancel_session()
        self.destroy()


# ═════════════════════════════════════════════════════════════════
#  Entry Point
# ═════════════════════════════════════════════════════════════════
def main() -> None:
    root = tk.Tk()
    root.minsize(1100, 650)
    EmotionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
