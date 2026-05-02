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
        tk.Label(
            title_frame,
            text="TouchDesigner Interface \u2014 "
                 "Adaptive Emotion Visualization",
            font=(UI_FONT, 14, "bold"),
            bg="#252526", fg="#CCCCCC",
            pady=8,
        ).pack()

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

        test_btn = ctk.CTkButton(
            test_frame, text="Load",
            font=(UI_FONT, 11),
            fg_color=BTN_PRIMARY, hover_color=BTN_PRIMARY_HOVER,
            text_color="white", corner_radius=6,
            width=70, height=28,
            command=self._load_test_image,
        )
        test_btn.pack(side=tk.RIGHT)

        # START / STOP
        btn_frame = tk.Frame(panel, bg=PANEL_BG)
        btn_frame.pack(padx=15, pady=(5, 15), fill=tk.X)

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
            print("[WARNING] Start the webcam before calibrating.")
            return
        if self.model is None:
            print("[WARNING] Load a model before calibrating.")
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
                ("Image files", "*.png;*.jpg;*.jpeg;*.bmp"),
                ("All files", "*.*"),
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
        self.dominant_emotion = best_label
        self.dominant_confidence = best_conf

        for emotion, widgets in self.prob_bars.items():
            prob = probs_dict.get(emotion, 0.0)
            pct = prob * 100
            widgets["pct"].configure(text=f"{pct:.0f}%")
            widgets["fill"].place_configure(relwidth=prob)

        if best_label:
            color_hex = EMOTION_HEX.get(best_label, "#FFFFFF")
            self.dominant_label.configure(
                text=f"{best_label.upper()} ({best_conf:.1%})",
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

            # FPS / latency
            t1 = time.time()
            self.current_latency_ms = (t1 - t0) * 1000
            self.current_fps = 1.0 / max(t1 - prev_time, 1e-6)
            prev_time = t1

            # Store data for GUI thread
            self.current_probs = probs_dict
            self.dominant_emotion = best_label
            self.dominant_confidence = best_conf

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
#  Entry Point
# ═════════════════════════════════════════════════════════════════
class CalibrationWizard(tk.Toplevel):
    """Modal dialog that guides the user through per-emotion calibration."""

    PREP_SECONDS = 3     # countdown before recording each emotion
    RECORD_SECONDS = 3   # seconds to record frames for each emotion
    TICK_MS = 100        # timer resolution

    # Friendly labels and hints shown to the user for each emotion
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
        self.geometry("520x480")
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
        # Title
        tk.Label(
            self, text="User Calibration",
            font=(UI_FONT, 18, "bold"),
            bg=BG_COLOR, fg=HEADING_COLOR,
        ).pack(pady=(18, 4))

        tk.Label(
            self,
            text="You will be asked to show each emotion in turn.\n"
                 "Hold each expression for a few seconds.",
            font=(UI_FONT, 10), bg=BG_COLOR, fg=TEXT_COLOR,
            justify="center",
        ).pack(pady=(0, 12))

        # User name
        name_frame = tk.Frame(self, bg=BG_COLOR)
        name_frame.pack(padx=30, fill=tk.X)
        tk.Label(
            name_frame, text="Your Name:",
            font=(UI_FONT, 11), bg=BG_COLOR, fg=TEXT_COLOR,
        ).pack(side=tk.LEFT)
        self.name_var = tk.StringVar(value="user")
        self.name_entry = tk.Entry(
            name_frame, textvariable=self.name_var,
            font=(UI_FONT, 11), width=20,
        )
        self.name_entry.pack(side=tk.LEFT, padx=(8, 0))

        # Emotion prompt (large)
        self.prompt_label = tk.Label(
            self, text="Press Start to begin",
            font=(UI_FONT, 26, "bold"),
            bg=BG_COLOR, fg="#FFD700",
        )
        self.prompt_label.pack(pady=(20, 4))

        # Hint text
        self.hint_label = tk.Label(
            self, text="",
            font=(UI_FONT, 11, "italic"),
            bg=BG_COLOR, fg="#AAAAAA",
        )
        self.hint_label.pack()

        # Countdown / status
        self.status_label = tk.Label(
            self, text="",
            font=(UI_FONT, 14),
            bg=BG_COLOR, fg=TEXT_COLOR,
        )
        self.status_label.pack(pady=(12, 4))

        # Progress bar
        self.progress_var = tk.DoubleVar(value=0.0)
        style = ttk.Style()
        style.configure(
            "Cal.Horizontal.TProgressbar",
            troughcolor=ACCENT, background="#4CAF50",
        )
        self.progress_bar = ttk.Progressbar(
            self, variable=self.progress_var,
            maximum=len(self._emotion_order),
            length=400, style="Cal.Horizontal.TProgressbar",
        )
        self.progress_bar.pack(pady=(8, 4))

        self.progress_text = tk.Label(
            self, text="0 / 5 emotions",
            font=(UI_FONT, 10), bg=BG_COLOR, fg=TEXT_COLOR,
        )
        self.progress_text.pack()

        # Buttons
        btn_frame = tk.Frame(self, bg=BG_COLOR)
        btn_frame.pack(pady=(16, 16))

        self.start_btn = tk.Button(
            btn_frame, text="Start",
            font=(UI_FONT, 12, "bold"),
            bg=BTN_START, fg="white",
            activebackground="#388E3C",
            relief=tk.FLAT, width=12,
            command=self._on_start,
        )
        self.start_btn.pack(side=tk.LEFT, padx=8)

        self.cancel_btn = tk.Button(
            btn_frame, text="Cancel",
            font=(UI_FONT, 12),
            bg=BTN_STOP, fg="white",
            activebackground="#D32F2F",
            relief=tk.FLAT, width=12,
            command=self._on_cancel,
        )
        self.cancel_btn.pack(side=tk.LEFT, padx=8)

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

        self.start_btn.configure(state=tk.DISABLED)
        self.name_entry.configure(state=tk.DISABLED)

        self._begin_prep()

    def _begin_prep(self) -> None:
        """Start the countdown phase for the current emotion."""
        emotion = self._emotion_order[self._current_idx]
        color = EMOTION_HEX.get(emotion, "#FFD700")
        hint = self.EMOTION_HINTS.get(emotion, "")

        self.prompt_label.configure(text=emotion.upper(), fg=color)
        self.hint_label.configure(text=hint)
        self._countdown = self.PREP_SECONDS
        self._phase = "prep"
        self.status_label.configure(
            text=f"Get ready... {self.PREP_SECONDS}", fg="#FFD700",
        )
        self._tick()

    def _begin_record(self) -> None:
        """Switch from prep countdown to frame recording."""
        self._phase = "record"
        self._frames_recorded = 0
        self.status_label.configure(text="Recording...", fg="#4CAF50")
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
                self.status_label.configure(
                    text=f"Recording... {recorded}/{needed}",
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
            text="Close", state=tk.NORMAL,
            command=self.destroy, bg=ACCENT,
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
