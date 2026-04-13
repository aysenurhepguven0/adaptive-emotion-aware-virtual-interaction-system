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
from ctypes import wintypes
import json
import os
import socket
import threading
import time
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
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

# Target display size for TouchDesigner Spout frames
SPOUT_DISPLAY_SIZE = (1024, 1024)

# Preview size for webcam/test image panel
WEBCAM_PREVIEW_SIZE = (320, 240)

# ── GUI colour palette ───────────────────────────────────────────
BG_COLOR = "#2B2B2B"
PANEL_BG = "#363636"
ACCENT = "#4A4A4A"
TEXT_COLOR = "#E0E0E0"
HEADING_COLOR = "#FFFFFF"
BORDER_COLOR = "#555555"
BTN_START = "#2E7D32"
BTN_STOP = "#C62828"
BAR_BG = "#4A4A4A"


# ── Hidden OpenGL context for Spout (Windows) ───────────────────
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
        self.root.configure(bg=BG_COLOR)
        self.root.state("zoomed")

        # ── State variables ──────────────────────────────────────
        self.running = False
        self.cap: Optional[cv2.VideoCapture] = None
        self.model: Optional[torch.nn.Module] = None
        self.class_names: Tuple[str, ...] = EMOTION_CLASSES
        self.transform: Optional[transforms.Compose] = None
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        if self.device.type == "cpu":
            torch.set_num_threads(min(4, os.cpu_count() or 1))
            torch.set_num_interop_threads(1)
        self.face_detector: Optional[cv2.CascadeClassifier] = None

        # Grad-CAM
        self.grad_cam_obj: Optional[GradCAM] = None
        self.grad_cam_enabled = True

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

        # Frame skipping for performance
        self._frame_count = 0
        self._cached_faces = ()
        self._cached_gradcam = None

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
        self._build_title_bar()
        self._build_panels()
        self._load_default_model()

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ─────────────────────────────────────────────────────────────
    #  Optional: Spout Receiver for TouchDesigner
    # ─────────────────────────────────────────────────────────────
    def _init_spout(self) -> None:
        """Check whether SpoutGL is available."""
        try:
            import SpoutGL                          # noqa: F401
            from OpenGL import GL as _GL            # noqa: F401
            self._spout_available = True
            print("[INFO] SpoutGL available – will receive TD frames.")
        except ImportError:
            self._spout_available = False
            print("[INFO] SpoutGL not found – Spout receiving disabled.")

    def _spout_receive_loop(self) -> None:
        """Background thread: receive TouchDesigner frames via Spout."""
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
        title = tk.Label(
            self.root,
            text="TouchDesigner Interface \u2014 "
                 "Adaptive Emotion Visualization",
            font=("Segoe UI", 16, "bold"),
            bg=BG_COLOR, fg=HEADING_COLOR,
            pady=10,
        )
        title.pack(side=tk.TOP, fill=tk.X)

    def _build_panels(self) -> None:
        container = tk.Frame(self.root, bg=BG_COLOR)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        container.columnconfigure(0, weight=2)
        container.columnconfigure(1, weight=5)
        container.columnconfigure(2, weight=2)

        self._build_control_panel(container)
        self._build_visualization_panel(container)
        self._build_monitoring_panel(container)

    # ── LEFT: Control Panel ──────────────────────────────────────
    def _build_control_panel(self, parent: tk.Frame) -> None:
        panel = tk.Frame(
            parent, bg=PANEL_BG,
            highlightbackground=BORDER_COLOR, highlightthickness=1,
        )
        panel.grid(row=0, column=0, sticky="nsew", padx=(0, 5))

        self._section_label(panel, "Control Panel")

        # Model Selection
        self._heading(panel, "Model Selection")
        self.model_var = tk.StringVar(value="ResNet-18")
        model_menu = ttk.Combobox(
            panel, textvariable=self.model_var,
            values=list(MODEL_OPTIONS.keys()),
            state="readonly", font=("Segoe UI", 11),
        )
        model_menu.pack(padx=15, pady=(0, 8), fill=tk.X)
        model_menu.bind("<<ComboboxSelected>>", self._on_model_change)

        # UDP Settings
        self._heading(panel, "UDP Settings")
        udp_frame = tk.Frame(panel, bg=PANEL_BG)
        udp_frame.pack(padx=15, fill=tk.X)

        tk.Label(
            udp_frame, text="Port:", bg=PANEL_BG, fg=TEXT_COLOR,
            font=("Segoe UI", 10),
        ).grid(row=0, column=0, sticky="w")
        self.port_var = tk.StringVar(value=str(DEFAULT_UDP_PORT))
        port_entry = tk.Entry(
            udp_frame, textvariable=self.port_var, width=8,
            font=("Segoe UI", 10),
        )
        port_entry.grid(row=0, column=1, padx=(5, 15))

        tk.Label(
            udp_frame, text="Host:", bg=PANEL_BG, fg=TEXT_COLOR,
            font=("Segoe UI", 10),
        ).grid(row=0, column=2, sticky="w")
        self.host_var = tk.StringVar(value=DEFAULT_UDP_IP)
        host_entry = tk.Entry(
            udp_frame, textvariable=self.host_var, width=12,
            font=("Segoe UI", 10),
        )
        host_entry.grid(row=0, column=3)

        # Sensitivity
        self._heading(panel, "Sensitivity")
        self.sensitivity_var = tk.DoubleVar(value=0.3)
        sens_slider = tk.Scale(
            panel, from_=0.0, to=1.0, resolution=0.05,
            orient=tk.HORIZONTAL, variable=self.sensitivity_var,
            bg=PANEL_BG, fg=TEXT_COLOR, troughcolor=ACCENT,
            highlightthickness=0, font=("Segoe UI", 9),
            command=self._on_sensitivity_change,
        )
        sens_slider.pack(padx=15, fill=tk.X)

        # Grad-CAM toggle
        self.gradcam_var = tk.BooleanVar(value=True)
        gradcam_chk = tk.Checkbutton(
            panel, text="Show Grad-CAM overlay",
            variable=self.gradcam_var, bg=PANEL_BG,
            fg=TEXT_COLOR, selectcolor=ACCENT,
            activebackground=PANEL_BG, activeforeground=TEXT_COLOR,
            font=("Segoe UI", 10),
            command=self._on_gradcam_toggle,
        )
        gradcam_chk.pack(padx=15, anchor="w", pady=(4, 0))

        # Webcam Preview
        self._heading(panel, "Webcam Preview")
        webcam_frame = tk.Frame(
            panel,
            bg="#1A1A1A",
            width=WEBCAM_PREVIEW_SIZE[0],
            height=WEBCAM_PREVIEW_SIZE[1],
        )
        webcam_frame.pack(padx=15, pady=(0, 8))
        webcam_frame.pack_propagate(False)

        self.webcam_canvas = tk.Label(
            webcam_frame, bg="#1A1A1A",
            text="Camera Off", fg="#666666",
            font=("Segoe UI", 12),
        )
        self.webcam_canvas.pack(fill=tk.BOTH, expand=True)

        # Test Image
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
            font=("Segoe UI", 10),
            command=self._on_test_image_toggle,
        )
        test_toggle.pack(side=tk.LEFT)

        test_btn = tk.Button(
            test_frame,
            text="Load Image",
            font=("Segoe UI", 10),
            bg=ACCENT,
            fg=TEXT_COLOR,
            activebackground="#5A5A5A",
            relief=tk.FLAT,
            command=self._load_test_image,
        )
        test_btn.pack(side=tk.RIGHT)

        # START / STOP
        btn_frame = tk.Frame(panel, bg=PANEL_BG)
        btn_frame.pack(padx=15, pady=(5, 15), fill=tk.X)

        self.start_btn = tk.Button(
            btn_frame, text="START", font=("Segoe UI", 14, "bold"),
            bg=BTN_START, fg="white", activebackground="#388E3C",
            relief=tk.FLAT, cursor="hand2",
            command=self._start,
        )
        self.start_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))

        self.stop_btn = tk.Button(
            btn_frame, text="STOP", font=("Segoe UI", 14, "bold"),
            bg=BTN_STOP, fg="white", activebackground="#D32F2F",
            relief=tk.FLAT, cursor="hand2",
            command=self._stop, state=tk.DISABLED,
        )
        self.stop_btn.pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(5, 0))

    # ── CENTER: Visualization ────────────────────────────────────
    def _build_visualization_panel(self, parent: tk.Frame) -> None:
        panel = tk.Frame(
            parent, bg="#0A0A0A",
            highlightbackground=BORDER_COLOR, highlightthickness=1,
        )
        panel.grid(row=0, column=1, sticky="nsew", padx=5)

        header = tk.Label(
            panel,
            text="Particle Visualization Area",
            font=("Segoe UI", 13, "italic"),
            bg="#0A0A0A", fg="#888888",
        )
        header.pack(pady=(8, 0))

        viz_frame = tk.Frame(panel, bg="#0A0A0A")
        viz_frame.pack(expand=True)

        self.viz_canvas = tk.Canvas(
            viz_frame,
            width=SPOUT_DISPLAY_SIZE[0],
            height=SPOUT_DISPLAY_SIZE[1],
            bg="#0A0A0A",
            highlightthickness=0,
        )
        self.viz_canvas.pack(padx=4, pady=4)
        self._spout_canvas_image = None

        # Dominant emotion overlay (big text on the center panel)
        self.viz_emotion_label = tk.Label(
            self.viz_canvas,
            text="",
            font=("Segoe UI", 48, "bold"),
            bg="#0A0A0A", fg="#FFD700",
        )
        self.viz_emotion_label.place(relx=0.5, rely=0.5, anchor="center")

        # Placeholder text when TD is not connected
        self.viz_placeholder = tk.Label(
            self.viz_canvas,
            text="Connect TouchDesigner via Spout\n"
                 "or run TD alongside this application.\n\n"
                 "UDP data is being sent to\n"
                 f"{DEFAULT_UDP_IP}:{DEFAULT_UDP_PORT}",
            font=("Segoe UI", 11),
            bg="#0A0A0A", fg="#555555",
            justify=tk.CENTER,
        )
        self.viz_placeholder.place(relx=0.5, rely=0.7, anchor="center")

        footer = tk.Label(
            panel,
            text="Real-time adaptive particle rendering",
            font=("Segoe UI", 9, "italic"),
            bg="#0A0A0A", fg="#555555",
        )
        footer.pack(side=tk.BOTTOM, pady=(0, 6))

    # ── RIGHT: Monitoring ────────────────────────────────────────
    def _build_monitoring_panel(self, parent: tk.Frame) -> None:
        panel = tk.Frame(
            parent, bg=PANEL_BG,
            highlightbackground=BORDER_COLOR, highlightthickness=1,
        )
        panel.grid(row=0, column=2, sticky="nsew", padx=(5, 0))

        self._section_label(panel, "Monitoring")

        # Emotion Probabilities
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
                font=("Segoe UI", 10, "bold"),
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
                row, text="0%", font=("Segoe UI", 9),
                fg=TEXT_COLOR, bg=PANEL_BG, width=5, anchor="w",
            )
            pct_label.pack(side=tk.LEFT)

            self.prob_bars[emotion] = {
                "container": bar_container,
                "fill": bar_fill,
                "pct": pct_label,
            }

        # Separator
        self._separator(panel)

        # Dominant Emotion
        self._heading(panel, "Dominant Emotion")
        self.dominant_label = tk.Label(
            panel, text="---",
            font=("Segoe UI", 22, "bold"),
            bg=PANEL_BG, fg="#FFD700",
        )
        self.dominant_label.pack(pady=(0, 4))

        # Separator
        self._separator(panel)

        # Performance
        self._heading(panel, "Performance")
        perf_frame = tk.Frame(panel, bg=PANEL_BG)
        perf_frame.pack(padx=15, fill=tk.X)

        self.fps_label = tk.Label(
            perf_frame, text="FPS: --",
            font=("Segoe UI", 11), fg=TEXT_COLOR, bg=PANEL_BG,
            anchor="w",
        )
        self.fps_label.pack(fill=tk.X)

        self.latency_label = tk.Label(
            perf_frame, text="Latency: -- ms",
            font=("Segoe UI", 11), fg=TEXT_COLOR, bg=PANEL_BG,
            anchor="w",
        )
        self.latency_label.pack(fill=tk.X)

        self.udp_label = tk.Label(
            perf_frame, text="UDP Recv: Idle",
            font=("Segoe UI", 11), fg=TEXT_COLOR, bg=PANEL_BG,
            anchor="w",
        )
        self.udp_label.pack(fill=tk.X)

        self.device_label = tk.Label(
            perf_frame,
            text=f"Device: {self.device}",
            font=("Segoe UI", 10), fg="#888888", bg=PANEL_BG,
            anchor="w",
        )
        self.device_label.pack(fill=tk.X, pady=(8, 0))

    # ── UI helpers ───────────────────────────────────────────────
    def _section_label(self, parent: tk.Frame, text: str) -> None:
        tk.Label(
            parent, text=text,
            font=("Segoe UI", 14, "bold"),
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
            font=("Segoe UI", 10), bg=PANEL_BG, fg="#AAAAAA",
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
        self.model, self.class_names = load_model_from_checkpoint(
            cfg["name"], checkpoint, self.device,
            class_names=EMOTION_CLASSES,
        )

        input_size = cfg["input_size"]
        grayscale = cfg["grayscale"]
        self.transform = build_transforms(
            input_size, grayscale=grayscale
        )["eval"]

        # Re-init Grad-CAM for the new model
        self._init_grad_cam(cfg["name"])
        print(f"[INFO] {display_name} loaded successfully.")

    def _init_grad_cam(self, model_name: str) -> None:
        if self.grad_cam_obj is not None:
            self.grad_cam_obj.release()
            self.grad_cam_obj = None
        try:
            target_layer = get_target_layer(self.model, model_name)
            self.grad_cam_obj = GradCAM(self.model, target_layer)
        except Exception as e:
            print(f"[WARNING] Grad-CAM init failed: {e}")
            self.grad_cam_obj = None

    # ─────────────────────────────────────────────────────────────
    #  Event Handlers
    # ─────────────────────────────────────────────────────────────
    def _on_model_change(self, event=None) -> None:
        name = self.model_var.get()
        was_running = self.running
        if was_running:
            self._stop()
        self._load_model(name)
        if was_running:
            self._start()

    def _on_sensitivity_change(self, value=None) -> None:
        self.sensitivity = self.sensitivity_var.get()

    def _on_gradcam_toggle(self) -> None:
        self.grad_cam_enabled = self.gradcam_var.get()

    def _on_test_image_toggle(self) -> None:
        self.use_test_image = self.use_test_image_var.get()
        if self.use_test_image and self.test_image_bgr is None:
            self.use_test_image = False
            self.use_test_image_var.set(False)
            print("[WARNING] No test image loaded.")
            return

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
        # Show raw preview only; processing starts with START
        self._latest_frame = self._cv2_to_tk(
            image, WEBCAM_PREVIEW_SIZE[0], WEBCAM_PREVIEW_SIZE[1],
        )
        self.webcam_canvas.configure(image=self._latest_frame, text="")
        self.webcam_canvas.image = self._latest_frame

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
                tensor = self.transform(face_pil).unsqueeze(0).to(
                    self.device
                )

                # Inference
                with torch.no_grad():
                    output = self.model(tensor)
                    prob_tensor = torch.softmax(output, dim=1)[0]

                for i, cls in enumerate(self.class_names):
                    probs_dict[cls] = float(prob_tensor[i])

                best_idx = prob_tensor.argmax().item()
                best_label = self.class_names[best_idx]
                best_conf = float(prob_tensor[best_idx])

                # Grad-CAM overlay on the face region
                if (
                    self.grad_cam_enabled
                    and self.grad_cam_obj is not None
                    and best_conf >= self.sensitivity
                ):
                    try:
                        tensor_gc = tensor.clone().requires_grad_(True)
                        heatmap = self.grad_cam_obj.generate(
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

                # Draw bounding box
                color = EMOTION_BGR.get(best_label, (0, 255, 0))
                cv2.rectangle(
                    display_frame, (x, y), (x + w, y + h), color, 2,
                )
                cv2.putText(
                    display_frame, "Face detected",
                    (x, y - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 1,
                )
                text = f"{best_label.upper()} -- {best_conf:.1%}"
                cv2.putText(
                    display_frame, text,
                    (x, y + h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    color, 2,
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
    def _start(self) -> None:
        if self.running:
            return

        # Read UDP settings
        try:
            self.udp_port = int(self.port_var.get())
        except ValueError:
            self.udp_port = DEFAULT_UDP_PORT
        self.udp_ip = self.host_var.get() or DEFAULT_UDP_IP

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
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("[ERROR] Cannot open webcam.")
                return
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_PREVIEW_SIZE[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_PREVIEW_SIZE[1])
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

        # Start Spout receiver thread
        if self._spout_available:
            self._spout_thread = threading.Thread(
                target=self._spout_receive_loop, daemon=True,
            )
            self._spout_thread.start()

        # Start GUI refresh
        self._refresh_gui()

    def _stop(self) -> None:
        self.running = False
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
        for emotion, widgets in self.prob_bars.items():
            widgets["pct"].configure(text="0%")
            widgets["fill"].place_configure(relwidth=0.0)
        self.dominant_label.configure(text="---", fg="#888888")
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

        while self.running:
            if self.use_test_image and self.test_image_bgr is not None:
                frame = self.test_image_bgr.copy()
            else:
                if self.cap is None:
                    break
                ret, frame = self.cap.read()
                if not ret:
                    break
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
                    tensor = self.transform(face_pil).unsqueeze(0).to(
                        self.device
                    )

                    # Inference
                    with torch.no_grad():
                        output = self.model(tensor)
                        prob_tensor = torch.softmax(output, dim=1)[0]

                    for i, cls in enumerate(self.class_names):
                        probs_dict[cls] = float(prob_tensor[i])

                    best_idx = prob_tensor.argmax().item()
                    best_label = self.class_names[best_idx]
                    best_conf = float(prob_tensor[best_idx])

                    # Grad-CAM: every 3rd frame, reuse cached otherwise
                    if (
                        self.grad_cam_enabled
                        and self.grad_cam_obj is not None
                        and best_conf >= self.sensitivity
                    ):
                        if self._frame_count % 3 == 1 or self._cached_gradcam is None:
                            try:
                                tensor_gc = tensor.clone().requires_grad_(True)
                                heatmap = self.grad_cam_obj.generate(
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

                    # Draw bounding box on display frame
                    color = EMOTION_BGR.get(best_label, (0, 255, 0))
                    cv2.rectangle(
                        display_frame, (x, y), (x + w, y + h), color, 2,
                    )
                    # "Face detected" label
                    cv2.putText(
                        display_frame, "Face detected",
                        (x, y - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 1,
                    )
                    # Emotion label
                    text = f"{best_label.upper()} -- {best_conf:.1%}"
                    cv2.putText(
                        display_frame, text,
                        (x, y + h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        color, 2,
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

            if self.use_test_image:
                time.sleep(0.03)

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
            # Also update center panel overlay
            self.viz_emotion_label.configure(
                text=self.dominant_emotion.upper(),
                fg=color,
            )
        else:
            self.dominant_label.configure(text="---", fg="#888888")
            self.viz_emotion_label.configure(text="")

        # Update center panel with Spout frames (if available)
        if self._spout_available and self._latest_spout_frame is not None:
            canvas_w, canvas_h = SPOUT_DISPLAY_SIZE
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
            self.viz_emotion_label.configure(text="")
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

        # Schedule next refresh (~33ms = ~30fps)
        self.root.after(33, self._refresh_gui)

    # ─────────────────────────────────────────────────────────────
    #  Cleanup
    # ─────────────────────────────────────────────────────────────
    def _on_close(self) -> None:
        self.running = False
        if self.cap is not None:
            self.cap.release()
        if self.grad_cam_obj is not None:
            self.grad_cam_obj.release()
        self.udp_sock.close()
        self.root.destroy()


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
