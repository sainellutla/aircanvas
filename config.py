import os

# --- Canvas / Frame ---
FRAME_W, FRAME_H = 1280, 720
CANVAS_W, CANVAS_H = FRAME_W, FRAME_H
TARGET_FPS = 30

# --- MediaPipe landmark indices ---
WRIST       = 0
THUMB_CMC   = 1; THUMB_MCP = 2; THUMB_IP = 3; THUMB_TIP = 4
INDEX_MCP   = 5; INDEX_PIP = 6; INDEX_DIP = 7;  INDEX_TIP = 8
MIDDLE_MCP  = 9; MIDDLE_PIP = 10; MIDDLE_DIP = 11; MIDDLE_TIP = 12
RING_MCP    = 13; RING_PIP = 14; RING_DIP = 15; RING_TIP = 16
PINKY_MCP   = 17; PINKY_PIP = 18; PINKY_DIP = 19; PINKY_TIP = 20

# --- Gesture thresholds ---
PINCH_THRESHOLD    = 0.055   # normalized distance index-tip <-> thumb-tip → pause draw
SPREAD_MIN         = 0.04    # thumb-index spread → min brush size
SPREAD_MAX         = 0.22    # thumb-index spread → max brush size
BRUSH_MIN          = 2
BRUSH_MAX          = 40
HOVER_FRAMES       = 8       # frames cursor must dwell over UI element to activate
FIST_FRAMES        = 12      # frames fist must be held to clear
SAVE_FRAMES        = 15      # frames 3-finger salute to save
BG_TOGGLE_FRAMES   = 10      # frames pinky-only to toggle background
SHAPE_CYCLE_FRAMES = 10      # frames 2-finger hold to cycle shape mode
SWIPE_PX           = 80      # pixels displacement to register undo/redo swipe
SWIPE_FRAMES       = 8       # window of frames for swipe
STICKY_HAND_FRAMES = 15      # frames to keep hand role assignment stable
MAX_UNDO_STACK     = 200     # cap on undo history
SMOOTH_ALPHA       = 0.75    # EMA weight for cursor smoothing (Catmull-Rom at commit handles aesthetics)

# --- Palette (BGR) ---
PALETTE_COLORS = [
    (0,   0,   255),   # Red
    (0,  128,  255),   # Orange
    (0,  255,  255),   # Yellow
    (0,  255,    0),   # Green
    (255,  0,    0),   # Blue
    (255,  0,  255),   # Magenta
    (255, 255, 255),   # White
    (0,   0,     0),   # Black
]
PALETTE_NAMES = ["Red", "Orange", "Yellow", "Green", "Blue", "Magenta", "White", "Black"]
ERASER_COLOR  = None   # sentinel

# --- UI geometry ---
UI_PANEL_H    = 90     # top strip height reserved for palette + controls
PALETTE_X0    = 10     # x of first swatch
PALETTE_Y0    = 10     # y of first swatch
SWATCH_SIZE   = 55     # width/height of each colour swatch
SWATCH_GAP    = 5      # gap between swatches
ERASER_X      = None   # computed in ui.py based on palette end

# Brush size slider (right of palette)
SLIDER_BRUSH_X1 = None   # computed in ui.py
SLIDER_Y0       = 15
SLIDER_H        = 25
SLIDER_W        = 160

# --- Opacity ---
OPACITY_DEFAULT = 1.0

# --- Shape modes ---
SHAPE_MODES = ["free", "line", "circle", "rect"]

# --- MediaPipe model ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "hand_landmarker.task")
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
NUM_HANDS  = 2

# --- Export ---
EXPORT_DIR = os.path.join(os.path.dirname(__file__), "exports")
