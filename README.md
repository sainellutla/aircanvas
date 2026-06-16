# Air Canvas

Draw in the air using hand gestures tracked through your webcam. Built with Python, OpenCV, and MediaPipe.

## Features

- Real-time hand tracking via MediaPipe Hands
- Freehand drawing with live stroke preview
- 8-color palette + eraser, selectable by hovering
- Adjustable brush size and opacity via gestures
- Undo / redo with swipe gestures
- Shape tools — snap freehand strokes to lines, circles, and rectangles
- Two-hand support — control hand manages palette while drawing hand draws
- Background toggle — draw over the live webcam feed or a solid canvas
- Save canvas as PNG with transparency preserved
- FPS display and gesture guide overlay
- Mode-specific cursors (brush, eraser, line, circle, rect, idle)

## Requirements

- Python 3.10+
- Webcam

## Installation

```bash
pip install -r requirements.txt
```

Dependencies: `opencv-python`, `mediapipe`, `numpy`

## Running

```bash
python main.py
```

On first launch the MediaPipe hand landmark model (~30 MB) is downloaded automatically to `models/`. Subsequent launches start immediately.

## Controls

### Gestures

| Gesture | Action |
|---|---|
| Index finger tip | Drawing cursor |
| **Pinch** (index + thumb) | Pause / resume drawing |
| **Open palm** (4+ fingers up) | Eraser mode |
| **Fist** — hold 12 frames | Clear canvas |
| **Swipe left** — while paused | Undo |
| **Swipe right** — while paused | Redo |
| **3-finger salute** — hold 15 frames | Save PNG to `exports/` |
| **Pinky only up** — hold 10 frames | Toggle webcam background |
| **2-finger hold** — hold 10 frames, while paused | Cycle shape mode (free → line → circle → rect) |
| **Thumb–index spread** | Brush size (continuous) |
| **Thumb only up** + vertical position | Opacity (up = opaque, down = transparent) |
| **Hover over colour swatch** — 8 frames | Select colour |
| **Second hand** index tip | Controls palette independently |

### Keyboard Shortcuts

| Key | Action |
|---|---|
| `Q` | Quit |
| `S` | Save PNG |
| `Z` | Undo |
| `Y` | Redo |
| `C` | Clear canvas |
| `G` | Toggle gesture guide overlay |
| `B` | Toggle webcam background |
| `F` | Toggle FPS display |

## Project Structure

```
aircanvas/
├── main.py              # Entry point and main loop
├── config.py            # Tunable constants (thresholds, colors, geometry)
├── hand_tracker.py      # MediaPipe HandLandmarker wrapper
├── gesture_detector.py  # Interprets landmarks into drawing commands
├── canvas.py            # Canvas layers, stroke history, undo/redo, save
├── ui.py                # Palette, sliders, overlays, per-mode cursors
├── shape_tools.py       # Catmull-Rom smoothing + shape snapping
└── requirements.txt
```

## How It Works

Each frame goes through five steps:

1. **Capture** — webcam frame is horizontally mirrored
2. **Detect** — MediaPipe identifies hand landmarks (21 points per hand)
3. **Interpret** — `GestureDetector` maps landmark positions to drawing commands and events
4. **Draw** — `Canvas` accumulates strokes on a BGRA layer; shape tools snap on commit
5. **Render** — canvas composited over background, UI drawn on top

Strokes are drawn to a preview layer in real time and committed to the canvas layer on finger lift, at which point Catmull-Rom smoothing is applied to brush strokes.

## Saved Files

Canvases are saved to `exports/canvas_YYYYMMDD_HHMMSS.png` with the alpha channel preserved. Export the file over a white background by saving as `.jpg` instead.
