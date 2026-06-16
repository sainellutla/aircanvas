import time

import cv2

from canvas import Canvas
from config import (
    CANVAS_H,
    CANVAS_W,
    FRAME_H,
    FRAME_W,
    PALETTE_COLORS,
    PALETTE_NAMES,
    TARGET_FPS,
    UI_PANEL_H,
)
from gesture_detector import GestureDetector, GestureState
from hand_tracker import HandTracker
from ui import UIRenderer, UIState


def main() -> None:
    print("Initialising Air Canvas …")
    tracker  = HandTracker()
    detector = GestureDetector(GestureState())
    canvas   = Canvas(CANVAS_W, CANVAS_H)
    renderer = UIRenderer()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    ui = UIState(
        color=PALETTE_COLORS[0],
        color_name=PALETTE_NAMES[0],
    )

    was_drawing    = False
    flash_duration = 60   # frames
    fps_history    = []

    print("Air Canvas running. Press Q to quit.")
    print("  G = gesture guide   B = background   F = fps   S = save")
    print("  Z = undo   Y = redo   C = clear")

    while True:
        frame_start = time.time()

        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        ts_ms = int(time.time() * 1000)

        hands  = tracker.detect(frame, ts_ms)
        result = detector.process(hands, (FRAME_W, FRAME_H))

        # ── Canvas dispatch ──────────────────────────────────────────────
        tool = (
            "eraser" if result.eraser_active
            else result.shape_mode if result.shape_mode != "free"
            else "brush"
        )

        if result.draw_active:
            cx, cy = result.cursor_px
            if not was_drawing:
                canvas.begin_stroke(cx, cy,
                                    color=None if result.eraser_active else result.color,
                                    size=result.brush_size,
                                    opacity=result.opacity,
                                    tool=tool)
            else:
                canvas.extend_stroke(cx, cy)
            was_drawing = True
        else:
            if was_drawing:
                canvas.commit_stroke(shape_mode=result.shape_mode)
            was_drawing = False

        # ── One-shot gesture events ──────────────────────────────────────
        if result.event == "clear":
            canvas.clear()
            ui.flash_msg = "CLEARED"
            ui.flash_count = flash_duration

        elif result.event == "undo":
            if canvas.undo():
                ui.flash_msg = "UNDO"
                ui.flash_count = flash_duration

        elif result.event == "redo":
            if canvas.redo():
                ui.flash_msg = "REDO"
                ui.flash_count = flash_duration

        elif result.event == "save":
            path = canvas.save()
            print(f"Saved: {path}")
            ui.flash_msg = "SAVED"
            ui.flash_count = flash_duration

        # ── Keyboard ────────────────────────────────────────────────────
        elapsed    = time.time() - frame_start
        wait_ms    = max(1, int((1 / TARGET_FPS - elapsed) * 1000))
        key        = cv2.waitKey(wait_ms) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("s"):
            path = canvas.save()
            print(f"Saved: {path}")
            ui.flash_msg = "SAVED"
            ui.flash_count = flash_duration
        elif key == ord("z"):
            if canvas.undo():
                ui.flash_msg = "UNDO"
                ui.flash_count = flash_duration
        elif key == ord("y"):
            if canvas.redo():
                ui.flash_msg = "REDO"
                ui.flash_count = flash_duration
        elif key == ord("c"):
            canvas.clear()
            ui.flash_msg = "CLEARED"
            ui.flash_count = flash_duration
        elif key == ord("g"):
            ui.show_guide = not ui.show_guide
        elif key == ord("b"):
            ui.bg_visible = not ui.bg_visible
        elif key == ord("f"):
            ui.show_fps = not ui.show_fps

        # Keep bg_visible in sync with gesture result
        if result.event not in ("undo", "redo", "save", "clear"):
            ui.bg_visible = result.bg_visible

        # ── Sync UI state from gesture result ────────────────────────────
        ui.color        = result.color or (200, 200, 200)
        ui.color_name   = result.color_name
        ui.brush_size   = result.brush_size
        ui.opacity      = result.opacity
        ui.draw_active  = result.draw_active
        ui.shape_mode   = result.shape_mode
        ui.eraser_active = result.eraser_active
        ui.show_guide   = result.show_guide or ui.show_guide
        ui.show_fps     = result.show_fps and ui.show_fps or ui.show_fps
        ui.hover_counts = detector.state.hover_counts

        # Flash countdown
        if ui.flash_count > 0:
            ui.flash_count -= 1

        # ── FPS ──────────────────────────────────────────────────────────
        frame_time = time.time() - frame_start
        if frame_time > 0:
            fps_history.append(1.0 / frame_time)
            if len(fps_history) > 30:
                fps_history.pop(0)
        ui.fps = sum(fps_history) / len(fps_history) if fps_history else 0.0

        # ── Render ───────────────────────────────────────────────────────
        bg = frame if ui.bg_visible else None
        composite = canvas.composite(bg)
        display   = renderer.render(composite, ui, result.cursor_px, result.control_cursor_px)

        cv2.imshow("Air Canvas", display)

    cap.release()
    tracker.close()
    cv2.destroyAllWindows()
    print("Air Canvas closed.")


if __name__ == "__main__":
    main()
