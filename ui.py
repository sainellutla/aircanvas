from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from config import (
    FRAME_H,
    FRAME_W,
    HOVER_FRAMES,
    PALETTE_COLORS,
    PALETTE_NAMES,
    PALETTE_X0,
    PALETTE_Y0,
    SWATCH_GAP,
    SWATCH_SIZE,
    UI_PANEL_H,
)

# Eraser swatch lives right after the colour swatches
_N_SWATCHES = len(PALETTE_COLORS) + 1   # +1 for eraser


def _swatch_rect(i: int) -> Tuple[int, int, int, int]:
    x = PALETTE_X0 + i * (SWATCH_SIZE + SWATCH_GAP)
    return x, PALETTE_Y0, x + SWATCH_SIZE, PALETTE_Y0 + SWATCH_SIZE


# Slider for brush size — placed to the right of the palette
_SLIDER_X0 = PALETTE_X0 + _N_SWATCHES * (SWATCH_SIZE + SWATCH_GAP) + 15
_SLIDER_Y0 = PALETTE_Y0 + 2
_SLIDER_W  = 150
_SLIDER_H  = 22

# Opacity slider below brush slider
_OPA_Y0 = _SLIDER_Y0 + _SLIDER_H + 10

# Clear button
_BTN_W = 70
_BTN_H = 30
_BTN_X0 = FRAME_W - _BTN_W - 12
_BTN_Y0 = PALETTE_Y0

GESTURE_GUIDE = [
    "INDEX TIP      draw cursor",
    "PINCH          pause / resume",
    "FIST (hold)    clear canvas",
    "SWIPE L        undo",
    "SWIPE R        redo",
    "OPEN PALM      eraser",
    "SPREAD         brush size",
    "THUMB UP/DN    opacity",
    "3-FINGER HOLD  save PNG",
    "PINKY ONLY     bg toggle",
    "2-FINGER HOLD  cycle shape",
    "2nd HAND       palette ctrl",
    "G key          toggle guide",
    "B / F / Q      bg / fps / quit",
]


@dataclass
class UIState:
    color: Tuple = (0, 0, 255)
    color_name: str = "Red"
    brush_size: int = 8
    opacity: float = 1.0
    draw_active: bool = False
    shape_mode: str = "free"
    eraser_active: bool = False
    show_guide: bool = False
    show_fps: bool = True
    bg_visible: bool = True
    fps: float = 0.0
    flash_msg: str = ""
    flash_count: int = 0
    hover_counts: Dict[str, int] = None


class UIRenderer:

    def render(
        self,
        frame: np.ndarray,
        state: UIState,
        cursor_px: Tuple[int, int],
        control_cursor_px: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        out = frame.copy()
        hover = state.hover_counts or {}

        self._draw_panel_bg(out)
        self._draw_palette(out, state.color, state.eraser_active, hover, cursor_px)
        self._draw_brush_slider(out, state.brush_size, hover, cursor_px)
        self._draw_opacity_slider(out, state.opacity, hover, cursor_px)
        self._draw_clear_btn(out, hover, cursor_px)
        self._draw_mode_label(out, state)
        self._draw_cursor_overlay(out, cursor_px, state)
        if control_cursor_px:
            self._draw_control_cursor(out, control_cursor_px)
        if state.show_guide:
            self._draw_guide(out)
        if state.show_fps:
            self._draw_fps(out, state.fps)
        if state.flash_count > 0:
            self._draw_flash(out, state.flash_msg)
        return out

    # ------------------------------------------------------------------

    def _draw_panel_bg(self, frame: np.ndarray) -> None:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (FRAME_W, UI_PANEL_H), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    def _draw_palette(
        self,
        frame: np.ndarray,
        active_color: Tuple,
        eraser_active: bool,
        hover: Dict,
        cursor_px: Tuple,
    ) -> None:
        cx, cy = cursor_px
        for i in range(_N_SWATCHES):
            x0, y0, x1, y1 = _swatch_rect(i)
            key = f"swatch_{i}"
            dwell = hover.get(key, 0)
            progress = min(dwell / HOVER_FRAMES, 1.0)

            if i < len(PALETTE_COLORS):
                color = PALETTE_COLORS[i]
                cv2.rectangle(frame, (x0, y0), (x1, y1), color, -1)
                # selection ring
                is_selected = (color == active_color and not eraser_active)
            else:
                # Eraser swatch
                cv2.rectangle(frame, (x0, y0), (x1, y1), (200, 200, 200), -1)
                cv2.putText(frame, "E", (x0 + 16, y1 - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 50, 50), 2)
                is_selected = eraser_active

            border_color = (255, 255, 255) if is_selected else (100, 100, 100)
            border_w = 3 if is_selected else 1
            cv2.rectangle(frame, (x0, y0), (x1, y1), border_color, border_w)

            # Dwell progress arc
            if 0 < progress < 1.0:
                angle = int(360 * progress)
                cx_s, cy_s = (x0 + x1) // 2, (y0 + y1) // 2
                cv2.ellipse(frame, (cx_s, cy_s), (SWATCH_SIZE // 2 - 2, SWATCH_SIZE // 2 - 2),
                            -90, 0, angle, (255, 255, 0), 2)

    def _draw_brush_slider(
        self,
        frame: np.ndarray,
        brush_size: int,
        hover: Dict,
        cursor_px: Tuple,
    ) -> None:
        x0, y0 = _SLIDER_X0, _SLIDER_Y0
        x1, y1 = x0 + _SLIDER_W, y0 + _SLIDER_H

        cv2.rectangle(frame, (x0, y0), (x1, y1), (70, 70, 70), -1)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (150, 150, 150), 1)

        from config import BRUSH_MAX, BRUSH_MIN
        fill_w = int((brush_size - BRUSH_MIN) / (BRUSH_MAX - BRUSH_MIN) * _SLIDER_W)
        cv2.rectangle(frame, (x0, y0), (x0 + fill_w, y1), (200, 200, 100), -1)

        label = f"Size: {brush_size}"
        cv2.putText(frame, label, (x0 + 4, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)

        cv2.putText(frame, "BRUSH", (x0, y0 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

    def _draw_opacity_slider(
        self,
        frame: np.ndarray,
        opacity: float,
        hover: Dict,
        cursor_px: Tuple,
    ) -> None:
        x0, y0 = _SLIDER_X0, _OPA_Y0
        x1, y1 = x0 + _SLIDER_W, y0 + _SLIDER_H

        cv2.rectangle(frame, (x0, y0), (x1, y1), (70, 70, 70), -1)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (150, 150, 150), 1)

        fill_w = int(opacity * _SLIDER_W)
        cv2.rectangle(frame, (x0, y0), (x0 + fill_w, y1), (100, 200, 200), -1)

        label = f"Opacity: {int(opacity * 100)}%"
        cv2.putText(frame, label, (x0 + 4, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)

        cv2.putText(frame, "OPACITY", (x0, y0 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

    def _draw_clear_btn(
        self,
        frame: np.ndarray,
        hover: Dict,
        cursor_px: Tuple,
    ) -> None:
        x0, y0, x1, y1 = _BTN_X0, _BTN_Y0, _BTN_X0 + _BTN_W, _BTN_Y0 + _BTN_H
        hovered = hover.get("btn_clear", 0) > 0
        bg = (60, 40, 40) if not hovered else (80, 60, 60)
        cv2.rectangle(frame, (x0, y0), (x1, y1), bg, -1)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (200, 100, 100), 1)
        cv2.putText(frame, "CLEAR", (x0 + 6, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 180, 180), 1)

    def _draw_mode_label(self, frame: np.ndarray, state: UIState) -> None:
        mode = "DRAW" if state.draw_active else "MOVE"
        tool = "ERASE" if state.eraser_active else state.shape_mode.upper()
        label = f"{mode} | {tool} | {state.color_name}"

        # Semi-transparent strip at bottom
        strip_h = 28
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, FRAME_H - strip_h), (FRAME_W, FRAME_H), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        cv2.putText(frame, label, (10, FRAME_H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (220, 220, 220), 1, cv2.LINE_AA)

    def _draw_cursor_overlay(
        self, frame: np.ndarray, cursor_px: Tuple, state: UIState
    ) -> None:
        cx, cy = cursor_px
        color = state.color if state.color else (200, 200, 200)

        if state.eraser_active:
            # Large hollow circle with ✕ — always shown when palm is open
            r = max(state.brush_size * 2, 16)
            cv2.circle(frame, (cx, cy), r, (255, 255, 255), 2, cv2.LINE_AA)
            d = int(r * 0.55)
            cv2.line(frame, (cx - d, cy - d), (cx + d, cy + d), (255, 80, 80), 2, cv2.LINE_AA)
            cv2.line(frame, (cx + d, cy - d), (cx - d, cy + d), (255, 80, 80), 2, cv2.LINE_AA)

        elif not state.draw_active:
            # Idle / paused: crosshair + optional dim shape icon
            if state.shape_mode != "free":
                self._draw_shape_icon(frame, cx, cy, state.shape_mode, color, dim=True)
            cv2.line(frame, (cx - 12, cy), (cx + 12, cy), (200, 200, 200), 1, cv2.LINE_AA)
            cv2.line(frame, (cx, cy - 12), (cx, cy + 12), (200, 200, 200), 1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 3, (200, 200, 200), -1)

        else:
            # Active drawing — dispatch by shape mode
            mode = state.shape_mode
            if mode == "free":
                # Brush: filled dot in current colour + white border
                r = max(state.brush_size, 4)
                cv2.circle(frame, (cx, cy), r, color, -1, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), r + 2, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                self._draw_shape_icon(frame, cx, cy, mode, color, dim=False)

    def _draw_shape_icon(
        self,
        frame: np.ndarray,
        cx: int,
        cy: int,
        mode: str,
        color: Tuple,
        dim: bool = False,
    ) -> None:
        """Draw a small mode-specific icon centred on (cx, cy)."""
        # Dim colours for idle state
        c = tuple(max(0, int(v * 0.45)) for v in color) if dim else color
        white = (120, 120, 120) if dim else (255, 255, 255)
        thick = 1 if dim else 2

        if mode == "line":
            cv2.line(frame, (cx - 16, cy), (cx + 16, cy), c, thick, cv2.LINE_AA)
            cv2.circle(frame, (cx - 16, cy), 3, white, -1)
            cv2.circle(frame, (cx + 16, cy), 3, white, -1)

        elif mode == "circle":
            cv2.circle(frame, (cx, cy), 18, c, thick, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 2, white, -1)

        elif mode == "rect":
            cv2.rectangle(frame, (cx - 18, cy - 12), (cx + 18, cy + 12), c, thick)
            cv2.circle(frame, (cx - 18, cy - 12), 3, white, -1)

    def _draw_control_cursor(
        self, frame: np.ndarray, pos: Tuple
    ) -> None:
        cv2.circle(frame, pos, 8, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(frame, pos, 2, (0, 255, 255), -1)

    def _draw_guide(self, frame: np.ndarray) -> None:
        panel_w, panel_h = 340, len(GESTURE_GUIDE) * 22 + 20
        px, py = (FRAME_W - panel_w) // 2, (FRAME_H - panel_h) // 2
        overlay = frame.copy()
        cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (px, py), (px + panel_w, py + panel_h), (150, 150, 150), 1)
        cv2.putText(frame, "GESTURE GUIDE", (px + 10, py + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 50), 1)
        for i, line in enumerate(GESTURE_GUIDE):
            cv2.putText(frame, line, (px + 10, py + 34 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

    def _draw_fps(self, frame: np.ndarray, fps: float) -> None:
        color = (0, 220, 0) if fps >= 25 else (0, 200, 200) if fps >= 15 else (0, 0, 220)
        cv2.putText(frame, f"FPS: {fps:.0f}", (FRAME_W - 110, UI_PANEL_H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    def _draw_flash(self, frame: np.ndarray, msg: str) -> None:
        (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        tx = (FRAME_W - tw) // 2
        ty = FRAME_H // 2
        cv2.putText(frame, msg, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 255, 100), 2, cv2.LINE_AA)
