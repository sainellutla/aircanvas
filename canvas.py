import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config import CANVAS_H, CANVAS_W, EXPORT_DIR, MAX_UNDO_STACK
from shape_tools import ShapeTool


@dataclass
class Stroke:
    points: List[Tuple[int, int]]
    color: Optional[Tuple[int, int, int]]   # BGR; None = eraser
    size: int
    opacity: float
    tool: str                               # "brush" | "eraser" | "line" | "circle" | "rect"
    smoothed_points: Optional[List[Tuple[int, int]]] = None
    shape_params: Optional[dict] = None


def _render_stroke(layer: np.ndarray, stroke: Stroke) -> None:
    alpha_val = int(stroke.opacity * 255)
    color_bgra = (*stroke.color, alpha_val) if stroke.color is not None else (0, 0, 0, 0)
    thickness = max(1, stroke.size * 2)

    if stroke.tool == "eraser":
        mask = np.zeros(layer.shape[:2], dtype=np.uint8)
        for pt in stroke.points:
            cv2.circle(mask, pt, stroke.size * 2, 255, -1)
        layer[mask > 0] = [0, 0, 0, 0]
        return

    if stroke.tool == "brush":
        pts = stroke.smoothed_points if stroke.smoothed_points else stroke.points
        if len(pts) == 1:
            cv2.circle(layer, pts[0], stroke.size, color_bgra, -1, cv2.LINE_AA)
        for i in range(1, len(pts)):
            cv2.line(layer, pts[i - 1], pts[i], color_bgra, thickness, cv2.LINE_AA)
        return

    if stroke.tool == "line" and stroke.shape_params:
        cv2.line(layer, stroke.shape_params["p1"], stroke.shape_params["p2"],
                 color_bgra, stroke.size, cv2.LINE_AA)
        return

    if stroke.tool == "circle" and stroke.shape_params:
        cv2.circle(layer, stroke.shape_params["center"], stroke.shape_params["radius"],
                   color_bgra, stroke.size, cv2.LINE_AA)
        return

    if stroke.tool == "rect" and stroke.shape_params:
        cv2.rectangle(layer, stroke.shape_params["tl"], stroke.shape_params["br"],
                      color_bgra, stroke.size)
        return


class Canvas:
    def __init__(self, w: int = CANVAS_W, h: int = CANVAS_H) -> None:
        self.w = w
        self.h = h
        self._canvas_layer = np.zeros((h, w, 4), dtype=np.uint8)
        self._preview_layer = np.zeros((h, w, 4), dtype=np.uint8)
        self._undo_stack: List[Stroke] = []
        self._redo_stack: List[Stroke] = []

        self._current_stroke: Optional[Stroke] = None
        self._prev_pt: Optional[Tuple[int, int]] = None

    # ------------------------------------------------------------------
    # Stroke lifecycle
    # ------------------------------------------------------------------

    def begin_stroke(
        self,
        x: int,
        y: int,
        color: Optional[Tuple[int, int, int]],
        size: int,
        opacity: float,
        tool: str,
    ) -> None:
        self._current_stroke = Stroke(
            points=[(x, y)],
            color=color,
            size=size,
            opacity=opacity,
            tool=tool,
        )
        self._prev_pt = (x, y)
        self._preview_layer[:] = 0
        # Draw initial dot immediately so the stroke is visible from the first frame
        stroke = self._current_stroke
        if stroke.tool == "brush" and stroke.color:
            alpha_val = int(stroke.opacity * 255)
            cv2.circle(self._preview_layer, (x, y), stroke.size,
                       (*stroke.color, alpha_val), -1, cv2.LINE_AA)
        elif stroke.tool == "eraser":
            mask = np.zeros(self._preview_layer.shape[:2], dtype=np.uint8)
            cv2.circle(mask, (x, y), stroke.size * 2, 255, -1)
            self._preview_layer[mask > 0] = [0, 0, 0, 128]

    def extend_stroke(self, x: int, y: int) -> None:
        stroke = self._current_stroke
        if stroke is None:
            return
        prev_pt = self._prev_pt
        stroke.points.append((x, y))

        if stroke.tool in ("brush", "eraser"):
            # Accumulate on preview layer — never zero it for brush/eraser
            self._draw_preview_segment(prev_pt, (x, y))
        else:
            # Shape tools: zero and redraw the live-snapped shape each frame
            self._preview_layer[:] = 0
            self._redraw_shape_preview()

        self._prev_pt = (x, y)

    def commit_stroke(self, shape_mode: str = "free") -> None:
        if self._current_stroke is None:
            return
        stroke = self._current_stroke
        self._current_stroke = None
        self._preview_layer[:] = 0

        if not stroke.points:
            return

        # Apply shape snapping / smoothing
        if stroke.tool == "brush" and len(stroke.points) >= 4:
            stroke.smoothed_points = ShapeTool.smooth_stroke(stroke.points)
        elif stroke.tool == "line" and len(stroke.points) >= 2:
            p1, p2 = ShapeTool.snap_line(stroke.points)
            stroke.shape_params = {"p1": p1, "p2": p2}
        elif stroke.tool == "circle" and len(stroke.points) >= 2:
            center, r = ShapeTool.snap_circle(stroke.points)
            stroke.shape_params = {"center": center, "radius": r}
        elif stroke.tool == "rect" and len(stroke.points) >= 2:
            tl, br = ShapeTool.snap_rect(stroke.points)
            stroke.shape_params = {"tl": tl, "br": br}

        _render_stroke(self._canvas_layer, stroke)

        self._undo_stack.append(stroke)
        self._redo_stack.clear()

        # Cap undo history
        while len(self._undo_stack) > MAX_UNDO_STACK:
            self._undo_stack.pop(0)

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def undo(self) -> bool:
        if not self._undo_stack:
            return False
        stroke = self._undo_stack.pop()
        self._redo_stack.append(stroke)
        self._rebuild_canvas()
        return True

    def redo(self) -> bool:
        if not self._redo_stack:
            return False
        stroke = self._redo_stack.pop()
        self._undo_stack.append(stroke)
        self._rebuild_canvas()
        return True

    def clear(self) -> None:
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._canvas_layer[:] = 0
        self._preview_layer[:] = 0
        self._current_stroke = None

    # ------------------------------------------------------------------
    # Compositing
    # ------------------------------------------------------------------

    def composite(self, bg_frame=None) -> np.ndarray:
        if bg_frame is not None:
            display = cv2.resize(bg_frame, (self.w, self.h))
        else:
            display = np.full((self.h, self.w, 3), 30, dtype=np.uint8)

        display = self._blend_layer(display, self._canvas_layer)
        display = self._blend_layer(display, self._preview_layer)
        return display

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None) -> str:
        os.makedirs(EXPORT_DIR, exist_ok=True)
        if path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(EXPORT_DIR, f"canvas_{ts}.png")
        if path.lower().endswith(".png"):
            cv2.imwrite(path, self._canvas_layer)
        else:
            white = np.full((self.h, self.w, 3), 255, dtype=np.uint8)
            out = self._blend_layer(white, self._canvas_layer)
            cv2.imwrite(path, out)
        return path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _draw_preview_segment(
        self, p0: Tuple[int, int], p1: Tuple[int, int]
    ) -> None:
        """Append one brush or eraser segment to the preview layer without clearing it."""
        stroke = self._current_stroke
        if stroke is None:
            return
        if stroke.tool == "brush" and stroke.color:
            alpha_val = int(stroke.opacity * 255)
            color_bgra = (*stroke.color, alpha_val)
            thickness = max(1, stroke.size * 2)
            cv2.line(self._preview_layer, p0, p1, color_bgra, thickness, cv2.LINE_AA)
        elif stroke.tool == "eraser":
            mask = np.zeros(self._preview_layer.shape[:2], dtype=np.uint8)
            cv2.line(mask, p0, p1, 255, stroke.size * 2)
            cv2.circle(mask, p1, stroke.size * 2, 255, -1)
            self._preview_layer[mask > 0] = [0, 0, 0, 128]

    def _redraw_shape_preview(self) -> None:
        """Clear preview and redraw the live-snapped shape (called every frame for shapes)."""
        stroke = self._current_stroke
        if stroke is None:
            return
        alpha_val = int(stroke.opacity * 255)
        color_bgra = (*stroke.color, alpha_val) if stroke.color else (0, 0, 0, 0)
        pts = stroke.points

        if stroke.tool == "line" and len(pts) >= 2:
            p1, p2 = ShapeTool.snap_line(pts)
            cv2.line(self._preview_layer, p1, p2, color_bgra, stroke.size, cv2.LINE_AA)
        elif stroke.tool == "circle" and len(pts) >= 2:
            center, r = ShapeTool.snap_circle(pts)
            cv2.circle(self._preview_layer, center, r,
                       color_bgra, stroke.size, cv2.LINE_AA)
        elif stroke.tool == "rect" and len(pts) >= 2:
            tl, br = ShapeTool.snap_rect(pts)
            cv2.rectangle(self._preview_layer, tl, br, color_bgra, stroke.size)

    def _rebuild_canvas(self) -> None:
        self._canvas_layer[:] = 0
        for stroke in self._undo_stack:
            _render_stroke(self._canvas_layer, stroke)

    @staticmethod
    def _blend_layer(bg: np.ndarray, layer: np.ndarray) -> np.ndarray:
        alpha = layer[:, :, 3:4].astype(np.float32) / 255.0
        rgb = layer[:, :, :3].astype(np.float32)
        out = (rgb * alpha + bg.astype(np.float32) * (1.0 - alpha))
        return np.clip(out, 0, 255).astype(np.uint8)

    @property
    def is_drawing(self) -> bool:
        return self._current_stroke is not None
