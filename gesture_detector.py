from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import (
    BG_TOGGLE_FRAMES,
    BRUSH_MAX,
    BRUSH_MIN,
    FIST_FRAMES,
    FRAME_H,
    FRAME_W,
    HOVER_FRAMES,
    INDEX_MCP,
    INDEX_TIP,
    MIDDLE_MCP,
    MIDDLE_TIP,
    OPACITY_DEFAULT,
    PALETTE_COLORS,
    PALETTE_NAMES,
    PALETTE_X0,
    PALETTE_Y0,
    PINCH_THRESHOLD,
    PINKY_MCP,
    PINKY_TIP,
    RING_MCP,
    RING_TIP,
    SAVE_FRAMES,
    SHAPE_CYCLE_FRAMES,
    SHAPE_MODES,
    SMOOTH_ALPHA,
    SPREAD_MAX,
    SPREAD_MIN,
    SWATCH_GAP,
    SWATCH_SIZE,
    SWIPE_FRAMES,
    SWIPE_PX,
    THUMB_IP,
    THUMB_MCP,
    THUMB_TIP,
    UI_PANEL_H,
)
from hand_tracker import HandData

STICKY_HAND_FRAMES = 15
_ERASER_SLOT = len(PALETTE_COLORS)   # index for the eraser swatch


@dataclass
class GestureState:
    draw_mode: bool = False
    shape_mode_idx: int = 0           # index into SHAPE_MODES
    color: Tuple = (0, 0, 255)        # current BGR colour
    color_name: str = "Red"
    brush_size: int = 8
    opacity: float = OPACITY_DEFAULT
    bg_visible: bool = True
    show_guide: bool = False
    show_fps: bool = True
    eraser_active: bool = False

    # Dwell counters keyed by element id
    hover_counts: Dict[str, int] = field(default_factory=dict)

    # Gesture dwell counters
    fist_count: int = 0
    save_count: int = 0
    bg_count: int = 0
    shape_cycle_count: int = 0

    # Swipe history
    x_history: deque = field(default_factory=lambda: deque(maxlen=SWIPE_FRAMES + 2))
    swipe_cooldown: int = 0           # frames before another swipe is recognised

    # Multi-hand sticky assignment
    draw_handedness: Optional[str] = None
    sticky_count: int = 0


@dataclass
class GestureResult:
    cursor_px: Tuple[int, int] = (0, 0)
    draw_active: bool = False
    brush_size: int = 8
    opacity: float = 1.0
    shape_mode: str = "free"
    color: Optional[Tuple] = (0, 0, 255)
    eraser_active: bool = False

    event: str = "idle"    # "draw"|"erase"|"idle"|"clear"|"undo"|"redo"|"save"
    bg_visible: bool = True
    show_guide: bool = False
    show_fps: bool = True
    color_name: str = "Red"

    # Second hand
    control_cursor_px: Optional[Tuple[int, int]] = None


# -------------------------------------------------------------------------
# Landmark geometry helpers
# -------------------------------------------------------------------------

def _dist_norm(lm, i, j) -> float:
    dx = lm[i].x - lm[j].x
    dy = lm[i].y - lm[j].y
    return (dx * dx + dy * dy) ** 0.5


def _finger_up(lm, tip_i: int, mcp_i: int) -> bool:
    return lm[tip_i].y < lm[mcp_i].y - 0.02


def _count_fingers_up(lm) -> int:
    pairs = [
        (INDEX_TIP, INDEX_MCP),
        (MIDDLE_TIP, MIDDLE_MCP),
        (RING_TIP, RING_MCP),
        (PINKY_TIP, PINKY_MCP),
    ]
    return sum(1 for tip, mcp in pairs if _finger_up(lm, tip, mcp))


def _is_fist(lm) -> bool:
    pairs = [
        (INDEX_TIP, INDEX_MCP),
        (MIDDLE_TIP, MIDDLE_MCP),
        (RING_TIP, RING_MCP),
        (PINKY_TIP, PINKY_MCP),
    ]
    return all(lm[tip].y > lm[mcp].y + 0.01 for tip, mcp in pairs)


def _is_pinky_only(lm) -> bool:
    pinky_up = _finger_up(lm, PINKY_TIP, PINKY_MCP)
    others_down = not any(_finger_up(lm, t, m) for t, m in [
        (INDEX_TIP, INDEX_MCP), (MIDDLE_TIP, MIDDLE_MCP), (RING_TIP, RING_MCP)
    ])
    return pinky_up and others_down


def _is_three_finger_salute(lm) -> bool:
    index_up  = _finger_up(lm, INDEX_TIP, INDEX_MCP)
    middle_up = _finger_up(lm, MIDDLE_TIP, MIDDLE_MCP)
    ring_up   = _finger_up(lm, RING_TIP, RING_MCP)
    pinky_down = not _finger_up(lm, PINKY_TIP, PINKY_MCP)
    thumb_down = lm[THUMB_TIP].y > lm[THUMB_MCP].y
    return index_up and middle_up and ring_up and pinky_down and thumb_down


def _is_two_finger_hold(lm) -> bool:
    index_up  = _finger_up(lm, INDEX_TIP, INDEX_MCP)
    middle_up = _finger_up(lm, MIDDLE_TIP, MIDDLE_MCP)
    ring_down  = not _finger_up(lm, RING_TIP, RING_MCP)
    pinky_down = not _finger_up(lm, PINKY_TIP, PINKY_MCP)
    return index_up and middle_up and ring_down and pinky_down


def _is_thumb_only(lm) -> bool:
    fingers_down = not any(_finger_up(lm, t, m) for t, m in [
        (INDEX_TIP, INDEX_MCP), (MIDDLE_TIP, MIDDLE_MCP),
        (RING_TIP, RING_MCP), (PINKY_TIP, PINKY_MCP),
    ])
    thumb_up = lm[THUMB_TIP].y < lm[THUMB_MCP].y - 0.02
    return thumb_up and fingers_down


# -------------------------------------------------------------------------
# Palette hit-testing
# -------------------------------------------------------------------------

def _palette_rects():
    rects = []
    n = len(PALETTE_COLORS) + 1   # +1 for eraser slot
    for i in range(n):
        x = PALETTE_X0 + i * (SWATCH_SIZE + SWATCH_GAP)
        rects.append((x, PALETTE_Y0, x + SWATCH_SIZE, PALETTE_Y0 + SWATCH_SIZE))
    return rects


_PALETTE_RECTS = _palette_rects()


def _point_in_rect(px, py, rect) -> bool:
    x0, y0, x1, y1 = rect
    return x0 <= px <= x1 and y0 <= py <= y1


# -------------------------------------------------------------------------
# GestureDetector
# -------------------------------------------------------------------------

class GestureDetector:
    def __init__(self, state: GestureState) -> None:
        self.state = state
        self._smooth_x: float = FRAME_W / 2
        self._smooth_y: float = FRAME_H / 2
        self._was_drawing: bool = False

    def process(
        self,
        hands: List[HandData],
        frame_wh: Tuple[int, int] = (FRAME_W, FRAME_H),
    ) -> GestureResult:
        s = self.state
        fw, fh = frame_wh
        result = GestureResult(
            shape_mode=SHAPE_MODES[s.shape_mode_idx],
            color=s.color,
            color_name=s.color_name,
            brush_size=s.brush_size,
            opacity=s.opacity,
            bg_visible=s.bg_visible,
            show_guide=s.show_guide,
            show_fps=s.show_fps,
            eraser_active=s.eraser_active,
        )

        if not hands:
            s.fist_count = 0
            s.save_count = 0
            s.bg_count = 0
            s.shape_cycle_count = 0
            s.swipe_cooldown = max(0, s.swipe_cooldown - 1)
            result.cursor_px = (int(self._smooth_x), int(self._smooth_y))
            result.draw_active = False
            result.event = "idle"
            return result

        # --- Role assignment ---
        draw_hand, ctrl_hand = self._assign_roles(hands)
        lm = draw_hand.landmarks
        px = draw_hand.px

        # --- Raw cursor (index tip) ---
        raw_x, raw_y = px[INDEX_TIP]
        self._smooth_x = SMOOTH_ALPHA * raw_x + (1 - SMOOTH_ALPHA) * self._smooth_x
        self._smooth_y = SMOOTH_ALPHA * raw_y + (1 - SMOOTH_ALPHA) * self._smooth_y
        cx, cy = int(self._smooth_x), int(self._smooth_y)
        result.cursor_px = (cx, cy)

        # --- Control hand cursor ---
        if ctrl_hand is not None:
            result.control_cursor_px = ctrl_hand.px[INDEX_TIP]

        # --- Pinch → draw/pause ---
        pinch_dist = _dist_norm(lm, INDEX_TIP, THUMB_TIP)
        draw_active = pinch_dist > PINCH_THRESHOLD

        # --- Brush size from thumb-index spread ---
        spread = _dist_norm(lm, THUMB_TIP, INDEX_TIP)
        s.brush_size = int(np.interp(spread, [SPREAD_MIN, SPREAD_MAX], [BRUSH_MIN, BRUSH_MAX]))
        result.brush_size = s.brush_size

        # --- Opacity: thumb-only gesture, y-pos ---
        if _is_thumb_only(lm):
            thumb_y = lm[THUMB_TIP].y
            s.opacity = float(np.clip(1.0 - (thumb_y - 0.1) / 0.8, 0.05, 1.0))
        result.opacity = s.opacity

        # --- Open palm = eraser ---
        fingers_up = _count_fingers_up(lm)
        is_open_palm = fingers_up >= 4

        # --- Fist = clear ---
        if _is_fist(lm):
            s.fist_count += 1
            if s.fist_count >= FIST_FRAMES:
                result.event = "clear"
                s.fist_count = 0
                return result
        else:
            s.fist_count = 0

        # --- Three-finger salute = save ---
        if _is_three_finger_salute(lm):
            s.save_count += 1
            if s.save_count >= SAVE_FRAMES:
                result.event = "save"
                s.save_count = 0
                return result
        else:
            s.save_count = 0

        # --- Pinky-only = background toggle ---
        if _is_pinky_only(lm):
            s.bg_count += 1
            if s.bg_count >= BG_TOGGLE_FRAMES:
                s.bg_visible = not s.bg_visible
                s.bg_count = 0
        else:
            s.bg_count = 0
        result.bg_visible = s.bg_visible

        # --- Two-finger hold = cycle shape mode (only while paused) ---
        if _is_two_finger_hold(lm) and not draw_active:
            s.shape_cycle_count += 1
            if s.shape_cycle_count >= SHAPE_CYCLE_FRAMES:
                s.shape_mode_idx = (s.shape_mode_idx + 1) % len(SHAPE_MODES)
                s.shape_cycle_count = 0
        else:
            s.shape_cycle_count = 0
        result.shape_mode = SHAPE_MODES[s.shape_mode_idx]

        # --- Swipe undo/redo (while paused) ---
        if not draw_active:
            s.x_history.append(cx)
            if s.swipe_cooldown == 0 and len(s.x_history) >= SWIPE_FRAMES:
                dx = s.x_history[-1] - s.x_history[0]
                if dx < -SWIPE_PX:
                    result.event = "undo"
                    s.swipe_cooldown = 20
                    return result
                elif dx > SWIPE_PX:
                    result.event = "redo"
                    s.swipe_cooldown = 20
                    return result
        else:
            s.x_history.clear()

        s.swipe_cooldown = max(0, s.swipe_cooldown - 1)

        # --- Palette / UI hover (using control hand if available, else draw hand) ---
        ui_cursor = result.control_cursor_px if result.control_cursor_px else (cx, cy)
        self._check_ui_hover(ui_cursor, result)

        # --- Eraser state ---
        s.eraser_active = is_open_palm
        result.eraser_active = s.eraser_active
        result.color = s.color
        result.color_name = s.color_name

        # --- Draw event ---
        in_canvas = cy > UI_PANEL_H
        if draw_active and in_canvas:
            result.draw_active = True
            if is_open_palm:
                result.event = "erase"
            else:
                result.event = "draw"
        else:
            result.draw_active = False
            result.event = "idle"

        return result

    # ------------------------------------------------------------------

    def _assign_roles(
        self, hands: List[HandData]
    ) -> Tuple[HandData, Optional[HandData]]:
        s = self.state
        if len(hands) == 1:
            s.draw_handedness = hands[0].handedness
            s.sticky_count = 0
            return hands[0], None

        # Two hands
        right = next((h for h in hands if h.handedness == "Right"), None)
        left  = next((h for h in hands if h.handedness == "Left"),  None)

        if s.draw_handedness is None:
            s.draw_handedness = "Right" if right else hands[0].handedness

        draw_h = next((h for h in hands if h.handedness == s.draw_handedness), hands[0])
        ctrl_h = next((h for h in hands if h is not draw_h), None)
        return draw_h, ctrl_h

    def _check_ui_hover(
        self, cursor: Tuple[int, int], result: GestureResult
    ) -> None:
        s = self.state
        cx, cy = cursor
        activated = False

        for i, rect in enumerate(_PALETTE_RECTS):
            key = f"swatch_{i}"
            if _point_in_rect(cx, cy, rect):
                s.hover_counts[key] = s.hover_counts.get(key, 0) + 1
                if s.hover_counts[key] == HOVER_FRAMES:
                    activated = True
                    if i == _ERASER_SLOT:
                        s.eraser_active = True
                    else:
                        s.eraser_active = False
                        s.color = PALETTE_COLORS[i]
                        s.color_name = PALETTE_NAMES[i]
                        s.shape_mode_idx = 0   # reset to free on colour pick
            else:
                s.hover_counts[key] = 0

        result.color = s.color
        result.color_name = s.color_name
        result.eraser_active = s.eraser_active
