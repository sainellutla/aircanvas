from typing import List, Optional, Tuple

import cv2
import numpy as np


class ShapeTool:

    @staticmethod
    def snap_line(
        points: List[Tuple[int, int]],
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        if len(points) < 2:
            return points[0], points[-1]
        pts = np.array(points, dtype=np.float32)
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        # Project first and last point onto the fitted line direction
        t_vals = [(p[0] - x0) * vx + (p[1] - y0) * vy for p in pts]
        t_min, t_max = min(t_vals), max(t_vals)
        p1 = (int(x0 + t_min * vx), int(y0 + t_min * vy))
        p2 = (int(x0 + t_max * vx), int(y0 + t_max * vy))
        return p1, p2

    @staticmethod
    def snap_circle(
        points: List[Tuple[int, int]],
    ) -> Tuple[Tuple[int, int], int]:
        if len(points) < 2:
            cx, cy = points[0]
            return (cx, cy), 10
        pts = np.array(points, dtype=np.float32)
        (cx, cy), r = cv2.minEnclosingCircle(pts)
        return (int(cx), int(cy)), max(int(r), 1)

    @staticmethod
    def snap_rect(
        points: List[Tuple[int, int]],
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        if len(points) < 2:
            return points[0], points[0]
        pts = np.array(points, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        return (x, y), (x + w, y + h)

    @staticmethod
    def smooth_stroke(
        points: List[Tuple[int, int]], n_interp: int = 5
    ) -> List[Tuple[int, int]]:
        if len(points) < 4:
            return points

        def _catmull_rom(p0, p1, p2, p3):
            p0 = np.array(p0, dtype=np.float64)
            p1 = np.array(p1, dtype=np.float64)
            p2 = np.array(p2, dtype=np.float64)
            p3 = np.array(p3, dtype=np.float64)
            result = []
            for i in range(n_interp):
                t = i / n_interp
                t2 = t * t
                t3 = t2 * t
                q = 0.5 * (
                    (2 * p1)
                    + (-p0 + p2) * t
                    + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                    + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
                )
                result.append((int(q[0]), int(q[1])))
            return result

        smoothed: List[Tuple[int, int]] = []
        pts = points
        # Duplicate endpoints so the spline passes through them
        pts = [pts[0]] + pts + [pts[-1]]
        for i in range(1, len(pts) - 2):
            smoothed.extend(_catmull_rom(pts[i - 1], pts[i], pts[i + 1], pts[i + 2]))
        smoothed.append(points[-1])
        return smoothed
