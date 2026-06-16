import os
import urllib.request
from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from config import MODEL_PATH, MODEL_URL, NUM_HANDS


@dataclass
class HandData:
    hand_index: int
    handedness: str                   # "Left" or "Right"
    landmarks: list                   # 21 NormalizedLandmark objects
    px: List[Tuple[int, int]] = field(default_factory=list)   # pixel coords


def _ensure_model() -> None:
    if not os.path.exists(MODEL_PATH):
        print(f"[HandTracker] Downloading MediaPipe model to {MODEL_PATH} ...")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[HandTracker] Download complete.")


class HandTracker:
    def __init__(self, num_hands: int = NUM_HANDS) -> None:
        _ensure_model()
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)

    def detect(self, bgr_frame, timestamp_ms: int) -> List[HandData]:
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        h, w = bgr_frame.shape[:2]
        hands: List[HandData] = []
        if not result.hand_landmarks:
            return hands

        for i, lm_list in enumerate(result.hand_landmarks):
            handedness = (
                result.handedness[i][0].category_name if result.handedness else "Right"
            )
            px = [(int(lm.x * w), int(lm.y * h)) for lm in lm_list]
            hands.append(HandData(
                hand_index=i,
                handedness=handedness,
                landmarks=lm_list,
                px=px,
            ))
        return hands

    def close(self) -> None:
        self._landmarker.close()
