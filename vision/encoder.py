import io
import base64
import logging

import numpy as np
from PIL import Image
from livekit import rtc

from .config import VisionConfig

logger = logging.getLogger("interview-agent.vision")


class FrameEncoder:
    def __init__(self, config: VisionConfig):
        self.max_size = config.max_image_size
        self.quality = config.image_quality
        self.fmt = config.image_format

    def encode(self, frame: rtc.VideoFrame) -> str:
        try:
            arr = np.frombuffer(frame.data, dtype=np.uint8).reshape(
                frame.height, frame.width, 4
            )
            img = Image.fromarray(arr[:, :, :3], "RGB")

            if max(img.width, img.height) > self.max_size:
                ratio = self.max_size / max(img.width, img.height)
                img = img.resize(
                    (int(img.width * ratio), int(img.height * ratio)),
                    Image.LANCZOS,
                )

            buf = io.BytesIO()
            img.save(buf, format=self.fmt.upper(), quality=self.quality)
            return base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception as e:
            logger.error("Frame encode error: %s", e)
            return ""

    def motion_score(self, current: rtc.VideoFrame, previous: rtc.VideoFrame | None) -> float:
        if previous is None:
            return 1.0
        try:
            c = np.frombuffer(current.data, dtype=np.uint8).reshape(
                current.height, current.width, 4
            )
            p = np.frombuffer(previous.data, dtype=np.uint8).reshape(
                previous.height, previous.width, 4
            )
            if c.shape != p.shape:
                return 1.0
            return float(np.mean(np.abs(c.astype(np.float32) - p.astype(np.float32))) / 255.0)
        except Exception:
            return 1.0
