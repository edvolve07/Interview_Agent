import asyncio
import logging
import time

from livekit import rtc
from livekit.agents import AgentSession

from .config import VisionConfig
from .encoder import FrameEncoder
from .provider import VisionProvider
from .context import VisionContext

logger = logging.getLogger("interview-agent.vision")


class VisionService:
    def __init__(
        self,
        provider: VisionProvider,
        encoder: FrameEncoder,
        context: VisionContext,
        config: VisionConfig,
    ):
        self._provider = provider
        self._encoder = encoder
        self._context = context
        self._config = config
        self._frame_queue: asyncio.Queue[tuple[rtc.VideoFrame, str]] = (
            asyncio.Queue()
        )
        self._task: asyncio.Task | None = None
        self._summary_counter = 0

    def start(self) -> None:
        self._task = asyncio.ensure_future(self._process_loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def queue_frame(self, frame: rtc.VideoFrame, source: str) -> None:
        try:
            self._frame_queue.put_nowait((frame, source))
        except asyncio.QueueFull:
            pass

    async def _process_loop(self) -> None:
        while True:
            try:
                frame, source = await self._frame_queue.get()
                await self._process_frame(frame, source)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Vision process error: %s", e)

    async def _process_frame(
        self, frame: rtc.VideoFrame, source: str
    ) -> None:
        try:
            b64 = self._encoder.encode(frame)
            if not b64:
                return

            b = self._analyze_brightness(frame)
            self._context.set_camera_status(
                True, f"{frame.width}x{frame.height}"
            )

            self._summary_counter += 1
            if self._summary_counter >= self._config.summary_interval:
                self._summary_counter = 0
                summary = await self._provider.summarize(b64, source)
                self._context.add_summary(summary, source)
                logger.info(
                    "Vision summary (%s): %s", source, summary
                )
        except Exception as e:
            logger.error("Frame processing error: %s", e)

    def _analyze_brightness(
        self, frame: rtc.VideoFrame
    ) -> float:
        try:
            import numpy as np

            arr = np.frombuffer(frame.data, dtype=np.uint8).reshape(
                frame.height, frame.width, 4
            )
            return float(arr[:, :, :3].mean()) / 255.0
        except Exception:
            return 0.5

    def get_context(self) -> str:
        return self._context.get_conversation_context()

    def get_camera_active(self) -> bool:
        return self._context._camera_active

    def get_screen_active(self) -> bool:
        return self._context._screen_active


def create_video_sampler(
    vision_service: VisionService,
    config: VisionConfig,
):
    last_sampled: float = 0.0
    prev_frame: list[rtc.VideoFrame | None] = [None]
    encoder = FrameEncoder(config)

    def sampler(frame: rtc.VideoFrame, session: AgentSession) -> bool:
        nonlocal last_sampled

        now = time.monotonic()
        if now - last_sampled < config.frame_interval:
            return False

        motion = encoder.motion_score(frame, prev_frame[0])
        if motion < config.motion_threshold:
            return False

        last_sampled = now
        prev_frame[0] = frame

        source = _detect_source(session)
        vision_service.queue_frame(frame, source)

        return True

    return sampler


def _detect_source(session: AgentSession) -> str:
    try:
        from livekit.agents.voice.agent_session import (
            AgentSession as _AS,
        )
    except ImportError:
        return "camera"
    return "camera"
