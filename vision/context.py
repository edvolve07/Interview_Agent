from .config import VisionConfig


class VisionContext:
    def __init__(self, config: VisionConfig):
        self._max_summaries = config.max_summaries
        self._summaries: list[str] = []
        self._camera_active = False
        self._screen_active = False
        self._camera_resolution = ""
        self._screen_resolution = ""

    def add_summary(self, text: str, source: str = "camera") -> None:
        if not text:
            return
        prefix = "[camera]" if source == "camera" else "[screen]"
        entry = f"{prefix} {text}"
        self._summaries.append(entry)
        if len(self._summaries) > self._max_summaries:
            self._summaries.pop(0)

    def set_camera_status(
        self, active: bool, resolution: str = ""
    ) -> None:
        self._camera_active = active
        if resolution:
            self._camera_resolution = resolution

    def set_screen_status(
        self, active: bool, resolution: str = ""
    ) -> None:
        self._screen_active = active
        if resolution:
            self._screen_resolution = resolution

    def get_conversation_context(self) -> str:
        parts = []
        if self._camera_active:
            parts.append(
                f"Camera is ON ({self._camera_resolution})"
                if self._camera_resolution
                else "Camera is ON"
            )
        else:
            parts.append("Camera is OFF")

        if self._screen_active:
            parts.append(
                f"Screen share is ON ({self._screen_resolution})"
                if self._screen_resolution
                else "Screen share is ON"
            )
        else:
            parts.append("No screen share")

        result = " | ".join(parts)

        if self._summaries:
            recent = self._summaries[-3:]
            result += "\nRecent observations:\n- " + "\n- ".join(recent)

        return result

    def get_recent_summaries(self, count: int = 3) -> list[str]:
        return self._summaries[-count:]

    def reset(self) -> None:
        self._summaries.clear()
        self._camera_active = False
        self._screen_active = False
        self._camera_resolution = ""
        self._screen_resolution = ""
