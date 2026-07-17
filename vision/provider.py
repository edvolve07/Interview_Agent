import logging
from abc import ABC, abstractmethod

from .config import VisionConfig

logger = logging.getLogger("interview-agent.vision")


class VisionProvider(ABC):
    @abstractmethod
    async def describe(
        self, image_base64: str, prompt: str = "Describe what you see in this image."
    ) -> str:
        ...

    @abstractmethod
    async def summarize(
        self,
        image_base64: str,
        source: str = "camera",
    ) -> str:
        ...

    @classmethod
    def from_config(cls, config: VisionConfig) -> "VisionProvider":
        provider = config.vision_provider.lower()
        if provider == "groq":
            return GroqVisionProvider(config)
        return NvidiaVisionProvider(config)


class NvidiaVisionProvider(VisionProvider):
    def __init__(self, config: VisionConfig):
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(
            api_key=config.nvidia_api_key,
            base_url=config.nvidia_base_url,
        )
        self._model = config.vision_model

    async def describe(self, image_base64: str, prompt: str = "") -> str:
        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt or "Describe what you see."},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=256,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            logger.error("NVIDIA vision describe failed: %s", e)
            return ""

    async def summarize(self, image_base64: str, source: str = "camera") -> str:
        prompt = (
            f"You are observing a {source} feed. "
            "Summarize what you see briefly in one sentence. "
            "Focus on objects, people, text, activities, and the environment."
        )
        return await self.describe(image_base64, prompt)


class GroqVisionProvider(VisionProvider):
    def __init__(self, config: VisionConfig):
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(
            api_key=config.groq_api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        self._model = config.groq_vision_model

    async def describe(self, image_base64: str, prompt: str = "") -> str:
        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt or "Describe what you see."},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=256,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            logger.error("Groq vision describe failed: %s", e)
            return ""

    async def summarize(self, image_base64: str, source: str = "camera") -> str:
        prompt = (
            f"You are observing a {source} feed. "
            "Summarize what you see briefly in one sentence. "
            "Focus on objects, people, text, activities, and the environment."
        )
        return await self.describe(image_base64, prompt)
