import asyncio
import logging
from collections import deque
from typing import Optional

import piper as piper_lib
from livekit.agents import utils
from livekit.agents import APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.tts import (
    TTS,
    AudioEmitter,
    ChunkedStream,
    SynthesizeStream,
    TTSCapabilities,
)

logger = logging.getLogger("piper-tts")

DEFAULT_VOICE = "en_US-lessac-low"


class PiperTTS(TTS):
    def __init__(
        self,
        *,
        voice: str = DEFAULT_VOICE,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
        sample_rate: int = 16000,
        num_channels: int = 1,
        noise_scale: float = 0.667,
        noise_w_scale: float = 0.8,
        length_scale: float = 1.0,
    ):
        super().__init__(
            capabilities=TTSCapabilities(streaming=False, aligned_transcript=False),
            sample_rate=sample_rate,
            num_channels=num_channels,
        )
        if model_path is not None:
            self._voice = None
            self._model_path = model_path
            self._config_path = config_path
        else:
            import pathlib
            base = pathlib.Path(__file__).parent / "voices" / voice
            self._model_path = str(base.with_suffix(".onnx"))
            self._config_path = str(base.with_suffix(".onnx.json"))
            self._voice = voice

        self._noise_scale = noise_scale
        self._noise_w_scale = noise_w_scale
        self._length_scale = length_scale
        self._piper_voice = None

    def _ensure_voice(self):
        if self._piper_voice is not None:
            return self._piper_voice
        self._piper_voice = piper_lib.PiperVoice.load(
            self._model_path,
            self._config_path,
        )
        actual_rate = getattr(self._piper_voice.config, "sample_rate", 16000)
        if actual_rate != self._sample_rate:
            logger.warning(
                "model sample_rate=%d differs from TTS sample_rate=%d, using model rate",
                actual_rate,
                self._sample_rate,
            )
        return self._piper_voice

    @property
    def model(self) -> str:
        return self._voice or self._model_path

    @property
    def provider(self) -> str:
        return "Piper"

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        return PiperChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SynthesizeStream:
        return PiperSynthesizeStream(tts=self, conn_options=conn_options)


class PiperChunkedStream(ChunkedStream):
    def __init__(self, *, tts: PiperTTS, input_text: str, conn_options: APIConnectOptions):
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: PiperTTS = tts

    async def _run(self, output_emitter: AudioEmitter) -> None:
        voice = self._tts._ensure_voice()
        sample_rate = getattr(voice.config, "sample_rate", 16000)

        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
        )

        loop = asyncio.get_event_loop()
        syn_config = piper_lib.SynthesisConfig(
            noise_scale=self._tts._noise_scale,
            noise_w_scale=self._tts._noise_w_scale,
            length_scale=self._tts._length_scale,
        )

        audio_chunks = await loop.run_in_executor(
            None,
            lambda: list(voice.synthesize(self._input_text, syn_config)),
        )

        for chunk in audio_chunks:
            output_emitter.push(chunk.audio_int16_bytes)

        output_emitter.flush()


class PiperSynthesizeStream(SynthesizeStream):
    def __init__(self, *, tts: PiperTTS, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: PiperTTS = tts

    async def _run(self, output_emitter: AudioEmitter) -> None:
        voice = self._tts._ensure_voice()
        sample_rate = getattr(voice.config, "sample_rate", 16000)

        pending_text: deque[str] = deque()
        input_done = asyncio.Event()

        async def _input_task():
            async for data in self._input_ch:
                if isinstance(data, SynthesizeStream._FlushSentinel):
                    input_done.set()
                    continue
                pending_text.append(data)
            input_done.set()

        input_task = asyncio.create_task(_input_task())

        await input_done.wait()
        full_text = "".join(pending_text)

        if not full_text.strip():
            await utils.aio.gracefully_cancel(input_task)
            return

        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
        )

        loop = asyncio.get_event_loop()
        syn_config = piper_lib.SynthesisConfig(
            noise_scale=self._tts._noise_scale,
            noise_w_scale=self._tts._noise_w_scale,
            length_scale=self._tts._length_scale,
        )

        audio_chunks = await loop.run_in_executor(
            None,
            lambda: list(voice.synthesize(full_text, syn_config)),
        )

        for chunk in audio_chunks:
            output_emitter.push(chunk.audio_int16_bytes)

        output_emitter.flush()
        await utils.aio.gracefully_cancel(input_task)
