import os
from dotenv import load_dotenv

from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.plugins import nvidia, silero, openai

load_dotenv()

class Assistant(Agent):
    def __init__(self):
        super().__init__(
            instructions="""
            You are a helpful AI assistant.
            Keep responses concise.
            """
        )

async def entrypoint(ctx: JobContext):

    await ctx.connect()

    session = AgentSession(
        vad=silero.VAD.load(),

        stt=nvidia.STT(),

        llm=openai.LLM(
            model="meta/llama-3.1-8b-instruct",
            api_key=os.environ["NVIDIA_API_KEY"],
            base_url="https://integrate.api.nvidia.com/v1",
        ),

        tts=nvidia.TTS(),
    )

    @session.on("conversation_item_added")
    def on_message(item):
        print(item)

    await session.start(
        room=ctx.room,
        agent=Assistant(),
    )

    await session.generate_reply(
        instructions="Your name is Agent 007"
    )

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint
        )
    )