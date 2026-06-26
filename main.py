import os
import json
import asyncio
import logging

import httpx
from dotenv import load_dotenv

from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.plugins import nvidia, silero, openai
from piper_tts import PiperTTS

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("interview-agent")

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8001")


class InterviewAgent(Agent):
    def __init__(self, category: str, user_identity: str):
        super().__init__(
            instructions=f"""
            You are an AI interviewer conducting a {category} interview.
            - Speak naturally, like a real human interviewer
            - Use conversational language and occasional fillers ("great", "interesting", "I see")
            - Ask one question at a time
            - After the candidate answers, give brief positive acknowledgment then ask the next question
            - Stay encouraging and warm
            - Never answer questions yourself
            - Keep responses to 1-3 sentences
            """
        )
        self.category = category
        self.user_identity = user_identity
        self.exchange_count = 0
        self.max_exchanges = 6
        self.current_emotion = {}


def send_data(room, payload: dict, identity: str = ""):
    room.local_participant.publish_data(
        json.dumps(payload).encode(),
        reliable=True,
        destination_identities=[identity] if identity else [],
    )


async def evaluate_user_response(
    session_id: str,
    transcript: str,
    exchange_count: int,
    current_prompt: str,
    category: str,
    emotion: dict = None,
) -> dict:
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{BACKEND_URL}/api/livekit/evaluate",
            json={
                "session_id": session_id,
                "transcript": transcript,
                "category": category,
                "exchange_count": exchange_count,
                "current_prompt": current_prompt,
                "emotion": emotion or {},
            },
        )
        resp.raise_for_status()
        return resp.json()


async def entrypoint(ctx: JobContext):
    metadata = json.loads(ctx.job.metadata or "{}")
    category = metadata.get("category", "Tell Me About Yourself")
    user_identity = metadata.get("userIdentity", "")

    logger.info("Starting interview: category=%s user=%s", category, user_identity)

    await ctx.connect()

    # Wait for the user participant to join the room
    # so the frontend's data channel listener is active
    if user_identity not in ctx.room.remote_participants:
        joined = asyncio.get_event_loop().create_future()

        @ctx.room.on("participant_connected")
        def on_participant_connected(participant):
            if participant.identity == user_identity and not joined.done():
                joined.set_result(True)

        await joined

    @ctx.room.on("data_received")
    def on_data_received(payload, participant):
        if participant and participant.identity != user_identity:
            return
        try:
            msg = json.loads(payload.decode())
            if msg.get("type") == "emotion":
                agent.current_emotion = msg.get("expressions", {})
        except Exception:
            pass

    send_data(ctx.room, {"type": "status", "status": "ready"})

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=nvidia.STT(),
        llm=openai.LLM(
            model="meta/llama-3.1-8b-instruct",
            api_key=os.environ["NVIDIA_API_KEY"],
            base_url="https://integrate.api.nvidia.com/v1",
        ),
        tts=PiperTTS(voice="en_US-lessac-low"),
    )

    agent = InterviewAgent(category, user_identity)
    current_question = ""

    @session.on("agent_state_changed")
    def on_agent_state(event):
        if event.new_state in ("listening", "thinking", "speaking"):
            send_data(ctx.room, {"type": "status", "status": event.new_state}, user_identity)

    @session.on("conversation_item_added")
    def on_conversation_item(event):
        item = event.item
        if item.role != "user" or not item.content:
            return

        agent.exchange_count += 1
        transcript = item.content

        async def process():
            nonlocal current_question
            try:
                result = await evaluate_user_response(
                    f"session-{user_identity}-{agent.exchange_count}",
                    transcript,
                    agent.exchange_count - 1,
                    current_question,
                    category,
                    emotion=agent.current_emotion,
                )

                current_question = result.get("next_prompt", "")
                is_last = result.get("is_last", False) or agent.exchange_count >= agent.max_exchanges

                send_data(
                    ctx.room,
                    {
                        "type": "evaluation",
                        "exchange_number": result.get("exchange_number", agent.exchange_count),
                        "transcript": transcript,
                        "evaluation": result.get("evaluation", {}),
                        "feedback": result.get("feedback", ""),
                        "strengths": result.get("strengths", []),
                        "improvements": result.get("improvements", []),
                        "next_prompt": current_question,
                        "is_last": is_last,
                    },
                    user_identity,
                )

                if is_last:
                    logger.info("Interview complete for user %s", user_identity)
                    send_data(ctx.room, {"type": "complete"}, user_identity)
                    ctx.shutdown()

            except Exception as e:
                logger.error("Processing error: %s", e)
                send_data(ctx.room, {"type": "status", "status": "ready"}, user_identity)

        asyncio.ensure_future(process())

    await session.start(room=ctx.room, agent=agent)

    await session.generate_reply(
        instructions=f"Greet the candidate warmly and ask the first interview question about {category}. Keep it natural and conversational."
    )

    try:
        await asyncio.Future()
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="interview-agent",
        )
    )
