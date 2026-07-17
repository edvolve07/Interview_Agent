import os
import json
import asyncio
import logging
import re
import time

from dotenv import load_dotenv
from openai import AsyncOpenAI

from livekit import rtc
from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.plugins import nvidia, silero, openai as lk_openai
from piper_tts import PiperTTS

from vision import (
    VisionConfig,
    VisionProvider,
    FrameEncoder,
    VisionContext,
    VisionService,
    create_video_sampler,
)

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("interview-agent")


class InterviewAgent(Agent):
    def __init__(self, category: str, user_identity: str):
        is_interview = _is_interview(category)
        role = "interviewer" if is_interview else "communication coach"
        context = f"conducting a {category} interview" if is_interview else f"practicing {category} communication scenarios"
        super().__init__(
            instructions=f"""
            You are an AI {role} {context}.
            - Speak naturally, like a real person in this situation
            - Use conversational language and occasional fillers ("great", "interesting", "I see")
            - Ask one question or prompt at a time
            - After the person responds, give brief positive acknowledgment then the next prompt
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


EVAL_CLIENT = AsyncOpenAI(
    api_key=os.environ["NVIDIA_API_KEY"],
    base_url="https://integrate.api.nvidia.com/v1",
)

EVAL_MODEL = "meta/llama-3.1-8b-instruct"

INTERVIEW_CATEGORIES = [
    "Tell Me About Yourself",
    "Behavioral Questions (STAR)",
    "Strengths & Weaknesses",
    "Why This Role / Company",
    "Technical Explanations",
    "Handling Difficult Questions",
    "Career Goals & Aspirations",
    "Salary & Negotiation Talk",
]


def _is_interview(category: str) -> bool:
    return category in INTERVIEW_CATEGORIES


def _clean_json(text: str) -> str:
    text = re.sub(r'```(?:json)?', '', text).strip()
    text = re.sub(r'`+$', '', text).strip()
    m = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', text)
    return m.group(1) if m else text


def _clamp(n, lo=0, hi=10):
    return max(lo, min(hi, int(n)))


def send_data(room, payload: dict, identity: str = ""):
    room.local_participant.publish_data(
        json.dumps(payload).encode(),
        reliable=True,
        destination_identities=[identity] if identity else [],
    )


def _build_eval_prompt(
    question: str,
    answer: str,
    category: str,
    video_context: str = "",
) -> str:
    video_section = ""
    if video_context:
        video_section = f"""
Video context during the response: {video_context}

- If the camera is ON and visible: assess eye contact, posture, lighting, and engagement
- If lighting is dim, note it may affect perceived confidence
- If screen share is ON: consider the person may be presenting, coding, or demonstrating
- Use vision observations (if available) to inform scores
"""

    is_interview = _is_interview(category)
    if is_interview:
        return f"""You are a strict interview coach evaluating a candidate's response.

Category: {category}
{video_section}
Question: {question}

Answer: {answer}

Score 0-10 each:
1. clarity — clear and articulate
2. structure — logical flow
3. conciseness — to the point
4. relevance — on-topic
5. confidence_tone — professional and confident

Also provide:
- strengths (1-2 items)
- improvements (1-2 items)
- feedback (one paragraph)
- next_prompt (follow-up question)
- real_world_tip (one actionable tip)

Return ONLY valid JSON:
{{"clarity":0,"structure":0,"conciseness":0,"relevance":0,"confidence_tone":0,"strengths":[],"improvements":[],"feedback":"","next_prompt":"","real_world_tip":""}}"""
    return f"""You are a communication coach evaluating someone's response.

Category: {category}
{video_section}
Prompt: {question}

Response: {answer}

Score 0-10 each:
1. clarity — clear and articulate
2. structure — logical flow
3. conciseness — to the point
4. relevance — on-topic
5. confidence_tone — appropriate for the context

Also provide:
- strengths (1-2 items)
- improvements (1-2 items)
- feedback (one paragraph)
- next_prompt (follow-up prompt)
- real_world_tip (one actionable tip)

Return ONLY valid JSON:
{{"clarity":0,"structure":0,"conciseness":0,"relevance":0,"confidence_tone":0,"strengths":[],"improvements":[],"feedback":"","next_prompt":"","real_world_tip":""}}"""


async def evaluate_user_response(
    transcript: str,
    exchange_count: int,
    current_prompt: str,
    category: str,
    video_context: str = "",
) -> dict:
    prompt = _build_eval_prompt(current_prompt, transcript, category, video_context)
    try:
        resp = await EVAL_CLIENT.chat.completions.create(
            model=EVAL_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024,
        )
        text = resp.choices[0].message.content or "{}"
        result = json.loads(_clean_json(text))
    except Exception as e:
        logger.error("Evaluation LLM call failed: %s", e)
        result = {}

    for key in ["clarity", "structure", "conciseness", "relevance", "confidence_tone"]:
        result[key] = _clamp(result.get(key, 5))
    for key in ["strengths", "improvements"]:
        if not isinstance(result.get(key), list):
            result[key] = [str(result[key])] if result.get(key) else []
    if not isinstance(result.get("feedback"), str) or not result["feedback"].strip():
        result["feedback"] = "Keep practicing to improve your communication skills."
    if not isinstance(result.get("next_prompt"), str) or not result["next_prompt"].strip():
        result["next_prompt"] = "Can you tell me more about a specific example from your experience?"
    if not isinstance(result.get("real_world_tip"), str) or not result["real_world_tip"].strip():
        result["real_world_tip"] = "In real situations, aim to be specific and provide concrete examples."

    return result


async def entrypoint(ctx: JobContext):
    metadata = json.loads(ctx.job.metadata or "{}")
    category = metadata.get("category", "Tell Me About Yourself")
    user_identity = metadata.get("userIdentity", "")

    logger.info("Starting interview: category=%s user=%s", category, user_identity)

    await ctx.connect()

    if user_identity not in ctx.room.remote_participants:
        joined = asyncio.get_event_loop().create_future()

        @ctx.room.on("participant_connected")
        def on_participant_connected(participant):
            if participant.identity == user_identity and not joined.done():
                joined.set_result(True)

        await joined

    # ── Vision pipeline ──────────────────────────────────────────
    vision_cfg = VisionConfig()

    vision_provider = VisionProvider.from_config(vision_cfg)
    frame_encoder = FrameEncoder(vision_cfg)
    vision_context = VisionContext(vision_cfg)
    vision_service = VisionService(
        vision_provider, frame_encoder, vision_context, vision_cfg
    )
    vision_service.start()

    video_sampler = create_video_sampler(vision_service, vision_cfg)

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
        llm=lk_openai.LLM(
            model=vision_cfg.voice_model,
            api_key=vision_cfg.nvidia_api_key,
            base_url=vision_cfg.nvidia_base_url,
        ),
        tts=PiperTTS(voice="en_US-lessac-low"),
        video_sampler=video_sampler,
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
                vision_ctx = vision_service.get_context()

                result = await evaluate_user_response(
                    transcript,
                    agent.exchange_count - 1,
                    current_question,
                    category,
                    video_context=vision_ctx,
                )

                current_question = result.get("next_prompt", "")
                is_last = result.get("is_last", False) or agent.exchange_count >= agent.max_exchanges

                eval_entry = {
                    "clarity": result.get("clarity", 0),
                    "structure": result.get("structure", 0),
                    "conciseness": result.get("conciseness", 0),
                    "relevance": result.get("relevance", 0),
                    "confidence_tone": result.get("confidence_tone", 0),
                }

                send_data(
                    ctx.room,
                    {
                        "type": "evaluation",
                        "exchange_number": agent.exchange_count,
                        "transcript": transcript,
                        "evaluation": eval_entry,
                        "feedback": result.get("feedback", ""),
                        "strengths": result.get("strengths", []),
                        "improvements": result.get("improvements", []),
                        "next_prompt": current_question,
                        "is_last": is_last,
                        "video_context": vision_ctx,
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

    vision_instructions = ""
    if vision_service.get_camera_active() or vision_service.get_screen_active():
        vision_instructions = (
            f"\n\nVisual context: {vision_service.get_context()}"
        )

    if _is_interview(category):
        await session.generate_reply(
            instructions=f"Greet the candidate warmly and ask the first interview question about {category}. Keep it natural and conversational.{vision_instructions}"
        )
    else:
        await session.generate_reply(
            instructions=f"Greet the person warmly and start the {category} communication scenario. Set up the situation naturally and prompt them to respond. Keep it natural and conversational.{vision_instructions}"
        )

    try:
        await asyncio.Future()
    except asyncio.CancelledError:
        pass
    finally:
        await vision_service.stop()


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="interview-agent",
        )
    )
