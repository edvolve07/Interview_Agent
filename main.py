import os
import json
import asyncio
import logging
import re
import time
from typing import Dict, Any, Optional

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
from prompts import (
    CORE_IDENTITY,
    MODES,
    SCENARIO_BEHAVIORS,
    get_mode_from_metadata,
    get_scenario_behavior,
    build_coaching_prompt,
    build_greeting_instructions,
)

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("interview-agent")


class InterviewAgent(Agent):
    def __init__(self, mode: str, category: str, user_identity: str):
        self.mode = mode
        self.category = category
        self.user_identity = user_identity
        self.exchange_count = 0
        self.max_exchanges = 6
        self.current_emotion = {}

        instructions = build_coaching_prompt(category, mode, user_identity, self.exchange_count)
        super().__init__(instructions=instructions)

        logger.info("Agent initialized - mode=%s category=%s user=%s", mode, category, user_identity)


EVAL_CLIENT = AsyncOpenAI(
    api_key=os.environ["NVIDIA_API_KEY"],
    base_url="https://integrate.api.nvidia.com/v1",
)

EVAL_MODEL = "meta/llama-3.1-8b-instruct"


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


async def evaluate_user_response(
    transcript: str,
    exchange_count: int,
    current_prompt: str,
    category: str,
    mode: str = "general",
    video_context: str = "",
) -> dict:
    """
    Evaluate user response with mode-appropriate criteria
    """
    # Build evaluation prompt based on mode
    if mode == "interview_prep":
        return await _evaluate_interview_response(transcript, exchange_count, current_prompt, category, video_context)
    else:
        return await _evaluate_general_response(transcript, exchange_count, current_prompt, category, video_context)


async def _evaluate_interview_response(
    transcript: str,
    exchange_count: int,
    current_prompt: str,
    category: str,
    video_context: str = "",
) -> dict:
    """Evaluation focused on interview communication skills"""
    video_section = ""
    if video_context:
        video_section = f"""
Video context during the response: {video_context}

- If the camera is ON and visible: assess eye contact, posture, lighting, and engagement
- If lighting is dim, note it may affect perceived confidence
- If screen share is ON: consider the person may be presenting, coding, or demonstrating
- Use vision observations (if available) to inform scores
"""

    prompt = f"""You are a communication coach evaluating interview communication skills.

Category: {category}
{video_section}
Interview Question/Prompt: {current_prompt}

User's Response: {transcript}

Evaluate on these dimensions (0-10 each):
1. clarity — clear and articulate expression
2. structure — logical flow and organization (for behavioral: STAR method)
3. conciseness — appropriately detailed without being verbose or too brief
4. relevance — stays on topic and addresses the question/prompt
5. confidence_tone — professional, assured, and appropriately assertive
6. engagement — shows interest and involvement in the conversation
7. listening_skills — demonstrates understanding through relevant responses
8. professionalism — maintains appropriate professional demeanor

Also provide:
- strengths (1-2 specific things done well)
- improvements (1-2 specific, actionable suggestions)
- feedback (one paragraph of coaching advice focused on interview communication)
- next_prompt (natural follow-up question or prompt)
- real_world_tip (one actionable tip for real interview situations)

Return ONLY valid JSON:
{{"clarity":0,"structure":0,"conciseness":0,"relevance":0,"confidence_tone":0,"engagement":0,"listening_skills":0,"professionalism":0,"strengths":[],"improvements":[],"feedback":"","next_prompt":"","real_world_tip":""}}"""

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

    # Ensure all expected keys exist with sensible defaults
    expected_keys = ["clarity", "structure", "conciseness", "relevance", "confidence_tone",
                     "engagement", "listening_skills", "professionalism", "strengths", "improvements",
                     "feedback", "next_prompt", "real_world_tip"]

    for key in expected_keys:
        if key not in result:
            if key in ["clarity", "structure", "conciseness", "relevance", "confidence_tone",
                       "engagement", "listening_skills", "professionalism"]:
                result[key] = 5
            elif key in ["strengths", "improvements"]:
                result[key] = []
            elif key in ["feedback", "next_prompt", "real_world_tip"]:
                result[key] = "Keep practicing to improve your communication skills." if key == "feedback" else (
                    "Can you tell me more about your experience?" if key == "next_prompt" else
                    "Focus on being clear and specific in your responses."
                )

    # Clamp numeric scores
    for key in ["clarity", "structure", "conciseness", "relevance", "confidence_tone",
                "engagement", "listening_skills", "professionalism"]:
        result[key] = _clamp(result.get(key, 5))

    # Ensure arrays are actually arrays
    for key in ["strengths", "improvements"]:
        if not isinstance(result.get(key), list):
            result[key] = [str(result[key])] if result.get(key) else []

    # Ensure strings are actually strings
    for key in ["feedback", "next_prompt", "real_world_tip"]:
        if not isinstance(result.get(key), str) or not result[key].strip():
            result[key] = "Keep practicing to improve your communication skills." if key == "feedback" else (
                "Can you tell me more about your experience?" if key == "next_prompt" else
                "Focus on being clear and specific in your responses."
            )

    return result


async def _evaluate_general_response(
    transcript: str,
    exchange_count: int,
    current_prompt: str,
    category: str,
    video_context: str = "",
) -> dict:
    """Evaluation focused on general communication skills"""
    video_section = ""
    if video_context:
        video_section = f"""
Video context during the response: {video_context}

- If the camera is ON and visible: assess eye contact, posture, lighting, and engagement
- If lighting is dim, note it may affect perceived confidence
- If screen share is ON: consider the person may be presenting, coding, or demonstrating
- Use vision observations (if available) to inform scores
"""

    prompt = f"""You are a communication coach evaluating general communication skills.

Scenario: {category}
{video_section}
Conversation Prompt: {current_prompt}

User's Response: {transcript}

Evaluate on these dimensions (0-10 each):
1. clarity — clear and articulate expression
2. coherence — logical and easy-to-follow flow of ideas
3. engagement — shows interest and involvement in the conversation
4. empathy — demonstrates understanding and consideration of others' perspectives
5. adaptability — adjusts communication style appropriately to the context
6. listening_skills — demonstrates understanding through relevant responses
7. confidence — expresses ideas with appropriate assurance (not over or under-confident)
8. authenticity — communicates genuinely and sincerely

Also provide:
- strengths (1-2 specific things done well)
- improvements (1-2 specific, actionable suggestions)
- feedback (one paragraph of coaching advice focused on general communication)
- next_prompt (natural follow-up question or prompt to continue the conversation)
- real_world_tip (one actionable tip for real-world situations like this)

Return ONLY valid JSON:
{{"clarity":0,"coherence":0,"engagement":0,"empathy":0,"adaptability":0,"listening_skills":0,"confidence":0,"authenticity":0,"strengths":[],"improvements":[],"feedback":"","next_prompt":"","real_world_tip":""}}"""

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

    # Ensure all expected keys exist with sensible defaults
    expected_keys = ["clarity", "coherence", "engagement", "empathy", "adaptability",
                     "listening_skills", "confidence", "authenticity", "strengths", "improvements",
                     "feedback", "next_prompt", "real_world_tip"]

    for key in expected_keys:
        if key not in result:
            if key in ["clarity", "coherence", "engagement", "empathy", "adaptability",
                       "listening_skills", "confidence", "authenticity"]:
                result[key] = 5
            elif key in ["strengths", "improvements"]:
                result[key] = []
            elif key in ["feedback", "next_prompt", "real_world_tip"]:
                result[key] = "Keep practicing to improve your communication skills." if key == "feedback" else (
                    "That's interesting! Can you tell me more about that?" if key == "next_prompt" else
                    "In real conversations, active listening and genuine curiosity go a long way."
                )

    # Clamp numeric scores
    for key in ["clarity", "coherence", "engagement", "empathy", "adaptability",
                "listening_skills", "confidence", "authenticity"]:
        result[key] = _clamp(result.get(key, 5))

    # Ensure arrays are actually arrays
    for key in ["strengths", "improvements"]:
        if not isinstance(result.get(key), list):
            result[key] = [str(result[key])] if result.get(key) else []

    # Ensure strings are actually strings
    for key in ["feedback", "next_prompt", "real_world_tip"]:
        if not isinstance(result.get(key), str) or not result[key].strip():
            result[key] = "Keep practicing to improve your communication skills." if key == "feedback" else (
                "That's interesting! Can you tell me more about that?" if key == "next_prompt" else
                "In real conversations, active listening and genuine curiosity go a long way."
            )

    return result


async def entrypoint(ctx: JobContext):
    metadata = json.loads(ctx.job.metadata or "{}")
    category = metadata.get("category", "Tell Me About Yourself")
    user_identity = metadata.get("userIdentity", "")

    logger.info("Starting session: category=%s mode=%s user=%s", category, metadata.get("mode", "general"), user_identity)

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

    mode = get_mode_from_metadata(metadata)
    agent = InterviewAgent(mode, category, user_identity)
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
        eliciting_prompt = current_question  # Store the prompt that elicited this response

        async def process():
            nonlocal current_question
            try:
                vision_ctx = vision_service.get_context()

                mode = agent.mode
                result = await evaluate_user_response(
                    transcript,
                    agent.exchange_count - 1,
                    eliciting_prompt,
                    category,
                    mode,
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

                # Add mode-specific fields to evaluation
                if mode == "interview_prep":
                    eval_entry.update({
                        "engagement": result.get("engagement", 0),
                        "listening_skills": result.get("listening_skills", 0),
                        "professionalism": result.get("professionalism", 0),
                    })
                else:
                    eval_entry.update({
                        "coherence": result.get("coherence", 0),
                        "engagement": result.get("engagement", 0),
                        "empathy": result.get("empathy", 0),
                        "adaptability": result.get("adaptability", 0),
                        "listening_skills": result.get("listening_skills", 0),
                        "confidence": result.get("confidence", 0),
                        "authenticity": result.get("authenticity", 0),
                    })

                send_data(
                    ctx.room,
                    {
                        "type": "evaluation",
                        "exchange_number": agent.exchange_count,
                        "eliciting_prompt": eliciting_prompt,
                        "transcript": transcript,
                        "evaluation": eval_entry,
                        "feedback": result.get("feedback", ""),
                        "strengths": result.get("strengths", []),
                        "improvements": result.get("improvements", []),
                        "next_prompt": current_question,
                        "is_last": is_last,
                        "video_context": vision_ctx,
                        "mode": mode,
                    },
                    user_identity,
                )

                if is_last:
                    logger.info("Session complete for user %s", user_identity)
                    send_data(ctx.room, {"type": "complete", "exchange_count": agent.exchange_count}, user_identity)
                    ctx.shutdown()

            except Exception as e:
                logger.error("Processing error: %s", e)
                send_data(ctx.room, {"type": "status", "status": "ready"}, user_identity)

        asyncio.ensure_future(process())

    await session.start(room=ctx.room, agent=agent)

    greeting = build_greeting_instructions(agent.mode, category)
    await session.generate_reply(instructions=greeting)

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