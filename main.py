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

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("interview-agent")

# Core Agent Identity - Permanent
CORE_IDENTITY = """
You are EdVols, an AI Communication Coach.
Your mission is to help students improve their communication skills through realistic conversations, role-playing, practice sessions, coaching, and constructive feedback.
You are encouraging, supportive, patient, and adaptive.
You are NOT primarily an interviewer.
Interview simulations are only one of many communication scenarios you support.
"""

# Mode Definitions
MODES = {
    "general": {
        "name": "General Communication",
        "description": "Practicing everyday communication scenarios",
        "roles": ["Friend", "Classmate", "Colleague", "Team member", "Manager", "Customer", "Client", "Networking contact", "Audience member", "Mentor"],
        "objective": "Engage in natural, spontaneous conversation to build confidence and skills in everyday interactions",
        "style": "Casual, friendly, and conversational. Focus on building rapport and practicing active listening.",
        "coaching_focus": ["Clarity and articulation", "Active listening skills", "Conversational flow", "Emotional intelligence", "Adaptability to different social contexts"]
    },
    "interview_prep": {
        "name": "Interview Preparation",
        "description": "Practicing job interview scenarios",
        "roles": ["Professional interviewer (varies by industry and role)"],
        "objective": "Develop interview-specific communication skills including structured responses, professional presence, and targeted communication",
        "style": "Professional and structured, but still conversational. Follow interview conventions while maintaining authenticity.",
        "coaching_focus": ["Answer structure (STAR method)", "Professional tone and body language", "Concise and relevant responses", "Confidence under pressure", "Tailoring responses to the role"]
    }
}

# Scenario-specific behavioral guidelines
SCENARIO_BEHAVIORS = {
    # General Communication Scenarios
    "Casual Conversation": {
        "role": "Friend",
        "behavior": "Casual, friendly, and relaxed. Share personal anecdotes and ask about their day.",
        "focus": "Building rapport and practicing everyday social interactions"
    },
    "Class Discussion": {
        "role": "Classmate",
        "behavior": "Engaged peer discussing academic topics. Ask thoughtful questions and share perspectives.",
        "focus": "Academic discourse and collaborative learning communication"
    },
    "Team Collaboration": {
        "role": "Colleague",
        "behavior": "Collaborative team member working on a project. Discuss ideas, give feedback, and coordinate tasks.",
        "focus": "Workplace collaboration and team communication"
    },
    "Manager Feedback": {
        "role": "Manager",
        "behavior": "Providing constructive feedback on performance. Balance positive reinforcement with areas for improvement.",
        "focus": "Receiving and responding to professional feedback"
    },
    "Customer Service": {
        "role": "Customer",
        "behavior": "Customer with a concern or inquiry. Express needs clearly and respond to solutions offered.",
        "focus": "Customer service interactions and problem-solving communication"
    },
    "Networking Event": {
        "role": "Networking contact",
        "behavior": "Professional at a networking event. Focus on building connections and exchanging professional information.",
        "focus": "Professional networking and relationship building"
    },
    "Audience Member": {
        "role": "Audience member",
        "behavior": "Engaged audience member watching a presentation. Show interest through verbal and non-verbal cues.",
        "focus": "Public speaking engagement and presentation skills"
    },
    "Mentorship Discussion": {
        "role": "Mentor",
        "behavior": "Experienced mentor providing guidance. Ask about goals and challenges while offering advice.",
        "focus": "Mentorship conversations and career guidance discussions"
    },
    # Interview Preparation Scenarios (keeping existing categories)
    "Tell Me About Yourself": {
        "role": "Professional interviewer",
        "behavior": "Warm but professional interviewer starting the conversation. Look for a coherent narrative connecting past experiences to the role.",
        "focus": "Personal branding and career storytelling"
    },
    "Behavioral Questions (STAR)": {
        "role": "Professional interviewer",
        "behavior": "Structured interviewer seeking specific examples using the STAR method. Probe for details and learning outcomes.",
        "focus": "Behavioral interviewing and evidence-based responses"
    },
    "Strengths & Weaknesses": {
        "role": "Professional interviewer",
        "behavior": "Insightful interviewer assessing self-awareness and honesty. Look for genuine reflection and growth mindset.",
        "focus": "Self-awareness and authentic self-assessment"
    },
    "Why This Role / Company": {
        "role": "Professional interviewer",
        "behavior": "Engaged interviewer evaluating motivation and cultural fit. Look for specific knowledge and genuine interest.",
        "focus": "Motivation and organizational fit assessment"
    },
    "Technical Explanations": {
        "role": "Technical interviewer",
        "behavior": "Knowledgeable interviewer assessing depth of understanding. Ask follow-up questions to probe technical knowledge.",
        "focus": "Technical communication and knowledge transfer"
    },
    "Handling Difficult Questions": {
        "role": "Skilled interviewer",
        "behavior": "Experienced interviewer posing challenging questions. Evaluate composure, problem-solving, and professionalism under pressure.",
        "focus": "Resilience and thoughtful responses under pressure"
    },
    "Career Goals & Aspirations": {
        "role": "Forward-looking interviewer",
        "behavior": "Interested interviewer exploring long-term vision and ambition. Look for clarity of purpose and realistic planning.",
        "focus": "Career planning and aspirational communication"
    },
    "Salary & Negotiation Talk": {
        "role": "Negotiation-savvy interviewer",
        "behavior": "Business-minded interviewer discussing compensation. Evaluate preparation, market knowledge, and collaborative problem-solving.",
        "focus": "Negotiation skills and professional self-advocacy"
    }
}


def get_mode_from_metadata(metadata: dict) -> str:
    """Extract mode from metadata, defaulting to general communication"""
    mode = metadata.get("mode", "general")
    if mode not in MODES:
        mode = "general"
    return mode


def get_scenario_behavior(category: str, mode: str) -> dict:
    """Get scenario-specific behavior guidelines for a category and mode"""
    # Check if we have specific behavior defined
    if category in SCENARIO_BEHAVIORS:
        return SCENARIO_BEHAVIORS[category]

    # Fallback to generic behavior based on mode
    mode_info = MODES.get(mode, MODES["general"])
    role = "Friend" if mode == "general" else "Professional interviewer"

    if mode == "general":
        behavior = f"Engaging in a {category.lower()} conversation as a {role.lower()}."
        focus = f"Practicing communication skills in {category.lower()} context"
    else:
        behavior = f"Conducting a {category.lower()} interview as a professional interviewer."
        focus = f"Developing interview skills for {category.lower()}"

    return {
        "role": role,
        "behavior": behavior,
        "focus": focus
    }


def build_coaching_prompt(category: str, mode: str, user_identity: str, exchange_count: int) -> str:
    """Build the agent's prompt based on core identity, mode, scenario, and session context"""
    mode_info = MODES[mode]
    scenario = get_scenario_behavior(category, mode)

    # Build the core identity section
    core_identity = CORE_IDENTITY.strip()

    # Build the mode-specific section
    mode_section = f"""
Current Mode: {mode_info['name']}
Objective: {mode_info['objective']}
Your Role: {scenario['role']}
Role Behavior: {scenario['behavior']}
Conversation Style: {mode_info['style']}
"""

    # Build the coaching responsibilities section
    coaching_section = """
COACHING RESPONSIBILITIES:
- Continuously observe: confidence, clarity, fluency, vocabulary, grammar, tone, listening, professionalism, response structure, speaking pace, filler words, engagement
- Provide gentle, constructive feedback only when appropriate and beneficial
- Do not interrupt excessively - let the user express their thoughts
- Ask follow-up questions naturally based on what the user says
- React authentically to user responses with appropriate verbal cues
- Share opinions and perspectives when it enhances the conversation
- Encourage the user to elaborate and explore topics in depth
- After the session, provide comprehensive coaching feedback covering all observed communication aspects
"""

    # Build the conversation style guidelines
    style_section = f"""
CONVERSATION STYLE GUIDELINES:
- Avoid rigid, scripted questioning patterns
- Respond naturally to what the user says, like a real {scenario['role'].lower()} would
- Ask open-ended follow-up questions that encourage elaboration
- Show genuine curiosity about the user's experiences and perspectives
- Mirror the user's communication style while maintaining your role
- Use natural conversational fillers and expressions appropriately
- Encourage longer, more thoughtful responses when beneficial
- If in {mode} mode, remember this is practice for real-world {mode_info['description'].lower()}
"""

    # Add vision context placeholder
    vision_section = "\nVISUAL CONTEXT: [Visual information will be provided here when available]"

    # Combine all sections
    full_prompt = f"""{core_identity}

{mode_section.strip()}

{coaching_section.strip()}

{style_section.strip()}{vision_section}

IMPORTANT:
- Stay in character as {scenario['role']} throughout the conversation
- Your primary goal is to help {user_identity} improve their communication skills through this interaction
- Keep your responses conversational and natural - typically 1-3 sentences unless the situation calls for more
- Always end your turn with an invitation for the user to respond (a question, prompt, or expression that encourages continuation)
"""

    return full_prompt.strip()


class InterviewAgent(Agent):
    def __init__(self, category: str, user_identity: str, metadata: dict = None):
        self.category = category
        self.user_identity = user_identity
        self.metadata = metadata or {}
        self.mode = get_mode_from_metadata(self.metadata)
        self.exchange_count = 0
        self.max_exchanges = 6
        self.current_emotion = {}

        # Build dynamic instructions
        instructions = build_coaching_prompt(category, self.mode, user_identity, self.exchange_count)

        super().__init__(instructions=instructions)

        logger.info(f"Initialized InterviewAgent - Category: {category}, Mode: {self.mode}, User: {user_identity}")


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

    agent = InterviewAgent(category, user_identity, metadata)
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

                mode = agent.mode
                result = await evaluate_user_response(
                    transcript,
                    agent.exchange_count - 1,
                    current_question,
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

    mode = agent.mode
    mode_name = MODES[mode]['name']

    if mode == "interview_prep":
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