"""
application/orchestration/voice_flow.py
Full voice pipeline (Rule 3, Rule 13).

  1. Transcribe audio → text (Whisper)
  2. Build memory context (RAG) + system prompt
  3. Get the LLM reply (provider chosen by config, via llm_client — no duplicated
     provider branch; this previously lived inline in voice.py)
  4. Save + optionally ingest to memory
  5. Convert reply to speech (ElevenLabs)
  6. Return transcript, reply text, conv id, and base64 audio
"""

from __future__ import annotations

import base64
import uuid

from fastapi import UploadFile

from audio.speech.speech_response import synthesise_reply
from audio.transcription.transcribe_audio import transcribe_audio
from config.model_settings import MODEL_FAST, VOICE_MAX_TOKENS
from model_runtime.clients import llm_client
from model_runtime.prompt_building.system_prompt_builder import build_system_prompt


async def run_voice_chat(file: UploadFile, conversation_id: str | None) -> dict:
    """Run the full transcribe → reason → speak pipeline. Returns a result dict."""
    user_text = await transcribe_audio(file)
    if not user_text:
        return {"error": "no speech detected"}

    conv_id = conversation_id or str(uuid.uuid4())

    system = build_system_prompt( "", "")

    messages = [{"role": "system", "content": system}]
    messages.append({"role": "user", "content": user_text})

    reply_text = await llm_client.complete(
        messages, model=MODEL_FAST, max_tokens=VOICE_MAX_TOKENS
    )

    audio_bytes = synthesise_reply(reply_text)

    return {
        "transcript": user_text,
        "reply": reply_text,
        "conv_id": conv_id,
        "audio_b64": base64.b64encode(audio_bytes).decode("utf-8"),
    }
