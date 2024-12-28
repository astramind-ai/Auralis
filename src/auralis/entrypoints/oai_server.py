#  Copyright (c) 2024 Astramind. Licensed under Apache License, Version 2.0.

import argparse
import base64
import json
import logging
import uuid
from typing import Optional

import aiohttp
import uvicorn
from auralis.common.definitions.protocols.openai import VoiceChatCompletionRequest, AudioSpeechGenerationRequest
from fastapi import FastAPI, Header, HTTPException, Response
from fastapi.responses import JSONResponse
from starlette.responses import StreamingResponse

from auralis.core.tts import TTS

app = FastAPI()

tts_engine: TTS

logger_str_to_logging={
    "info": logging.INFO,
    "warn": logging.WARNING,
    "err": logging.ERROR
}

def start_tts_engine(args, logging_level):
    global tts_engine
    tts_engine = (TTS(
        scheduler_max_concurrency=args.max_concurrency,
        vllm_logging_level=logging_level)
    .from_pretrained(
        args.model, gpt_model=args.gpt_model
    ))

@app.post("/v1/audio/speech")
async def generate_audio(request: AudioSpeechGenerationRequest):

    try:
        # Create TTSRequest with default params and auralis overrides
        tts_request = request.to_tts_request()

        output = await tts_engine.generate_speech_async(tts_request)
        output = output.change_speed(request.speed)
        audio_bytes = output.to_bytes(request.response_format)

        return Response(content=audio_bytes, media_type=f"audio/{request.response_format}")

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Error generating audio: {str(e)}"})


@app.post("/v1/chat/completions")
async def chat_completions(request: VoiceChatCompletionRequest, authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        return JSONResponse(
            status_code=400,
            content={"error": "Authorization header with Bearer token is required"}
        )
    try:
        # Rest of the parameters
        openai_api_key = authorization[len("Bearer "):]
        modalities = request.modalities
        num_of_token_to_vocalize = request.vocalize_at_every_n_words

        # Initialize TTS request with auralis parameters
        tts_request = request.to_tts_request(text='')

        # Prepare OpenAI request
        openai_request_data = request.to_openai_request()

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}"
        }

        request_id = uuid.uuid4().hex

        # Validate modalities
        valid_modalities = ['text', 'audio']
        if not all(m in valid_modalities for m in modalities):
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid modalities. Must be one or more of {valid_modalities}"}
            )

        async def stream_generator():
            accumulated_content = ""

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(request.openai_api_url, json=openai_request_data, headers=headers) as resp:
                        if resp.status != 200:
                            error_response = await resp.text()
                            raise HTTPException(status_code=resp.status, detail=error_response)

                        async for line in resp.content:
                            if not line:
                                continue

                            line = line.decode("utf-8").strip()
                            if not line.startswith("data:"):
                                continue

                            data_str = line[5:].strip()
                            if data_str == "[DONE]":
                                break

                            try:
                                data = json.loads(data_str)
                                content = data.get("choices", [{}])[0].get("delta", {}).get("content", "")

                                if content:
                                    accumulated_content += content
                                    # Only yield text if text modality is requested
                                    if 'text' in modalities:
                                        yield f"data: {json.dumps(data)}\n\n"

                                    if len(accumulated_content.split()) >= num_of_token_to_vocalize:
                                        # Only generate and yield audio if audio modality is requested
                                        if 'audio' in modalities:
                                            tts_request.text = accumulated_content
                                            tts_request.infer_language()
                                            audio_output = await tts_engine.generate_speech_async(tts_request)
                                            audio_base64 = base64.b64encode(audio_output.to_bytes()).decode("utf-8")
                                            yield f"data: {json.dumps({'id': request_id, 'object': 'audio.chunk', 'data': audio_base64})}\n\n"

                                        accumulated_content = ""
                                elif 'text' in modalities:
                                    # Other non-content text events only if text modality is requested
                                    yield f"data: {json.dumps(data)}\n\n"

                            except json.JSONDecodeError:
                                continue

                # Process any remaining content for audio if needed
                if accumulated_content and 'audio' in modalities:
                    tts_request.text = accumulated_content
                    tts_request.infer_language()
                    audio_output = await tts_engine.generate_speech_async(tts_request)
                    audio_base64 = base64.b64encode(audio_output.to_bytes()).decode("utf-8")
                    yield f"data: {json.dumps({'id': request_id, 'object': 'audio.chunk', 'data': audio_base64})}\n\n"

                # Send completion messages only if text modality is requested
                if 'text' in modalities:
                    yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'choices': [{'delta': {}, 'index': 0, 'finish_reason': 'stop'}]})}\n\n"
                yield "data: [DONE]\n\n"

            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            finally:
                pass

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Error in chat completions: {str(e)}"})

def main():
    parser = argparse.ArgumentParser(description="Auralis TTS FastAPI Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--model",  type=str, default='AstraMindAI/xttsv2', help="The base model to run")
    parser.add_argument("--gpt_model", type=str, default='AstraMindAI/xtts2-gpt', help="The gpt model to load alongside the base model, if present")
    parser.add_argument("--max_concurrency", type=int, default=8, help="The concurrency value that is used in the TTS Engine, it is directly connected to the memory consumption")
    parser.add_argument("--vllm_logging_level", type=str, default='warn', help="The vllm logging level, could be one of [info | warn | err]")

    args = parser.parse_args()

    # Initialize the TTS engine
    logging_level = logger_str_to_logging.get(args.vllm_logging_level, None)
    if not logging_level:
        raise ValueError("The logging level for vllm was not correct, please choose between ['info' | 'warn' | 'err']")

    start_tts_engine(args, logging_level)

    uvicorn.run(
        "auralis.entrypoints.oai_server:app",
        host=args.host,
        port=args.port,
    )

if __name__ == "__main__":
    main()