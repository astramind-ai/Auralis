import argparse
import base64
import json
import logging
import uuid
from typing import Optional

import aiohttp
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Response
from fastapi.responses import JSONResponse
from starlette.responses import StreamingResponse

from auralis.core.tts import TTS
from auralis.common.definitions.openai import VoiceChatCompletionRequest, AudioSpeechGenerationRequest

# Initialize FastAPI application
app = FastAPI()

# Global TTS engine instance
tts_engine: Optional[TTS] = None

# Mapping of logging level strings to their corresponding logging constants
logger_str_to_logging = {
    "info": logging.INFO,
    "warn": logging.WARNING,
    "err": logging.ERROR
}

def start_tts_engine(args, logging_level):
    """Initialize the Text-to-Speech engine with specified parameters

    Args:
        args: Parsed command line arguments containing model configurations
        logging_level: Logging level for the TTS engine
    """
    global tts_engine
    tts_engine = (TTS(
        scheduler_max_concurrency=args.max_concurrency,
        vllm_logging_level=logging_level)
    .from_pretrained(
        args.model, gpt_model=args.gpt_model
    ))

@app.on_event("startup")
async def startup_event():
    """Initialize the TTS engine on FastAPI server startup with default parameters"""
    global tts_engine
    if tts_engine is None:
        # Use default arguments for startup
        args = argparse.Namespace(
            model='AstraMindAI/xttsv2',
            gpt_model='AstraMindAI/xtts2-gpt',
            max_concurrency=8,
            vllm_logging_level='warn'
        )
        logging_level = logger_str_to_logging.get(args.vllm_logging_level)
        start_tts_engine(args, logging_level)

@app.post("/v1/audio/speech")
async def generate_audio(request: AudioSpeechGenerationRequest):
    """Generate audio from text using the TTS engine

    Args:
        request: Audio speech generation request containing text and parameters

    Returns:
        Response containing generated audio in requested format

    Raises:
        HTTPException: If TTS engine is not initialized or generation fails
    """
    if tts_engine is None:
        raise HTTPException(status_code=500, detail="TTS engine not initialized")

    try:
        # Create TTSRequest with default params and auralis overrides
        tts_request = request.to_tts_request()

        # Generate speech and adjust speed
        output = await tts_engine.generate_speech_async(tts_request)
        output = output.change_speed(request.speed)
        audio_bytes = output.to_bytes(request.response_format)

        return Response(content=audio_bytes, media_type=f"audio/{request.response_format}")

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Error generating audio: {str(e)}"})

@app.post("/v1/chat/completions")
async def chat_completions(request: VoiceChatCompletionRequest, authorization: Optional[str] = Header(None)):
    """Handle chat completions with optional audio generation

    Args:
        request: Voice chat completion request containing prompt and parameters
        authorization: Bearer token for API authentication

    Returns:
        StreamingResponse containing text and/or audio chunks

    Raises:
        HTTPException: If TTS engine is not initialized or request fails
    """
    if tts_engine is None:
        raise HTTPException(status_code=500, detail="TTS engine not initialized")

    # Validate authorization header
    if not authorization or not authorization.startswith("Bearer "):
        return JSONResponse(
            status_code=400,
            content={"error": "Authorization header with Bearer token is required"}
        )
    try:
        # Extract request parameters
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

        # Validate requested modalities
        valid_modalities = ['text', 'audio']
        if not all(m in valid_modalities for m in modalities):
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid modalities. Must be one or more of {valid_modalities}"}
            )

        async def stream_generator():
            """Generator function for streaming text and audio responses

            Yields:
                JSON-formatted strings containing text chunks and/or audio data
            """
            accumulated_content = ""

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(request.openai_api_url, json=openai_request_data, headers=headers) as resp:
                        if resp.status != 200:
                            error_response = await resp.text()
                            raise HTTPException(status_code=resp.status, detail=error_response)

                        # Process streaming response line by line
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
                                    # Accumulate content and handle text/audio generation
                                    accumulated_content += content
                                    # Stream text if requested
                                    if 'text' in modalities:
                                        yield f"data: {json.dumps(data)}\n\n"

                                    # Generate audio when word threshold is reached
                                    if len(accumulated_content.split()) >= num_of_token_to_vocalize:
                                        if 'audio' in modalities:
                                            tts_request.text = accumulated_content
                                            tts_request.infer_language()
                                            audio_output = await tts_engine.generate_speech_async(tts_request)
                                            audio_base64 = base64.b64encode(audio_output.to_bytes()).decode("utf-8")
                                            yield f"data: {json.dumps({'id': request_id, 'object': 'audio.chunk', 'data': audio_base64})}\n\n"

                                        accumulated_content = ""
                                elif 'text' in modalities:
                                    # Stream non-content text events
                                    yield f"data: {json.dumps(data)}\n\n"

                            except json.JSONDecodeError:
                                continue

                # Process any remaining content for audio
                if accumulated_content and 'audio' in modalities:
                    tts_request.text = accumulated_content
                    tts_request.infer_language()
                    audio_output = await tts_engine.generate_speech_async(tts_request)
                    audio_base64 = base64.b64encode(audio_output.to_bytes()).decode("utf-8")
                    yield f"data: {json.dumps({'id': request_id, 'object': 'audio.chunk', 'data': audio_base64})}\n\n"

                # Send completion messages for text modality
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
    """Main function to configure and start the FastAPI server"""
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description="Auralis TTS FastAPI Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--model", type=str, default='AstraMindAI/xttsv2', help="The base model to run")
    parser.add_argument("--gpt_model", type=str, default='AstraMindAI/xtts2-gpt', help="The gpt model to load alongside the base model, if present")
    parser.add_argument("--max_concurrency", type=int, default=8, help="The concurrency value that is used in the TTS Engine, it is directly connected to the memory consumption")
    parser.add_argument("--vllm_logging_level", type=str, default='warn', help="The vllm logging level, could be one of [info | warn | err]")

    args = parser.parse_args()

    # Initialize the TTS engine
    logging_level = logger_str_to_logging.get(args.vllm_logging_level, None)
    if not logging_level:
        raise ValueError("The logging level for vllm was not correct, please choose between ['info' | 'warn' | 'err']")

    start_tts_engine(args, logging_level)

    # Start the FastAPI server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
    )

if __name__ == "__main__":
    main()