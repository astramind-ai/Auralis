#  Copyright (c) 2024 Astramind. Licensed under Apache License, Version 2.0.

from pathlib import Path

import requests
import base64
import json
import os
import asyncio
import threading
from queue import Queue
from typing import Generator
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from auralis import TTSOutput


class AudioPlayer:
    def __init__(self):
        self.audio_queue = Queue()
        self.playing = True
        self.player_thread = threading.Thread(target=self._play_audio_worker)
        self.player_thread.daemon = True
        self.player_thread.start()

    def _play_audio_worker(self):
        while self.playing:
            try:
                audio_data = self.audio_queue.get()
                if audio_data is None:  # Sentinel value to stop
                    break
                TTSOutput(array=audio_data).play()
            except Exception as e:
                print(f"Error playing audio: {e}")
            finally:
                self.audio_queue.task_done()

    def add_audio(self, audio_data):
        self.audio_queue.put(audio_data)

    def stop(self):
        self.playing = False
        self.audio_queue.put(None)  # Add sentinel value
        self.player_thread.join()


def process_stream_response(url: str, headers: dict, payload: dict):
    """
    Process streaming response from TTS server with async audio playback
    """
    audio_player = AudioPlayer()

    try:
        with requests.post(url, json=payload, headers=headers, stream=True) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue

                line = line.decode('utf-8')
                if not line.startswith('data: '):
                    continue

                data_str = line[6:]  # Remove 'data: ' prefix
                if data_str == '[DONE]':
                    break

                try:
                    data = json.loads(data_str)

                    # Handle text chunks
                    if data.get("object") == "chat.completion.chunk":
                        content = data.get("choices", [{}])[0].get("delta", {}).get("content")
                        if content:
                            print(f"{content}", end="", flush=True)

                    # Handle audio chunks
                    elif data.get("object") == "audio.chunk":
                        audio_data = base64.b64decode(data["data"])
                        audio_player.add_audio(audio_data)

                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse JSON data: {e}")
                    continue
    finally:
        audio_player.stop()


def process_stream_with_openai(
        client: OpenAI,
        model: str,
        prompt: str,
        auralis_params: dict = None
):
    """
    Process stream using OpenAI SDK with Auralis parameters

    Args:
        client: OpenAI client instance
        model: Model to use for text generation
        prompt: Text prompt for generation
        auralis_params: Optional dictionary of Auralis-specific parameters
    """
    audio_player = AudioPlayer()

    try:
        # Start Auralis request
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            modalities=['text', 'audio'],
            extra_body={
                **auralis_params  # Include all Auralis parameters
            }
        )

        for chunk in stream:
            if not chunk:
                continue
            if getattr(chunk, 'object', None) == 'audio.chunk':
                # Handle audio chunk
                audio_data = base64.b64decode(chunk.data)
                audio_player.add_audio(audio_data)
            elif hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                # Handle text chunk
                print(f"{chunk.choices[0].delta.content}", end="", flush=True)

    finally:
        audio_player.stop()

def generate_from_streaming_source(client, audio_data, auralis_params):
    auralis_params.update({
        # this should be your text generation endpoint, so not where you are running the auralis but the LLM endpoint (OAI, Ollama...)
        'openai_api_url': "http://127.0.0.1:8001/v1/chat/completions",
        'speaker_files': [audio_data],
        'vocalize_at_every_n_words': 40,
    })
    # Process stream using OpenAI
    process_stream_with_openai(
        client=client,
        model='meta-llama/Llama-3.2-1B-Instruct',
        prompt="Tell me a story about a brave knight",
        auralis_params=auralis_params
    )

def vocalize_text_with_tts_endpoint(client, audio_data, auralis_params):
    speech_file_path = Path(__file__).parent / "speech.mp3"

    response = client.audio.speech.create(
        model="xttsv2", # it doesn't actually matter
        voice=[audio_data],
        input="Today is a wonderful day to build something people love!",
        response_format='mp3',
        speed=1.0,
        extra_body={**auralis_params}
    )
    response.stream_to_file(speech_file_path)


def main():
    # Read reference audio
    with open("../tests/resources/audio_samples/female.wav", "rb") as f:
        audio_data = base64.b64encode(f.read()).decode('utf-8')
        # or with OAI
    client = OpenAI(
            api_key="your-openai-api-key",
            base_url="http://127.0.0.1:8000/v1/",  # insert the auralis endpoint, NOT the LLM generation endpoint
    )
    auralis_params = {
        # this should be your text generation endpoint, so not where you are running the auralis but the LLM endpoint (OAI, Ollama...)
        'enhance_speech': True,
        'sound_norm_refs': False,
        'max_ref_length': 60,
        'gpt_cond_len': 30,
        'gpt_cond_chunk_len': 4,
        'temperature': 0.75,
        'top_p': 0.85,
        'top_k': 50,
        'repetition_penalty': 5.0,
        'length_penalty': 1.0,
        'do_sample': True,
        'language': "auto"
    }
    generate_from_streaming_source(client, audio_data, auralis_params)
    vocalize_text_with_tts_endpoint(client, audio_data, auralis_params)

if __name__ == '__main__':
    main()
