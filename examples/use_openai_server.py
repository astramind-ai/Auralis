from pathlib import Path
import requests
import base64
import json
import threading
from queue import Queue
from typing import Generator, Optional

from openai import OpenAI
from auralis import TTSOutput

# AudioPlayer class handles asynchronous audio playback using a queue system
class AudioPlayer:
    def __init__(self):
        # Initialize queue for audio data and create a daemon thread for playback
        self.audio_queue = Queue()
        self.playing = True
        self.player_thread = threading.Thread(target=self._play_audio_worker)
        self.player_thread.daemon = True
        self.player_thread.start()

    def _play_audio_worker(self):
        # Worker thread that continuously processes audio data from the queue
        while self.playing:
            try:
                audio_data = self.audio_queue.get()
                if audio_data is None:  # Sentinel value to stop the thread
                    break
                TTSOutput(array=audio_data).play()
            except Exception as e:
                print(f"Error playing audio: {e}")
            finally:
                self.audio_queue.task_done()

    def add_audio(self, audio_data):
        # Add new audio data to the playback queue
        self.audio_queue.put(audio_data)

    def stop(self):
        # Gracefully stop the audio player and wait for thread completion
        self.playing = False
        self.audio_queue.put(None)  # Send sentinel value to stop the worker
        self.player_thread.join()

def generate_llm_response(client: OpenAI, model: str, prompt: str) -> Optional[str]:
    """Generate text response from LLM using the LLM endpoint

    Args:
        client: OpenAI client instance
        model: Name of the language model to use
        prompt: Input text prompt

    Returns:
        Generated text response or None if generation fails
    """
    try:
        LLM_ENDPOINT = "http://localhost:8001/v1/chat/completions"

        # Format the chat template manually for Llama model
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ]

        response = requests.post(
            LLM_ENDPOINT,
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "temperature": 1.0,
                "max_tokens": 100,
                "stop": ["</s>"],
                # Custom chat template for formatting the prompt
                "chat_template": """{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% if system_message %}{{ system_message }}

{% endif %}{% for message in loop_messages %}{% if message['role'] == 'user' %}[INST] {{ message['content'] }} [/INST]
{% elif message['role'] == 'assistant' %}{{ message['content'] }}
{% endif %}{% endfor %}"""
            },
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {client.api_key}"
            }
        )

        if response.status_code == 200:
            response_data = response.json()
            return response_data['choices'][0]['message']['content']
        else:
            print(f"Error generating LLM response: {response.text}")
            return None
    except Exception as e:
        print(f"Error in LLM generation: {e}")
        return None

def process_stream_with_openai(client: OpenAI, model: str, prompt: str, auralis_params: dict = None):
    """Process streaming audio and text responses from the model

    Args:
        client: OpenAI client instance
        model: Model name to use
        prompt: Input prompt
        auralis_params: Parameters for audio processing
    """
    audio_player = AudioPlayer()

    try:
        print("Starting streaming request...")
        AUDIO_ENDPOINT = "http://127.0.0.1:8000/v1/chat/completions"

        # Send streaming request with both text and audio modalities
        response = requests.post(
            AUDIO_ENDPOINT,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
                "modalities": ["text", "audio"],
                "vocalize_at_every_n_words": auralis_params.get('vocalize_at_every_n_words', 40),
                "speaker_files": auralis_params.get('speaker_files')
            },
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {client.api_key}"
            },
            stream=True
        )

        print(f"Server response status: {response.status_code}")
        if response.status_code != 200:
            print(f"Server error: {response.text}")
            return

        # Process streaming response, handling both audio and text chunks
        for line in response.iter_lines():
            if not line:
                continue

            try:
                line = line.decode('utf-8')
                if not line.startswith('data: '):
                    continue

                data_str = line[6:]
                if data_str == '[DONE]':
                    break

                data = json.loads(data_str)

                # Handle audio chunks
                if data.get('object') == 'audio.chunk':
                    audio_data = base64.b64decode(data['data'])
                    audio_player.add_audio(audio_data)
                    print("Audio chunk processed")
                # Handle text chunks
                elif 'choices' in data and data['choices'][0].get('delta', {}).get('content'):
                    content = data['choices'][0]['delta']['content']
                    print(content, end="", flush=True)

            except Exception as inner_e:
                print(f"Error processing chunk: {inner_e}")

    except Exception as e:
        print(f"Error in stream processing: {str(e)}")
        if hasattr(e, 'response'):
            print(f"Response details: {e.response.text if hasattr(e.response, 'text') else 'No response text'}")
    finally:
        audio_player.stop()

def generate_from_streaming_source(client, audio_data, auralis_params):
    """Generate streaming audio and text from a given audio source

    Args:
        client: OpenAI client instance
        audio_data: Reference audio data
        auralis_params: Audio processing parameters
    """
    print("\nPreparing streaming parameters...")
    auralis_params.update({
        'openai_api_url': "http://127.0.0.1:8000/v1/chat/completions",
        'speaker_files': [audio_data],
        'vocalize_at_every_n_words': 40,
    })

    try:
        process_stream_with_openai(
            client=client,
            model='meta-llama/Llama-3.2-1B',
            prompt="Tell me a story about a brave adventurer",
            auralis_params=auralis_params
        )
    except Exception as e:
        print(f"Error in generate_from_streaming_source: {e}")

def vocalize_text_with_tts_endpoint(client, audio_data, auralis_params, text_to_vocalize: str):
    """Convert text to speech using the TTS endpoint

    Args:
        client: OpenAI client instance
        audio_data: Reference audio data
        auralis_params: Audio processing parameters
        text_to_vocalize: Text to convert to speech
    """
    if not text_to_vocalize:
        print("No text to vocalize")
        return

    speech_file_path = Path(__file__).parent / "speech.wav"

    try:
        print("Creating speech request...")
        # Generate speech using streaming response
        with client.audio.speech.with_streaming_response.create(
            model="xttsv2",
            voice=[audio_data],
            input=text_to_vocalize,
            response_format="wav",
            speed=1.0,
            extra_body={
                **auralis_params,
                "speaker_files": [audio_data]
            }
        ) as response:
            print("Writing response to file...")
            # Save audio chunks to file
            with open(speech_file_path, 'wb') as f:
                for chunk in response.iter_bytes():
                    if chunk:
                        f.write(chunk)
            print(f"Audio saved to {speech_file_path}")
    except Exception as e:
        print(f"Error in speech generation: {e}")
        if hasattr(e, 'response'):
            print(f"Response details: {e.response.text if hasattr(e.response, 'text') else 'No response text'}")


def main():
    try:
        print("Starting application...")
        # Read reference audio file
        with open("../tests/resources/audio_samples/female.wav", "rb") as f:
            audio_data = base64.b64encode(f.read()).decode('utf-8')

        # Initialize OpenAI client for audio endpoints
        client = OpenAI(
            api_key="your-openai-api-key",
            base_url="http://127.0.0.1:8000/v1/",
        )

        # Initialize separate client for LLM endpoint
        llm_client = OpenAI(
            api_key="your-openai-api-key",
            base_url="http://localhost:8001/v1/",
        )

        # Set up audio processing parameters
        auralis_params = {
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

        # Generate streaming audio and text
        print("\nStarting streaming source generation...")
        generate_from_streaming_source(client, audio_data, auralis_params)

        # Generate LLM response and convert to speech
        print("\nGenerating LLM response for vocalization...")
        llm_response = generate_llm_response(
            client=llm_client,
            model='meta-llama/Llama-3.2-1B',
            prompt="Tell me an interesting fact about space exploration"
        )

        print("\nStarting speech generation...")
        if llm_response:
            vocalize_text_with_tts_endpoint(client, audio_data, auralis_params, llm_response)
        else:
            print("Failed to generate LLM response for vocalization")

    except Exception as e:
        print(f"Application error: {str(e)}")

if __name__ == '__main__':
    main()