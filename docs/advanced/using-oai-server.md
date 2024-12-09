# Auralis Text-to-Speech API Documentation  
  
Auralis is a text-to-speech system that provides two main functionalities: direct text-to-speech conversion and streaming text-to-speech generation integrated with language models. This documentation explains how to interact with both endpoints.  
  
## Core Concepts  
  
Auralis works by accepting reference audio files that are used to clone voices. These reference files should be high-quality recordings of speech, ideally 6-60 seconds long. The system can then generate speech in that voice, either directly from text or while streaming responses from a language model.  
  
## Start the engine  
```commandline  
auralis.openai --host 127.0.0.1 --port 8000 --model AstraMindAI/xttsv2 --gpt_model AstraMindAI/xtts2-gpt --max_concurrency 8 --vllm_logging_level warn  
```  
  
## API Endpoints  
  
### 1. Direct Text-to-Speech Endpoint  
  
This endpoint converts provided text directly to speech using a cloned voice.  
  
**Endpoint:** `/v1/audio/speech`  
**Method:** POST  
  
#### Python Example  
```python  
from openai import OpenAI  
  
client = OpenAI(  
 api_key="your-api-key",  # Can be any string when using local Auralis base_url="http://127.0.0.1:8000/v1/"  # Your Auralis endpoint)  
with open(reference_audio_path, "rb") as f:  
 audio_data = base64.b64encode(f.read()).decode('utf-8')  # Configure speech parameters  
speech_params = {  
 "enhance_speech": True, "sound_norm_refs": False, "max_ref_length": 60, "gpt_cond_len": 30, "gpt_cond_chunk_len": 4, "temperature": 0.75, "top_p": 0.85, "top_k": 50, "repetition_penalty": 5.0, "length_penalty": 1.0, "do_sample": True, "language": "auto"}  
  
# Generate speech  
response = client.audio.speech.create(  
 model="xttsv2", voice=[audio_data], input=text, response_format="mp3", speed=1.0, extra_body=speech_params)  
  
# Save the audio file  
response.stream_to_file("output.mp3")  
```  
### 2. Streaming Text-to-Speech Endpoint  
  
This endpoint handles real-time text generation and speech synthesis simultaneously.  
  
**Endpoint:** `/v1/chat/completions`  
**Method:** POST  
  
#### Python Example  
```python  
with open(reference_audio_path, "rb") as f:  
 audio_data = base64.b64encode(f.read()).decode('utf-8')  # Configure streaming parameters  
streaming_params = {  
 "openai_api_url": "http://your-llm-endpoint:8001/v1/chat/completions", "speaker_files": [audio_data], "vocalize_at_every_n_words": 40, "enhance_speech": True, "sound_norm_refs": False, "max_ref_length": 60, "gpt_cond_len": 30, "gpt_cond_chunk_len": 4, "temperature": 0.75, "top_p": 0.85, "top_k": 50, "repetition_penalty": 5.0, "length_penalty": 1.0, "do_sample": True, "language": "auto"}  
  
# Start streaming request  
stream = client.chat.completions.create(  
 model="your-llm-model", messages=[{"role": "user", "content": prompt}], stream=True, modalities=["text", "audio"], extra_body=streaming_params)  
audio_chunks = []  
# Process the stream  
for chunk in stream:  
 if not chunk: continue if getattr(chunk, 'object', None) == 'audio.chunk': # Process audio audio_chunks.append(base64.b64decode(chunk.data)) elif hasattr(chunk.choices[0].delta, 'content'): # Process text print(chunk.choices[0].delta.content, end="", flush=True)#Do something with audio_chunks   
```  
## Response Format  
  
### Direct TTS Response  
Returns the audio file in the specified format (mp3, wav, etc.)  
  
### Streaming Response  
Returns a stream of Server-Sent Events (SSE) with the following format:  
  
```  
data: {"object": "chat.completion.chunk", "choices": [{"delta": {"content": "Text chunk"}}]}  
data: {"object": "audio.chunk", "data": "base64_encoded_audio_chunk"}  
data: [DONE]  
```