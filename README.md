[![](https://dcbadge.limes.pink/api/server/https://discord.gg/BEMVTmcPEs)](https://discord.gg/https://discord.gg/BEMVTmcPEs)

# Auralis 🌌 (/auˈralis/)

Transform text into natural speech (with voice cloning) at warp speed. Process an entire novel in minutes, not hours.

## What is Auralis? 🚀

Auralis is a text-to-speech engine that makes voice generation practical for real-world use:

- Convert the entire first Harry Potter book to speech in 10 minutes (**realtime factor of ≈ 0.02x!** )
- Automatically enhance the reference quality, you can register them even with a low quality mic!
- It can be configured to have a small memory footprint (scheduler_max_concurrency)
- Process multiple requests simultaneously
- Stream long texts piece by piece

## Quick Start ⭐

1. Create a new Conda environment:
   ```bash
   conda create -n auralis_env python=3.10 -y
   ```

2. Activate the environment:
   ```bash
   conda activate auralis_env
   ```

3. Install Auralis:
   ```bash
   pip install auralis
   ```

and then you can try it out via **python**

```python
from auralis import TTS, TTSRequest

# Initialize
tts = TTS().from_pretrained("AstraMindAI/xttsv2", gpt_model='AstraMindAI/xtts2-gpt')

# Generate speech
request = TTSRequest(
    text="Hello Earth! This is Auralis speaking.",
    speaker_files=['reference.wav']
)

output = tts.generate_speech(request)
output.save('hello.wav')
```

or via **cli** using the openai compatible server
```commandline
auralis.openai --host 127.0.0.1 --port 8000 --model AstraMindAI/xttsv2 --gpt_model AstraMindAI/xtts2-gpt --max_concurrency 8 --vllm_logging_level warn  
```
You can see [here](https://github.com/astramind-ai/Auralis/tree/main/docs/USING_OAI_SERVER.md) for a more in-depth explanation or try it out with this [example](https://github.com/astramind-ai/Auralis/tree/main/examples/use_openai_server.py)

## Key Features 🛸

### Speed & Efficiency
- Processes long texts rapidly using smart batching
- Runs on consumer GPUs without memory issues
- Handles multiple requests in parallel

### Easy Integration
- Simple Python API
- Streaming support for long texts
- Built-in audio enhancement
- Automatic language detection

### Audio Quality
- Voice cloning from short samples
- Background noise reduction
- Speech clarity enhancement
- Volume normalization

## XTTSv2 Finetunes

You can use your own XTTSv2 finetunes by simply converting them from the standard coqui checkpoint format to our safetensor format. Use [this script](https://github.com/astramind-ai/Auralis/blob/main/src/auralis/models/xttsv2/utils/checkpoint_converter.py):
```commandline
python checkpoint_converter.py path/to/checkpoint.pth --output_dir path/to/output
```

it will create two folders, one with the core xttsv2 checkpoint and one with the gtp2 component. Then create a TTS instance with 
```python
tts = TTS().from_pretrained("som/core-xttsv2_model", gpt_model='some/xttsv2-gpt_model')
```

## Examples & Usage 🚀

### Basic Examples ⭐

<details>
<summary><b>Simple Text Generation</b></summary>

```python
from auralis import TTS, TTSRequest

# Initialize
tts = TTS().from_pretrained("AstraMindAI/xttsv2", gpt_model='AstraMindAI/xtts2-gpt')
# Basic generation
request = TTSRequest(
    text="Hello Earth! This is Auralis speaking.",
    speaker_files=["speaker.wav"]
)
output = tts.generate_speech(request)
output.save("hello.wav")
```
</details>

<details>
<summary><b>Working with TTSRequest</b> 🎤</summary>

```python
# Basic request
request = TTSRequest(
    text="Hello world!",
    speaker_files=["speaker.wav"]
)

# Enhanced audio processing
request = TTSRequest(
    text="Pristine audio quality",
    speaker_files=["speaker.wav"],
    audio_config=AudioPreprocessingConfig(
        normalize=True,
        trim_silence=True,
        enhance_speech=True,
        enhance_amount=1.5
    )
)

# Language-specific request
request = TTSRequest(
    text="Bonjour le monde!",
    speaker_files=["speaker.wav"],
    language="fr"
)

# Streaming configuration
request = TTSRequest(
    text="Very long text...",
    speaker_files=["speaker.wav"],
    stream=True,
)

# Generation parameters
request = TTSRequest(
    text="Creative variations",
    speaker_files=["speaker.wav"],
    temperature=0.8,
    top_p=0.9,
    top_k=50
)
```
</details>

<details>
<summary><b>Working with TTSOutput</b> 🎧</summary>

```python
# Load audio file
output = TTSOutput.from_file("input.wav")

# Format conversion
output.bit_depth = 32
output.channel = 2
tensor_audio = output.to_tensor()
audio_bytes = output.to_bytes()



# Audio processing
resampled = output.resample(target_sr=44100)
faster = output.change_speed(1.5)
num_samples, sample_rate, duration = output.get_info()

# Combine multiple outputs
combined = TTSOutput.combine_outputs([output1, output2, output3])

# Playback and saving
output.play()  # Play audio
output.preview()  # Smart playback (Jupyter/system)
output.save("processed.wav", sample_rate=44100)
```
</details>

### Synchronous Advanced Examples 🌟

<details>
<summary><b>Batch Text Processing</b></summary>

```python
# Process multiple texts with same voice
texts = ["First paragraph.", "Second paragraph.", "Third paragraph."]
requests = [
    TTSRequest(
        text=text,
        speaker_files=["speaker.wav"]
    ) for text in texts
]

# Sequential processing with progress
outputs = []
for i, req in enumerate(requests, 1):
    print(f"Processing text {i}/{len(requests)}")
    outputs.append(tts.generate_speech(req))

# Combine all outputs
combined = TTSOutput.combine_outputs(outputs)
combined.save("combined_output.wav")
```
</details>

<details>
<summary><b>Book Chapter Processing</b></summary>

```python
def process_book(chapter_file: str, speaker_file: str):
    # Read chapter
    with open(chapter_file, 'r') as f:
        chapter = f.read()
    
    # You can pass the whole book, auralis will take care of splitting
    
    request = TTSRequest(
            text=chapter,
            speaker_files=[speaker_file],
            audio_config=AudioPreprocessingConfig(
                enhance_speech=True,
                normalize=True
            )
        )
        
    output = tts.generate_speech(request)
    
    output.play()
    output.save("chapter_output.wav")
```
</details>

### Asynchronous Examples 🛸

<details>
<summary><b>Basic Async Generation</b></summary>

```python
import asyncio
from auralis import TTS, TTSRequest

async def generate_speech():
    tts = TTS().from_pretrained("AstraMindAI/xttsv2", gpt_model='AstraMindAI/xtts2-gpt')
    
    request = TTSRequest(
        text="Async generation example",
        speaker_files=["speaker.wav"]
    )
    
    output = await tts.generate_speech_async(request)
    output.save("async_output.wav")

asyncio.run(generate_speech())
```
</details>

<details>
<summary><b>Parallel Processing</b></summary>

```python
async def generate_parallel():
    tts = TTS().from_pretrained("AstraMindAI/xttsv2", gpt_model='AstraMindAI/xtts2-gpt')
    
    # Create multiple requests
    requests = [
        TTSRequest(
            text=f"This is voice {i}",
            speaker_files=[f"speaker_{i}.wav"]
        ) for i in range(3)
    ]
    
    # Process in parallel
    coroutines = [tts.generate_speech_async(req) for req in requests]
    outputs = await asyncio.gather(*coroutines, return_exceptions=True)
    
    # Handle results
    valid_outputs = [
        out for out in outputs 
        if not isinstance(out, Exception)
    ]
    
    combined = TTSOutput.combine_outputs(valid_outputs)
    combined.save("parallel_output.wav")

asyncio.run(generate_parallel())
```
</details>

<details>
<summary><b>Async Streaming with Multiple Requests</b></summary>

```python
async def stream_multiple_texts():
    tts = TTS().from_pretrained("AstraMindAI/xttsv2", gpt_model='AstraMindAI/xtts2-gpt')
    
    # Prepare streaming requests
    texts = [
        "First long text...",
        "Second long text...",
        "Third long text..."
    ]
    
    requests = [
        TTSRequest(
            text=text,
            speaker_files=["speaker.wav"],
            stream=True,
        ) for text in texts
    ]
    
    # Process streams in parallel
    coroutines = [tts.generate_speech_async(req) for req in requests]
    streams = await asyncio.gather(*coroutines)
    
    # Collect outputs
    output_container = {i: [] for i in range(len(requests))}
    
    async def process_stream(idx, stream):
        async for chunk in stream:
            output_container[idx].append(chunk)
            print(f"Processed chunk for text {idx+1}")
            
    # Process all streams
    await asyncio.gather(
        *(process_stream(i, stream) 
          for i, stream in enumerate(streams))
    )
    
    # Save results
    for idx, chunks in output_container.items():
        TTSOutput.combine_outputs(chunks).save(
            f"text_{idx}_output.wav"
        )

asyncio.run(stream_multiple_texts())
```
</details>


## Core Classes 🌟

<details>
<summary><b>TTSRequest</b> - Unified request container with audio enhancement 🎤</summary>

```python
@dataclass
class TTSRequest:
    """Container for TTS inference request data"""
    # Request metadata
    text: Union[AsyncGenerator[str, None], str, List[str]]

    speaker_files: Union[List[str], bytes]  # Path to the speaker audio file

    enhance_speech: bool = True
    audio_config: AudioPreprocessingConfig = field(default_factory=AudioPreprocessingConfig)
    language: SupportedLanguages = "auto"
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    load_sample_rate: int = 22050
    sound_norm_refs: bool = False

    # Voice conditioning parameters
    max_ref_length: int = 60
    gpt_cond_len: int = 30
    gpt_cond_chunk_len: int = 4

    # Generation parameters
    stream: bool = False
    temperature: float = 0.75
    top_p: float = 0.85
    top_k: int = 50
    repetition_penalty: float = 5.0
    length_penalty: float = 1.0
    do_sample: bool = True
```

### Examples

```python
# Basic usage
request = TTSRequest(
    text="Hello world!",
    speaker_files=["reference.wav"]
)

# With custom audio enhancement
request = TTSRequest(
    text="Hello world!",
    speaker_files=["reference.wav"],
    audio_config=AudioPreprocessingConfig(
        normalize=True,
        trim_silence=True,
        enhance_speech=True,
        enhance_amount=1.5
    )
)

# Streaming long text
request = TTSRequest(
    text="Very long text...",
    speaker_files=["reference.wav"],
    stream=True,
)
```

### Features
- Automatic language detection
- Audio preprocessing & enhancement
- Flexible input handling (strings, lists, generators)
- Configurable generation parameters
- Caching for efficient processing

</details>

<details>
<summary><b>TTSOutput</b> - Unified output container for audio processing 🎧</summary>

```python
@dataclass
class TTSOutput:
    array: np.ndarray
    sample_rate: int
```

### Methods

#### Format Conversion
```python
output.to_tensor()      # → torch.Tensor
output.to_bytes()       # → bytes (wav/raw)
output.from_tensor()    # → TTSOutput
output.from_file()      # → TTSOutput
```

#### Audio Processing
```python
output.combine_outputs()  # Combine multiple outputs
output.resample()        # Change sample rate
output.get_info()        # Get audio properties
output.change_speed()    # Modify playback speed
```

#### File & Playback
```python
output.save()           # Save to file
output.play()          # Play audio
output.display()       # Show in Jupyter
output.preview()       # Smart playback
```

### Examples

```python
# Load and process
output = TTSOutput.from_file("input.wav")
output = output.resample(target_sr=44100)
output.save("output.wav")

# Combine multiple outputs
combined = TTSOutput.combine_outputs([output1, output2, output3])

# Change playback speed
faster = output.change_speed(1.5)
```

</details>


## Languages 🌍

XTTSv2 Supports: English, Spanish, French, German, Italian, Portuguese, Polish, Turkish, Russian, Dutch, Czech, Arabic, Chinese (Simplified), Hungarian, Korean, Japanese, Hindi

## Performance Details 📊

Processing speeds on NVIDIA 3090:
- Short phrases (< 100 chars): ~1 second
- Medium texts (< 1000 chars): ~5-10 seconds
- Full books (~500K chars @ concurrency 36): ~10 minutes

Memory usage:
- Base: ~2.5GB VRAM concurrency = 1
- ~ 5.3GB VRAM concurrency = 20


## Contributions

**Join Our Community!**

We welcome and appreciate any contributions to our project! To ensure a smooth and efficient process, please take a moment to review our [Contribution Guideline](https://github.com/astramind-ai/Auralis/blob/main/CONTRIBUTING.md). By following these guidelines, you'll help us review and accept your contribution quickly. Thank you for your support!


## Learn More 🔭

- [Technical Deep Dive](https://www.astramind.ai/post/auralis)
- [Adding Custom Models](docs/ADDING_MODELS.md)

## License

The codebase is released under Apache 2.0, feel free to use it in your projects.

The XTTSv2 model (and the files under auralis/models/xttsv2/components/tts) are licensed under the [Coqui AI License](https://coqui.ai/cpml).
