# Quick Start Guide

This guide will help you get started with Auralis quickly.

## Installation

Install Auralis using pip:

```bash
pip install auralis
```

For development installation with all extras:

```bash
git clone https://github.com/yourusername/Auralis
cd Auralis
pip install -e ".[dev,test]"
```

## Basic Usage

### Simple Text-to-Speech

```python
from auralis import TTS

# Initialize TTS with default settings
tts = TTS()

# Generate speech
audio = tts.generate("Hello, welcome to Auralis!")

# Save to file
audio.save("welcome.wav")
```

### Multi-Language Support

```python
# Generate speech in different languages
tts = TTS()

# Spanish
audio_es = tts.generate(
    "¡Hola, bienvenido a Auralis!",
    language="es"
)

# Japanese
audio_ja = tts.generate(
    "Auralisへようこそ！",
    language="ja"
)
```

### Voice Cloning

```python
# Clone a voice from a reference audio
tts = TTS()

audio = tts.generate(
    "This is a cloned voice speaking.",
    reference_audio="speaker.wav"
)
```

## Advanced Features

### Batch Processing

```python
# Process multiple texts in batch
texts = [
    "First sentence to synthesize.",
    "Second sentence to synthesize.",
    "Third sentence to synthesize."
]

audios = tts.generate_batch(texts)
```

### Performance Optimization

```python
# Enable VLLM optimization
tts = TTS(use_vllm=True)

# Configure batch size and cache
tts.configure(
    batch_size=4,
    use_kv_cache=True
)
```

### Custom Audio Configuration

```python
from auralis.models.xttsv2 import XTTSAudioConfig

# Create custom audio config
audio_config = XTTSAudioConfig(
    sample_rate=44100,
    mel_channels=80,
    hop_length=256
)

# Initialize TTS with audio config
tts = TTS(audio_config=audio_config)
```

## Next Steps

- Check out the [Performance Tuning Guide](../advanced/performance-tuning.md) for optimization
- Learn about [Model Configuration](../api/models/xttsv2.md#configuration)
- Explore [Deployment Options](../advanced/deployment.md)

!!! tip "Best Practices"
    - Always use appropriate logging for production deployments
    - Consider batch processing for multiple texts
    - Enable VLLM for better performance
    - Use voice cloning carefully to match your use case