# TTS Core API

This page documents the core TTS functionality of Auralis.

## TTS Class

::: auralis.core.tts.TTS
    options:
        show_root_heading: true
        show_source: true
        members:
            - __init__
            - generate
            - generate_batch
            - configure

## Usage Examples

### Basic Usage

```python
from auralis import TTS

# Initialize TTS
tts = TTS()

# Generate speech
audio = tts.generate("Hello, world!")
audio.save("output.wav")
```

### Batch Processing

```python
# Generate multiple audio files
texts = [
    "First sentence.",
    "Second sentence.",
    "Third sentence."
]

audios = tts.generate_batch(texts)
```

### Configuration

```python
# Configure TTS settings
tts.configure(
    batch_size=4,
    use_vllm=True,
    use_kv_cache=True
)
```

## See Also

- [Base Model Documentation](base.md): Base model implementation
- [XTTSv2 Documentation](../models/xttsv2.md): XTTSv2 model details
- [Performance Tuning](../../advanced/performance-tuning.md): Optimization guide