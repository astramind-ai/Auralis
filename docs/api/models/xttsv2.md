# XTTSv2 Model

This page documents the XTTSv2 model implementation.

## Model Architecture

::: auralis.models.xttsv2.XTTSv2
    options:
        show_root_heading: true
        show_source: true

## Configuration

::: auralis.models.xttsv2.config.xttsv2_config.XTTSConfig
    options:
        show_root_heading: true
        show_source: true

::: auralis.models.xttsv2.config.xttsv2_gpt_config.XTTSGPTConfig
    options:
        show_root_heading: true
        show_source: true

## Components

### Encoders and Decoders

The XTTSv2 model uses several specialized components for text and audio processing:

- **Latent Encoder**: Converts input text into latent representations
- **Perceiver Encoder**: Processes and conditions the latent representations
- **HiFi-GAN Decoder**: Generates high-fidelity audio from processed representations

## VLLM Integration

::: auralis.models.xttsv2.components.vllm.hijack.ExtendedSamplingParams
    options:
        show_root_heading: true
        show_source: true

::: auralis.models.xttsv2.components.vllm.hidden_state_collector.HiddenStatesCollector
    options:
        show_root_heading: true
        show_source: true

## Usage Examples

```python
from auralis import TTS
from auralis.models.xttsv2 import XTTSv2Config

# Create custom config
config = XTTSv2Config(
    hidden_size=1024,
    num_hidden_layers=30
)

# Initialize with config
tts = TTS(model_config=config)

# Generate speech
audio = tts.generate("Hello, world!")
```

## Performance Tips

!!! tip "Optimization"
    - Use VLLM for faster inference
    - Enable KV cache for repeated generations
    - Batch similar length inputs together
    - Configure appropriate hidden size for your use case 