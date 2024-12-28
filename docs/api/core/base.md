# Base Model

This page documents the base model functionality that all TTS models in Auralis inherit from.

## BaseAsyncTTSEngine

::: auralis.models.base.BaseAsyncTTSEngine
    options:
        show_root_heading: true
        show_source: true
        members:
            - get_generation_context
            - process_tokens_to_speech
            - conditioning_config

## Implementation Guide

When implementing a new model, you need to inherit from `BaseAsyncTTSEngine` and implement these key methods:

### get_generation_context

```python
async def get_generation_context(self, text: str, **kwargs):
    """Prepare model for generation.
    
    Args:
        text: Input text to process
        **kwargs: Additional arguments
        
    Returns:
        tuple: (generators, other_params)
    """
    # Your implementation here
```

### process_tokens_to_speech

```python
async def process_tokens_to_speech(self, tokens, **kwargs):
    """Convert tokens to audio.
    
    Args:
        tokens: Input tokens
        **kwargs: Additional arguments
        
    Returns:
        numpy.ndarray: Audio waveform
    """
    # Your implementation here
```

### conditioning_config

```python
@property
def conditioning_config(self):
    """Define conditioning parameters.
    
    Returns:
        dict: Conditioning configuration
    """
    return {
        "use_speaker_embedding": True,
        "use_language_embedding": True
    }
```

## Example Implementation

```python
from auralis.models.base import BaseAsyncTTSEngine

class MyCustomEngine(BaseAsyncTTSEngine):
    """Custom TTS engine implementation."""
    
    async def get_generation_context(self, text, **kwargs):
        # Process text and prepare generators
        return generators, params
        
    async def process_tokens_to_speech(self, tokens, **kwargs):
        # Generate audio from tokens
        return audio_waveform
        
    @property
    def conditioning_config(self):
        return {
            "use_speaker_embedding": True
        }
```

## See Also

- [Adding Models Guide](../../advanced/adding-models.md): Complete guide to adding new models
- [XTTSv2 Implementation](../models/xttsv2.md): Example model implementation