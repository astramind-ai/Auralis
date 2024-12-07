# Welcome to Auralis

Auralis is a high-performance text-to-speech (TTS) library that improves upon x-tts with optimized inference and enhanced features.

## Features

- **High Performance**: Optimized inference with VLLM integration
- **Multi-Language Support**: Support for 17+ languages out of the box
- **Easy Integration**: Simple API for both basic and advanced use cases
- **Flexible Architecture**: Easy to extend with new models and components
- **Production Ready**: Built-in logging, metrics, and error handling

## Quick Installation

```bash
pip install auralis
```

## Basic Usage

```python
from auralis import TTS

# Initialize TTS
tts = TTS()

# Generate speech
audio = tts.generate("Hello, world!")

# Save to file
audio.save("output.wav")
```

## Why Auralis?

!!! tip "Key Benefits"
    - 🚀 **Faster Inference**: Optimized for speed with VLLM
    - 🌍 **Language Support**: 17+ languages supported
    - 🎯 **Easy to Use**: Simple, intuitive API
    - 📊 **Monitoring**: Built-in logging and metrics
    - 🔧 **Extensible**: Easy to add new models

## Project Structure

```
auralis/
├── core/           # Core TTS functionality
├── models/         # Model implementations
│   └── xttsv2/    # XTTSv2 model and components
├── common/         # Shared utilities
└── docs/          # Documentation
```

## Getting Started

Check out our [Quick Start Guide](getting-started/quickstart.md) to begin using Auralis in your projects.

## Documentation

- [Performance Tuning](advanced/performance-tuning.md): Optimize for production
- [Deployment Guide](advanced/deployment.md): Deploy in production
- [Adding Models](advanced/adding-models.md): Extend with your models
- [Using OAI Server](advanced/using-oai-server.md): OpenAI API compatibility
- [Logging Reference](api/common/logging.md): Logging system
- [Documentation Guide](contributing/documentation.md): Contribute to docs

## Need Help?

- Check the documentation sections above
- Open an issue on GitHub
- Join our community discussions 