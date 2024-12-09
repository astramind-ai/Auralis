# How to Add Your Own Models to Auralis

So, you want to bring your own TTS models into the mix? Sweet! Here's how you can plug them into Auralis.

## Step-by-Step Guide

### 1. Create a New Engine Class

Your model needs to inherit from `BaseAsyncTTSEngine` and implement a few methods.
Create a new folder under auralis/models/[your_model] and inside putall of your files
```python
from auralis.models.base import BaseAsyncTTSEngine
from auralis.models.registry import MODEL_REGISTRY

class MyCustomEngine(BaseAsyncTTSEngine):
    # Implement the required methods here
```

And in your main code, before the call to the model (or in the init file of the model folder) 
```python
from auralis.models.registry import register_model
from auralis.models.your_new_model import YourModelArch

register_model("yourmodel", YourModelArch) # the lower caps name, must be the same in the model arch under model type and also the same on the config file
```

### 2. Implement Required Methods

You'll need to implement:

- `get_generation_context`: Prepares your model for generation and returns genetators alognside as other parameter.
- `process_tokens_to_speech`: Converts tokens into audio.
- `conditioning_config`: Defines how your model handles conditioning like speaker embeddings.

Check out `xttsv2_engine.py` for inspiration.

### 3. Implement HF methods such as form pretrained and and make a compatible (fast) tokenier if not already present

To do this steps you can also take inspiration from our xtts implementation

### 4. Update the TTS Class

Make sure the `TTS` class can initialize your model:

```python
if config['model_type'] == 'my_custom_model':
    self.tts_engine = MyCustomEngine.from_pretrained(model_name_or_path, **kwargs)
```

### 5. Handle Conditioning (If Needed)

If your model uses speaker embeddings or other conditioning data, make sure to handle them in `get_generation_context`.

### 6. Test Your Model

Fire up some tests to make sure everything works smoothly.

## Tips and Tricks

- **Async All the Way**: Since Auralis is async, your methods should be too.
- **Semaphore Control**: Use semaphores if your model has heavy computation to manage concurrency.
- **Executor for CPU Tasks**: Use `ThreadPoolExecutor` for CPU-bound tasks to keep the event loop snappy.

## Need Help?

Don't hesitate to reach out if you get stuck. Open an issue, and I'll be happy to help!
