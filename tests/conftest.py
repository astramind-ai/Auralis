import pytest

import os
@pytest.fixture
def default_test_params():
    return {
        "tts_model": "AstraMindAI/xttsv2",
        "gpt_model": "AstraMindAI/xtts2-gpt",
        "text": "Hello, how are you? today is a good day, I'll probably have a good day too",
        "speaker_file": os.path.join(os.getcwd(),"..", "resources","audio_samples","female.wav"),
        "n_iterations_mem_leak": 100,
        "n_iterations_parallel_requests": 5,
    }