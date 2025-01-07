from auralis.common.definitions.dto.requests import TTSRequest
from auralis.core.tts import TTS
import pytest


@pytest.mark.timeout(300)
def test_tts_sync_generator(default_test_params):
    tts = TTS().from_pretrained(default_test_params['tts_model'], gpt_model=default_test_params['gpt_model'])

    request = TTSRequest(
        text=default_test_params['text'],
        speaker_files=[default_test_params['speaker_file']],
        stream=True,
    )
    for _ in range(default_test_params['n_iterations_parallel_requests']): # Consume n results
        generator = tts.generate_speech(request)

        output_list = []
        # Consume the results
        for result in generator:
            if isinstance(result, Exception):
                raise result
            else:
                output_list.append(result)

    assert len(output_list) > 0

@pytest.mark.timeout(300)
def test_tts_generation(default_test_params):
    # Create a TTS request
    tts = TTS().from_pretrained(default_test_params['tts_model'], gpt_model=default_test_params['gpt_model'])

    request = TTSRequest(
        text=default_test_params['text'],
        speaker_files=[default_test_params['speaker_file']],
        stream=False,
    )
    for _ in range(default_test_params['n_iterations_parallel_requests']): # Consume 5 results
        audio = tts.generate_speech(request)

    assert len(audio.array) > 0
