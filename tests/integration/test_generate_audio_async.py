import asyncio

import pytest

from auralis.common.definitions.dto.requests import TTSRequest
from auralis.core.tts import TTS

@pytest.mark.asyncio
async def test_tts_async_multiple_concurrent_generation(default_test_params):
    tts = TTS().from_pretrained(default_test_params['tts_model'], gpt_model=default_test_params['gpt_model'])

    request = TTSRequest(
        text=default_test_params['text'],
        speaker_files=[default_test_params['speaker_file']],
        stream=False,
    )
    # Create requests
    async_requests = [
        TTSRequest(
            text=default_test_params['text'],
            speaker_files=[default_test_params['speaker_file']],
            stream=True,
        ) for _ in range(default_test_params['n_iterations_parallel_requests'])  # Creating 5 requests
    ]
    requests = [
        TTSRequest(
            text=default_test_params['text'],
            speaker_files=[default_test_params['speaker_file']],
            stream=False,
        ) for _ in range(default_test_params['n_iterations_parallel_requests'])  # Creating 5 requests
    ]

    async def process_stream(request, idx):
        try:
            generator = tts.generate_speech_async(request)
            chunks = []
            async for chunk in generator:
                chunks.append(chunk)
            return chunks
        except Exception as e:
            print(f"Error in request {idx}: {e}")
            return None

    ## Process streams concurrently
    tasks = [process_stream(req, i) for i, req in enumerate(async_requests)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for result in results:
        assert len(result) > 0

    non_streaming_async_task = [tts.generate_speech_async(req) for req in requests]
    result_not_streaming = await asyncio.gather(*non_streaming_async_task, return_exceptions=True)

    assert len(result_not_streaming) > 0


