from auralis.common.definitions.dto.requests import TTSRequest
from auralis.core.tts import TTS
import pytest
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor


@pytest.mark.asyncio
async def test_tts_high_concurrency_async(default_test_params):
    """Test multiple concurrent async requests"""
    tts = TTS(scheduler_max_concurrency=10).from_pretrained(
        default_test_params['tts_model'],
        gpt_model=default_test_params['gpt_model']
    )

    requests = []
    for i in range(10):  # Create 10 different requests
        request = TTSRequest(
            text=f"{default_test_params['text']} {i}",
            speaker_files=[default_test_params['speaker_file']],
            stream=False
        )
        requests.append(request)

    # Launch all requests concurrently
    start_time = time.time()
    tasks = [tts.generate_speech_async(req) for req in requests]
    results = await asyncio.gather(*tasks)
    end_time = time.time()

    # Verify results
    assert len(results) == 10
    assert all(len(audio.array) > 0 for audio in results)

    # Check that concurrent processing was faster than sequential
    concurrent_time = end_time - start_time

    # Now do it sequentially for comparison
    start_time = time.time()
    for req in requests:
        audio = await tts.generate_speech_async(req)
        assert len(audio.array) > 0
    sequential_time = time.time() - start_time

    assert concurrent_time < sequential_time


def test_tts_mixed_streaming_requests_sequential(default_test_params):
    """Test mix of streaming and non-streaming requests sequentially"""
    tts = TTS(scheduler_max_concurrency=10).from_pretrained(
        default_test_params['tts_model'],
        gpt_model=default_test_params['gpt_model']
    )

    # Create mix of streaming and non-streaming requests
    requests = []
    for i in range(5):
        requests.extend([
            TTSRequest(
                text=default_test_params['text'],
                speaker_files=[default_test_params['speaker_file']],
                stream=True
            ),
            TTSRequest(
                text=default_test_params['text'],
                speaker_files=[default_test_params['speaker_file']],
                stream=False
            )
        ])

    results = []
    for req in requests:
        if req.stream:
            chunks = []
            for chunk in tts.generate_speech(req):
                chunks.append(chunk)
            results.append(chunks)
        else:
            results.append(tts.generate_speech(req))

    # Verify results
    assert len(results) == 10
    for result in results:
        if isinstance(result, list):  # streaming result
            assert len(result) > 0
            assert all(len(chunk.array) > 0 for chunk in result)
        else:  # non-streaming result
            assert len(result.array) > 0

@pytest.mark.asyncio
async def test_tts_stress_test(default_test_params):
    """Stress test with many concurrent requests and varying text lengths"""
    tts = TTS(scheduler_max_concurrency=20).from_pretrained(
        default_test_params['tts_model'],
        gpt_model=default_test_params['gpt_model']
    )

    # Create requests with varying text lengths
    requests = []
    text_multipliers = [1, 2, 3, 4, 5]  # Different text lengths
    for i in range(20):  # 20 concurrent requests
        text_mult = text_multipliers[i % len(text_multipliers)]
        request = TTSRequest(
            text=default_test_params['text'] * text_mult,
            speaker_files=[default_test_params['speaker_file']],
            stream=i % 2 == 0  # Alternate between streaming and non-streaming
        )
        requests.append(request)

    async def process_request(req):
        if req.stream:
            chunks = []
            async for chunk in await tts.generate_speech_async(req):
                chunks.append(chunk)
            return chunks
        else:
            return await tts.generate_speech_async(req)

    # Launch all requests concurrently
    start_time = time.time()
    tasks = [process_request(req) for req in requests]
    results = await asyncio.gather(*tasks)
    processing_time = time.time() - start_time

    # Verify results
    assert len(results) == 20
    for result in results:
        if isinstance(result, list):  # streaming result
            assert len(result) > 0
            assert all(len(chunk.array) > 0 for chunk in result)
        else:  # non-streaming result
            assert len(result.array) > 0

    # Log performance metrics
    print(f"Processed {len(requests)} concurrent requests in {processing_time:.2f} seconds")
    print(f"Average time per request: {processing_time / len(requests):.2f} seconds")