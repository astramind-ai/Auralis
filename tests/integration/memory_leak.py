import pytest
import asyncio
import torch
import psutil
import os
from auralis import TTS, TTSRequest


@pytest.mark.asyncio
async def test_memory_leak_tts_async(default_test_params):

    tts = TTS().from_pretrained(default_test_params['tts_model'], gpt_model=default_test_params['gpt_model'])

    request = TTSRequest(
        text=default_test_params['text'],
        speaker_files=[default_test_params['speaker_file']],
        stream=False,
    )
    last_consumed_memory = torch.cuda.memory_allocated()
    prev_memory = last_consumed_memory
    for _ in range(default_test_params['n_iterations_mem_leak']):
        prev_memory = last_consumed_memory
        result = await tts.generate_speech_async(request)
        print(f"VRAM: {torch.cuda.memory_allocated() / 1024 / 1024 / 1024} GB")
        last_consumed_memory = torch.cuda.memory_allocated()

        del result
        torch.cuda.empty_cache()


def test_memory_leak_tts(default_test_params):

    tts = TTS().from_pretrained(default_test_params['tts_model'], gpt_model=default_test_params['gpt_model'])

    request = TTSRequest(
        text=default_test_params['text'],
        speaker_files=[default_test_params['speaker_file']],
        stream=False,
    )

    last_consumed_memory = torch.cuda.memory_allocated()
    prev_memory = last_consumed_memory
    for _ in range(default_test_params['n_iterations_mem_leak']):  # 100 iterazioni
        prev_memory = last_consumed_memory
        result = tts.generate_speech(request)
        print(f"VRAM: {torch.cuda.memory_allocated() / 1024 / 1024 / 1024} GB")
        last_consumed_memory = torch.cuda.memory_allocated()
        del result
        torch.cuda.empty_cache()

    assert last_consumed_memory - prev_memory < 10 * 1024 * 1024 # 10 MB