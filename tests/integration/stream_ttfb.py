import asyncio
import time

import pytest

from auralis import TTS, TTSRequest, TTSOutput

@pytest.mark.asyncio
async def main(default_test_params):
    tts = TTS().from_pretrained(default_test_params['tts_model'], gpt_model=default_test_params['gpt_model'])

    # Crea multiple richieste parallele
    requests = [
        TTSRequest(
            text=default_test_params['text'],
            speaker_files=[default_test_params['speaker_file']],
            stream=id % 2 == 0,
        ) for id in range(default_test_params['n_iterations_parallel_requests'])  # x parallel request
    ]

    # Funzione per processare una singola richiesta
    async def process_request(request, idx):
        print(f"Starting request {idx}")
        start_time = time.perf_counter()
        is_first = True

        stream = await tts.generate_speech_async(request=request)
        chunks = []
        if not request.stream:
            end_time = time.perf_counter()
            print(f"Request {idx} - Time to first chunk: {end_time - start_time:.2f}s")
            return end_time, stream
        async for chunk in stream:
            if is_first:
                is_first = False
                end_time = time.perf_counter()
                print(f"Request {idx} - Time to first chunk: {end_time - start_time:.2f}s")
            chunks.append(chunk)
            print(f"Request {idx} - Got chunk of size {len(chunk.array)}")

        final_end_time = time.perf_counter()
        print(f"Request {idx} completed in {final_end_time - start_time:.2f}s")
        return end_time, TTSOutput.combine_outputs(chunks)
    # execute requests
    ttfb = {}
    out={}
    for idx, request in enumerate(requests):
        out_tuple = await process_request(request, idx)
        ttfb.update({idx:out_tuple[0]})
        out.update({idx:out_tuple[1]})
        print(f"Time to make {'streaming' if request.stream else 'non-streaming'} request {idx}: {ttfb[idx]-ttfb[0]:.2f}s")

    print("Combined output size: ", len(out[0].array))


if __name__ == "__main__":
    asyncio.run(main())