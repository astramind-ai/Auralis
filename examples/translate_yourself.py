#  Copyright (c) 2024 Astramind. Licensed under Apache License, Version 2.0.

from auralis import TTS, TTSRequest

tts = TTS(scheduler_max_concurrency=12).from_pretrained("AstraMindAI/xttsv2", gpt_model='AstraMindAI/xtts2-gpt')

request = TTSRequest(
    text="愛しい彼女へ "
         "あなたの笑顔は私の人生を照らす光です。"
         "毎日あなたと過ごせることが私の幸せです。"
         "あなたは私の心の中で一番大切な人です。"
         "いつも一緒にいてくれて、"
         "ありがとう。"
         "愛を込めて",
    speaker_files=["your_voice.ogg"],
)

output = tts.generate_speech(request)

output.play()
