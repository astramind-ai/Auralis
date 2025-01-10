import asyncio
import base64
import time
import uuid
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

import ebooklib
import gradio as gr
import torch
import torchaudio
from ebooklib import epub
from bs4 import BeautifulSoup

from auralis import TTS, TTSRequest, TTSOutput, AudioPreprocessingConfig, setup_logger

logger = setup_logger(__file__)

tts = TTS()
model_path = "AstraMindAI/xttsv2" # change this if you have a different model
gpt_model = "AstraMindAI/xtts2-gpt"
try:
    tts = tts.from_pretrained(model_path, gpt_model=gpt_model)
    logger.info(f"Successfully loaded model {model_path}")
except Exception as e:
    logger.error(f"Failed to load model: {e}. Ensure that the model exists at {model_path}")

# Create a temporary directory to store short-named files
temp_dir = Path("/tmp/auralis")
temp_dir.mkdir(exist_ok=True)

def shorten_filename(original_path: str) -> str:
    """Copies the given file to a temporary directory with a shorter, random filename."""
    ext = Path(original_path).suffix
    short_name = "file_" + uuid.uuid4().hex[:8] + ext
    short_path = temp_dir / short_name
    shutil.copyfile(original_path, short_path)
    return str(short_path)

def extract_text_from_epub(epub_path: str) -> str:
    """
    Extracts text from an EPUB file.
    """
    # Ensure the path is shortened to avoid filename too long error
    epub_short_path = shorten_filename(epub_path)

    book = epub.read_epub(epub_short_path)
    chapters = []
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            html_content = item.get_content().decode('utf-8')
            soup = BeautifulSoup(html_content, 'html.parser')
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            chapters.append(text)
    return '\n\n'.join(chapters).replace('»', '"').replace('«', '"')

def text_from_file(txt_file_path):
    # Shorten filename before reading
    txt_short_path = shorten_filename(txt_file_path)
    with open(txt_short_path, 'r') as f:
        text = f.read()
    return text

def clone_voice(audio_path: str):
    """Clone a voice from an audio path."""
    # Shorten filename before reading
    audio_short_path = shorten_filename(audio_path)
    with open(audio_short_path, "rb") as f:
        audio_data = base64.b64encode(f.read()).decode('utf-8')
    return audio_data

def process_text_and_generate(input_text, ref_audio_files, speed, enhance_speech, temperature, top_p, top_k, repetition_penalty, language, *args):
    """Process text and generate audio."""
    log_messages = ""
    if not ref_audio_files:
        log_messages += "Please provide at least one reference audio!\n"
        return None, log_messages

    # clone voices from all file paths (shorten them)
    base64_voices = ref_audio_files[:5]

    request = TTSRequest(
      text=input_text,
      speaker_files=base64_voices,
      stream=False,
      enhance_speech=enhance_speech,
      temperature=temperature,
      top_p=top_p,
      top_k=top_k,
      repetition_penalty=repetition_penalty,
      language=language,
    )

    try:
        with torch.no_grad():
            output = tts.generate_speech(request)
            if output:
                if speed != 1:
                    output.change_speed(speed)
                log_messages += f"✅ Successfully Generated audio\n"
                return (output.sample_rate, output.array), log_messages
            else:
                log_messages += "❌ No output was generated. Check that the model was correctly loaded\n"
                return None, log_messages
    except Exception as e:
        logger.error(f"Error: {e}")
        log_messages += f"❌ An Error occured: {e}\n"
        return None, log_messages

def build_gradio_ui():
    """Builds and launches the Gradio UI for Auralis."""
    with gr.Blocks(title="Auralis TTS Demo", theme="soft") as ui:

        gr.Markdown(
          """
          # Auralis Text-to-Speech Demo 🌌
          Convert text to speech with advanced voice cloning and enhancement.
          """
        )

        with gr.Tab("Text to Speech"):
          with gr.Row():
            with gr.Column():
              input_text = gr.Text(label="Enter Text Here", placeholder="Write the text you want to convert...")
              ref_audio_files = gr.Files(label="Reference Audio Files", file_types=["audio"])
              with gr.Accordion("Advanced settings", open=False):
                  speed = gr.Slider(label="Playback speed", minimum=0.5, maximum=2.0, value=1.0, step=0.1)
                  enhance_speech = gr.Checkbox(label="Enhance Reference Speech", value=False)
                  temperature = gr.Slider(label="Temperature", minimum=0.5, maximum=1.0, value=0.75, step=0.05)
                  top_p = gr.Slider(label="Top P", minimum=0.5, maximum=1.0, value=0.85, step=0.05)
                  top_k = gr.Slider(label="Top K", minimum=0, maximum=100, value=50, step=10)
                  repetition_penalty = gr.Slider(label="Repetition penalty", minimum=1.0, maximum=10.0, value=5.0, step=0.5)
                  language = gr.Dropdown(label="Target Language", choices=[
                      "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru",
                      "nl", "cs", "ar", "zh-cn", "hu", "ko", "ja", "hi", "auto",
                  ], value="auto")
              generate_button = gr.Button("Generate Speech")
            with gr.Column():
                audio_output = gr.Audio(label="Generated Audio")
                log_output = gr.Text(label="Log Output")

          generate_button.click(
            process_text_and_generate,
            inputs=[input_text, ref_audio_files, speed, enhance_speech, temperature, top_p, top_k, repetition_penalty, language],
            outputs=[audio_output, log_output],
          )

        with gr.Tab("File to Speech"):
          with gr.Row():
            with gr.Column():
              file_input = gr.File(label="Text / Ebook File", file_types=["text", ".epub"])
              ref_audio_files_file = gr.Files(label="Reference Audio Files", file_types=["audio"])
              with gr.Accordion("Advanced settings", open=False):
                  speed_file = gr.Slider(label="Playback speed", minimum=0.5, maximum=2.0, value=1.0, step=0.1)
                  enhance_speech_file = gr.Checkbox(label="Enhance Reference Speech", value=False)
                  temperature_file = gr.Slider(label="Temperature", minimum=0.5, maximum=1.0, value=0.75, step=0.05)
                  top_p_file = gr.Slider(label="Top P", minimum=0.5, maximum=1.0, value=0.85, step=0.05)
                  top_k_file = gr.Slider(label="Top K", minimum=0, maximum=100, value=50, step=10)
                  repetition_penalty_file = gr.Slider(label="Repetition penalty", minimum=1.0, maximum=10.0, value=5.0, step=0.5)
                  language_file = gr.Dropdown(label="Target Language", choices=[
                      "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru",
                      "nl", "cs", "ar", "zh-cn", "hu", "ko", "ja", "hi", "auto",
                  ], value="auto")
              generate_button_file = gr.Button("Generate Speech from File")
            with gr.Column():
                audio_output_file = gr.Audio(label="Generated Audio")
                log_output_file = gr.Text(label="Log Output")

          def process_file_and_generate(file_input, ref_audio_files_file, speed_file, enhance_speech_file, temperature_file, top_p_file, top_k_file, repetition_penalty_file, language_file):
              if file_input:
                  file_extension = Path(file_input.name).suffix
                  if file_extension == '.epub':
                      input_text = extract_text_from_epub(file_input.name)
                  elif file_extension == '.txt':
                      input_text = text_from_file(file_input.name)
                  else:
                      return None, "Unsupported file format, it needs to be either .epub or .txt"

                  return process_text_and_generate(input_text, ref_audio_files_file, speed_file, enhance_speech_file,
                                                   temperature_file, top_p_file, top_k_file, repetition_penalty_file, language_file)
              else:
                  return None, "Please provide an .epub or .txt file!"

          generate_button_file.click(
            process_file_and_generate,
            inputs=[file_input, ref_audio_files_file, speed_file, enhance_speech_file, temperature_file, top_p_file, top_k_file, repetition_penalty_file, language_file],
            outputs=[audio_output_file, log_output_file],
          )

        with gr.Tab("Clone With Microfone"):
          with gr.Row():
            with gr.Column():
              input_text_mic = gr.Text(label="Enter Text Here", placeholder="Write the text you want to convert...")
              mic_ref_audio = gr.Audio(label="Record Reference Audio", sources=["microphone"])

              with gr.Accordion("Advanced settings", open=False):
                  speed_mic = gr.Slider(label="Playback speed", minimum=0.5, maximum=2.0, value=1.0, step=0.1)
                  enhance_speech_mic = gr.Checkbox(label="Enhance Reference Speech", value=True)
                  temperature_mic = gr.Slider(label="Temperature", minimum=0.5, maximum=1.0, value=0.75, step=0.05)
                  top_p_mic = gr.Slider(label="Top P", minimum=0.5, maximum=1.0, value=0.85, step=0.05)
                  top_k_mic = gr.Slider(label="Top K", minimum=0, maximum=100, value=50, step=10)
                  repetition_penalty_mic = gr.Slider(label="Repetition penalty", minimum=1.0, maximum=10.0, value=5.0, step=0.5)
                  language_mic = gr.Dropdown(label="Target Language", choices=[
                      "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru",
                      "nl", "cs", "ar", "zh-cn", "hu", "ko", "ja", "hi", "auto",
                  ], value="auto")
              generate_button_mic = gr.Button("Generate Speech")
            with gr.Column():
                audio_output_mic = gr.Audio(label="Generated Audio")
                log_output_mic = gr.Text(label="Log Output")

          import hashlib

          def process_mic_and_generate(input_text_mic, mic_ref_audio, speed_mic, enhance_speech_mic, temperature_mic, top_p_mic, top_k_mic, repetition_penalty_mic, language_mic):
              if mic_ref_audio:
                  data = str(time.time()).encode("utf-8")
                  hash = hashlib.sha1(data).hexdigest()[:10]
                  output_path = temp_dir / (f"mic_{hash}.wav")

                  torch_audio = torch.from_numpy(mic_ref_audio[1].astype(float))
                  try:
                      torchaudio.save(str(output_path), torch_audio.unsqueeze(0), mic_ref_audio[0])
                      return process_text_and_generate(input_text_mic, [Path(output_path)], speed_mic, enhance_speech_mic, temperature_mic, top_p_mic, top_k_mic, repetition_penalty_mic, language_mic)
                  except Exception as e:
                      logger.error(f"Error saving audio file: {e}")
                      return None, f"Error saving audio file: {e}"
              else:
                  return None, "Please record an audio!"

          generate_button_mic.click(
            process_mic_and_generate,
            inputs=[input_text_mic, mic_ref_audio, speed_mic, enhance_speech_mic, temperature_mic, top_p_mic, top_k_mic, repetition_penalty_mic, language_mic],
            outputs=[audio_output_mic, log_output_mic],
          )

        # gr.Examples(
        # [
        #     [
        #         "I need to stop procrastinating and get my life together; ...Googles How long would it take a snail to travel around the world?",
        #         ['../tests/resources/audio_samples/female.wav'],
        #         1, False, 0.75, 0.85, 50, 5.0, 'auto'
        #     ],
        # ],
        #     inputs=[input_text, ref_audio_files, speed, enhance_speech, temperature, top_p, top_k, repetition_penalty, language],
        # )

    return ui

if __name__ == "__main__":
    ui = build_gradio_ui()
    ui.launch(debug=True)
