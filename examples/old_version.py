import torch
from auralis import TTS, TTSRequest
from auralis.common.definitions.enhancer import AudioPreprocessingConfig
import os


def coqui_v200_model(model_path, speaker_path, output_path):
    """
      Tests loading and generating audio with a Coqui XTTS v2.0.0 model using Auralis.

      Args:
          model_path (str): Path to the converted XTTS model directory
          speaker_path (str): Path to the reference speaker audio file.
          output_path (str): Path to save the generated audio.
      """
    try:
        # Load the model, with the tokenizer size which is set to 6153 for v2.0.0, and the gpt model
        tts = TTS().from_pretrained(model_path, gpt_model="path_to/gpt", tokenizer_size=6153)

        # Create a TTS request
        request = TTSRequest(
            text="This is a test using an old Coqui XTTS v2.0.0 model with a specific tokenizer size!",
            speaker_files=[speaker_path],
            audio_config=AudioPreprocessingConfig(
                enhance_speech=True,
                normalize=True,
            )
        )

        # Generate audio and save it
        output = tts.generate_speech(request)
        output.save(output_path)

        print(f"Successfully generated audio and saved to: {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    # Define paths
    model_path = "path_to/core_xttsv2/"  # Path to your converted model directory
    speaker_path = "../tests/resources/audio_samples/female.wav" # Path to a reference speaker audio file
    output_path = "output_v200.wav"

    # Run the test
    coqui_v200_model(model_path, speaker_path, output_path)