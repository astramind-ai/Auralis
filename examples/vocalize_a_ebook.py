#  Copyright (c) 2024 Astramind. Licensed under Apache License, Version 2.0.

import time

import asyncio
import ebooklib
import torch
from ebooklib import epub
from bs4 import BeautifulSoup
import os

from auralis import TTS, TTSRequest, TTSOutput


def extract_text_from_epub(epub_path, output_path=None):
    """
    Extracts text from an EPUB file and optionally saves it to a text file.

    Args:
        epub_path (str): Path to the EPUB file
        output_path (str, optional): Path where to save the text file

    Returns:
        str: The extracted text
    """
    # Load the book
    book = epub.read_epub(epub_path)

    # List to hold extracted text
    chapters = []

    # Extract text from each chapter
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            # Get HTML content
            html_content = item.get_content().decode('utf-8')

            # Use BeautifulSoup to extract text
            soup = BeautifulSoup(html_content, 'html.parser')

            # Remove scripts and styles
            for script in soup(["script", "style"]):
                script.decompose()

            # Get text
            text = soup.get_text()

            # Clean text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)

            chapters.append(text)

    # Join all chapters
    full_text = '\n\n'.join(chapters)

    # Save text if output path is specified
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_text)

    return full_text .replace('»', '"').replace('«', '"')


def process_multiple_epubs(input_folder, output_folder):
    """
    Process all EPUB files in a folder.

    Args:
        input_folder (str): Folder containing EPUB files
        output_folder (str): Folder where to save text files
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each EPUB file in the folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.epub'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.txt")

            try:
                print(f"Processing: {filename}")
                extract_text_from_epub(input_path, output_path)
                print(f"Successfully processed: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

def main():
    text = extract_text_from_epub("book.epub")

    speaker_file = 'sample_voice.wav'
    # Initialize the engine, you can experiment with the scheduler_max_concurrency parameter to optimize the performance
    tts = TTS(
        scheduler_max_concurrency=12).from_pretrained("AstraMindAI/xttsv2", gpt_model="AstraMindAI/xtts2-gpt")
    req = TTSRequest(
            text=text,
            language="auto",
            temperature=0.75,
            repetition_penalty=6.5,
            speaker_files=[speaker_file],
            stream=True
        )

    start_time = time.time()

    # Execute requests in a generator to get audio instantly
    result_generator = tts.generate_speech(req)
    out_list = []
    for out in result_generator:
        out_list.append(out)
        # Play the audio
    print(f"Execution time: {time.time() - start_time:.2f} seconds")

    # Save the audio to a file
    TTSOutput.combine_outputs(out_list).save("your_book.wav")



if __name__ == "__main__":
    main()