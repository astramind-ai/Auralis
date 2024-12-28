import os
from pathlib import Path


def combine_python_files(root_dir: str, output_file: str):
    """
    Ricorsivamente trova tutti i file .py e li appende a un unico file
    con riferimento al path originale.

    Args:
        root_dir: Directory da cui partire per la ricerca
        output_file: File di output dove verranno combinati i risultati
    """
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    relative_path = file_path.relative_to(root_dir)

                    outfile.write(f"\n\n{'=' * 80}\n")
                    outfile.write(f"# File: {relative_path}\n")
                    outfile.write(f"# Path: {file_path}\n")
                    outfile.write('=' * 80 + "\n\n")

                    try:
                        with open(file_path, 'r', encoding='utf-8') as infile:
                            outfile.write(infile.read())
                    except Exception as e:
                        outfile.write(f"# Error reading file: {e}\n")


if __name__ == "__main__":
    # Esempio di utilizzo
    root_directory = "/home/marco/PycharmProjects/betterVoiceCraft/Auralis/src"  # Directory corrente
    output_file = "combined_python_files.txt"
    combine_python_files(root_directory, output_file)