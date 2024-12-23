import os


def merge_files(directory, output_file):
    # Apre il file di output in modalità scrittura
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # Attraversa ricorsivamente la directory
        for root, dirs, files in os.walk(directory):
            for file in files:
                # Ignora il file di output se è nella directory
                if file == os.path.basename(output_file):
                    continue

                # Costruisce il path completo del file
                file_path = os.path.join(root, file)

                try:
                    # Scrive l'header con nome file e path
                    outfile.write(f"### File: {file}\n")
                    outfile.write(f"### Path: {file_path}\n")
                    outfile.write("=" * 80 + "\n\n")

                    # Copia il contenuto del file
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        outfile.write(infile.read())

                    # Aggiunge separatore tra i file
                    outfile.write("\n\n" + "=" * 80 + "\n\n")

                except Exception as e:
                    print(f"Errore nel processare {file_path}: {e}")


if __name__ == "__main__":
    # Directory da processare
    directory = "./auralis"  # Directory corrente, modifica come necessario

    # File di output
    output_file = "merged_files.txt"

    merge_files(directory, output_file)
    print(f"Files uniti in {output_file}")