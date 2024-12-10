import os

def trova_e_aggiungi_file_python(cartella_iniziale, file_output):
    """
    Trova ricorsivamente tutti i file Python nelle sottocartelle e aggiunge
    il loro contenuto a un file di output, insieme al path relativo.

    Args:
        cartella_iniziale: Il percorso della cartella radice da cui iniziare la ricerca.
        file_output: Il percorso del file di output in cui scrivere i risultati.
    """
    try:
        with open(file_output, "w") as outfile:
            for cartella_corrente, sottocartelle, files in os.walk(cartella_iniziale):
                for file in files:
                    if file.endswith(".py"):
                        path_relativo = os.path.relpath(os.path.join(cartella_corrente, file), cartella_iniziale)
                        outfile.write(f"# File: {path_relativo}\n")

                        try:
                            with open(os.path.join(cartella_corrente, file), "r") as infile:
                                contenuto = infile.read()
                                outfile.write(contenuto)
                                outfile.write("\n\n")
                        except UnicodeDecodeError:
                            print(f"Impossibile decodificare il file: {path_relativo}. Il file verrà saltato.")
                            outfile.write(f"# Impossibile decodificare il file: {path_relativo}\n\n")
                        except Exception as e:
                            print(f"Errore durante la lettura del file {path_relativo}: {e}")
                            outfile.write(f"# Errore durante la lettura del file: {path_relativo}\n\n")

    except Exception as e:
        print(f"Errore durante la creazione del file di output: {e}")

# Utilizzo
cartella_iniziale = "."  # Sostituisci con il percorso della tua cartella radice
file_output = "file_python_concatenati.txt"  # Sostituisci con il percorso del tuo file di output

trova_e_aggiungi_file_python(cartella_iniziale, file_output)

print(f"File Python concatenati e salvati in: {file_output}")