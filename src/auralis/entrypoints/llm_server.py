import subprocess

def start_vllm_server():
    command = [
        "vllm",
        "serve",
        "meta-llama/Llama-3.2-1B",
        "--host=127.0.0.1",
        "--port=8001",
    ]
    try:
        subprocess.run(command, check=True)  # check=True will raise an error if the command fails
        print("vLLM server finished (this is unlikely for a server).")
    except FileNotFoundError:
        print("Error: The 'vllm' command was not found.")
    except subprocess.CalledProcessError as e:
        print(f"Error starting vLLM server: {e}")

if __name__ == "__main__":
    start_vllm_server()