#  Copyright (c) 2024 Astramind. Licensed under Apache License, Version 2.0.

import argparse
import os

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import save_file


def download_repo_files(repo_id, output_path, exclude_extensions=['*.safetensors']):
    """
    Downloads all files from a GitHub repository except specified extensions.

    Args:
        owner (str): GitHub repository owner
        repo (str): Repository name
        exclude_extensions (list): List of file extensions to exclude
    """
    # Create base directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    snapshot_download(repo_id=repo_id, ignore_patterns=exclude_extensions, local_dir=output_path)



def convert_checkpoint(pytorch_checkpoint_path, output_dir, args):
    """
    Convert PyTorch checkpoint to SafeTensors format, mapping weights to GPT2 or XTTSv2 models
    based on specific substrings.

    Args:
        pytorch_checkpoint_path: Path to input PyTorch checkpoint
        output_dir: Directory to save the output SafeTensors files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "gpt"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "core_xttsv2"), exist_ok=True)

    # Load PyTorch checkpoint
    checkpoint = torch.load(pytorch_checkpoint_path, map_location='cpu', weights_only=False) # to avoid warning

    # Initialize dictionaries for different models
    gpt2_weights = {}
    xtts_weights = {}

    # List of substrings to identify GPT2 weights
    gpt2_substrings = [
       'ln_1.weight', 'ln_1.bias', 'attn.c_attn.weight', 'attn.c_attn.bias', 'attn.c_proj.weight',
        'attn.c_proj.bias', 'ln_2.weight', 'ln_2.bias', 'mlp.c_fc.weight',
        'mlp.c_fc.bias', 'mlp.c_proj.weight', 'mlp.c_proj.bias', 'ln_f.weight',
        'ln_f.bias', 'mel_head.weight', 'mel_head.bias'

    ]
    ignore_in_check_components = ['mel_embedding.weight', 'mel_pos_embedding.emb.weight']
    # mel_emb -> wte.emb.weight, mel_pos_emb -> wpe.emb.weight
    ignore_keys_from_training = {"torch_mel_spectrogram_style_encoder", "torch_mel_spectrogram_dvae", "dvae"}

    all_sub_str = gpt2_substrings + ignore_in_check_components
    # Separate weights based on substrings
    for key, tensor in checkpoint['model'].items():
        # Check if any GPT2 substring is in the key
        if any(substring in key for substring in ignore_keys_from_training):
            continue # skip training layers
        key = key.replace('xtts.', '')

        is_gpt2_weight = any(substring in key for substring in all_sub_str)

        if is_gpt2_weight:

            if 'mel_embedding.weight' in key:
                key = 'gpt.wte.weight'
            elif 'mel_pos_embedding.emb.weight' in key:
                key = 'gpt.wpe.emb.weight'
            elif 'mel_head' in key:
                key = key.replace('gpt.', '')
            else:
                key = key.replace('gpt.gpt.', 'gpt.')
            # Use a modded name for GPT-2 weights
            gpt2_weights[key] = tensor
        elif 'final_norm' in key:
            gpt2_weights[key.replace('gpt.', '')] = tensor
            xtts_weights[key.replace('gpt.', '')] = tensor
        else:
            # All other weights go to XTTS
            xtts_weights[key.replace('gpt.', '')] = tensor

    # Check if all the weights keys are matched
    assert all(any(substr in key for key in gpt2_weights.keys()) for substr in gpt2_substrings), \
        f"Missing substrings: {[substr for substr in gpt2_substrings if not any(substr in key for key in gpt2_weights.keys())]}"


    gpt2_path = os.path.join(output_dir, "gpt", 'gpt2_model.safetensors')
    save_file(gpt2_weights, gpt2_path)
    download_repo_files("AstraMindAI/xtts2-gpt", os.path.join(output_dir, "gpt"))
    print(f"Saved XTTSv2 GPT-2 weights to {gpt2_path}")
    print(f"XTTSv2 GPT-2 weights: {list(gpt2_weights.keys())}")

    # Save XTTS weights if any exist
    if xtts_weights:
        xtts_path = os.path.join(output_dir, 'core_xttsv2', 'xtts-v2.safetensors')
        save_file(xtts_weights, xtts_path)
        download_repo_files("AstraMindAI/xttsv2", os.path.join(output_dir, "core_xttsv2"))
        print(f"Saved XTTSv2 weights to {xtts_path}")
        print(f"XTTSv2 weights: {list(xtts_weights.keys())}")

def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch checkpoint to SafeTensors format')
    parser.add_argument('checkpoint_path', type=str, help='Path to PyTorch checkpoint file')
    parser.add_argument('--output_dir', type=str, default=os.getcwd(),
                        help='Output directory (defaults to current working directory)')

    args = parser.parse_args()

    # Verify checkpoint file exists
    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint file '{args.checkpoint_path}' does not exist")
        return

    # Convert the checkpoint
    convert_checkpoint(args.checkpoint_path, args.output_dir, args)

if __name__ == '__main__':
    main()