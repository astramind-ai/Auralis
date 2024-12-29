import os
import json
import shutil
import argparse
from typing import Dict, Any, Optional, Tuple
import torch
from safetensors.torch import save_file
from huggingface_hub import snapshot_download

def analyze_model_architecture(model_state: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Analyzes model architecture from weights."""
    architecture = {}

    # Extract vocab and hidden size from text embedding
    for key, tensor in model_state.items():
        if 'text_embedding.weight' in key:
            vocab_size, hidden_size = tensor.shape
            architecture.update({
                'vocab_size': vocab_size,
                'number_text_tokens': vocab_size,
                'hidden_size': hidden_size,
                'decoder_input_dim': hidden_size
            })
            break

    # Count unique layer indices
    max_layer = -1
    for key in model_state.keys():
        if 'gpt.gpt.h.' in key and 'attn.c_attn.weight' in key:
            # Split by dots and look for the number after 'h'
            parts = key.split('.')
            for i, part in enumerate(parts):
                if part == 'h' and i + 1 < len(parts):
                    try:
                        layer_num = int(parts[i + 1])
                        max_layer = max(max_layer, layer_num)
                    except ValueError:
                        continue
    
    architecture['num_hidden_layers'] = max_layer + 1

    # Analyze attention structure from weight dimensions
    for key, tensor in model_state.items():
        if 'attn.c_attn.weight' in key:
            hidden_size, triple_size = tensor.shape
            if triple_size != 3 * hidden_size:
                continue
            if hidden_size % 64 == 0:
                architecture['num_attention_heads'] = hidden_size // 64
            else:
                architecture['num_attention_heads'] = 1
                print(f"Warning: hidden_size={hidden_size} not multiple of 64, setting num_attention_heads=1")
            architecture['n_inner'] = architecture['hidden_size'] * 4
            break

    # Extract audio token config from mel head
    for key, tensor in model_state.items():
        if 'mel_head.weight' in key:
            num_outputs, _ = tensor.shape
            architecture.update({
                'num_audio_tokens': num_outputs,
                'max_audio_tokens': num_outputs - 421,
                'start_audio_token': num_outputs - 2,
                'stop_audio_token': num_outputs - 1
            })
            break

    return architecture

def extract_original_values(checkpoint: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extracts important values from original checkpoint."""
    config_locations = ['config', 'model_config', 'training_config', 'model_args']

    for loc in config_locations:
        if loc in checkpoint:
            config = checkpoint[loc]
            important_values = {}

            training_keys = [
                'gpt_max_text_tokens',
                'gpt_max_audio_tokens',
                'gpt_max_prompt_tokens',
                'gpt_code_stride_len',
                'output_hop_length',
                'input_sample_rate',
                'output_sample_rate'
            ]

            preserve_keys = [
                'languages',
                'audio_config',
                'speaker_embeddings',
                'use_speaker_embedding',
                'use_multi_speaker',
                'multi_speaker_config',
                'custom_tokens',
                'speaker_dim',
                'd_vector_dim',
                'tokenizer_config'
            ]

            for key in training_keys + preserve_keys:
                if key in config:
                    important_values[key] = config[key]

            if loc == 'config' and 'model_args' in checkpoint: # NEW handle model args keys
                  model_args_config = checkpoint['model_args']
                  for key in ['use_masking_gt_prompt_approach', 'use_perceiver_resampler']:
                    if key in model_args_config:
                          important_values[key] = model_args_config[key]


            return important_values if important_values else None

    return None

def create_xtts_core_config(model_architecture: Dict[str, Any],
                          original_values: Optional[Dict[str, Any]] = None,
                          gpt_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Creates the config for core XTTS model."""
    config = {
        "model_type": "xtts",
        "architectures": ["XttsGPT"],

        "audio_config": {
            "fmax": 8000,
            "fmin": 0,
            "hop_length": original_values.get('output_hop_length', 256),
            "mel_channels": 80,
            "mel_norms_file": None,
            "n_fft": 1024,
            "output_sample_rate": original_values.get('output_sample_rate', 24000),
            "power": 1.0,
            "sample_rate": original_values.get('input_sample_rate', 22050),
            "win_length": 1024
        },

        "d_vector_dim": original_values.get('d_vector_dim', 512),
        "decoder_input_dim": model_architecture['hidden_size'],
        "num_chars": 255,

        "duration_const": 102400,
        "output_hop_length": original_values.get('output_hop_length', 256),
        "input_sample_rate": original_values.get('input_sample_rate', 22050),
        "output_sample_rate": original_values.get('output_sample_rate', 24000),

        "gpt": {"model_type": "xtts_gpt"},
        "gpt_config": gpt_config,
        "gpt_code_stride_len": original_values.get('gpt_code_stride_len', 1024),
        "cond_d_vector_in_each_upsampling_layer": True,

        "auto_map": {
            "AutoConfig": "AstraMindAI/xtts2--xtts2_config.XTTSConfig",
            "AutoModelForCausalLM": "AstraMindAI/xtts2--xtts2_modeling.Xtts",
            "AutoTokenizer": "AstraMindAI/xtts2--tokenizer.XTTSTokenizerFast"
        },

        "languages": original_values.get('languages', [
            "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru",
            "nl", "cs", "ar", "zh-cn", "hu", "ko", "ja", "hi"
        ]),

        "tokenizer_file": "",
        "transformers_version": "4.46.0"
    }

    if original_values:
        for key, value in original_values.items():
            if key == 'audio_config':
                config['audio_config'].update(value)
            else:
                config[key] = value

    return config

def create_auralis_config(model_architecture: Dict[str, Any],
                         original_values: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Creates Auralis-compatible GPT configuration."""
    config = {
        "model_type": "xtts_gpt",
        "architectures": ["XttsGPT"],

        "vocab_size": model_architecture['vocab_size'],
        "hidden_size": model_architecture['hidden_size'],
        "num_hidden_layers": model_architecture['num_hidden_layers'],
        "num_attention_heads": model_architecture['num_attention_heads'],
        "n_inner": model_architecture['n_inner'],

        "number_text_tokens": model_architecture['vocab_size'],
        "num_audio_tokens": model_architecture['num_audio_tokens'],
        "max_audio_tokens": model_architecture['max_audio_tokens'],
        "start_audio_token": model_architecture['start_audio_token'],
        "stop_audio_token": model_architecture['stop_audio_token'],

        "max_text_tokens": original_values.get('gpt_max_text_tokens', 402),
        "max_prompt_tokens": original_values.get('gpt_max_prompt_tokens', 70),
        "activation_function": "gelu_new",
        "attn_pdrop": 0.1,
        "layer_norm_epsilon": 1e-5,
        "initializer_range": 0.02,
        "use_masking_gt_prompt_approach": True,
        "use_perceiver_resampler": True,

        "kv_cache": True,
        "enable_redaction": False,
        "reorder_and_upcast_attn": False,
        "scale_attn_by_inverse_layer_idx": False,

        "auto_map": {
            "AutoConfig": "AstraMindAI/xtts2-gpt--gpt_config.XTTSGPTConfig",
            "AutoModelForCausalLM": "AstraMindAI/xtts2-gpt--xtts2_gpt_modeling.XttsGPT",
            "AutoTokenizer": "AstraMindAI/xtts2-gpt--tokenizer.XTTSTokenizerFast"
        }
    }

    if original_values:
        for key, value in original_values.items():
            if key not in ['vocab_size', 'hidden_size', 'num_hidden_layers', 'num_attention_heads', 'n_inner',
                           'number_text_tokens', 'num_audio_tokens', 'max_audio_tokens', 'start_audio_token',
                           'stop_audio_token']:
                config[key] = value

    return config

def convert_model_weights(model_state: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Converts model weights into correct formats."""
    gpt_weights = {}
    xtts_weights = {}

    gpt_patterns = [
        'ln_1.weight', 'ln_1.bias',
        'attn.c_attn.weight', 'attn.c_attn.bias',
        'attn.c_proj.weight', 'attn.c_proj.bias',
        'ln_2.weight', 'ln_2.bias',
        'mlp.c_fc.weight', 'mlp.c_fc.bias',
        'mlp.c_proj.weight', 'mlp.c_proj.bias',
        'ln_f.weight', 'ln_f.bias',
        'mel_head.weight', 'mel_head.bias'
    ]

    ignore_patterns = [
        'mel_embedding.weight',
        'mel_pos_embedding.emb.weight'
    ]

    training_ignore = {
        "torch_mel_spectrogram_style_encoder",
        "torch_mel_spectrogram_dvae",
        "dvae"
    }

    for key, tensor in model_state.items():
        if any(pattern in key for pattern in training_ignore):
            continue

        key = key.replace('xtts.', '')
        is_gpt_weight = any(pattern in key for pattern in gpt_patterns + ignore_patterns)

        if is_gpt_weight:
            if 'mel_embedding.weight' in key:
                new_key = 'gpt.wte.weight'
            elif 'mel_pos_embedding.emb.weight' in key:
                new_key = 'gpt.wpe.emb.weight'
            elif 'mel_head' in key:
                new_key = key.replace('gpt.', '')
            else:
                new_key = key.replace('gpt.gpt.', 'gpt.')

            gpt_weights[new_key] = tensor
        elif 'final_norm' in key:
            clean_key = key.replace('gpt.', '')
            gpt_weights[clean_key] = tensor
            xtts_weights[clean_key] = tensor
        else:
            xtts_weights[key.replace('gpt.', '')] = tensor

    missing_patterns = [
        pattern for pattern in gpt_patterns
        if not any(pattern in key for key in gpt_weights.keys())
    ]
    if missing_patterns:
        raise ValueError(f"Missing required GPT patterns: {missing_patterns}")

    return gpt_weights, xtts_weights

def save_model_weights(gpt_weights: Dict[str, torch.Tensor],
                      xtts_weights: Dict[str, torch.Tensor],
                      output_dir: str) -> Tuple[str, str]:
    """Saves model weights in SafeTensors format."""
    gpt_dir = os.path.join(output_dir, "gpt")
    xtts_dir = os.path.join(output_dir, "core_xttsv2")
    os.makedirs(gpt_dir, exist_ok=True)
    os.makedirs(xtts_dir, exist_ok=True)

    gpt_path = os.path.join(gpt_dir, 'gpt2_model.safetensors')
    save_file(gpt_weights, gpt_path)
    print(f"GPT weights saved to: {gpt_path}")
    print(f"GPT weight keys: {list(gpt_weights.keys())}")

    xtts_path = os.path.join(xtts_dir, 'xtts-v2.safetensors')
    save_file(xtts_weights, xtts_path)
    print(f"XTTS weights saved to: {xtts_path}")
    print(f"XTTS weight keys: {list(xtts_weights.keys())}")

    return gpt_path, xtts_path

def save_configs(output_dir: str, checkpoint: Dict[str, Any]) -> Tuple[str, str, str, str]:
    """Creates and saves all necessary configurations."""
    model_architecture = analyze_model_architecture(checkpoint['model'])
    original_values = extract_original_values(checkpoint)

    gpt_config = create_auralis_config(model_architecture, original_values)
    xtts_config = create_xtts_core_config(model_architecture, original_values, gpt_config)

    gpt_dir = os.path.join(output_dir, "gpt")
    xtts_dir = os.path.join(output_dir, "core_xttsv2")
    os.makedirs(gpt_dir, exist_ok=True)
    os.makedirs(xtts_dir, exist_ok=True)

    gpt_config_path = os.path.join(gpt_dir, "config.json")
    gpt_backup_path = os.path.join(gpt_dir, "config.original.json")

    with open(gpt_config_path, 'w', encoding='utf-8') as f:
        json.dump(gpt_config, f, indent=2)
    shutil.copy2(gpt_config_path, gpt_backup_path)

    xtts_config_path = os.path.join(xtts_dir, "config.json")
    xtts_backup_path = os.path.join(xtts_dir, "config.original.json")

    with open(xtts_config_path, 'w', encoding='utf-8') as f:
        json.dump(xtts_config, f, indent=2)
    shutil.copy2(xtts_config_path, xtts_backup_path)

    return gpt_config_path, gpt_backup_path, xtts_config_path, xtts_backup_path

def download_safely(repo_id: str, output_dir: str, config_path: str):
    """Downloads repo files while preserving local configs."""
    with open(config_path, 'r', encoding='utf-8') as f:
        our_config = json.load(f)

    snapshot_download(
        repo_id=repo_id,
        ignore_patterns=['*.safetensors', 'config.json'],
        local_dir=output_dir
    )

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(our_config, f, indent=2)

def convert_checkpoint(pytorch_checkpoint_path: str, output_dir: str):
    """Main conversion process."""
    print(f"Loading checkpoint from: {pytorch_checkpoint_path}")
    checkpoint = torch.load(pytorch_checkpoint_path, map_location='cpu')

    print("\nCreating and saving configurations...")
    gpt_config_path, gpt_backup, xtts_config_path, xtts_backup = save_configs(
        output_dir, checkpoint
    )
    print(f"GPT config: {gpt_config_path} (backup: {gpt_backup})")
    print(f"XTTS config: {xtts_config_path} (backup: {xtts_backup})")

    print("\nConverting model weights...")
    gpt_weights, xtts_weights = convert_model_weights(checkpoint['model'])

    print("\nSaving weights...")
    gpt_weights_path, xtts_weights_path = save_model_weights(
        gpt_weights, xtts_weights, output_dir
    )

    print("\nDownloading additional files...")
    download_safely(
        "AstraMindAI/xtts2-gpt",
        os.path.join(output_dir, "gpt"),
        gpt_config_path
    )
    download_safely(
        "AstraMindAI/xttsv2",
        os.path.join(output_dir, "core_xttsv2"),
        xtts_config_path
    )

    print("\nConversion completed successfully!")
    print("Generated files:")
    print(f"- GPT config: {gpt_config_path}")
    print(f"- GPT weights: {gpt_weights_path}")
    print(f"- XTTS config: {xtts_config_path}")
    print(f"- XTTS weights: {xtts_weights_path}")
    print(f"- Backup configs: {gpt_backup}, {xtts_backup}")

def main():
    parser = argparse.ArgumentParser(
        description='Convert PyTorch checkpoint to Auralis format while preserving configurations'
    )
    parser.add_argument(
        'checkpoint_path',
        help='Path to the PyTorch checkpoint file'
    )
    parser.add_argument(
        '--output_dir',
        default=os.getcwd(),
        help='Output directory (defaults to current working directory)'
    )

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint file '{args.checkpoint_path}' does not exist")
        return

    convert_checkpoint(args.checkpoint_path, args.output_dir)

if __name__ == '__main__':
    main()
