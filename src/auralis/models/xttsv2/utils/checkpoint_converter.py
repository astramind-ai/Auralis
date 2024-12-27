import os
import json
import shutil
import argparse
from typing import Dict, Any, Optional, Tuple
import torch
from safetensors.torch import save_file
from huggingface_hub import snapshot_download

def analyze_model_architecture(model_state: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """
    Analyzes the model architecture from weights to extract fundamental parameters.
    This function ensures we use actual model parameters rather than defaults.

    Args:
        model_state: Dictionary containing model state/weights

    Returns:
        Dictionary containing extracted architecture parameters
    """
    architecture = {}

    # Extract vocabulary size and model dimensions from text embedding
    for key, tensor in model_state.items():
        if 'text_embedding.weight' in key:
            vocab_size, hidden_size = tensor.shape
            architecture.update({
                'vocab_size': vocab_size,
                'number_text_tokens': vocab_size,  # Auralis uses both names
                'hidden_size': hidden_size,
                'decoder_input_dim': hidden_size
            })
            break

    # Analyze layer structure by counting unique layer indices
    max_layer = -1
    for key in model_state.keys():
        if 'gpt.gpt.h.' in key and 'attn.c_attn.weight' in key:
            layer_num = int(key.split('.')[3])
            max_layer = max(max_layer, layer_num)
    architecture['num_hidden_layers'] = max_layer + 1

    # Analyze attention structure from weight dimensions
    for key, tensor in model_state.items():
        if 'attn.c_attn.weight' in key:
            # Questa weight shape dovrebbe essere [hidden_size, 3 * hidden_size] per Q, K, V
            hidden_size, triple_size = tensor.shape
            if triple_size != 3 * hidden_size:
                # Non è la solita shape? Allora saltiamo
                continue

            # Se hidden_size è divisibile per 64, assumiamo dimensioni testa = 64
            if hidden_size % 64 == 0:
                architecture['num_attention_heads'] = hidden_size // 64
            else:
                # Fallback: se non è divisibile per 64, si può o
                # 1) Forzare 1 sola testa (poco utile)
                # 2) Oppure lanciare un'eccezione o un warning
                architecture['num_attention_heads'] = 1
                print(f"Warning: hidden_size={hidden_size} non è multiplo di 64, impostato num_attention_heads=1")

            # n_inner (dimensione del feed-forward) segue di solito hidden_size * 4 in GPT
            architecture['n_inner'] = architecture['hidden_size'] * 4

            # Abbiamo trovato e settato i parametri, usciamo dal loop
            break

    # Extract audio token configuration from mel head
    # This determines the audio vocabulary and special tokens
    for key, tensor in model_state.items():
        if 'mel_head.weight' in key:
            num_outputs, _ = tensor.shape
            architecture.update({
                'num_audio_tokens': num_outputs,
                'max_audio_tokens': num_outputs - 421,  # Standard XTTS spacing
                'start_audio_token': num_outputs - 2,   # Second to last token
                'stop_audio_token': num_outputs - 1     # Last token
            })
            break

    return architecture

def extract_original_values(checkpoint: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Extracts important values from the original checkpoint that should be preserved,
    particularly fine-tuning specific configurations.

    Args:
        checkpoint: The complete model checkpoint

    Returns:
        Dictionary of preserved configuration values or None if not found
    """
    config_locations = ['config', 'model_config', 'training_config', 'model_args']

    for loc in config_locations:
        if loc in checkpoint:
            config = checkpoint[loc]
            important_values = {}

            # Critical training parameters that affect model behavior
            training_keys = [
                'gpt_max_text_tokens',
                'gpt_max_audio_tokens',
                'gpt_max_prompt_tokens',
                'gpt_code_stride_len',
                'output_hop_length',
                'input_sample_rate',
                'output_sample_rate'
            ]

            # Core model configuration parameters
            preserve_keys = [
                'languages',           # Language support
                'audio_config',        # Audio processing settings
                'speaker_embeddings',  # Speaker-related configurations
                'use_speaker_embedding',
                'use_multi_speaker',
                'multi_speaker_config',
                'custom_tokens',
                'speaker_dim',
                'd_vector_dim',
                'tokenizer_config'     # Tokenizer settings
            ]

            # Extract all important values
            for key in training_keys + preserve_keys:
                if key in config:
                    important_values[key] = config[key]

            return important_values if important_values else None

    return None

def create_xtts_core_config(model_architecture: Dict[str, Any],
                          original_values: Optional[Dict[str, Any]] = None,
                          gpt_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Creates the configuration for the core XTTS model.
    Integrates architectural parameters with audio processing settings
    and preserves custom configurations.

    Args:
        model_architecture: Parameters extracted from weights
        original_values: Original configuration to preserve
        gpt_config: GPT configuration to include

    Returns:
        Complete XTTS configuration dictionary
    """
    config = {
        "model_type": "xtts",
        "architectures": ["XttsGPT"],

        # Audio processing configuration - use originals or defaults
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

        # Model parameters - from architecture or originals
        "d_vector_dim": original_values.get('d_vector_dim', 512),
        "decoder_input_dim": model_architecture['hidden_size'],
        "num_chars": 255,

        # Processing parameters
        "duration_const": 102400,
        "output_hop_length": original_values.get('output_hop_length', 256),
        "input_sample_rate": original_values.get('input_sample_rate', 22050),
        "output_sample_rate": original_values.get('output_sample_rate', 24000),

        # GPT configuration
        "gpt": {"model_type": "xtts_gpt"},
        "gpt_config": gpt_config,  # Use the exact same GPT config
        "gpt_code_stride_len": original_values.get('gpt_code_stride_len', 1024),
        "cond_d_vector_in_each_upsampling_layer": True,

        # Auto-mapping configuration
        "auto_map": {
            "AutoConfig": "AstraMindAI/xtts2--xtts2_config.XTTSConfig",
            "AutoModelForCausalLM": "AstraMindAI/xtts2--xtts2_modeling.Xtts",
            "AutoTokenizer": "AstraMindAI/xtts2--tokenizer.XTTSTokenizerFast"
        },

        # Language support - use original or default list
        "languages": original_values.get('languages', [
            "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru",
            "nl", "cs", "ar", "zh-cn", "hu", "ko", "ja", "hi"
        ]),

        "tokenizer_file": "",
        "transformers_version": "4.46.0"
    }

    # Integrate any remaining original values
    if original_values:
        for key, value in original_values.items():
            if key == 'audio_config':
                config['audio_config'].update(value)
            else:
                config[key] = value

    return config

def create_auralis_config(model_architecture: Dict[str, Any],
                         original_values: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Creates the Auralis-compatible GPT configuration.
    Uses dynamically extracted architecture parameters while preserving
    original values for non-architectural settings.

    Args:
        model_architecture: Extracted model architecture parameters
        original_values: Original configuration values to preserve

    Returns:
        Complete GPT configuration dictionary
    """
    config = {
        "model_type": "xtts_gpt",
        "architectures": ["XttsGPT"],

        # Model architecture parameters (extracted from weights)
        "vocab_size": model_architecture['vocab_size'],
        "hidden_size": model_architecture['hidden_size'],
        "num_hidden_layers": model_architecture['num_hidden_layers'],
        "num_attention_heads": model_architecture['num_attention_heads'],
        "n_inner": model_architecture['n_inner'],

        # Token configuration
        "number_text_tokens": model_architecture['vocab_size'],
        "num_audio_tokens": model_architecture['num_audio_tokens'],
        "max_audio_tokens": model_architecture['max_audio_tokens'],
        "start_audio_token": model_architecture['start_audio_token'],
        "stop_audio_token": model_architecture['stop_audio_token'],

        # Model behavior settings - use originals or defaults
        "max_text_tokens": original_values.get('gpt_max_text_tokens', 402),
        "max_prompt_tokens": original_values.get('gpt_max_prompt_tokens', 70),
        "activation_function": "gelu_new",
        "attn_pdrop": 0.1,
        "layer_norm_epsilon": 1e-5,
        "initializer_range": 0.02,
        "use_masking_gt_prompt_approach": True,
        "use_perceiver_resampler": True,

        # Performance settings
        "kv_cache": True,
        "enable_redaction": False,
        "reorder_and_upcast_attn": False,
        "scale_attn_by_inverse_layer_idx": False,

        # Auralis auto-mapping
        "auto_map": {
            "AutoConfig": "AstraMindAI/xtts2-gpt--gpt_config.XTTSGPTConfig",
            "AutoModelForCausalLM": "AstraMindAI/xtts2-gpt--xtts2_gpt_modeling.XttsGPT",
            "AutoTokenizer": "AstraMindAI/xtts2-gpt--tokenizer.XTTSTokenizerFast"
        }
    }

    # Integrate preserved values
    if original_values:
        for key, value in original_values.items():
            if key not in ['vocab_size', 'hidden_size', 'num_hidden_layers', 'num_attention_heads', 'n_inner',
                           'number_text_tokens', 'num_audio_tokens', 'max_audio_tokens', 'start_audio_token',
                           'stop_audio_token']:
                config[key] = value

    return config

def convert_model_weights(model_state: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Converts model weights into the correct formats for GPT and XTTS components.
    Handles weight renaming and separation according to Auralis conventions.

    Args:
        model_state: Dictionary containing model weights

    Returns:
        Tuple of (gpt_weights, xtts_weights)
    """
    gpt_weights = {}
    xtts_weights = {}

    # Define patterns for identifying different types of weights
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

    # Process each weight tensor
    for key, tensor in model_state.items():
        # Skip training-specific layers
        if any(pattern in key for pattern in training_ignore):
            continue

        # Clean up key name
        key = key.replace('xtts.', '')
        is_gpt_weight = any(pattern in key for pattern in gpt_patterns + ignore_patterns)

        if is_gpt_weight:
            # Handle special cases for GPT weights
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
            # final_norm goes to both sets
            clean_key = key.replace('gpt.', '')
            gpt_weights[clean_key] = tensor
            xtts_weights[clean_key] = tensor
        else:
            # All other weights go to XTTS
            xtts_weights[key.replace('gpt.', '')] = tensor

    # Verify all required GPT patterns are present
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
    """
    Saves model weights in SafeTensors format in the correct directory structure.
    Creates all necessary directories and provides feedback on the process.

    Args:
        gpt_weights: Dictionary of GPT model weights
        xtts_weights: Dictionary of XTTS model weights
        output_dir: Base output directory

    Returns:
        Tuple of (gpt_weights_path, xtts_weights_path)
    """
    # Create directory structure
    gpt_dir = os.path.join(output_dir, "gpt")
    xtts_dir = os.path.join(output_dir, "core_xttsv2")
    os.makedirs(gpt_dir, exist_ok=True)
    os.makedirs(xtts_dir, exist_ok=True)

    # Save GPT weights
    gpt_path = os.path.join(gpt_dir, 'gpt2_model.safetensors')
    save_file(gpt_weights, gpt_path)
    print(f"GPT weights saved to: {gpt_path}")
    print(f"GPT weight keys: {list(gpt_weights.keys())}")

    # Save XTTS weights
    xtts_path = os.path.join(xtts_dir, 'xtts-v2.safetensors')
    save_file(xtts_weights, xtts_path)
    print(f"XTTS weights saved to: {xtts_path}")
    print(f"XTTS weight keys: {list(xtts_weights.keys())}")

    return gpt_path, xtts_path

def save_configs(output_dir: str, checkpoint: Dict[str, Any]) -> Tuple[str, str, str, str]:
    """
    Creates and saves all necessary configurations for Auralis compatibility.
    Handles both GPT and XTTS configurations with backups.

    Args:
        output_dir: Base output directory
        checkpoint: Full PyTorch checkpoint dictionary

    Returns:
        Tuple of (gpt_config_path, gpt_backup_path, xtts_config_path, xtts_backup_path)
    """
    # Extract configurations
    model_architecture = analyze_model_architecture(checkpoint['model'])
    original_values = extract_original_values(checkpoint)

    # Create configurations
    gpt_config = create_auralis_config(model_architecture, original_values)
    xtts_config = create_xtts_core_config(model_architecture, original_values, gpt_config)
    # Prepare directories
    gpt_dir = os.path.join(output_dir, "gpt")
    xtts_dir = os.path.join(output_dir, "core_xttsv2")
    os.makedirs(gpt_dir, exist_ok=True)
    os.makedirs(xtts_dir, exist_ok=True)

    # Save GPT configuration and backup
    gpt_config_path = os.path.join(gpt_dir, "config.json")
    gpt_backup_path = os.path.join(gpt_dir, "config.original.json")

    with open(gpt_config_path, 'w', encoding='utf-8') as f:
        json.dump(gpt_config, f, indent=2)
    shutil.copy2(gpt_config_path, gpt_backup_path)

    # Save XTTS configuration and backup
    xtts_config_path = os.path.join(xtts_dir, "config.json")
    xtts_backup_path = os.path.join(xtts_dir, "config.original.json")

    with open(xtts_config_path, 'w', encoding='utf-8') as f:
        json.dump(xtts_config, f, indent=2)
    shutil.copy2(xtts_config_path, xtts_backup_path)

    return gpt_config_path, gpt_backup_path, xtts_config_path, xtts_backup_path

def download_safely(repo_id: str, output_dir: str, config_path: str):
    """
    Downloads repository files while preserving our local configurations.
    Ensures downloaded files don't overwrite our custom configurations.

    Args:
        repo_id: Hugging Face repository ID
        output_dir: Directory to save downloaded files
        config_path: Path to our configuration file to preserve
    """
    # Backup existing configuration
    with open(config_path, 'r', encoding='utf-8') as f:
        our_config = json.load(f)

    # Download files, excluding safetensors and config
    snapshot_download(
        repo_id=repo_id,
        ignore_patterns=['*.safetensors', 'config.json'],
        local_dir=output_dir
    )

    # Restore our configuration
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(our_config, f, indent=2)

def convert_checkpoint(pytorch_checkpoint_path: str, output_dir: str):
    """
    Main conversion process that handles both configurations and weights.
    Ensures proper directory structure and file organization.

    Args:
        pytorch_checkpoint_path: Path to the input PyTorch checkpoint
        output_dir: Base output directory for converted files
    """
    # Load checkpoint
    print(f"Loading checkpoint from: {pytorch_checkpoint_path}")
    checkpoint = torch.load(pytorch_checkpoint_path, map_location='cpu')

    # Save configurations
    print("\nCreating and saving configurations...")
    gpt_config_path, gpt_backup, xtts_config_path, xtts_backup = save_configs(
        output_dir, checkpoint
    )
    print(f"GPT config: {gpt_config_path} (backup: {gpt_backup})")
    print(f"XTTS config: {xtts_config_path} (backup: {xtts_backup})")

    # Convert and save weights
    print("\nConverting model weights...")
    gpt_weights, xtts_weights = convert_model_weights(checkpoint['model'])

    print("\nSaving weights...")
    gpt_weights_path, xtts_weights_path = save_model_weights(
        gpt_weights, xtts_weights, output_dir
    )

    # Download additional files safely
    print("\nDownloading additional files...")
    # First handle GPT files
    download_safely(
        "AstraMindAI/xtts2-gpt",
        os.path.join(output_dir, "gpt"),
        gpt_config_path
    )
    # Then handle XTTS files
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
    """
    Main entry point with argument parsing and default paths for testing.
    Handles command-line arguments and initiates the conversion process.
    """
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

    # Default paths for testing
    import sys
    if len(sys.argv) == 1:
        # If no arguments provided, use default test paths
        default_paths = [
            '/home/astramind-giacomo/Downloads/coqui_2_0_0/model.pth',
            '--output_dir', '/home/astramind-giacomo/Downloads/'
        ]
        sys.argv.extend(default_paths)

    args = parser.parse_args()

    # Verify input file exists
    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint file '{args.checkpoint_path}' does not exist")
        return

    # Convert the checkpoint
    convert_checkpoint(args.checkpoint_path, args.output_dir)

if __name__ == '__main__':
    main()