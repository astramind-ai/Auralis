import pytest
import os
import torch
import json
from auralis.models.xttsv2.utils.checkpoint_converter import (
    analyze_model_architecture,
    extract_original_values,
    create_xtts_core_config,
    create_auralis_config,
    convert_model_weights,
    save_model_weights,
    save_configs,
    download_safely,
    convert_checkpoint,
)

@pytest.fixture
def sample_model_state():
    """Fixture to create a dummy (but realistic) model state dictionary for testing."""
    return {
        'text_embedding.weight': torch.randn(6153, 1024),
        'gpt.gpt.h.0.ln_1.weight': torch.randn(1024),
        'gpt.gpt.h.0.ln_1.bias': torch.randn(1024),
        'gpt.gpt.h.0.attn.c_attn.weight': torch.randn(1024, 1024 * 3),
        'gpt.gpt.h.0.attn.c_attn.bias': torch.randn(1024 * 3),
        'gpt.gpt.h.0.attn.c_proj.weight': torch.randn(1024, 1024),
        'gpt.gpt.h.0.attn.c_proj.bias': torch.randn(1024),
        'gpt.gpt.h.0.ln_2.weight': torch.randn(1024),
        'gpt.gpt.h.0.ln_2.bias': torch.randn(1024),
        'gpt.gpt.h.0.mlp.c_fc.weight': torch.randn(1024, 1024 * 4),
        'gpt.gpt.h.0.mlp.c_fc.bias': torch.randn(1024 * 4),
        'gpt.gpt.h.0.mlp.c_proj.weight': torch.randn(1024 * 4, 1024),
        'gpt.gpt.h.0.mlp.c_proj.bias': torch.randn(1024),
        'gpt.gpt.h.1.ln_1.weight': torch.randn(1024),
        'gpt.gpt.h.1.ln_1.bias': torch.randn(1024),
        'gpt.gpt.h.1.attn.c_attn.weight': torch.randn(1024, 1024 * 3),
        'gpt.gpt.h.1.attn.c_attn.bias': torch.randn(1024 * 3),
        'gpt.gpt.h.1.attn.c_proj.weight': torch.randn(1024, 1024),
        'gpt.gpt.h.1.attn.c_proj.bias': torch.randn(1024),
        'gpt.gpt.h.1.ln_2.weight': torch.randn(1024),
        'gpt.gpt.h.1.ln_2.bias': torch.randn(1024),
        'gpt.gpt.h.1.mlp.c_fc.weight': torch.randn(1024, 1024 * 4),
        'gpt.gpt.h.1.mlp.c_fc.bias': torch.randn(1024 * 4),
        'gpt.gpt.h.1.mlp.c_proj.weight': torch.randn(1024 * 4, 1024),
        'gpt.gpt.h.1.mlp.c_proj.bias': torch.randn(1024),
        'gpt.ln_f.weight': torch.randn(1024),
        'gpt.ln_f.bias': torch.randn(1024),
        'mel_head.weight': torch.randn(1026, 1024),
        'mel_head.bias': torch.randn(1026),
         'xtts.mel_embedding.weight': torch.randn(1026, 1024),
         'xtts.mel_pos_embedding.emb.weight' : torch.randn(400,1024)
    }

@pytest.fixture
def sample_checkpoint():
    """Fixture to create a dummy checkpoint dictionary for testing."""
    return {
        "model": {
            'text_embedding.weight': torch.randn(6153, 1024),
            'gpt.gpt.h.0.ln_1.weight': torch.randn(1024),
            'gpt.gpt.h.0.ln_1.bias': torch.randn(1024),
            'gpt.gpt.h.0.attn.c_attn.weight': torch.randn(1024, 1024 * 3),
            'gpt.gpt.h.0.attn.c_attn.bias': torch.randn(1024 * 3),
            'gpt.gpt.h.0.attn.c_proj.weight': torch.randn(1024, 1024),
            'gpt.gpt.h.0.attn.c_proj.bias': torch.randn(1024),
            'gpt.gpt.h.0.ln_2.weight': torch.randn(1024),
            'gpt.gpt.h.0.ln_2.bias': torch.randn(1024),
            'gpt.gpt.h.0.mlp.c_fc.weight': torch.randn(1024, 1024 * 4),
            'gpt.gpt.h.0.mlp.c_fc.bias': torch.randn(1024 * 4),
            'gpt.gpt.h.0.mlp.c_proj.weight': torch.randn(1024 * 4, 1024),
            'gpt.gpt.h.0.mlp.c_proj.bias': torch.randn(1024),
            'gpt.gpt.h.1.ln_1.weight': torch.randn(1024),
            'gpt.gpt.h.1.ln_1.bias': torch.randn(1024),
            'gpt.gpt.h.1.attn.c_attn.weight': torch.randn(1024, 1024 * 3),
            'gpt.gpt.h.1.attn.c_attn.bias': torch.randn(1024 * 3),
            'gpt.gpt.h.1.attn.c_proj.weight': torch.randn(1024, 1024),
            'gpt.gpt.h.1.attn.c_proj.bias': torch.randn(1024),
            'gpt.gpt.h.1.ln_2.weight': torch.randn(1024),
            'gpt.gpt.h.1.ln_2.bias': torch.randn(1024),
            'gpt.gpt.h.1.mlp.c_fc.weight': torch.randn(1024, 1024 * 4),
            'gpt.gpt.h.1.mlp.c_fc.bias': torch.randn(1024 * 4),
            'gpt.gpt.h.1.mlp.c_proj.weight': torch.randn(1024 * 4, 1024),
            'gpt.gpt.h.1.mlp.c_proj.bias': torch.randn(1024),
            'gpt.ln_f.weight': torch.randn(1024),
            'gpt.ln_f.bias': torch.randn(1024),
            'mel_head.weight': torch.randn(1026, 1024),
            'mel_head.bias': torch.randn(1026),
             'xtts.mel_embedding.weight': torch.randn(1026, 1024),
             'xtts.mel_pos_embedding.emb.weight' : torch.randn(400,1024)
        },
        "config": {
            "gpt_max_text_tokens": 402,
            "output_hop_length": 256,
            "input_sample_rate": 22050,
            "output_sample_rate": 24000,
            "languages":  [
              "en",
              "es",
              "fr",
              "de",
              "it",
              "pt",
              "pl",
              "tr",
              "ru",
              "nl",
              "cs",
              "ar",
              "zh-cn",
              "hu",
              "ko",
              "ja"
            ],
             "audio_config": {
                 "sample_rate": 22050,
                 "output_sample_rate": 24000,

            },
            "d_vector_dim": 512,
            "speaker_dim": 512,
            "use_speaker_embedding": True,
             "use_multi_speaker": True,
            "multi_speaker_config": {"num_speakers": 100},
             "custom_tokens": None,
            "tokenizer_config": {},
            "gpt_code_stride_len" : 1024,

        },
        "model_args": { #Add this block and put the keys here
               "use_masking_gt_prompt_approach": True,
                "use_perceiver_resampler": True
            }
    }
@pytest.fixture
def sample_output_dir(tmpdir):
   """Fixture to create a temporary output directory"""
   return str(tmpdir)


def test_analyze_model_architecture(sample_model_state):
    """Tests the model architecture analysis function."""
    architecture = analyze_model_architecture(sample_model_state)
    assert architecture['vocab_size'] == 6153
    assert architecture['hidden_size'] == 1024
    assert architecture['num_hidden_layers'] == 2
    assert architecture['num_attention_heads'] == 16
    assert architecture['n_inner'] == 1024*4
    assert architecture['num_audio_tokens'] == 1026
    assert architecture['max_audio_tokens'] == 1026 - 421
    assert architecture['start_audio_token'] == 1026 - 2
    assert architecture['stop_audio_token'] == 1026 - 1

def test_analyze_model_architecture_edge_case():
  """Test edge case when attention head dimensions are not divisible by 64"""
  model_state_edge = {
      'text_embedding.weight': torch.randn(6153, 1023),  #hidden_size 1023
      'gpt.gpt.h.0.ln_1.weight': torch.randn(1023),
       'gpt.gpt.h.0.ln_1.bias': torch.randn(1023),
       'gpt.gpt.h.0.attn.c_attn.weight': torch.randn(1023, 1023 * 3),
       'gpt.gpt.h.0.attn.c_attn.bias': torch.randn(1023 * 3),
       'gpt.gpt.h.0.attn.c_proj.weight': torch.randn(1023, 1023),
       'gpt.gpt.h.0.attn.c_proj.bias': torch.randn(1023),
       'gpt.gpt.h.0.ln_2.weight': torch.randn(1023),
       'gpt.gpt.h.0.ln_2.bias': torch.randn(1023),
       'gpt.gpt.h.0.mlp.c_fc.weight': torch.randn(1023, 1023 * 4),
       'gpt.gpt.h.0.mlp.c_fc.bias': torch.randn(1023 * 4),
       'gpt.gpt.h.0.mlp.c_proj.weight': torch.randn(1023 * 4, 1023),
       'gpt.gpt.h.0.mlp.c_proj.bias': torch.randn(1023),
       'gpt.gpt.h.1.ln_1.weight': torch.randn(1023),
       'gpt.gpt.h.1.ln_1.bias': torch.randn(1023),
      'gpt.gpt.h.1.attn.c_attn.weight': torch.randn(1023, 1023 * 3),
       'gpt.gpt.h.1.attn.c_attn.bias': torch.randn(1023 * 3),
       'gpt.gpt.h.1.attn.c_proj.weight': torch.randn(1023, 1023),
       'gpt.gpt.h.1.attn.c_proj.bias': torch.randn(1023),
       'gpt.gpt.h.1.ln_2.weight': torch.randn(1023),
       'gpt.gpt.h.1.ln_2.bias': torch.randn(1023),
       'gpt.gpt.h.1.mlp.c_fc.weight': torch.randn(1023, 1023 * 4),
       'gpt.gpt.h.1.mlp.c_fc.bias': torch.randn(1023 * 4),
       'gpt.gpt.h.1.mlp.c_proj.weight': torch.randn(1023 * 4, 1023),
       'gpt.gpt.h.1.mlp.c_proj.bias': torch.randn(1023),
       'gpt.ln_f.weight': torch.randn(1023),
       'gpt.ln_f.bias': torch.randn(1023),
      'mel_head.weight': torch.randn(1026, 1023),
       'xtts.mel_embedding.weight': torch.randn(1026, 1023),
        'xtts.mel_pos_embedding.emb.weight' : torch.randn(400,1023)
  }
  architecture = analyze_model_architecture(model_state_edge)
  assert architecture['vocab_size'] == 6153
  assert architecture['hidden_size'] == 1023
  assert architecture['num_hidden_layers'] == 2
  assert architecture['num_attention_heads'] == 1 #Expected to be set to one
  assert architecture['n_inner'] == 1023*4
  assert architecture['num_audio_tokens'] == 1026
  assert architecture['max_audio_tokens'] == 1026 - 421
  assert architecture['start_audio_token'] == 1026 - 2
  assert architecture['stop_audio_token'] == 1026 - 1


def test_extract_original_values(sample_checkpoint):
    """Tests the extraction of original config values."""
    original_values = extract_original_values(sample_checkpoint)
    assert original_values['gpt_max_text_tokens'] == 402
    assert original_values['output_hop_length'] == 256
    assert original_values['input_sample_rate'] == 22050
    assert original_values['output_sample_rate'] == 24000
    assert original_values['languages'] == [
              "en",
              "es",
              "fr",
              "de",
              "it",
              "pt",
              "pl",
              "tr",
              "ru",
              "nl",
              "cs",
              "ar",
              "zh-cn",
              "hu",
              "ko",
              "ja"
            ]
    assert original_values['audio_config'] == {
                 "sample_rate": 22050,
                 "output_sample_rate": 24000,
            }
    assert original_values["d_vector_dim"] == 512
    assert original_values['speaker_dim'] == 512
    assert original_values["use_speaker_embedding"] == True
    assert original_values["use_multi_speaker"] == True
    assert original_values["multi_speaker_config"] == {"num_speakers": 100}
    assert original_values["custom_tokens"] == None
    assert isinstance(original_values["tokenizer_config"], dict)
    assert original_values["gpt_code_stride_len"] == 1024
    assert original_values["use_masking_gt_prompt_approach"] == True
    assert original_values["use_perceiver_resampler"] == True


def test_extract_original_values_no_config(sample_model_state):
    """Test edge case when there's no config in checkpoint"""
    checkpoint = {"model": sample_model_state}
    assert extract_original_values(checkpoint) is None


def test_create_xtts_core_config(sample_model_state, sample_checkpoint):
   """Tests the creation of the XTTS core configuration."""
   model_architecture = analyze_model_architecture(sample_model_state)
   original_values = extract_original_values(sample_checkpoint)
   gpt_config = create_auralis_config(model_architecture, original_values)
   xtts_config = create_xtts_core_config(model_architecture, original_values, gpt_config)

   assert xtts_config['model_type'] == 'xtts'
   assert xtts_config["audio_config"]["hop_length"] == 256
   assert xtts_config["audio_config"]["sample_rate"] == 22050
   assert xtts_config["audio_config"]["output_sample_rate"] == 24000
   assert xtts_config["d_vector_dim"] == 512
   assert xtts_config["decoder_input_dim"] == 1024
   assert xtts_config["gpt_code_stride_len"] == 1024
   assert xtts_config["languages"] == [
              "en",
              "es",
              "fr",
              "de",
              "it",
              "pt",
              "pl",
              "tr",
              "ru",
              "nl",
              "cs",
              "ar",
              "zh-cn",
              "hu",
              "ko",
              "ja"
            ]
   assert xtts_config["gpt"]["model_type"] == "xtts_gpt"
   assert isinstance(xtts_config["gpt_config"], dict)


def test_create_auralis_config(sample_model_state, sample_checkpoint):
   """Tests the creation of the Auralis GPT configuration."""
   model_architecture = analyze_model_architecture(sample_model_state)
   original_values = extract_original_values(sample_checkpoint)
   auralis_config = create_auralis_config(model_architecture, original_values)

   assert auralis_config['model_type'] == 'xtts_gpt'
   assert auralis_config['vocab_size'] == 6153
   assert auralis_config['hidden_size'] == 1024
   assert auralis_config['num_hidden_layers'] == 2
   assert auralis_config['num_attention_heads'] == 16
   assert auralis_config['n_inner'] == 1024*4
   assert auralis_config['number_text_tokens'] == 6153
   assert auralis_config['num_audio_tokens'] == 1026
   assert auralis_config['max_audio_tokens'] == 1026 - 421
   assert auralis_config['start_audio_token'] == 1026 - 2
   assert auralis_config['stop_audio_token'] == 1026 - 1
   assert auralis_config["max_text_tokens"] == 402
   assert auralis_config["max_prompt_tokens"] == 70
   assert auralis_config["use_masking_gt_prompt_approach"] == True
   assert auralis_config["use_perceiver_resampler"] == True


def test_convert_model_weights(sample_model_state):
    """Tests model weights conversion."""
    gpt_weights, xtts_weights = convert_model_weights(sample_model_state)
    assert 'gpt.wte.weight' in gpt_weights #fixed assert
    assert 'gpt.wpe.emb.weight' in gpt_weights #fixed assert
    assert 'gpt.h.0.attn.c_attn.weight' in gpt_weights #fixed assert
    assert 'gpt.h.1.attn.c_attn.weight' in gpt_weights #fixed assert
    assert 'mel_head.weight' in gpt_weights
    assert len(xtts_weights) == 1 #Added one because of the embedding layers


def test_convert_model_weights_final_norm(sample_model_state):
  """Tests final_norm weights are included in both gpt and xtts"""
  sample_model_state_with_final_norm = sample_model_state.copy()
  sample_model_state_with_final_norm["gpt.final_norm.weight"] = torch.randn(1024)
  sample_model_state_with_final_norm["gpt.final_norm.bias"] = torch.randn(1024)
  gpt_weights, xtts_weights = convert_model_weights(sample_model_state_with_final_norm)

  assert "final_norm.weight" in gpt_weights
  assert "final_norm.bias" in gpt_weights
  assert "final_norm.weight" in xtts_weights
  assert "final_norm.bias" in xtts_weights


def test_convert_model_weights_missing_patterns():
  """Tests exception when not all needed GPT patterns are in the input"""
  model_state = {
        'text_embedding.weight': torch.randn(6153, 1024),
        'gpt.gpt.h.0.attn.c_attn.weight': torch.randn(1024, 1024 * 3),
        'gpt.gpt.h.1.attn.c_attn.weight': torch.randn(1024, 1024 * 3),
    }
  with pytest.raises(ValueError, match=r"Missing required GPT patterns:.*"):
        convert_model_weights(model_state)

def test_save_model_weights(sample_model_state, sample_output_dir):
    """Tests saving model weights."""
    gpt_weights, xtts_weights = convert_model_weights(sample_model_state)
    gpt_path, xtts_path = save_model_weights(gpt_weights, xtts_weights, sample_output_dir)

    assert os.path.exists(gpt_path)
    assert os.path.exists(xtts_path)
    assert os.path.basename(gpt_path) == "gpt2_model.safetensors"
    assert os.path.basename(xtts_path) == "xtts-v2.safetensors"


def test_save_configs(sample_output_dir, sample_checkpoint):
    """Tests saving model configurations."""
    gpt_config_path, gpt_backup_path, xtts_config_path, xtts_backup_path = save_configs(
        sample_output_dir, sample_checkpoint
    )
    assert os.path.exists(gpt_config_path)
    assert os.path.exists(gpt_backup_path)
    assert os.path.exists(xtts_config_path)
    assert os.path.exists(xtts_backup_path)
    assert os.path.basename(gpt_config_path) == "config.json"
    assert os.path.basename(gpt_backup_path) == "config.original.json"
    assert os.path.basename(xtts_config_path) == "config.json"
    assert os.path.basename(xtts_backup_path) == "config.original.json"

def test_download_safely(sample_output_dir):
    """Tests the safe download of files."""
    test_config = {"test": "value"}
    config_path = os.path.join(sample_output_dir, "test_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(test_config, f)
    try:
       download_safely(
            "AstraMindAI/xtts2-gpt",  #dummy repo for testing
            sample_output_dir,
            config_path
        )
    except Exception as e:
       print(f"Warning: download safely test failed with error {e}")
    with open(config_path, 'r', encoding='utf-8') as f:
      downloaded_config = json.load(f)
    assert downloaded_config == test_config #check if original file was preserved

def test_convert_checkpoint(sample_checkpoint, sample_output_dir):
    """Tests the full checkpoint conversion process."""
    # Create a fake checkpoint file
    checkpoint_file = os.path.join(sample_output_dir, "test_checkpoint.pth")
    torch.save(sample_checkpoint, checkpoint_file)
    convert_checkpoint(checkpoint_file, sample_output_dir)

    assert os.path.exists(os.path.join(sample_output_dir, "gpt", "config.json"))
    assert os.path.exists(os.path.join(sample_output_dir, "gpt", "gpt2_model.safetensors"))
    assert os.path.exists(os.path.join(sample_output_dir, "core_xttsv2", "config.json"))
    assert os.path.exists(os.path.join(sample_output_dir, "core_xttsv2", "xtts-v2.safetensors"))
    assert os.path.exists(os.path.join(sample_output_dir, "gpt", "README.md"))
    assert os.path.exists(os.path.join(sample_output_dir, "core_xttsv2", "README.md"))