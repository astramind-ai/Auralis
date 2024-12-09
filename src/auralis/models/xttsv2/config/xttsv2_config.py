from dataclasses import asdict, dataclass
from typing import Dict, Optional, List
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


@dataclass
class GPTAudioConfig:
    """Configuration for GPT audio processing parameters.
    
    This class defines the basic audio processing parameters used by the GPT
    component of the XTTS model.

    Attributes:
        mel_channels (int): Number of mel-spectrogram channels. Defaults to 80.
        sample_rate (int): Input audio sampling rate in Hz. Defaults to 22050.
        output_sample_rate (int): Output audio sampling rate in Hz. Defaults to 24000.
    """
    mel_channels: int = 80
    sample_rate: int = 22050
    output_sample_rate: int = 24000

@dataclass
class XTTSAudioConfig:
    """Configuration for XTTS audio processing parameters.
    
    This class defines the complete set of audio processing parameters used
    throughout the XTTS model, including mel-spectrogram generation and
    normalization.

    Attributes:
        sample_rate (int): Input audio sampling rate in Hz. Defaults to 22050.
        output_sample_rate (int): Output audio sampling rate in Hz. Defaults to 24000.
        mel_channels (int): Number of mel-spectrogram channels. Defaults to 80.
        hop_length (int): Number of samples between STFT columns. Defaults to 256.
        win_length (int): Window size for STFT. Defaults to 1024.
        n_fft (int): FFT size. Defaults to 1024.
        fmin (int): Minimum frequency for mel scale. Defaults to 0.
        fmax (int): Maximum frequency for mel scale. Defaults to 8000.
        power (float): Power of the magnitude spectrogram. Defaults to 1.0.
        mel_norms_file (Optional[str]): Path to mel-spectrogram normalization file.
    """
    sample_rate: int = 22050
    output_sample_rate: int = 24000
    mel_channels: int = 80
    hop_length: int = 256
    win_length: int = 1024
    n_fft: int = 1024
    fmin: int = 0
    fmax: int = 8000
    power: float = 1.0
    mel_norms_file: Optional[str] = None


class XTTSGPTConfig(PretrainedConfig):
    """Configuration for the GPT component of XTTS.
    
    This class defines the architecture and behavior of the GPT model used in XTTS.
    It inherits from HuggingFace's PretrainedConfig for compatibility with the
    transformers library.

    The GPT model is responsible for generating audio tokens from text tokens,
    with support for various conditioning signals and architectural features.

    Attributes:
        hidden_size (int): Size of hidden layers. Defaults to 1024.
        n_inner (int): Size of feed-forward inner layer. Defaults to 4096.
        num_hidden_layers (int): Number of transformer layers. Defaults to 30.
        num_attention_heads (int): Number of attention heads. Defaults to 16.
        vocab_size (int): Size of text vocabulary. Defaults to 6681.
        number_text_tokens (int): Explicit text token vocabulary size.
        start_text_token (Optional[int]): Token ID for text start.
        stop_text_token (Optional[int]): Token ID for text end.
        num_audio_tokens (int): Size of audio token vocabulary. Defaults to 1026.
        start_audio_token (int): Token ID for audio start. Defaults to 1024.
        stop_audio_token (int): Token ID for audio end. Defaults to 1025.
        max_audio_tokens (int): Maximum audio sequence length. Defaults to 605.
        max_text_tokens (int): Maximum text sequence length. Defaults to 402.
        max_prompt_tokens (int): Maximum prompt sequence length. Defaults to 70.
        use_masking_gt_prompt_approach (bool): Whether to use masking. Defaults to True.
        use_perceiver_resampler (bool): Whether to use perceiver. Defaults to True.
        kv_cache (bool): Whether to use KV cache. Defaults to True.
        enable_redaction (bool): Whether to enable redaction. Defaults to False.
        audio_config (Optional[Dict]): Audio processing configuration.
    """
    model_type = "xtts_gpt"

    def __init__(
            self,
            # Model architecture
            hidden_size: int = 1024,  # gpt_n_model_channels in original
            n_inner: int = 4096,
            num_hidden_layers: int = 30,  # gpt_layers in original
            num_attention_heads: int = 16,  # gpt_n_heads in original

            # Tokenizer settings
            vocab_size: int = 6681,  # gpt_number_text_tokens in original
            number_text_tokens: int = 6681,  # Explicit text token vocabulary size
            start_text_token: Optional[int] = None,
            stop_text_token: Optional[int] = None,

            # Audio token settings
            num_audio_tokens: int = 1026,  # gpt_num_audio_tokens in original
            start_audio_token: int = 1024,  # gpt_start_audio_token in original
            stop_audio_token: int = 1025,  # gpt_stop_audio_token in original

            # Sequence length settings
            max_audio_tokens: int = 605,  # gpt_max_audio_tokens in original
            max_text_tokens: int = 402,  # gpt_max_text_tokens in original
            max_prompt_tokens: int = 70,  # gpt_max_prompt_tokens in original
            gpt_max_audio_tokens: int = 605,  # Used for generation

            # Model behavior settings
            use_masking_gt_prompt_approach: bool = True,  # gpt_use_masking_gt_prompt_approach in original
            use_perceiver_resampler: bool = True,  # gpt_use_perceiver_resampler in original
            kv_cache: bool = True,
            enable_redaction: bool = False,

            # GPT batch settings
            gpt_batch_size: int = 1,

            # Audio processing
            audio_config: Optional[Dict] = None,

            # Architecture specifics
            layer_norm_epsilon: float = 1e-5,
            initializer_range: float = 0.02,
            add_cross_attention: bool = False,
            scale_attn_by_inverse_layer_idx: bool = False,
            reorder_and_upcast_attn: bool = False,

            # Size settings for the decoder
            decoder_input_dim: int = 1024,
            architectures=["XttsGPT"],
            auto_map={
                "AutoConfig": "AstraMindAI/xtts2-gpt--gpt_config.XTTSGPTConfig",
                "AutoModelForCausalLM": "AstraMindAI/xtts2-gpt--xtts2_gpt_modeling.XttsGPT",
            },
            activation_function: str = "gelu",
            attn_pdrop: float = 0.1,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.architectures = architectures
        self.auto_map = auto_map
        self.audio_config = GPTAudioConfig(
            **audio_config if audio_config is not None else {}
        )
        self.activation_function = activation_function
        self.attn_pdrop = attn_pdrop
        self.hidden_size = hidden_size
        self.n_inner = n_inner
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        self.vocab_size = vocab_size
        self.number_text_tokens = number_text_tokens
        self.start_text_token = start_text_token
        self.stop_text_token = stop_text_token

        self.num_audio_tokens = num_audio_tokens
        self.start_audio_token = start_audio_token
        self.stop_audio_token = stop_audio_token

        self.max_audio_tokens = max_audio_tokens
        self.max_text_tokens = max_text_tokens
        self.max_prompt_tokens = max_prompt_tokens
        self.gpt_max_audio_tokens = gpt_max_audio_tokens

        self.use_masking_gt_prompt_approach = use_masking_gt_prompt_approach
        self.use_perceiver_resampler = use_perceiver_resampler
        self.kv_cache = kv_cache
        self.enable_redaction = enable_redaction

        self.gpt_batch_size = gpt_batch_size

        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.add_cross_attention = add_cross_attention
        self.scale_attn_by_inverse_layer_idx = scale_attn_by_inverse_layer_idx
        self.reorder_and_upcast_attn = reorder_and_upcast_attn

        self.decoder_input_dim = decoder_input_dim

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary format.

        Returns:
            Dict: Configuration dictionary including audio config.
        """
        output = super().to_dict()
        output["audio_config"] = asdict(self.audio_config)
        return output

    @classmethod
    def from_dict(cls, config_dict: Dict, *args, **kwargs) -> "XTTSGPTConfig":
        """Create configuration from dictionary.

        Args:
            config_dict (Dict): Configuration dictionary.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            XTTSGPTConfig: Configuration instance.
        """
        return cls(**config_dict)


class XTTSConfig(PretrainedConfig):
    """Configuration for the complete XTTS model.
    
    This class defines the configuration for all XTTS components except the GPT
    model (which has its own config). It includes settings for audio processing,
    model architecture, training, tokenization, and language support.

    Attributes:
        audio_config (XTTSAudioConfig): Audio processing configuration.
        input_sample_rate (int): Input audio sampling rate. Defaults to 22050.
        output_sample_rate (int): Output audio sampling rate. Defaults to 24000.
        output_hop_length (int): Output hop length. Defaults to 256.
        decoder_input_dim (int): Decoder input dimension. Defaults to 1024.
        d_vector_dim (int): Speaker embedding dimension. Defaults to 512.
        cond_d_vector_in_each_upsampling_layer (bool): Whether to condition each
            upsampling layer. Defaults to True.
        gpt_code_stride_len (int): GPT code stride length. Defaults to 1024.
        duration_const (int): Duration constant. Defaults to 102400.
        tokenizer_file (str): Path to tokenizer file.
        num_chars (int): Number of characters. Defaults to 255.
        languages (List[str]): Supported language codes.
        gpt (XTTSGPTConfig): GPT model configuration.
    """
    model_type = "xtts"

    def __init__(
            self,
            # Audio settings
            audio_config: Optional[Dict] = None,
            input_sample_rate: int = 22050,
            output_sample_rate: int = 24000,
            output_hop_length: int = 256,

            # Model architecture
            decoder_input_dim: int = 1024,
            d_vector_dim: int = 512,
            cond_d_vector_in_each_upsampling_layer: bool = True,

            # Training settings
            gpt_code_stride_len: int = 1024,
            duration_const: int = 102400,

            # Tokenizer settings
            tokenizer_file: str = "",
            num_chars: int = 255,

            # Language support
            languages: Optional[List[str]] = None,

            # GPT configuration
            gpt_config: Optional[Dict] = None,
            architectures=["Xtts"],
            auto_map = {
                       "AutoConfig": "AstraMindAI/xtts2--xtts2_config.XTTSConfig",
                       "AutoModelForCausalLM": "AstraMindAI/xtts2--xtts2_modeling.Xtts",
                   },
            **kwargs
    ):
        super().__init__(**kwargs)
        self.architectures = architectures
        self.auto_map = auto_map
        # Initialize audio config
        self.audio_config = XTTSAudioConfig(
            **audio_config if audio_config is not None else {}
        )

        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.output_hop_length = output_hop_length

        self.decoder_input_dim = decoder_input_dim
        self.d_vector_dim = d_vector_dim
        self.cond_d_vector_in_each_upsampling_layer = cond_d_vector_in_each_upsampling_layer

        self.gpt_code_stride_len = gpt_code_stride_len
        self.duration_const = duration_const

        self.tokenizer_file = tokenizer_file
        self.num_chars = num_chars

        # Initialize GPT config
        self.gpt = XTTSGPTConfig(**gpt_config if gpt_config is not None else {})

        if languages is None:
            self.languages = [
                "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru",
                "nl", "cs", "ar", "zh-cn", "hu", "ko", "ja", "hi"
            ]
        else:
            self.languages = languages

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary format.

        Returns:
            Dict: Configuration dictionary including audio and GPT configs.
        """
        output = super().to_dict()
        output["audio_config"] = asdict(self.audio_config)
        output["gpt_config"] = self.gpt.to_dict()
        return output

    @classmethod
    def from_dict(cls, config_dict: Dict, *args, **kwargs) -> "XTTSConfig":
        """Create configuration from dictionary.

        Args:
            config_dict (Dict): Configuration dictionary.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            XTTSConfig: Configuration instance.
        """
        if "gpt_config" in config_dict:
            gpt_config = config_dict["gpt_config"]
            config_dict = {k: v for k, v in config_dict.items() if k != "gpt_config"}
            return cls(gpt_config=gpt_config, **config_dict)
        return cls(**config_dict)