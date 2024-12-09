import io
import uuid

from dataclasses import dataclass
from pathlib import Path
from typing import Union, AsyncGenerator, Optional, List, Literal, get_args, Callable

import langid
import librosa
import soundfile as sf

import functools
import hashlib
import json
from functools import lru_cache
from dataclasses import asdict, field

from cachetools import LRUCache


def hash_params(*args, **kwargs):
    """Create a hash from function parameters for caching.

    Args:
        *args: Variable positional arguments to hash.
        **kwargs: Variable keyword arguments to hash.

    Returns:
        str: MD5 hash of the stringified parameters.
    """
    params_str = json.dumps([str(arg) for arg in args], sort_keys=True)
    return hashlib.md5(params_str.encode()).hexdigest()


def cached_processing(maxsize=128):
    """Decorator for caching audio processing results.

    Implements an LRU cache for audio processing functions to avoid
    reprocessing the same audio files with the same configuration.

    Args:
        maxsize (int, optional): Maximum size of the LRU cache. Defaults to 128.

    Returns:
        Callable: Decorated function with caching behavior.
    """
    def decorator(func):
        # Create cache storage
        cache = LRUCache(maxsize=maxsize)
        @functools.wraps(func)
        def wrapper(self, audio_path: str, audio_config: AudioPreprocessingConfig, *args, **kwargs):
            # Create hash from the two parameters we care about
            params_dict = {
                'audio_path': audio_path,
                'config': asdict(audio_config)
            }
            cache_key = hash_params(params_dict)

            # Check cache
            if result := cache.get(cache_key):
                return result

            # If not in cache, process and store
            result = func(self, audio_path, audio_config, *args, **kwargs)
            cache.__setitem__(cache_key, result)
            return result

        return wrapper

    return decorator

from auralis.common.definitions.enhancer import EnhancedAudioProcessor, AudioPreprocessingConfig

SupportedLanguages = Literal[
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
        "ja",
        "hi",
        "auto",
        ""
    ]

@lru_cache(maxsize=1024)
def get_language(text: str):
    """Detect the language of input text.

    Uses langid for language detection and handles special cases like
    Chinese (zh-cn).

    Args:
        text (str): Text to detect language for.

    Returns:
        str: Detected language code.
    """
    detected_language =  langid.classify(text)[0].strip()
    if detected_language == "zh":
        # we use zh-cn
        detected_language = "zh-cn"
    return detected_language

def validate_language(language: str) -> SupportedLanguages:
    """Validate that a language code is supported.

    Args:
        language (str): Language code to validate.

    Returns:
        SupportedLanguages: Validated language code.

    Raises:
        ValueError: If the language is not supported.
    """
    supported = get_args(SupportedLanguages)
    if language not in supported:
        raise ValueError(
            f"Language {language} not supported. Must be one of {supported}"
        )
    return language # type: ignore

@dataclass
class TTSRequest:
    """Container for TTS inference request data.
    
    This class encapsulates all parameters needed for text-to-speech synthesis,
    including text input, speaker reference files, and generation parameters.
    It also handles audio preprocessing and language detection.

    Attributes:
        text (Union[AsyncGenerator[str, None], str, List[str]]): Input text to synthesize.
        speaker_files (Union[Union[str,List[str]], Union[bytes,List[bytes]]]): Reference audio for voice cloning.
        context_partial_function (Optional[Callable]): Optional function for context preparation.
        start_time (Optional[float]): Request start time.
        enhance_speech (bool): Whether to apply speech enhancement.
        audio_config (AudioPreprocessingConfig): Audio preprocessing configuration.
        language (SupportedLanguages): Language code for synthesis.
        request_id (str): Unique request identifier.
        load_sample_rate (int): Sample rate for loading audio files.
        sound_norm_refs (bool): Whether to normalize reference audio.
        max_ref_length (int): Maximum reference audio length in seconds.
        gpt_cond_len (int): Length of GPT conditioning.
        gpt_cond_chunk_len (int): Length of each conditioning chunk.
        stream (bool): Whether to stream the output.
        temperature (float): Sampling temperature.
        top_p (float): Nucleus sampling parameter.
        top_k (int): Top-k sampling parameter.
        repetition_penalty (float): Penalty for token repetition.
        length_penalty (float): Penalty for sequence length.
        do_sample (bool): Whether to use sampling for generation.
    """
    # Request metadata
    text: Union[AsyncGenerator[str, None], str, List[str]]

    speaker_files: Union[Union[str,List[str]], Union[bytes,List[bytes]]]
    context_partial_function: Optional[Callable] = None

    start_time: Optional[float] = None
    enhance_speech: bool = False
    audio_config: AudioPreprocessingConfig = field(default_factory=AudioPreprocessingConfig)
    language: SupportedLanguages = "auto"
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    load_sample_rate: int = 22050
    sound_norm_refs: bool = False

    # Voice conditioning parameters
    max_ref_length: int = 60
    gpt_cond_len: int = 30
    gpt_cond_chunk_len: int = 4

    # Generation parameters
    stream: bool = False
    temperature: float = 0.75
    top_p: float = 0.85
    top_k: int = 50
    repetition_penalty: float = 5.0
    length_penalty: float = 1.0
    do_sample: bool = True

    def __post_init__(self):
        """Initialize request after dataclass creation.
        
        Performs language detection if needed and sets up audio preprocessing.
        """
        if self.language == 'auto' and len(self.text) > 0:
            self.language = get_language(self.text)

        validate_language(self.language)
        self.processor = EnhancedAudioProcessor(self.audio_config)
        if isinstance(self.speaker_files, list) and self.enhance_speech:
            self.speaker_files = [self.preprocess_audio(f, self.audio_config) for f in self.speaker_files]

    def infer_language(self):
        """Infer the language of the input text if not specified.
        
        Updates the language attribute based on text content if set to 'auto'.
        """
        if self.language == 'auto':
            self.language = get_language(self.text)

    @cached_processing()
    def preprocess_audio(self, audio_source: Union[str, bytes], audio_config: AudioPreprocessingConfig) -> str:
        """Preprocess audio files for voice cloning.

        Applies audio enhancement and preprocessing according to the configuration.
        Results are cached to avoid reprocessing the same files.

        Args:
            audio_source (Union[str, bytes]): Path to audio file or audio data.
            audio_config (AudioPreprocessingConfig): Preprocessing configuration.

        Returns:
            str: Path to the processed audio file.

        Note:
            Processed files are stored in /tmp/auralis with unique identifiers.
        """
        try:
            temp_dir = Path("/tmp/auralis")
            temp_dir.mkdir(exist_ok=True)
            if isinstance(audio_source, str):
                audio_source = Path(audio_source)
                audio, sr = librosa.load(audio_source, sr=self.audio_config.sample_rate)
            else:
                audio, sr = librosa.load(io.BytesIO(audio_source), sr=self.audio_config.sample_rate)
            processed = self.processor.process(audio)

            output_path = temp_dir / (f"{hash(audio_source) if isinstance(audio_source, bytes) else audio_source.stem}"
                                      f"{uuid.uuid4().hex}"
                                      f"{'.wav' if isinstance(audio_source, bytes) else audio_source.suffix}")
            sf.write(output_path, processed, sr)
            return str(output_path)

        except Exception as e:
            print(f"Error processing audio: {e}. Using original file.")
            return audio_source

    def copy(self):
        """Create a deep copy of the request.

        Returns:
            TTSRequest: New instance with the same attributes.
        """
        copy_fields = {
            'text': self.text,
            'speaker_files': self.speaker_files,
            'enhance_speech': self.enhance_speech,
            'audio_config': self.audio_config,
            'language': self.language,
            'request_id': self.request_id,
            'load_sample_rate': self.load_sample_rate,
            'sound_norm_refs': self.sound_norm_refs,
            'max_ref_length': self.max_ref_length,
            'gpt_cond_len': self.gpt_cond_len,
            'gpt_cond_chunk_len': self.gpt_cond_chunk_len,
            'stream': self.stream,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'top_k': self.top_k,
            'repetition_penalty': self.repetition_penalty,
            'length_penalty': self.length_penalty,
            'do_sample': self.do_sample
        }

        return TTSRequest(**copy_fields)
