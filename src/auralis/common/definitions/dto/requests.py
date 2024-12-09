#  Copyright (c) 2024 Astramind. Licensed under Apache License, Version 2.0.

import functools
import hashlib
import io
import json
import uuid
from dataclasses import asdict, field
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Union, AsyncGenerator, Optional, List, Literal, get_args, Callable, Dict

import langid
import librosa
import soundfile as sf
from cachetools import LRUCache

from auralis.common.definitions.batch.batchable_item import BatchableItem
from auralis.common.logging.logger import setup_logger
from auralis.common.definitions.enhancer import EnhancedAudioProcessor, AudioPreprocessingConfig

logger = setup_logger(__name__)

def hash_params(*args, **kwargs):
    """
    Convert args and kwargs to a JSON string and hash it.

    Parameters
    ----------
    *args : tuple
        Positional arguments to be hashed
    **kwargs : dict
        Keyword arguments to be hashed

    Returns
    -------
    str
        The hash string
    """
    params_str = json.dumps([str(arg) for arg in args], sort_keys=True)
    return hashlib.md5(params_str.encode()).hexdigest()



def cached_processing(maxsize=128):

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
    """
    Detect the language of a given text using langid.

    Args:
        text (str): The text to detect the language from.

    Returns:
        str: The detected language as an ISO 639-1 language code.

    Notes:
        Langid is used to detect the language. If the detected language is Chinese
        ("zh"), it is replaced with "zh-cn" since we use Mandarin Chinese as our
        Chinese language variant.

    """
    detected_language =  langid.classify(text)[0].strip()
    if detected_language == "zh":
        # we use zh-cn
        detected_language = "zh-cn"
    return detected_language

def validate_language(language: str) -> SupportedLanguages:
    """
    Validate that the provided language is supported.

    Args:
        language (str): The language code to validate.

    Returns:
        SupportedLanguages: The validated language code.

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
    """
    Data class representing a Text-to-Speech (TTS) request.

    Attributes:
        text (Union[AsyncGenerator[str, None], str, List[str]]): The text or texts to be converted into speech.
        speaker_files (Union[Union[str, List[str]], Union[bytes, List[bytes]]]): The speaker audio files or data.
        context_partial_function (Optional[Callable]): A partial function for additional context processing.
        start_time (Optional[float]): The start time for the TTS operation.
        enhance_speech (bool): Flag to indicate if speech enhancement should be applied.
        audio_config (AudioPreprocessingConfig): Configuration for audio preprocessing.
        language (SupportedLanguages): The language of the text, defaults to 'auto' for automatic detection.
        request_id (str): Unique identifier for the request.
        load_sample_rate (int): The sample rate for loading audio files.
        sound_norm_refs (bool): Flag to indicate if sound normalization references should be used.
        max_ref_length (int): Maximum reference length for voice conditioning.
        gpt_cond_len (int): Length of GPT conditioning.
        gpt_cond_chunk_len (int): Chunk length for GPT conditioning.
        stream (bool): Flag to indicate if the output should be streamed.
        temperature (float): Sampling temperature for generation.
        top_p (float): Top-p sampling parameter for generation.
        top_k (int): Top-k sampling parameter for generation.
    """
    # Request metadata
    text: Union[AsyncGenerator[str, None], str, List[str]]

    speaker_files: Union[Union[str,List[str]], Union[bytes,List[bytes]]] = None
    context_partial_function: Optional[Callable] = None

    start_time: Optional[float] = None
    split_text: bool = True
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


    def _validation_for_model(self):
         pass
        # for now just a placeholder, but it'll check form the model registry some predefined values

    def __post_init__(self):

        if self.language == 'auto' and len(self.text) > 0:
            self.language = get_language(self.text)

        validate_language(self.language)
        self.processor = EnhancedAudioProcessor(self.audio_config)

        if isinstance(self.speaker_files, list) and self.enhance_speech:
            if len(self.speaker_files) > 5:
                logger.warning(f"You provided alist of {len(self.speaker_files)} speaker files "
                               f"but only 5 are supported. we'll take the first 5.")
                self.speaker_files = self.speaker_files[:5]  # FIXME(mlinmg): for xttsv2 might need adjustments later

            self.speaker_files = [self.preprocess_audio(f, self.audio_config) for f in self.speaker_files]

        if self.max_ref_length > 60:
            logger.warning(f"Maximum reference length is set to {self.max_ref_length}. "
                           f"We hard limit this to 60 seconds.") # FIXME(mlinmg): for xttsv2 might need adjustments later
            self.max_ref_length = 60


    def infer_language(self):
        """
        Infer the language of the text if it is set to 'auto'.

        If the language is set to 'auto', this method will infer the language of the
        text using the `langid` library. The inferred language is then stored in the
        `language` attribute.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self.language == 'auto':
            self.language = get_language(self.text)

    @cached_processing()
    def preprocess_audio(self, audio_source: Union[str, bytes], audio_config: AudioPreprocessingConfig) -> str:
        """
        Preprocesses an audio source (either a file path or a bytes object).

        The audio is processed using the `processor` attribute, which is an instance of
        `EnhancedAudioProcessor` with the `audio_config` parameter. The output is saved to
        a temporary file in `/tmp/auralis` and the path to the file is returned.

        If an error occurs during processing, the original file is returned.

        Parameters
        ----------
        audio_source : Union[str, bytes]
            The audio source to preprocess.
        audio_config : AudioPreprocessingConfig
            The configuration for the audio preprocessing.

        Returns
        -------
        str
            The path to the preprocessed audio file.
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

        """
        Creates a shallow copy of the current request.

        Returns:
            A new instance of TTSRequest with the same fields as the current instance.
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
