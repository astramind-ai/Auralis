from typing import Literal, Union, AsyncGenerator, List

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
TextStringOrGenerator = Union[AsyncGenerator[str, None], str, List[str]]
SpeakerFiles = Union[Union[str,List[str]], Union[bytes,List[bytes]]]