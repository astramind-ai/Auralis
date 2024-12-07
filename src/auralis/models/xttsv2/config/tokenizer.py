import re
from typing import List, Optional, Union, Dict, Any
from functools import cached_property

import pypinyin
import torch
from hangul_romanize import Transliter
from hangul_romanize.rule import academic
from num2words import num2words
from spacy.lang.ar import Arabic
from spacy.lang.en import English
from spacy.lang.es import Spanish
from spacy.lang.ja import Japanese
from spacy.lang.zh import Chinese
from transformers import PreTrainedTokenizerFast, BatchEncoding
from transformers.tokenization_utils_base import TruncationStrategy, PaddingStrategy
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import TemplateProcessing

from auralis.models.xttsv2.components.tts.layers.xtts.zh_num2words import TextNorm as zh_num2words

import cutlet

def get_spacy_lang(lang):
    """Get spaCy language model for text processing.
    
    This function returns the appropriate spaCy language model based on the
    input language code. For languages without specific models, it defaults
    to English which provides basic tokenization capabilities.

    Args:
        lang (str): Language code (e.g., 'zh', 'ja', 'ar', 'es').

    Returns:
        spacy.Language: Initialized spaCy language model.
    """
    if lang == "zh":
        return Chinese()
    elif lang == "ja":
        return Japanese()
    elif lang == "ar":
        return Arabic()
    elif lang == "es":
        return Spanish()
    else:
        # For most languages, English does the job
        return English()


def find_best_split_point(text: str, target_pos: int, window_size: int = 30) -> int:
    """Find the optimal point to split text near a target position.
    
    This function analyzes text around the target position to find the most
    natural break point, considering punctuation marks, whitespace, and other
    language markers. It uses a scoring system that prioritizes different types
    of breaks and their distance from the target position.

    Args:
        text (str): Input text to analyze.
        target_pos (int): Target position around which to find split point.
        window_size (int, optional): Size of text window to analyze. Defaults to 30.

    Returns:
        int: Position of the best split point.

    Notes:
        The function uses a prioritized list of markers:
        1. Strong breaks (periods, exclamation marks, question marks)
        2. Medium breaks (commas, closing brackets)
        3. Weak breaks (spaces, special characters)
        
        Each marker type has a priority score, and the final position is chosen
        based on both the marker priority and proximity to target position.
    """
    # Define split markers by priority
    markers = [
        # Strong breaks (longest pause)
        (r'[.!?؟။။။]+[\s]*', 1.0),  # Periods, exclamation, question (multi-script)
        (r'[\n\r]+\s*[\n\r]+', 1.0),  # Multiple newlines
        (r'[:|;；：；][\s]*', 0.9),  # Colons, semicolons (multi-script)

        # Medium breaks
        (r'[,，،、][\s]*', 0.8),  # Commas (multi-script)
        (r'[)}\]）】』»›》\s]+', 0.7),  # Closing brackets/parentheses
        (r'[-—−]+[\s]*', 0.7),  # Dashes

        # Weak breaks
        (r'\s+[&+=/\s]+\s+', 0.6),  # Special characters with spaces
        (r'[\s]+', 0.5),  # Any whitespace as last resort
    ]

    # Calculate window boundaries
    start = max(0, target_pos - window_size)
    end = min(len(text), target_pos + window_size)
    window = text[start:end]

    best_pos = target_pos
    best_score = 0

    for pattern, priority in markers:
        matches = list(re.finditer(pattern, window))
        for match in matches:
            # Calculate position score based on distance from target
            pos = start + match.end()
            distance = abs(pos - target_pos)
            distance_score = 1 - (distance / (window_size * 2))

            # Combine priority and position scores
            score = priority * distance_score

            if score > best_score:
                best_score = score
                best_pos = pos

    return best_pos


def split_sentence(text: str, lang: str, text_split_length: int = 250) -> List[str]:
    """Split text into natural sentences optimized for TTS.
    
    This function performs intelligent text splitting that considers language
    structure, sentence boundaries, and optimal split points. It uses spaCy
    for initial sentence detection and handles special cases like long
    sentences that exceed the target length.

    !!! note "Language Support"
        The function supports multiple languages through spaCy models:
        
        - English (default fallback)
        - Chinese (zh)
        - Japanese (ja)
        - Arabic (ar)
        - Spanish (es)

    !!! tip "Split Point Selection"
        Split points are chosen based on priority:

        1. Strong breaks (1.0):
           - Periods, exclamation marks, question marks
           - Multiple newlines
           - Colons, semicolons
        
        2. Medium breaks (0.8-0.7):
           - Commas
           - Closing brackets/parentheses
           - Dashes
        
        3. Weak breaks (0.6-0.5):
           - Special characters with spaces
           - Any whitespace

    !!! example "Usage Example"
        ```python
        text = "This is a long sentence. It needs to be split. For TTS processing."
        splits = split_sentence(text, "en", text_split_length=50)
        # Result: ["This is a long sentence.", "It needs to be split.", "For TTS processing."]
        ```

    Args:
        text (str): Input text to split.
        lang (str): Language code for text processing.
        text_split_length (int, optional): Target length for text splits.
            Defaults to 250.

    Returns:
        List[str]: List of text splits optimized for TTS processing.

    See Also:
        - [`find_best_split_point`][auralis.models.xttsv2.config.tokenizer.find_best_split_point]: Split point finder
        - [`get_spacy_lang`][auralis.models.xttsv2.config.tokenizer.get_spacy_lang]: Language model selector
    """
    text = text.strip()
    if len(text) <= text_split_length:
        return [text]

    nlp = get_spacy_lang(lang)
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")

    # Get base sentences using spaCy
    doc = nlp(text)
    sentences = list(doc.sents)

    splits = []
    current_split = []
    current_length = 0

    for sent in sentences:
        sentence_text = str(sent).strip()
        sentence_length = len(sentence_text)

        # If sentence fits in current split
        if current_length + sentence_length <= text_split_length:
            current_split.append(sentence_text)
            current_length += sentence_length + 1

        # Handle long sentences
        elif sentence_length > text_split_length:
            # Add current split if exists
            if current_split:
                splits.append(" ".join(current_split))
                current_split = []
                current_length = 0

            # Split long sentence at optimal points
            remaining = sentence_text
            while len(remaining) > text_split_length:
                split_pos = find_best_split_point(
                    remaining,
                    text_split_length,
                    window_size=30
                )

                # Add split and continue with remainder
                splits.append(remaining[:split_pos].strip())
                remaining = remaining[split_pos:].strip()

            # Handle remaining text
            if remaining:
                current_split = [remaining]
                current_length = len(remaining)

        # Start new split
        else:
            splits.append(" ".join(current_split))
            current_split = [sentence_text]
            current_length = sentence_length

    # Add final split if needed
    if current_split:
        splits.append(" ".join(current_split))

    cleaned_sentences = [s[:-1]+' ' if s.endswith('.') else s for s in splits if s] # prevents annoying sounds in italian
    # Clean up splits
    return cleaned_sentences

_whitespace_re = re.compile(r"\s+")

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = {
    "en": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("mrs", "misess"),
            ("mr", "mister"),
            ("dr", "doctor"),
            ("st", "saint"),
            ("co", "company"),
            ("jr", "junior"),
            ("maj", "major"),
            ("gen", "general"),
            ("drs", "doctors"),
            ("rev", "reverend"),
            ("lt", "lieutenant"),
            ("hon", "honorable"),
            ("sgt", "sergeant"),
            ("capt", "captain"),
            ("esq", "esquire"),
            ("ltd", "limited"),
            ("col", "colonel"),
            ("ft", "fort"),
        ]
    ],
    "es": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("sra", "señora"),
            ("sr", "señor"),
            ("dr", "doctor"),
            ("dra", "doctora"),
            ("st", "santo"),
            ("co", "compañía"),
            ("jr", "junior"),
            ("ltd", "limitada"),
        ]
    ],
    "fr": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("mme", "madame"),
            ("mr", "monsieur"),
            ("dr", "docteur"),
            ("st", "saint"),
            ("co", "compagnie"),
            ("jr", "junior"),
            ("ltd", "limitée"),
        ]
    ],
    "de": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("fr", "frau"),
            ("dr", "doktor"),
            ("st", "sankt"),
            ("co", "firma"),
            ("jr", "junior"),
        ]
    ],
    "pt": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("sra", "senhora"),
            ("sr", "senhor"),
            ("dr", "doutor"),
            ("dra", "doutora"),
            ("st", "santo"),
            ("co", "companhia"),
            ("jr", "júnior"),
            ("ltd", "limitada"),
        ]
    ],
    "it": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            # ("sig.ra", "signora"),
            ("sig", "signore"),
            ("dr", "dottore"),
            ("st", "santo"),
            ("co", "compagnia"),
            ("jr", "junior"),
            ("ltd", "limitata"),
        ]
    ],
    "pl": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("p", "pani"),
            ("m", "pan"),
            ("dr", "doktor"),
            ("sw", "święty"),
            ("jr", "junior"),
        ]
    ],
    "ar": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            # There are not many common abbreviations in Arabic as in English.
        ]
    ],
    "zh": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            # Chinese doesn't typically use abbreviations in the same way as Latin-based scripts.
        ]
    ],
    "cs": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("dr", "doktor"),  # doctor
            ("ing", "inženýr"),  # engineer
            ("p", "pan"),  # Could also map to pani for woman but no easy way to do it
            # Other abbreviations would be specialized and not as common.
        ]
    ],
    "ru": [
        (re.compile("\\b%s\\b" % x[0], re.IGNORECASE), x[1])
        for x in [
            ("г-жа", "госпожа"),  # Mrs.
            ("г-н", "господин"),  # Mr.
            ("д-р", "доктор"),  # doctor
            # Other abbreviations are less common or specialized.
        ]
    ],
    "nl": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("dhr", "de heer"),  # Mr.
            ("mevr", "mevrouw"),  # Mrs.
            ("dr", "dokter"),  # doctor
            ("jhr", "jonkheer"),  # young lord or nobleman
            # Dutch uses more abbreviations, but these are the most common ones.
        ]
    ],
    "tr": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("b", "bay"),  # Mr.
            ("byk", "büyük"),  # büyük
            ("dr", "doktor"),  # doctor
            # Add other Turkish abbreviations here if needed.
        ]
    ],
    "hu": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("dr", "doktor"),  # doctor
            ("b", "bácsi"),  # Mr.
            ("nőv", "nővér"),  # nurse
            # Add other Hungarian abbreviations here if needed.
        ]
    ],
    "ko": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            # Korean doesn't typically use abbreviations in the same way as Latin-based scripts.
        ]
    ],
}

def expand_abbreviations_multilingual(text, lang="en"):
    if lang in _abbreviations:
        for regex, replacement in _abbreviations[lang]:
            text = re.sub(regex, replacement, text)
    return text

_symbols_multilingual = {
    "en": [
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " and "),
            ("@", " at "),
            ("%", " percent "),
            ("#", " hash "),
            ("$", " dollar "),
            ("£", " pound "),
            ("°", " degree "),
        ]
    ],
    "es": [
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " y "),
            ("@", " arroba "),
            ("%", " por ciento "),
            ("#", " numeral "),
            ("$", " dolar "),
            ("£", " libra "),
            ("°", " grados "),
        ]
    ],
    "fr": [
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " et "),
            ("@", " arobase "),
            ("%", " pour cent "),
            ("#", " dièse "),
            ("$", " dollar "),
            ("£", " livre "),
            ("°", " degrés "),
        ]
    ],
    "de": [
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " und "),
            ("@", " at "),
            ("%", " prozent "),
            ("#", " raute "),
            ("$", " dollar "),
            ("£", " pfund "),
            ("°", " grad "),
        ]
    ],
    "pt": [
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " e "),
            ("@", " arroba "),
            ("%", " por cento "),
            ("#", " cardinal "),
            ("$", " dólar "),
            ("£", " libra "),
            ("°", " graus "),
        ]
    ],
    "it": [
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " e "),
            ("@", " chiocciola "),
            ("%", " per cento "),
            ("#", " cancelletto "),
            ("$", " dollaro "),
            ("£", " sterlina "),
            ("°", " gradi "),
        ]
    ],
    "pl": [
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " i "),
            ("@", " małpa "),
            ("%", " procent "),
            ("#", " krzyżyk "),
            ("$", " dolar "),
            ("£", " funt "),
            ("°", " stopnie "),
        ]
    ],
    "ar": [
        # Arabic
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " و "),
            ("@", " على "),
            ("%", " في المئة "),
            ("#", " رقم "),
            ("$", " دولار "),
            ("£", " جنيه "),
            ("°", " درجة "),
        ]
    ],
    "zh": [
        # Chinese
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " 和 "),
            ("@", " 在 "),
            ("%", " 百分之 "),
            ("#", " 号 "),
            ("$", " 美元 "),
            ("£", " 英镑 "),
            ("°", " 度 "),
        ]
    ],
    "cs": [
        # Czech
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " a "),
            ("@", " na "),
            ("%", " procento "),
            ("#", " křížek "),
            ("$", " dolar "),
            ("£", " libra "),
            ("°", " stupně "),
        ]
    ],
    "ru": [
        # Russian
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " и "),
            ("@", " собака "),
            ("%", " процентов "),
            ("#", " номер "),
            ("$", " доллар "),
            ("£", " фунт "),
            ("°", " градус "),
        ]
    ],
    "nl": [
        # Dutch
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " en "),
            ("@", " bij "),
            ("%", " procent "),
            ("#", " hekje "),
            ("$", " dollar "),
            ("£", " pond "),
            ("°", " graden "),
        ]
    ],
    "tr": [
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " ve "),
            ("@", " at "),
            ("%", " yüzde "),
            ("#", " diyez "),
            ("$", " dolar "),
            ("£", " sterlin "),
            ("°", " derece "),
        ]
    ],
    "hu": [
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " és "),
            ("@", " kukac "),
            ("%", " százalék "),
            ("#", " kettőskereszt "),
            ("$", " dollár "),
            ("£", " font "),
            ("°", " fok "),
        ]
    ],
    "ko": [
        # Korean
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " 그리고 "),
            ("@", " 에 "),
            ("%", " 퍼센트 "),
            ("#", " 번호 "),
            ("$", " 달러 "),
            ("£", " 파운드 "),
            ("°", " 도 "),
        ]
    ],
}

def expand_symbols_multilingual(text, lang="en"):
    if lang in _symbols_multilingual:
        for regex, replacement in _symbols_multilingual[lang]:
            text = re.sub(regex, replacement, text)
            text = text.replace("  ", " ")  # Ensure there are no double spaces
    return text.strip()

_ordinal_re = {
    "en": re.compile(r"([0-9]+)(st|nd|rd|th)"),
    "es": re.compile(r"([0-9]+)(º|ª|er|o|a|os|as)"),
    "fr": re.compile(r"([0-9]+)(º|ª|er|re|e|ème)"),
    "de": re.compile(r"([0-9]+)(st|nd|rd|th|º|ª|\.(?=\s|$))"),
    "pt": re.compile(r"([0-9]+)(º|ª|o|a|os|as)"),
    "it": re.compile(r"([0-9]+)(º|°|ª|o|a|i|e)"),
    "pl": re.compile(r"([0-9]+)(º|ª|st|nd|rd|th)"),
    "ar": re.compile(r"([0-9]+)(ون|ين|ث|ر|ى)"),
    "cs": re.compile(r"([0-9]+)\.(?=\s|$)"),  # In Czech, a dot is often used after the number to indicate ordinals.
    "ru": re.compile(r"([0-9]+)(-й|-я|-е|-ое|-ье|-го)"),
    "nl": re.compile(r"([0-9]+)(de|ste|e)"),
    "tr": re.compile(r"([0-9]+)(\.|inci|nci|uncu|üncü|\.)"),
    "hu": re.compile(r"([0-9]+)(\.|adik|edik|odik|edik|ödik|ödike|ik)"),
    "ko": re.compile(r"([0-9]+)(번째|번|차|째)"),
}
_number_re = re.compile(r"[0-9]+")
# noinspection Annotator
_currency_re = {
    "USD": re.compile(r"((\$[0-9\.\,]*[0-9]+)|([0-9\.\,]*[0-9]+\$))"),
    "GBP": re.compile(r"((£[0-9\.\,]*[0-9]+)|([0-9\.\,]*[0-9]+£))"),
    "EUR": re.compile(r"(([0-9\.\,]*[0-9]+€)|((€[0-9\.\,]*[0-9]+)))"),
}

_comma_number_re = re.compile(r"\b\d{1,3}(,\d{3})*(\.\d+)?\b")
_dot_number_re = re.compile(r"\b\d{1,3}(\.\d{3})*(\,\d+)?\b")
_decimal_number_re = re.compile(r"([0-9]+[.,][0-9]+)")

def _remove_commas(m):
    text = m.group(0)
    if "," in text:
        text = text.replace(",", "")
    return text

def _remove_dots(m):
    text = m.group(0)
    if "." in text:
        text = text.replace(".", "")
    return text

def _expand_decimal_point(m, lang="en"):
    amount = m.group(1).replace(",", ".")
    return num2words(float(amount), lang=lang if lang != "cs" else "cz")

def _expand_currency(m, lang="en", currency="USD"):
    amount = float((re.sub(r"[^\d.]", "", m.group(0).replace(",", "."))))
    full_amount = num2words(amount, to="currency", currency=currency, lang=lang if lang != "cs" else "cz")

    and_equivalents = {
        "en": ", ",
        "es": " con ",
        "fr": " et ",
        "de": " und ",
        "pt": " e ",
        "it": " e ",
        "pl": ", ",
        "cs": ", ",
        "ru": ", ",
        "nl": ", ",
        "ar": ", ",
        "tr": ", ",
        "hu": ", ",
        "ko": ", ",
    }

    if amount.is_integer():
        last_and = full_amount.rfind(and_equivalents.get(lang, ", "))
        if last_and != -1:
            full_amount = full_amount[:last_and]

    return full_amount

def _expand_ordinal(m, lang="en"):
    return num2words(int(m.group(1)), ordinal=True, lang=lang if lang != "cs" else "cz")

def _expand_number(m, lang="en"):
    return num2words(int(m.group(0)), lang=lang if lang != "cs" else "cz")

def expand_numbers_multilingual(text, lang="en"):
    if lang == "zh":
        text = zh_num2words()(text)
    else:
        if lang in ["en", "ru"]:
            text = re.sub(_comma_number_re, _remove_commas, text)
        else:
            text = re.sub(_dot_number_re, _remove_dots, text)
        try:
            text = re.sub(_currency_re["GBP"], lambda m: _expand_currency(m, lang, "GBP"), text)
            text = re.sub(_currency_re["USD"], lambda m: _expand_currency(m, lang, "USD"), text)
            text = re.sub(_currency_re["EUR"], lambda m: _expand_currency(m, lang, "EUR"), text)
        except Exception as e:
            pass
        if lang != "tr":
            text = re.sub(_decimal_number_re, lambda m: _expand_decimal_point(m, lang), text)
        if lang in _ordinal_re:
            text = re.sub(_ordinal_re[lang], lambda m: _expand_ordinal(m, lang), text)
        text = re.sub(_number_re, lambda m: _expand_number(m, lang), text)
    return text

def lowercase(text):
    return text.lower()

def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text)

def multilingual_cleaners(text, lang):
    text = text.replace('"', "")
    if lang == "tr":
        text = text.replace("İ", "i")
        text = text.replace("Ö", "ö")
        text = text.replace("Ü", "ü")
    text = lowercase(text)
    text = expand_numbers_multilingual(text, lang)
    text = expand_abbreviations_multilingual(text, lang)
    text = expand_symbols_multilingual(text, lang=lang)
    text = collapse_whitespace(text)
    return text

def basic_cleaners(text):
    """Basic pipeline that lowercases and collapses whitespace without transliteration."""
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text

def chinese_transliterate(text):
    return "".join(
        [p[0] for p in pypinyin.pinyin(text, style=pypinyin.Style.TONE3, heteronym=False, neutral_tone_with_five=True)]
    )

def japanese_cleaners(text, katsu):
    text = katsu.romaji(text)
    text = lowercase(text)
    return text

def korean_transliterate(text, transliter):
    return transliter.translit(text)

# Fast Tokenizer Class

class XTTSTokenizerFast(PreTrainedTokenizerFast):
    """
    Fast Tokenizer implementation for XTTS model using HuggingFace's PreTrainedTokenizerFast
    """

    def __init__(
            self,
            vocab_file: str = None,
            tokenizer_object: Optional[Tokenizer] = None,
            unk_token: str = "[UNK]",
            pad_token: str = "[PAD]",
            bos_token: str = "[START]",
            eos_token: str = "[STOP]",
            auto_map: dict = {"AutoTokenizer": ["AstraMindAI/xtts2-gpt--tokenizer.XTTSTokenizerFast", None]},
            clean_up_tokenization_spaces: bool = True,
            **kwargs
    ):
        if tokenizer_object is None and vocab_file is not None:
            tokenizer_object = Tokenizer.from_file(vocab_file)

        if tokenizer_object is not None:
            # Configure the tokenizer
            tokenizer_object.pre_tokenizer = WhitespaceSplit()
            tokenizer_object.post_processor = TemplateProcessing(
                single=f"{bos_token} $A {eos_token}",
                special_tokens=[
                    (bos_token, tokenizer_object.token_to_id(bos_token)),
                    (eos_token, tokenizer_object.token_to_id(eos_token)),
                ],
            )

        super().__init__(
            tokenizer_object=tokenizer_object,
            unk_token=unk_token,
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs
        )

        # Character limits per language
        self.char_limits = {
            "en": 250, "de": 253, "fr": 273, "es": 239,
            "it": 213, "pt": 203, "pl": 224, "zh": 82,
            "ar": 166, "cs": 186, "ru": 182, "nl": 251,
            "tr": 226, "ja": 71, "hu": 224, "ko": 95,
        }

        # Initialize language tools
        self._katsu = None
        self._korean_transliter = Transliter(academic)

        # Ensure pad_token_id is set
        if self.pad_token_id is None:
            self.pad_token_id = self.tokenizer.token_to_id(self.pad_token)

    @cached_property
    def katsu(self):
        if self._katsu is None:
            self._katsu = cutlet.Cutlet()
        return self._katsu

    def preprocess_text(self, text: str, lang: str) -> str:
        """Apply text preprocessing for language"""
        base_lang = lang.split("-")[0]  # remove region
        if base_lang in {"ar", "cs", "de", "en", "es", "fr", "hu", "it",
                         "nl", "pl", "pt", "ru", "tr", "zh", "ko"}:
            text = multilingual_cleaners(text, base_lang)
            if base_lang == "zh":
                text = chinese_transliterate(text)
            if base_lang == "ko":
                text = korean_transliterate(text, self._korean_transliter)
        elif base_lang == "ja":
            text = japanese_cleaners(text, self.katsu)
        else:
            text = basic_cleaners(text)
        return text

    def batch_encode_with_split(self, texts: Union[str, List[str]], lang: Union[str, List[str]],
                                **kwargs) -> torch.Tensor:
        """
        Split texts into smaller chunks based on language character limits and encode them using HuggingFace fast tokenizer.
        strictly mimic the xttsv2 tokenizer
        """
        # Convert single inputs to lists
        if isinstance(texts, str):
            texts = [texts]
        if isinstance(lang, str):
            lang = [lang]
        # Ensure lang list matches texts list
        if len(lang) == 1 and len(texts) > 1:
            lang = lang * len(texts)

        # Check if texts and lang have the same length
        if len(texts) != len(lang):
            raise ValueError(f"Number of texts ({len(texts)}) does not match number of languages ({len(lang)}).")

        chunk_list = []
        max_splits = 0

        # For each text, split into chunks based on character limit
        for text, text_lang in zip(texts, lang):
            # Get language character limit
            base_lang = text_lang.split("-")[0]
            char_limit = self.char_limits.get(base_lang, 250)

            # Clean and preprocess
            #text = self.preprocess_text(text, text_lang) we do this in the hidden function

            # Split text into sentences/chunks based on language
            chunk_list = split_sentence(text, base_lang, text_split_length=char_limit)

        # Ensure the tokenizer is a fast tokenizer
        if not self.is_fast:
            raise ValueError("The tokenizer must be a fast tokenizer.")

        # Encode all chunks using the fast tokenizer
        encoding: BatchEncoding = self(
            chunk_list,
            lang = lang,
            add_special_tokens=False,
            padding=False,
            **kwargs
        )

        # The 'input_ids' tensor will have shape [total_chunks, max_sequence_length]
        return encoding['input_ids']  # Tensor of shape [total_chunks, sequence_length]

    def _batch_encode_plus(
            self,
            batch_text_or_text_pairs,
            add_special_tokens: bool = True,
            padding_strategy=PaddingStrategy.DO_NOT_PAD,
            truncation_strategy=TruncationStrategy.DO_NOT_TRUNCATE,
            max_length: Optional[int] = None,
            stride: int = 0,
            is_split_into_words: bool = False,
            pad_to_multiple_of: Optional[int] = None,
            return_tensors: Optional[str] = None,
            return_token_type_ids: Optional[bool] = None,
            return_attention_mask: Optional[bool] = None,
            return_overflowing_tokens: bool = False,
            return_special_tokens_mask: bool = False,
            return_offsets_mapping: bool = False,
            return_length: bool = False,
            verbose: bool = True,
            **kwargs
    ) -> Dict[str, Any]:
        """
        Override batch encoding to handle language-specific preprocessing
        """
        lang = kwargs.pop("lang", ["en"] * len(batch_text_or_text_pairs))
        if isinstance(lang, str):
            lang = [lang]
        # Ensure lang list matches texts list
        if len(lang) == 1 and len(batch_text_or_text_pairs) > 1:
            lang = lang * len(batch_text_or_text_pairs)

        # Check if batch_text_or_text_pairs and lang have the same length
        if len(batch_text_or_text_pairs) != len(lang):
            raise ValueError(f"Number of texts ({len(batch_text_or_text_pairs)}) does not match number of languages ({len(lang)}).")

        # Preprocess each text in the batch with its corresponding language
        processed_texts = []
        for text, text_lang in zip(batch_text_or_text_pairs, lang):
            if isinstance(text, str):
                # Check length and preprocess
                #self.check_input_length(text, text_lang)
                processed_text = self.preprocess_text(text, text_lang)

                # Format text with language tag and spaces
                base_lang = text_lang.split("-")[0]
                lang_code = "zh-cn" if base_lang == "zh" else base_lang
                processed_text = f"[{lang_code}]{processed_text}"
                processed_text = processed_text.replace(" ", "[SPACE]")

                processed_texts.append(processed_text)
            else:
                processed_texts.append(text)

        # Call the parent class's encoding method with processed texts
        return super()._batch_encode_plus(
            processed_texts,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            is_split_into_words=is_split_into_words,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            **kwargs
        )


    def __call__(
            self,
            text: Union[str, List[str]],
            lang: Union[str, List[str]] = "en",
            add_special_tokens: bool = True,
            padding: Union[bool, str, PaddingStrategy] = False,
            truncation: Union[bool, str, TruncationStrategy] = False,
            max_length: Optional[int] = None,
            stride: int = 0,
            return_tensors: Optional[str] = None,
            return_token_type_ids: Optional[bool] = None,
            return_attention_mask: Optional[bool] = True,
            **kwargs
    ):
        """
        Main tokenization method
        """
        # Convert single string to list for batch processing
        if isinstance(text, str):
            text = [text]
        if isinstance(lang, str):
            lang = [lang]
        # Ensure lang list matches texts list
        if len(lang) == 1 and len(text) > 1:
            lang = lang * len(text)

        # Ensure text and lang lists have same length
        if len(text) != len(lang):
            raise ValueError(f"Number of texts ({len(text)}) does not match number of languages ({len(lang)}).")

        # Convert padding strategy
        if isinstance(padding, bool):
            padding_strategy = PaddingStrategy.LONGEST if padding else PaddingStrategy.DO_NOT_PAD
        else:
            padding_strategy = PaddingStrategy(padding)

        # Convert truncation strategy
        if isinstance(truncation, bool):
            truncation_strategy = TruncationStrategy.LONGEST_FIRST if truncation else TruncationStrategy.DO_NOT_TRUNCATE
        else:
            truncation_strategy = TruncationStrategy(truncation)

        # Use the batch encoding method
        encoded = self._batch_encode_plus(
            text,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            return_tensors=return_tensors,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            lang=lang,
            **kwargs
        )

        return encoded
