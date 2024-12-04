#  Copyright (c) 2024 Astramind. Licensed under Apache License, Version 2.0.

import base64
from dataclasses import fields
from typing import List, Optional, Dict, Any, Literal

from pydantic import BaseModel, Field, field_validator

from auralis.common.definitions.dto.requests import TTSRequest


class ChatCompletionMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

tts_defaults = {field.name: field.default for field in fields(TTSRequest)}

class VoiceChatCompletionRequest(BaseModel):
    # Chat completion fields
    model: str
    messages: List[ChatCompletionMessage]
    speaker_files: List[str] = Field(..., description="List of base64-encoded audio files")
    modalities: List[Literal["text", "audio"]] = Field(
        default=["text", "audio"],
        description="Output modalities to return"
    )
    openai_api_url: Optional[str] = Field(
        default=None,
        description="Custom OpenAI API endpoint to make the LLM reqeust to"
    )
    vocalize_at_every_n_words: int = Field(
        default=100,
        ge=1,
        description="Number of words after which to generate audio"
    )
    stream: bool = Field(default=True)

    # TTSRequest parameters usando i defaults dalla dataclass
    enhance_speech: bool = Field(default=tts_defaults['enhance_speech'])
    language: str = Field(default=tts_defaults['language'])
    max_ref_length: int = Field(default=tts_defaults['max_ref_length'])
    gpt_cond_len: int = Field(default=tts_defaults['gpt_cond_len'])
    gpt_cond_chunk_len: int = Field(default=tts_defaults['gpt_cond_chunk_len'])
    temperature: float = Field(default=tts_defaults['temperature'])
    top_p: float = Field(default=tts_defaults['top_p'])
    top_k: int = Field(default=tts_defaults['top_k'])
    repetition_penalty: float = Field(default=tts_defaults['repetition_penalty'])
    length_penalty: float = Field(default=tts_defaults['length_penalty'])
    do_sample: bool = Field(default=tts_defaults['do_sample'])

    @field_validator('openai_api_url')
    def validate_oai_url(cls, v):
        if v is None:
            raise ValueError("You should always give a url for the text generation")
        return v

    @field_validator('stream')
    def validate_stream(cls, v):
        if not v:
            raise ValueError('Streaming should be enabled! For non-streaming conversion use the audio endpoint')
        return v

    @field_validator('speaker_files')
    def validate_speaker_files(cls, v):
        if not v:
            raise ValueError("At least one speaker file is required")
        for file in v:
            try:
                base64.b64decode(file)
            except Exception:
                raise ValueError(f"Invalid base64 encoding in speaker file")
        return v

    @field_validator('modalities')
    def validate_modalities(cls, v):
        valid_modalities = ["text", "audio"]
        if not all(m in valid_modalities for m in v):
            raise ValueError(f"Invalid modalities. Must be one or more of {valid_modalities}")
        return v

    def to_tts_request(self, text: str = "") -> TTSRequest:
        """Convert to TTSRequest with decoded speaker files"""
        speaker_data_list = [base64.b64decode(f) for f in self.speaker_files]

        return TTSRequest(
            text=text,
            stream=False,
            speaker_files=speaker_data_list,
            enhance_speech=self.enhance_speech,
            language=self.language,
            max_ref_length=self.max_ref_length,
            gpt_cond_len=self.gpt_cond_len,
            gpt_cond_chunk_len=self.gpt_cond_chunk_len,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            length_penalty=self.length_penalty,
            do_sample=self.do_sample
        )

    def to_openai_request(self) -> Dict[str, Any]:
        """Convert to OpenAI API compatible request format"""
        oai_dict = {
            k: v for k, v in self.model_dump().items()
            if k not in ["speaker_files", "openai_api_url", "vocalize_at_every_n_words", 'modalities'] and
               not k in tts_defaults.keys()
        }
        oai_dict.update({"stream": True})
        return oai_dict


class AudioSpeechGenerationRequest(BaseModel):
        # Chat completion fields
        input: str = Field(..., description="The textual input to convert")
        model: str = Field(..., description="The model to use for conversion")
        voice: List[str] = Field(..., description="List of base64-encoded audio files")
        response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(
            default='mp3', description="List of base64-encoded audio files"
        )
        speed: float = Field(default=1.0, description="List of base64-encoded audio files"),

        # TTSRequest parameters
        enhance_speech: bool = Field(default=tts_defaults['enhance_speech'])
        language: str = Field(default=tts_defaults['language'])
        max_ref_length: int = Field(default=tts_defaults['max_ref_length'])
        gpt_cond_len: int = Field(default=tts_defaults['gpt_cond_len'])
        gpt_cond_chunk_len: int = Field(default=tts_defaults['gpt_cond_chunk_len'])
        temperature: float = Field(default=tts_defaults['temperature'])
        top_p: float = Field(default=tts_defaults['top_p'])
        top_k: int = Field(default=tts_defaults['top_k'])
        repetition_penalty: float = Field(default=tts_defaults['repetition_penalty'])
        length_penalty: float = Field(default=tts_defaults['length_penalty'])
        do_sample: bool = Field(default=tts_defaults['do_sample'])

        @field_validator('voice')
        def validate_speaker_files(cls, v):
            if not v:
                raise ValueError("At least one voice file is required")
            for file in v:
                try:
                    base64.b64decode(file)
                except Exception:
                    raise ValueError(f"Invalid base64 encoding in voice file")
            return v

        def to_tts_request(self) -> TTSRequest:
            """Convert to TTSRequest with decoded speaker files"""
            speaker_data_list = [base64.b64decode(f) for f in self.voice]

            return TTSRequest(
                text=self.input,
                stream=False,
                speaker_files=speaker_data_list,
                enhance_speech=self.enhance_speech,
                language=self.language,
                max_ref_length=self.max_ref_length,
                gpt_cond_len=self.gpt_cond_len,
                gpt_cond_chunk_len=self.gpt_cond_chunk_len,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                repetition_penalty=self.repetition_penalty,
                length_penalty=self.length_penalty,
                do_sample=self.do_sample
            )
