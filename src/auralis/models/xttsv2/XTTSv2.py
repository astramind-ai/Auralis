#  Copyright (c) 2024 Astramind. Licensed under Apache License, Version 2.0.

import asyncio
import functools
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, List, Tuple, Union, AsyncGenerator

import librosa
import numpy as np
import torch
import torchaudio
from torch import nn
from vllm import AsyncLLMEngine, AsyncEngineArgs, TokensPrompt
from vllm.multimodal import MultiModalDataDict
from vllm.sampling_params import RequestOutputKind
from vllm.utils import Counter

from .components.tts.layers.xtts.hifigan_decoder import HifiDecoder
from .components.tts.layers.xtts.latent_encoder import ConditioningEncoder
from .components.tts.layers.xtts.perceiver_encoder import PerceiverResampler
from .components.vllm_mm_gpt import LearnedPositionEmbeddings
from .config.tokenizer import XTTSTokenizerFast
from .config.xttsv2_config import XTTSConfig
from .config.xttsv2_gpt_config import XTTSGPTConfig
from ..base import BaseAsyncTTSEngine
from ..registry import register_tts_model, SupportedModelTypes
from ...common.definitions.dto.output import TTSOutput
from ...common.definitions.dto.requests import TTSRequest
from ...common.definitions.scheduler.context import GenerationContext
from ...common.logging.logger import setup_logger
from ...common.utilities import wav_to_mel_cloning, load_audio
from ...common.vllm.hidden_state_collector import HiddenStatesCollector
from ...common.vllm.hijack import ExtendedSamplingParams, LogitsRepetitionPenalizer


@register_tts_model(
    model_type=SupportedModelTypes.XTTSv2,
    uses_vllm=True,
    supported_languages = ['en', 'es', 'fr', 'de', 'it', 'pt', 'pl', 'tr', 'ru',
                           'nl', 'cs', 'ar', 'zh-cn', 'ja', 'hu', 'ko', 'hi'],
    context_generation_max_length = ()


)
class XTTSv2Engine(BaseAsyncTTSEngine):
    """Async XTTS model implementation using VLLM's AsyncEngine."""

    model_type: str = "xtts"
    uses_vllm = True

    def __init__(self,
                 hifi_config: XTTSConfig,
                 gpt_config: XTTSGPTConfig,
                 pipeline_parallel_size: int = 1,
                 tensor_parallel_size: int = 1,
                 **kwargs):
        super().__init__()

        self.max_gb_for_vllm_model = None
        self.logger = setup_logger(__file__)
        self.logger.info("Initializing XTTSv2Engine...")

        self.gpt_model = kwargs.pop('gpt_model')
        self.hifi_config = hifi_config
        self.gpt_config = gpt_config
        self.mel_bos_token_id = gpt_config.start_audio_token
        self.mel_eos_token_id = gpt_config.stop_audio_token
        self.tp = tensor_parallel_size
        self.pp = pipeline_parallel_size
        self.tokenizer = XTTSTokenizerFast.from_pretrained(self.gpt_model)
        self.request_counter = Counter()

        self.max_concurrency = kwargs.pop('max_concurrency', 10)
        semaphore_concurrency = max(1,self.max_concurrency // 6) * self.tp
        self.executor = ThreadPoolExecutor(max_workers=semaphore_concurrency)  # For CPU-bound tasks

        # Register buffer before creating modules
        self.register_buffer("mel_stats", torch.ones(80))

        # Initialize all nn.Module components
        self.conditioning_encoder = ConditioningEncoder(
            gpt_config.audio_config.mel_channels,
            gpt_config.hidden_size,
            num_attn_heads=gpt_config.num_attention_heads
        )

        self.text_embedding = nn.Embedding(
            gpt_config.number_text_tokens,
            gpt_config.hidden_size
        )

        self.text_pos_embedding = (
            LearnedPositionEmbeddings(
                gpt_config.max_text_tokens + 2,
                gpt_config.hidden_size,
                supports_pp=False
            )
            if gpt_config.max_audio_tokens != -1
            else functools.partial(gpt_config.null_position_embeddings, dim=gpt_config.hidden_size)
        )

        self.conditioning_perceiver = PerceiverResampler(
            dim=gpt_config.hidden_size,
            depth=2,
            dim_context=gpt_config.hidden_size,
            num_latents=32,
            dim_head=64,
            heads=8,
            ff_mult=4,
            use_flash_attn=False,
        )

        # Initialize HiFi-GAN decoder
        self.hifigan_decoder = HifiDecoder(
            input_sample_rate=self.hifi_config.input_sample_rate,
            output_sample_rate=self.hifi_config.output_sample_rate,
            output_hop_length=self.hifi_config.output_hop_length,
            ar_mel_length_compression=self.hifi_config.gpt_code_stride_len,
            decoder_input_dim=self.hifi_config.decoder_input_dim,
            d_vector_dim=self.hifi_config.d_vector_dim,
            cond_d_vector_in_each_upsampling_layer=self.hifi_config.cond_d_vector_in_each_upsampling_layer,
        )

        self.final_norm = nn.LayerNorm(gpt_config.hidden_size, eps=1e-5, bias=True)

        # Kept for model loading purposes
        self.text_head = nn.Linear(gpt_config.hidden_size, gpt_config.number_text_tokens, bias=True)

        self.get_memory_usage_curve()

        # Initialize VLLM engine at the end, settings its concurrency
        self.init_vllm_engine(self.max_concurrency)

        # Semaphore for concurrency control of the encoding process
        self.eval()

    def get_memory_usage_curve(self):
        # empirically found values
        x = np.array([2, 5, 10, 16])
        y = np.array([1.25, 1.35, 1.45, 1.625])

        # polynomial fit
        coefficients = np.polyfit(x, y, 2)

        # create a polynomial object
        self.max_gb_for_vllm_model = (coefficients[0] * self.max_concurrency ** 2 +
                    coefficients[1] * self.max_concurrency +
                    coefficients[2])

    def half(self):
        self.logger.warning("Cannot call .half() on XTTSv2Engine. it will be ignored.")
        # We cannot permit downcasting since it will throw an error while padding
        return

    def to(self, *args, **kwargs):
        # Block downcasting
        dtype = kwargs.get('dtype', None)
        if dtype == torch.float16 or dtype == torch.bfloat16:
            self.logger.warning("Cannot cast to half precision. Ignoring the request.")
            kwargs['dtype'] = torch.float32
        elif len(args) > 0 and (args[0] == torch.float16 or args[0] == torch.bfloat16):
            self.logger.warning("Cannot cast to half precision. Ignoring the request.")
            args = list(args)
            args[0] = torch.float32
            args = tuple(args)
        return super().to(*args, **kwargs)

    def init_vllm_engine(self, concurrency):
        """Initialize models with AsyncVLLMEngine."""
        max_seq_num = concurrency
        mem_utils = self.get_memory_percentage(self.max_gb_for_vllm_model * 1024 ** 3) #
        if not mem_utils:
            raise RuntimeError("Could not find the memory usage for the VLLM model initialization.")
        engine_args = AsyncEngineArgs(
            model=self.gpt_model,
            tensor_parallel_size=self.tp,
            pipeline_parallel_size=self.pp,
            dtype="auto",
            max_model_len=self.gpt_config.max_text_tokens +
                          self.gpt_config.max_audio_tokens +
                          32 + 5 + 3, # this is from the xttsv2 code, 32 is the conditioning sql
            gpu_memory_utilization=mem_utils,
            trust_remote_code=True,
            enforce_eager=True,
            limit_mm_per_prompt={"audio": 1}, # even if more audio are present, they'll be condendesed into one
            max_num_seqs=max_seq_num,
            disable_log_stats=True, # temporary fix for the log stats, there is a known bug in vllm that will be fixed in the next relaese
            max_num_batched_tokens=(self.gpt_config.max_text_tokens +
                                    self.gpt_config.max_audio_tokens +
                                    32 + 5 + 3) * max_seq_num,
            #We round to the nearest multiple of 32 and multiply by max_seq_num to get the max batched number (arbitrary) of tokens
        )
        self.logger.info(f"Initializing VLLM engine with args: {engine_args}")
        self.llm_engine = AsyncLLMEngine.from_engine_args(engine_args)

    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path: str,
            torch_dtype: torch.dtype = torch.float32,
            device_map: Optional[str] = "auto",
            tensor_parallel_size: int = 1,
            pipeline_parallel_size: int = 1,
            **kwargs,
    ) -> nn.Module:
        """Load pretrained XTTS model from HuggingFace Hub."""
        from huggingface_hub import hf_hub_download
        import json
        import os

        # Download and load configs
        if not os.path.exists(pretrained_model_name_or_path):
            config_file = hf_hub_download(
                repo_id=pretrained_model_name_or_path,
                filename="config.json"
            )
            with open(config_file, 'r') as f:
                config = json.load(f)

        else:
            # Load from local path
            with open(os.path.join(pretrained_model_name_or_path, "config.json"), 'r') as f:
                config = json.load(f)

        # Initialize configs
        gpt_config = XTTSGPTConfig(**config['gpt_config'])
        hifi_config = XTTSConfig(**config)

        # Initialize model
        model = cls(
            hifi_config=hifi_config,
            gpt_config=gpt_config,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            **kwargs
        )

        # Load model weights
        if not os.path.exists(pretrained_model_name_or_path):
            hifigan_weights = hf_hub_download(
                repo_id=pretrained_model_name_or_path,
                filename="xtts-v2.safetensors"
            )
        else:
            hifigan_weights = os.path.join(pretrained_model_name_or_path, "xtts-v2.safetensors")

        import safetensors.torch

        # Load HiFi-GAN weights
        hifigan_state = safetensors.torch.load_file(hifigan_weights)
        model.load_state_dict(hifigan_state)

        # Set model properties
        model.config = config

        # Cast model to specified dtype
        model = model.to(torch_dtype)
        model = model.to('cuda')

        return model

    async def _get_speaker_embedding(self, audio, sr):
        audio_16k = torchaudio.functional.resample(audio, sr, 16000)
        return (
                self.hifigan_decoder.speaker_encoder.forward(
                    audio_16k.to(self.device), l2_norm=True).unsqueeze(-1)
                .to(self.device)
            )

    async def _merge_conditioning(self,
                                  text_conditioning: List[torch.Tensor],
                                  audio_conditioning: torch.Tensor) -> List[torch.Tensor]:
        cond_latents = []
        for text_embedding in text_conditioning:
            # Concatenate along sequence dimension
            cond_latents.append((torch.cat([audio_conditioning, text_embedding], dim=1).squeeze(0)
                                 .to(self.llm_engine.engine.model_config.dtype)))
        return cond_latents

    def get_gpt_cond_latents(self, audio, sr, length: int = 30, chunk_length: int = 6):
        """Compute the conditioning latents for the GPT model from the given audio."""
        if sr != 22050:
            audio = torchaudio.functional.resample(audio, sr, 22050)
        if length > 0:
            audio = audio[:, : 22050 * length]
        if self.gpt_config.use_perceiver_resampler:
            style_embs = []
            for i in range(0, audio.shape[1], 22050 * chunk_length):
                audio_chunk = audio[:, i: i + 22050 * chunk_length]

                # if the chunk is too short ignore it
                if audio_chunk.size(-1) < 22050 * 0.33:
                    continue

                mel_chunk = wav_to_mel_cloning(
                    audio_chunk,
                    mel_norms=self.mel_stats.cpu(),
                    n_fft=2048,
                    hop_length=256,
                    win_length=1024,
                    power=2,
                    normalized=False,
                    sample_rate=22050,
                    f_min=0,
                    f_max=8000,
                    n_mels=80,
                )
                style_emb = self.get_style_emb(mel_chunk.to(self.device), None)
                style_embs.append(style_emb)

            # mean style embedding
            cond_latent = torch.stack(style_embs).mean(dim=0)
        else:
            mel = wav_to_mel_cloning(
                audio,
                mel_norms=self.mel_stats.cpu(),
                n_fft=4096,
                hop_length=1024,
                win_length=4096,
                power=2,
                normalized=False,
                sample_rate=22050,
                f_min=0,
                f_max=8000,
                n_mels=80,
            )
            cond_latent = self.get_style_emb(mel.to(self.device))
        return cond_latent.transpose(1, 2)

    async def get_conditioning_latents(
            self,
            audio_reference,
            max_ref_length=30,
            gpt_cond_len=6,
            gpt_cond_chunk_len=6,
            librosa_trim_db=None,
            sound_norm_refs=False,
            load_sr=22050,
    ):
        """Get the conditioning latents for the GPT model from the given audio."""
        # Deal with multiple references
        assert (isinstance(audio_reference, bytes) or
                isinstance(audio_reference, str) or
                isinstance(audio_reference, list)), \
            f"audio_reference must be a string, byte or a list but it is {type(audio_reference)}"

        if not isinstance(audio_reference, list):
            audio_paths = [audio_reference]
        else:
            audio_paths = audio_reference

        speaker_embeddings = []
        audios = []
        for file_path in audio_paths:
            audio = load_audio(file_path, load_sr)
            audio = audio[:, : load_sr * max_ref_length].to(self.device).to(self.dtype)
            if sound_norm_refs:
                audio = (audio / torch.abs(audio).max()) * 0.75
            if librosa_trim_db is not None:
                audio = librosa.effects.trim(audio, top_db=librosa_trim_db)[0]

            # Compute latents for the decoder
            speaker_embedding = await self._get_speaker_embedding(audio, load_sr)
            speaker_embeddings.append(speaker_embedding)

            audios.append(audio)

        # Merge all the audios and compute the latents for the GPT
        full_audio = torch.cat(audios, dim=-1)
        gpt_cond_latents = self.get_gpt_cond_latents(
            full_audio, load_sr, length=gpt_cond_len, chunk_length=gpt_cond_chunk_len
        )  # [1, 1024, T]

        speaker_embedding = torch.stack(speaker_embeddings)
        speaker_embedding = speaker_embedding.mean(dim=0)

        return gpt_cond_latents, speaker_embedding

    def get_style_emb(self, cond_input: torch.Tensor, return_latent: Optional[bool] = False) -> torch.Tensor:
        """Get conditioning embeddings from mel spectrograms."""
        if not return_latent:
            if cond_input.ndim == 4:
                cond_input = cond_input.squeeze(1)
            conds = self.conditioning_encoder(cond_input)

            if hasattr(self, 'conditioning_perceiver'):
                conds = self.conditioning_perceiver(
                    conds.permute(0, 2, 1)
                ).transpose(1, 2) # (b,d,32)
        else:
            conds = cond_input.unsqueeze(1)
        return conds

    async def prepare_text_tokens_async(self, text: str, language: str, split_text=False) \
            -> Tuple[List[Union[int, List[int]]], List[torch.Tensor]]:
        """Prepare text tokens for the given text and language."""
        self.logger.debug(f"Preparing text tokens for text: {text}")
        async def elaborate_tokens(text_tokens: List[int]) -> torch.Tensor:
            text_tokens.insert(0, self.tokenizer.bos_token_id)
            text_tokens.append(self.tokenizer.eos_token_id)
            return torch.tensor(text_tokens).unsqueeze(0).to(self.text_embedding.weight.device)

        async def embed_tokens(text_tokens: Union[torch.Tensor, List[torch.Tensor]]) -> List[torch.Tensor]:
            embeds = []
            if isinstance(text_tokens, list):
                for list_element in text_tokens:
                    embeds.append(self.text_embedding(list_element) + self.text_pos_embedding(list_element))
            else:
                embeds.append(self.text_embedding(text_tokens) + self.text_pos_embedding(text_tokens))
            return embeds

        fake_tokens_for_audio_generation = []
        if split_text:
            text_tokens = self.tokenizer.batch_encode_with_split(text, lang=[language])
            for idx, text_token in enumerate(text_tokens):
                text_tokens[idx] = await elaborate_tokens(text_token)
                fake_tokens_for_audio_generation.append([1] * len(text_token))
        else:
            text_tokens = self.tokenizer(text, lang=[language])['input_ids'][0]
            text_tokens = await elaborate_tokens(text_tokens)
            fake_tokens_for_audio_generation = [1] * len(text_tokens)
        return fake_tokens_for_audio_generation, await embed_tokens(text_tokens)


    async def prepare_inputs_async(self, text: str, language: str, speaker_file: List[Union[str, Path]],
                                   max_ref_length: int, gpt_cond_len: int, gpt_cond_chunk_len: int, split_text: bool) \
            -> Tuple[List[List[int]], List[torch.Tensor], torch.Tensor]:
        """Prepare input text with conditioning tokens. Return combined conditioning latents"""
        # Tokenize text based on the language
        text_tokens, text_embeddings = await self.prepare_text_tokens_async(text, language, split_text)

        # Load the speaker file and convert it to a tensor
        gpt_cond_latent, speaker_embeddings = await self.get_audio_conditioning(
            speaker_file,
            max_ref_length,
            gpt_cond_len,
            gpt_cond_chunk_len
        )

        cond_latents = await self._merge_conditioning(text_embeddings, gpt_cond_latent)

        return text_tokens, cond_latents, speaker_embeddings

    async def get_audio_conditioning(
            self,
            audio_reference: [str, Path],
            max_ref_length=30,
            gpt_cond_len=6,
            gpt_cond_chunk_len=6,
            librosa_trim_db=None,
            sound_norm_refs=False,
            load_sr=22050,
    ):
        """Async version of get_conditioning_latents with concurrency control."""
        # Run the original get_conditioning_latents in executor
        result = await self.get_conditioning_latents(
            audio_reference,
            max_ref_length,
            gpt_cond_len,
            gpt_cond_chunk_len,
            librosa_trim_db,
            sound_norm_refs,
            load_sr
        )
        return result

    async def get_model_logits(
            self,
            token_ids: List[int],
            conditioning: MultiModalDataDict,
            request_id: str,
    ) -> torch.Tensor:
        """
        Get model logits for a request with retry logic for empty hidden states.

        Args:
            token_ids: Input token IDs
            conditioning: Conditioning data
            request_id: Unique request ID
        """
        request_id = f"{request_id}_logits"


        # Reset token_ids on each attempt
        token_ids = ([self.mel_bos_token_id] + list(token_ids) + [self.mel_eos_token_id] * 4)
        # we need 5 eos tokens

        engine_inputs = TokensPrompt(prompt_token_ids=token_ids)
        conditioning['audio']['sequence_length'] = len(token_ids)

        engine_inputs["multi_modal_data"] = conditioning

        hidden_states_collector = HiddenStatesCollector()
        # Bind the collector to this request
        bound_collector = hidden_states_collector.bind_to_request(request_id)

        # Set up sampling parameters with the bound collector
        sampling_params = ExtendedSamplingParams(
            detokenize=False,
            request_id=request_id,
            max_tokens=1,
            hidden_state_collector=bound_collector,
            output_kind=RequestOutputKind.FINAL_ONLY
        )

        # Generate with unique request ID
        generator = self.llm_engine.generate(
            prompt=engine_inputs,
            sampling_params=sampling_params,
            request_id=request_id
        )

        async for output in generator:  # consume the generator
            if output.finished:
                pass

        # Get the collected hidden states
        hidden_states = await hidden_states_collector.get_hidden_states(request_id)

        if hidden_states is None:
            raise RuntimeError(
                f"No hidden states collected for request {request_id}. "
                f"This should never happen! Please report this issue on GitHub."
            )
        start_of_audio_hs = conditioning["audio"]["embeds"].shape[0] # type: ignore
        # Successfully got hidden states
        return self.final_norm(hidden_states[start_of_audio_hs:-5, ...].unsqueeze(0).to(self.device).to(self.dtype))

    async def conditioning_phase(self,
                                     request: TTSRequest,
                                     #gpt_cond_latent: Optional[torch.Tensor] = None,
                                     #speaker_embeddings: Optional[torch.Tensor] = None,
                                     ) -> List[GenerationContext]:
        #if gpt_cond_latent is None or speaker_embeddings is None: # TODO to be re-enabled for streming efficency
        # Prepare input with conditioning
        tokens_list, gpt_embed_inputs, speaker_embeddings = await self.prepare_inputs_async(
            request.text,
            request.language,
            request.speaker_files,
            request.max_ref_length,
            request.gpt_cond_len,
            request.gpt_cond_chunk_len,
            split_text=True  # Split text to avoid OOM on big texts
        )
        generation_context = [
            GenerationContext.from_request(request,
                                           tokens=tokens,
                                           decoding_embeddings_modifier=gpt_embed_input,
                                           speaker_embeddings=speaker_embeddings)
                              for tokens, gpt_embed_input in zip(tokens_list, gpt_embed_inputs)]
        #else:
        #    tokens_list, text_embeddings = await self.prepare_text_tokens_async(request.text,
        #                                                                        request.language,
        #                                                                        split_text=True)
        #    gpt_embed_inputs = await self._merge_conditioning(text_embeddings, gpt_cond_latent)

        # Start all requests in parallel
        return generation_context

    async def phonetic_phase(self, context: GenerationContext) -> GenerationContext:
        sampling_params = ExtendedSamplingParams(
            temperature=context.temperature,
            top_p=context.top_p,
            detokenize=False,
            request_id=uuid.uuid4(),
            top_k=context.top_k,
            logits_processors=[LogitsRepetitionPenalizer(context.repetition_penalty)],
            repetition_penalty=1.0,  # Since we're handling repetition penalty manually
            max_tokens=self.gpt_config.gpt_max_audio_tokens,
            ignore_eos=True,  # Ignore the tokenizer eos token since it is for textual generation
            stop_token_ids=[self.mel_eos_token_id],
            output_kind=RequestOutputKind.FINAL_ONLY
        )

        engine_inputs = TokensPrompt(prompt_token_ids=context.tokens)
        if context.decoding_embeddings_modifier is not None:
            engine_inputs["multi_modal_data"] = {
                "audio": {
                    "embeds": context.decoding_embeddings_modifier,
                    "is_logits_only_mode": False,
                    "sequence_length": len(context.text)
                }
            }
        request_id =f"{context.request_id}"
        # Get audio token generator from VLLM
        token_generator = self.llm_engine.generate(
            prompt=engine_inputs,
            sampling_params=sampling_params,
            request_id=request_id,
        )

        async for output in token_generator:

            if output.finished:
                # get the hidden states
                context.spectrogram = await self.get_model_logits(
                    list(output.outputs[0].token_ids),
                    {
                        "audio": {
                            'embeds': context.decoding_embeddings_modifier,  # Use multimodal data for conditioning
                            "is_logits_only_mode": True,
                            "sequence_length": False # will be inserted later in the decoding process
                        },
                    },
                    output.request_id
                )
                context.tokens = list(output.outputs[0].token_ids)

        return context

    async def speech_phase(
            self,
            context: GenerationContext,
    ) -> AsyncGenerator[TTSOutput, None]:
        """
        Process tokens to speech using a vocoder
        """

        async with self.cuda_memory_manager():
                wav = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda: self.hifigan_decoder(
                        context.spectrogram,
                        g=context.speaker_embeddings
                    ).cpu().detach().numpy().squeeze()
                ) # noqa

        # yield the audio output
        yield TTSOutput(array= wav,
                        start_time = context.start_time,
                        token_length = len(context.tokens)
                        )

    async def shutdown(self):
        self.llm_engine.shutdown_background_loop()

