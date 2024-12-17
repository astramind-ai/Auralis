#  Copyright (c) 2024 Astramind. Licensed under Apache License, Version 2.0.

import asyncio
import functools
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Tuple, Union

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
from ...common.definitions.scheduler.contexts import ConditioningContext, PhoneticContext, SpeechContext
from ...common.definitions.types.generator import Tokens
from ...common.logging.logger import setup_logger
from ...common.utilities import wav_to_mel_cloning, load_audio
from ...common.vllm.hidden_state_collector import HiddenStatesCollector
from ...common.vllm.hijack import ExtendedSamplingParams, LogitsRepetitionPenalizer

def mock_context_data(ctx):
    """Return the worst case scenario for profiling of the context generation part."""

    # one since we enforce mono, 60s since we enforce 60s max reference lenght
    placeholder_audio_tensor = [torch.zeros((1, ctx['input_sample_rate'] * 60),
                                            device=ctx['device'],
                                            # 5 is the hard limit for reference files
                                            dtype=ctx['dtype'])] * 5 * ctx['concurrences'][0]
    return ConditioningContext(
        tokens=[torch.zeros(
            (1, ctx['max_sizes'][0]),
            dtype=torch.long,
            device=ctx['device'])
               ] * ctx['concurrences'][0],
        speaker_files=placeholder_audio_tensor,
        start_time=time.time(),
        request_id=uuid.uuid4().hex
    )

def mock_synth_data(ctx):
    """Return the worst case scenario for profiling."""
    max_seq_len = ctx['max_sizes'][2] # start and eos tokens and the conditioning sql
    placeholder_tensor = torch.zeros(
        (ctx['concurrences'][2], max_seq_len, ctx['hidden_size']),
        device=ctx['device'],
        dtype=ctx['dtype']
    )
    decoding_conditioning = torch.zeros(
        (ctx['concurrences'][2], 512, 1),
        device=ctx['device'],
        dtype=ctx['dtype']
    )
    mock_tokens = torch.zeros(
        (ctx['concurrences'][2]* max_seq_len),
        dtype=torch.long,
        device=ctx['device']
    )
    return SpeechContext(
        spectrogram=placeholder_tensor,
        speaker_embeddings=decoding_conditioning,
        tokens=mock_tokens,
        start_time=time.time(),
        request_id=uuid.uuid4().hex
    )


@register_tts_model(
    model_type=SupportedModelTypes.XTTSv2,
    uses_vllm=True,
    supported_languages = ['en', 'es', 'fr', 'de', 'it', 'pt', 'pl', 'tr', 'ru',
                           'nl', 'cs', 'ar', 'zh-cn', 'ja', 'hu', 'ko', 'hi'],
    fake_data_factories = (mock_context_data, None, mock_synth_data),
)
class XTTSv2Engine(BaseAsyncTTSEngine):
    """Async XTTS model implementation using VLLM's AsyncEngine."""

    model_type: str = "xtts"

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

        self.max_concurrency = kwargs.pop('scheduler_max_concurrency', 10)
        self.first_and_third_concurrency = max(1, self.max_concurrency // 6) * self.tp

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
        self.init_vllm_engine(self.max_concurrency, kwargs.get('device', 'cuda'))

        self.eval()

    async def _merge_conditioning(self,
                                  text_conditioning: List[torch.Tensor],
                                  audio_conditioning: torch.Tensor) -> List[torch.Tensor]:
        cond_latents = []
        for text_embedding in text_conditioning:
            # Concatenate along sequence dimension
            cond_latents.append((torch.cat([audio_conditioning, text_embedding], dim=1).squeeze(0)
                                 .to(self.llm_engine.engine.model_config.dtype)))
        return cond_latents

    async def _get_speaker_embedding(self,
                                     audio_list: List[torch.Tensor],
                                     sr: int) -> (
            Tuple)[torch.Tensor, torch.Tensor]: # ([bs, embeddings], [, audio])
        # here we could not batch the inputs, because we cannot guarantee that the audio is the same length
        # we would have to modify the model to accept a padding mask

        audios=[]
        for audio in audio_list:
            audio_16k = torchaudio.functional.resample(audio, sr, 16000)
            audios.append (
                self.hifigan_decoder.speaker_encoder.forward(
                    audio_16k.to(self.device), l2_norm=True).unsqueeze(-1)
                .to(self.device)
            )
        return torch.stack(audios).mean(dim=0), torch.cat(audio_list, dim=-1)

    @property
    def config(self):
        return vars(self.gpt_config) | vars(self.hifi_config) | {
            "concurrences": (self.first_and_third_concurrency,) * 3,
            "max_sizes":(
            (vars(self.gpt_config)['max_text_tokens']),
            None,
            vars(self.gpt_config)['max_text_tokens'] + vars(self.gpt_config)['max_audio_tokens'] + 32 + 5  # start and eos tokens and the conditioning sql
            ),
            "dtype": self.dtype,
            "device": self.device
        }

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

    def init_vllm_engine(self, concurrency, device):
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
            device=device,
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

        # Cast model to specified dtype
        model = model.to(torch_dtype)
        model = model.to('cuda')

        return model


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
            context: ConditioningContext,
            librosa_trim_db=None,
            sound_norm_refs=False,
            load_sr=22050,
    ):
        """Get the conditioning latents for the GPT model from the given audio."""
        # Deal with multiple references
        assert (isinstance(context.speaker_files, bytes) or
                isinstance(context.speaker_files, str) or
                isinstance(context.speaker_files, list) or
                isinstance(context.speaker_files, torch.Tensor)# for profiling
                ), \
            f"speaker_files must be a string, byte or a list but it is {type(context.speaker_files)}"

        if not isinstance(context.speaker_files, list):
            audio_paths = [context.speaker_files]
        else:
            audio_paths = context.speaker_files

        audios = []
        for file_path in audio_paths:
            audio = load_audio(file_path, load_sr) if not isinstance(file_path, torch.Tensor) else file_path
            audio = audio[:, : load_sr * context.max_ref_length].to(self.device).to(self.dtype)
            if sound_norm_refs:
                audio = (audio / torch.abs(audio).max()) * 0.75
            if librosa_trim_db is not None:
                audio = librosa.effects.trim(audio, top_db=librosa_trim_db)[0]
            audios.append(audio)

        # Compute latents for the decoder
        speaker_embedding, full_audio = await self._get_speaker_embedding(audios, load_sr)

        # Merge all the audios and compute the latents for the GPT
        gpt_cond_latents = self.get_gpt_cond_latents(
            full_audio, load_sr, length=context.gpt_cond_len, chunk_length=context.gpt_cond_chunk_len
        )  # [1, 1024, T]

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

    async def preprocess_inputs(self, request: TTSRequest) -> List[ConditioningContext]:
        """
        Preprocess a TTSRequest to prepare it for text-to-speech generation.

        This method handles the token elaboration, adding special beginning-of-sequence
        and end-of-sequence tokens to the provided text tokens, and converts them
        into a suitable tensor format for subsequent processing.

        Args:
            request (TTSRequest): The text-to-speech request containing the necessary
                                  information for processing, including the input text.

        Returns:
            ConditioningContext: The context required for the generation process,
                               including the processed tokens.
        """
        async def elaborate_tokens(text_tokens: List[int]) -> torch.Tensor:

            text_tokens.insert(0, self.tokenizer.bos_token_id)
            text_tokens.append(self.tokenizer.eos_token_id)
            return torch.tensor(text_tokens).unsqueeze(0).to(self.text_embedding.weight.device)

        if isinstance(request.text, str):
            self.logger.debug(f"Preparing text tokens for text: {request.text}")

            if request.split_text:
                conditioning_context = []
                text_tokens = self.tokenizer.batch_encode_with_split(request.text, lang=[request.language])
                for text_token in text_tokens:
                    conditioning_context.append(
                        ConditioningContext.from_request(
                            request, tokens=await elaborate_tokens(text_token)
                        )
                    )
                return conditioning_context
            else:
                text_tokens = self.tokenizer(request.text, lang=[request.language])['input_ids'][0]
                text_tokens = await elaborate_tokens(text_tokens)
                return [ConditioningContext.from_request(request, tokens=text_tokens)]


    async def prepare_text_tokens_and_embeddings_async(self, context: ConditioningContext) \
            -> Tuple[List[Tokens], List[torch.Tensor]]:
        """
        Prepare the text tokens and their embeddings asynchronously.

        This method takes a GenerationContext, and asynchronously prepares the text
        tokens and their embeddings. It first embeds the text tokens using the
        text_embedding and text_pos_embedding layers. It then prepares the tokens
        and their embeddings by calling prepare_token_and_textual_embeddings on each
        of the tokens.

        Args:
            context (ConditioningContext): The GenerationContext containing the text
                                         tokens and other information.

        Returns:
            Tuple[List[Union[int, List[int]]], List[torch.Tensor]]: A tuple containing
                the prepared tokens and their embeddings.
        """

        async def embed_tokens(text_tokens: Union[torch.Tensor, List[torch.Tensor]]) -> List[torch.Tensor]:
            return self.text_embedding(text_tokens) + self.text_pos_embedding(text_tokens)

        async def prepare_token_and_textual_embeddings(tokens: Union[torch.Tensor, List[torch.Tensor]]) -> \
            Tuple[List[Union[int, List[int]]], List[torch.Tensor]]:
            return [1] * tokens.shape[-1], await embed_tokens(tokens)

        if isinstance(context.tokens, torch.Tensor):
            context.tokens = [context.tokens]

        prepared_tokens = []
        prepared_embeddings = []
        for token in context.tokens:
            results = await prepare_token_and_textual_embeddings(token)
            prepared_tokens.append(results[0])
            prepared_embeddings.append(results[1])

        return prepared_tokens, prepared_embeddings


    async def prepare_inputs_async(self,
                                   context: ConditioningContext) \
            -> (Tuple)[List[List[int]], List[torch.Tensor], torch.Tensor]:
        """Prepare input text with conditioning tokens. Return combined conditioning latents"""
        # Tokenize text based on the language
        text_tokens, text_embeddings = await self.prepare_text_tokens_and_embeddings_async(context)

        # Load the speaker file and convert it to a tensor
        gpt_cond_latent, speaker_embeddings = await self.get_audio_conditioning(
            context
        )

        cond_latents = await self._merge_conditioning(text_embeddings, gpt_cond_latent)
        return text_tokens, cond_latents, speaker_embeddings

    async def get_audio_conditioning(
            self,
            context: ConditioningContext,
            librosa_trim_db=None,
            sound_norm_refs=False,
            load_sr=22050,
    ):
        """Async version of get_conditioning_latents with concurrency control."""

        # Run the original get_conditioning_latents in executor
        result = await self.get_conditioning_latents(
            context,
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

    async def conditioning_phase(
            self,
            context: ConditioningContext,
    ) -> List[PhoneticContext]:
        """
        Performs the conditioning phase for text-to-speech generation.

        This method prepares input tokens, GPT embeddings, and speaker embeddings
        for one or multiple generation contexts. It processes them concurrently
        and creates new contexts for each generation combination.

        Args:
            contexts: Single GenerationContext containing the input parameters for generation.

        Returns:
            List[PhoneticContext]: List of new generation contexts with
            prepared tokens, embeddings, and modifiers. Each context is ready for
            the generation phase.
        """
        def is_nested(lst):
            """
            Check if a list contains any nested lists.

            Args:
                lst: A list of items.

            Returns:
                bool: True if the list contains any nested lists, False otherwise.
            """
            return any(isinstance(item, list) for item in lst)

        # Unpack results into separate lists
        tokens, gpt_embed_input, speaker_embeddings = await self.prepare_inputs_async(context)

        # Create new contexts for each generation combination
        contexts_for_generations = []
        for token_seq, single_gpt_embed_input in zip(tokens, gpt_embed_input):

                contexts_for_generations.append(
                    PhoneticContext(
                    request_id=context.request_id,
                    start_time=context.start_time,
                    tokens=token_seq[0] if is_nested(token_seq) else token_seq,
                    decoding_embeddings_modifier=single_gpt_embed_input,
                    speaker_embeddings=speaker_embeddings
                ))

        return contexts_for_generations

    async def phonetic_phase(self, context: PhoneticContext) -> SpeechContext:
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
                    "sequence_length": len(context.tokens)
                }
            }

        request_id =f"{context.request_id}"
        # Get audio token generator from VLLM
        token_generator = self.llm_engine.generate(
            prompt=engine_inputs,
            sampling_params=sampling_params,
            request_id=request_id,
        )

        speech_context = None
        async for output in token_generator:
            if output.finished:
                # get the hidden states
                speech_context = SpeechContext(
                    spectrogram = await self.get_model_logits(
                    list(output.outputs[0].token_ids),
                    {
                        "audio": {
                            'embeds': context.decoding_embeddings_modifier,  # Use multimodal data for conditioning
                            "is_logits_only_mode": True,
                            "sequence_length": False # will be inserted later in the decoding process
                        },
                    },
                    output.request_id
                ),
                speaker_embeddings=context.speaker_embeddings,
                tokens = list(output.outputs[0].token_ids),
                request_id=context.request_id,
                start_time=context.start_time
                )
        if speech_context is None:
            raise RuntimeError(
                f"No audio tokens generated for request {request_id}. "
                f"This should never happen! Please report this issue on GitHub."
            )
        return speech_context

    async def speech_phase(
            self,
            context: SpeechContext,
    ) -> TTSOutput:
        """
        Process tokens to speech using a vocoder
        """

        wav = (await asyncio.to_thread(self.hifigan_decoder, # to thread since is a blocking call
                        context.spectrogram,
                        g=context.speaker_embeddings
                    )).cpu().detach().numpy().squeeze()


        # yield the audio output
        return TTSOutput(parent_request_id=context.request_id,
                        array= wav,
                        is_finished = True,
                        end_time=time.time(),
                        start_time = context.start_time,
                        token_length = len(context.tokens) if context.tokens is not None else 0
                        )

    async def shutdown(self):
        self.llm_engine.shutdown_background_loop()

