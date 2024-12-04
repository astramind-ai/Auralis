#  Copyright (c) 2024 Astramind. Licensed under Apache License, Version 2.0.

import functools
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List
from typing import Optional, Union, Iterable, Tuple, Mapping

import torch
import torch.nn as nn
from torch import Tensor
from transformers import GPT2Config
from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group
from vllm.inputs import InputContext, INPUT_REGISTRY, DecoderOnlyInputs, token_inputs, DummyData
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding, ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.gpt2 import GPT2Block
from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsPP
from vllm.model_executor.models.utils import make_layers, make_empty_intermediate_tensors_factory
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalKwargs
from vllm.multimodal.inputs import PlaceholderRange
from vllm.multimodal.utils import consecutive_placeholder_ranges
from vllm.sequence import IntermediateTensors, SequenceData
from vllm.utils import is_list_of

PrefillLength= Union[int, List[int]]
TokenPosition= Union[int, List[int]]
TokenId = Union[Union[torch.Tensor,int], List[Union[torch.Tensor,int]]]

@dataclass
class TokenPositionAndPrefillTuple:
    prefill_len: Optional[PrefillLength] = None
    pos_id: Optional[TokenPosition] = None
    token_id: Optional[TokenId] = None

    def update_(self,
                prefill_len: Optional[PrefillLength] = None,
                pos_id: Optional[TokenPosition] = None,
                token_id: Optional[TokenId] = None):
        if prefill_len is not None:
            self.prefill_len=prefill_len
        if pos_id is not None:
            self.pos_id=pos_id
        if token_id is not None:
            self.token_id= token_id
        return self


class PositionalEmbeddingsCorrecter:
    """Corrects positional embeddings for XTTS model,
    since they have a different length than the text embeddings.
    This class tracks tokens both by request_id and position for vLLM compatibility.
    """

    def __init__(self):
        # Maps request_id to its prefill length
        self.request_tracker_dict: Dict[str, TokenPositionAndPrefillTuple] = defaultdict(lambda: TokenPositionAndPrefillTuple())
        # Maps token_position pairs to their request_id
        self.token_to_request: Dict[str, str] = {}

    def init_request_id_prefill(self, request_id: str, prefill_len: PrefillLength, nex_token: torch.Tensor):
        """Initialize a request_id with its prefill length."""
        self.request_tracker_dict[request_id] = TokenPositionAndPrefillTuple(prefill_len, prefill_len)
        self.token_to_request[f"{nex_token}_{prefill_len}"] = request_id

    def get_by_request_id(self, request_id: str) -> TokenPositionAndPrefillTuple:
        """Retrieve the prefill length for a given request_id."""
        return self.request_tracker_dict.get(request_id, None)

    def get_by_next_token(self,
                          next_token_ids: List[int],
                          next_position_ids: List[int]
                          ) -> List[Optional[TokenPositionAndPrefillTuple]]:
        """Retrieve prefill lengths for given token and position pairs.

        Args:
            next_token_ids: List of token IDs
            next_position_ids: List of position IDs, corresponding to token IDs

        Returns:
            List of prefill lengths for each token-position pair

        Raises:
            ValueError: If no valid token mappings are found
        """
        prefill_lengths = []
        assert len(next_token_ids) == len(next_position_ids), "Token and position lists must have the same length"
        if len(next_token_ids) == 0:
            return prefill_lengths
        for next_token_id, next_position_id in zip(next_token_ids, next_position_ids):
            token_key = f"{next_token_id}_{next_position_id}"
            if token_key in self.token_to_request:
                request_id = self.token_to_request[token_key]
                prefill_lengths.append(self.request_tracker_dict[request_id].update_(token_id=next_token_id))

        if not prefill_lengths:
            raise ValueError(f"No valid mappings found for token pairs")
        return prefill_lengths

    def _invalidate_previous_mapping(self, request_id: str):
        """Remove all token mappings associated with a given request_id.

        This prevents memory leaks from old token mappings and ensures
        we don't have stale token-to-request associations.
        """
        # Find all token keys that map to this request_id
        keys_to_remove = [
            token_key for token_key, req_id in self.token_to_request.items()
            if req_id == request_id
        ]

        # Remove all found mappings
        for token_key in keys_to_remove:
            del self.token_to_request[token_key]

    def _get_pos_id_and_update (self, request_id: str):
        """Get the position ID for a given request_id and update it."""
        tuple_prefill_token = self.get_by_request_id(request_id)
        # Update the position ID
        self.request_tracker_dict[request_id] = TokenPositionAndPrefillTuple(tuple_prefill_token.prefill_len, tuple_prefill_token.pos_id + 1)
        return tuple_prefill_token.pos_id + 1


    def associate_new_tokens(self, request_id: str, next_token_id: int):
        """Associate a new token-position pair with a request_id.

        Before creating the new association, it removes all previous
        token mappings for this request_id to maintain consistency.

        Args:
            request_id: The request identifier
            next_token_id: The token ID to associate
        """
        pos_id = self._get_pos_id_and_update(request_id)

        # Clean up old mappings first
        self._invalidate_previous_mapping(request_id)

        # Create new mapping
        self.token_to_request[f"{next_token_id}_{pos_id}"] = request_id

    def clear_request(self, request_id: str):
        """Remove all data associated with a request_id.

        This includes both the prefill length tracking and any token mappings.
        """
        if request_id in self.request_tracker_dict:
            # First remove all token mappings
            self._invalidate_previous_mapping(request_id)
            # Then remove the request tracking
            del self.request_tracker_dict[request_id]

class LearnedPositionEmbeddings(nn.Module):
    def __init__(self, seq_len, model_dim, init=0.02, relative=False, supports_pp=False):
        super().__init__()
        # nn.Embedding
        self.emb = VocabParallelEmbedding(seq_len, model_dim) if supports_pp else nn.Embedding(seq_len, model_dim)
        # Initializing this way is standard for GPT-2
        self.emb.weight.data.normal_(mean=0.0, std=init)
        self.relative = relative
        self.seq_len = seq_len

    def forward(self, x):
        sl = x.shape[1]
        if self.relative:
            start = random.randint(sl, self.seq_len) - sl
            indices = torch.arange(start, start + sl, device=x.device)
            # Validate indices
            assert (indices < self.seq_len).all() and (indices >= 0).all(), \
                f"Invalid position indices in forward: min={indices.min().item()}, max={indices.max().item()}, valid_range=[0,{self.seq_len-1}]"
            return self.emb(indices)
        else:
            indices = torch.arange(0, sl, device=x.device)
            # Validate indices
            assert (indices < self.seq_len).all(), \
                f"Sequence length {sl} exceeds maximum position embedding length {self.seq_len}"
            return self.emb(indices)

    def get_fixed_embedding(self, ind: torch.Tensor, dev: torch.device) -> torch.Tensor:
        """Get position embeddings with batch support.

        Args:
            ind: Position indices tensor. Can be single or batched
                 Shape: [..., seq_len] or [seq_len]
            dev: Target device for the embeddings

        Returns:
            Position embeddings tensor matching input shape plus embedding dimension
            Shape: [batch_size, seq_len, model_dim] or [1, 1, model_dim]
        """
        # Validation of indices to prevent unknown errors
        assert (ind < self.seq_len).all(), \
            f"Position indices out of range. Found max={ind.max().item()}, but maximum allowed is {self.seq_len-1}"
        assert (ind >= 0).all(), \
            f"Negative position indices found. Min value={ind.min().item()}"

        if ind.shape[0] > 1:

            return self.emb(ind)
        else:
            #assert ind.dim() <= 2, f"Single input should have 1 or 2 dimensions, got {ind.dim()}"
            return self.emb(torch.tensor([ind], device=dev)).unsqueeze(0)



def get_xtts_max_audio_tokens(ctx: InputContext) -> int:
    """Calculate maximum audio tokens based on text context and audio duration."""
    return 32 # the conditoning perciever output


def dummy_seq_data_for_xtts(
        ctx: InputContext,
        seq_len: int,
        audio_count: int,
):
    """Create dummy sequence data for XTTS profiling."""
    # Calculate audio token space needed
    conditioning_lenght = (32 # the conditioning perceiver output length in the sql (which is fixed)
                           +
                           1) # the start audio token

    return SequenceData.from_prompt_token_counts(
        (1, conditioning_lenght * audio_count),
        (0, seq_len - conditioning_lenght * audio_count)),{
        "audio":
            consecutive_placeholder_ranges(num_items=audio_count,
                                           item_size=conditioning_lenght)
    }


def dummy_conditioning_for_xtts(
        ctx: InputContext,
        seq_len: int,
        audio_count: int,
) -> dict:
    """Create dummy conditioning data for XTTS."""
    return {
        "audio": {
            "embeds":[
                torch.zeros(
                    (seq_len, ctx.model_config.hf_config.hidden_size),
                    dtype=ctx.model_config.dtype) for _ in range(audio_count)
            ],
            "is_logits_only_mode": False,
            "sequence_length": -1,
        }
    }


def dummy_data_for_xtts(
        ctx: InputContext,
        seq_len: int,
        mm_counts: Mapping[str, int],
):
    """Create complete dummy data for XTTS profiling."""
    audio_count = mm_counts["audio"]
    seq_data, ranges = dummy_seq_data_for_xtts(ctx, seq_len, audio_count)
    cond_data = dummy_conditioning_for_xtts(ctx, seq_len, audio_count)
    return DummyData(seq_data, cond_data, ranges)


def input_mapper_for_xtts(ctx: InputContext, data: Union[Dict, List[Tensor]]) -> MultiModalKwargs:
    """Map input data to XTTS format."""

    if not isinstance(data, list):
        data = [data]

    if len(data) == 0:
        return MultiModalKwargs()

    assert is_list_of(data, dict, check="all"), (f"Expected a list of dictionaries, "
                                                 f"but got a list of {[type(dat) for dat in data if type(dat) != dict][0]}")

    embeds = [dat["embeds"] for dat in data]
    is_logits_only_mode = [dat.get("is_logits_only_mode", False) for dat in data]
    sequence_length = [dat.get("sequence_length", -1) for dat in data]
    return MultiModalKwargs(
        {
            "cond_latents": embeds,
            "is_logits_only_mode": is_logits_only_mode,
            "sequence_length": sequence_length
        }
    )




def input_processor_for_xtts2_gpt(ctx: InputContext, inputs: DecoderOnlyInputs):
    """
    We'll accomodate for the extra contditioning token and for the start audio token,
    we actually insert a -1 repeated for the differecne in length between the conditioning and the tokenized text
    and then we add 1 for the start audio token
    Args:
        ctx:
        inputs:

    Returns:

    """
    multi_modal_data = inputs.get("multi_modal_data")
    if multi_modal_data is None or "audio" not in multi_modal_data:
        raise ValueError("Missing audio data in multi-modal inputs")

    audio_dict = multi_modal_data['audio']
    audio = audio_dict.get('embeds')

    is_last_decoding_pass = audio_dict.get("is_logits_only_mode", False)

    prompt_token_ids = inputs.get("prompt_token_ids")

    if not is_last_decoding_pass:
        # we fill everything with 1 since we don't actually needs text token ids, it would mess up in the sampling step
        new_token_ids = ([1] * (audio.shape[0])) + [ctx.model_config.hf_config.start_audio_token] # add the start audio generation token
    else:
        new_token_ids = ([1] * audio.shape[0]) + prompt_token_ids
    # the encoding had already been done externally to reuse the embeddings for later use but we
    # account for the new token that will be added before generation
    new_prompt = None
    return token_inputs(prompt_token_ids=new_token_ids,
                        prompt=new_prompt,
                        multi_modal_data=multi_modal_data,
                        multi_modal_placeholders={'audio':[PlaceholderRange(offset=0, length=len(new_token_ids))]})


@MULTIMODAL_REGISTRY.register_input_mapper("audio", input_mapper_for_xtts)
@MULTIMODAL_REGISTRY.register_max_multimodal_tokens("audio", get_xtts_max_audio_tokens)
@INPUT_REGISTRY.register_dummy_data(dummy_data_for_xtts)
@INPUT_REGISTRY.register_input_processor(input_processor_for_xtts2_gpt)
class XttsGPT(nn.Module, SupportsMultiModal, SupportsPP):
    def __init__( # type: ignore
            self,
            vllm_config: VllmConfig,
            prefix: str,
            cache_config: Optional[CacheConfig] = None,
            quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = vllm_config
        self.gpt_config = self.config.model_config.hf_config
        self.quant_config = quant_config

        self.prefix_sequence_dict: Dict[str, torch.Tensor] = {}
        # Core GPT components
        self.gpt = GPT2Model(
            self.gpt_config,
            cache_config,
            quant_config,
            prefix="gpt"
        )

        self.final_norm =  nn.LayerNorm(self.gpt_config.hidden_size, bias=True, eps=self.gpt_config.layer_norm_epsilon)
        # Output head for mel tokens
        self.mel_head = ParallelLMHead(
            self.gpt_config.num_audio_tokens,
            self.gpt_config.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix="mel_head"
        )
        self.audio_start_generation_token = self.gpt_config.start_audio_token

        self.gpt.audio_start_generation_token = self.audio_start_generation_token


        # Initialize logits processor and sampler
        logit_scale = getattr(self.gpt_config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.gpt_config.num_audio_tokens,
                                                self.gpt_config.num_audio_tokens,
                                                logit_scale)
        self.sampler = Sampler()

        self.positional_embeddings_correcter = PositionalEmbeddingsCorrecter()

    @staticmethod
    def _check_is_logits_only_mode(is_logits_only_mode) -> torch.Tensor:

        # First check if it's a boolean
        if isinstance(is_logits_only_mode, bool):
            return torch.tensor([is_logits_only_mode])

        # Then check if it's a tensor
        if torch.is_tensor(is_logits_only_mode):
            # if it's a scalar tensor, return the value
            if is_logits_only_mode.numel() == 1:
                return is_logits_only_mode
            # for non-scalar tensors, check if all elements are the same
            return is_logits_only_mode

        # Fallback
        return torch.tensor([bool(is_logits_only_mode)])

    @staticmethod
    def find_len_of_sequence(
            positions_ids: torch.Tensor,
            index: torch.Tensor
    ) -> torch.Tensor:
        """
        Starting from the index, it goes backward in the positions until it finds a jump higher than 1.
        This function is tensorized for efficiency.

        Args:
        positions_ids: Tensor of position IDs
        index: Tensor of indices to start searching from

        Returns:
        Tensor of sequence lengths
        """
        # Ensure index is a tensor
        if not isinstance(index, torch.Tensor):
            index = torch.tensor(index, device=positions_ids.device)

        # Create a mask for valid positions (from 0 to index for each element)
        mask = torch.arange(positions_ids.size(0), device=positions_ids.device).unsqueeze(0) <= index

        # Calculate differences between adjacent positions
        diffs = positions_ids[1:] - positions_ids[:-1]

        # Pad the diffs tensor to match the original size
        diffs = torch.cat([torch.ones(1, device=positions_ids.device), diffs])

        # Find where the difference is different from 1 and is within the valid range
        jumps = (diffs != 1) & mask

        # Get the indices of the jumps
        jump_indices = jumps.nonzero()

        # If no jumps are found, return the index itself (full length)
        if jump_indices.numel() == 0:
            return torch.tensor([0], device=positions_ids.device)

        # Get the last jump for each index
        last_jumps = jump_indices[:, 1].reshape(-1, jump_indices.size(0))[:, -1]

        # Calculate the sequence lengths
        return last_jumps

    def _maybe_correct_positions(self,
                                 input_ids: torch.Tensor,
                                 positions: torch.Tensor,
                                 conditioning_inputs_list: List[torch.Tensor]):
        correct_positions_ids = self.positional_embeddings_correcter.get_by_next_token(input_ids.tolist(),
                                                                                       positions.tolist())
        if len(correct_positions_ids) > 0:
            position_and_id_tensor = torch.cat(
                [positions.unsqueeze(0), input_ids.unsqueeze(0)],
                dim=0
            )

            index_2d = torch.tensor(
                [(correct_positions_id.pos_id, correct_positions_id.token_id) for
                 correct_positions_id in correct_positions_ids],
                device=positions.device
            )

            prefill_len_token = torch.tensor(
                [correct_positions_id.prefill_len for correct_positions_id in correct_positions_ids],
                device=positions.device)

            position_and_id_expanded = position_and_id_tensor.unsqueeze(-1)
            index_2d_expanded = index_2d.T.unsqueeze(1)

            matches = (position_and_id_expanded == index_2d_expanded).all(dim=0)
            matching_indices = matches.any(dim=1).nonzero().squeeze(1)

            if not isinstance(conditioning_inputs_list, list) or len(conditioning_inputs_list) < 1:
                # this is the case where all the tokens are a "second iter" token,
                # so we don't have mixed stages in the batch
                return 1 + positions - prefill_len_token
            # Iterate through all matching indices
            for idx, seq_idx in enumerate(matching_indices):

                # Ensure we have corresponding conditioning input
                if (isinstance(conditioning_inputs_list, list) and
                        len(conditioning_inputs_list) > 0 and
                        idx < len(conditioning_inputs_list)):
                    end_pos = seq_idx + 1
                    start_pos = self.find_len_of_sequence(positions, seq_idx)  # type: ignore

                    # Apply correction only to the relevant part of the sequence
                    positions[start_pos:end_pos] = 1 + positions[start_pos:end_pos] - \
                                                   correct_positions_ids[
                                                       idx].prefill_len

            return positions

    def _apply_op_to_seq_in_batch(self,
                                  input_ids: torch.Tensor,
                                  positions: torch.Tensor,
                                  conditioning_inputs_list: List[torch.Tensor],
                                  is_logit_only_mode: torch.Tensor,
                                  seq_len: Union[torch.Tensor],
                                  is_profiling_run: bool = False
                                  ) -> Tuple[List[int], torch.Tensor, torch.Tensor]:
        """
        Apply different ops to the tensors sequence in the batch
        Returns:
            - List of starting indexes
            - A tensor for the logit only mode
            - A mask to reinsert the tokens in the correct position for the logit only mode
            - Modified input IDs
            - Modified positions
        """
        if is_profiling_run:
            return [], input_ids, positions

        # Pre-allocate lists for better memory efficiency
        starting_indexes = []

        # Find all end markers at once
        end_markers = (input_ids == self.audio_start_generation_token).nonzero(as_tuple=True)[0]

        if len(end_markers) == 0:
            positions = self._maybe_correct_positions(input_ids, positions, conditioning_inputs_list)
            return [], input_ids, positions

        # Create mask for valid conditioning inputs
        cond_latent_mask = torch.tensor([
            isinstance(cond_latent, torch.Tensor) and cond_latent.dim() > 1
            for cond_latent in conditioning_inputs_list
        ], device=input_ids.device)

        effective_indexes = cond_latent_mask.nonzero(as_tuple=True)[0]

        # Pre-calculate all sequence lengths
        sequence_lengths = torch.tensor([
            cond.shape[(0 if cond.dim == 1 else 1)] if isinstance(cond, torch.Tensor) and cond.dim() > 1
            else 0 for cond in conditioning_inputs_list
        ], device=input_ids.device)

        # Create masks for efficient tensor operations
        keep_mask = torch.ones(len(input_ids), dtype=torch.bool, device=input_ids.device)
        non_logit_mask = torch.ones_like(keep_mask)

        cumulative_offset = 0

        for idx, end_marker in zip(effective_indexes, end_markers):
            # Calculate effective positions
            end_pos = end_marker.item() - cumulative_offset
            start_pos = end_pos - sequence_lengths[idx].item()
            start_pos_for_masking = start_pos + cumulative_offset

            # Store original starting index
            starting_indexes.append(start_pos_for_masking)

            if is_logit_only_mode[idx]:
                # here the logic is a bit messy:
                # in the og implementation, the treats the embedding for the star tof generation token differently.
                # during the autoregressive token generation phase they use the token embeddings of the start
                # of generation token as input for the position embeddings, but in the logit only mode they use the
                # position id of the start of generation token as input for the position embeddings

                non_logit_mask[start_pos_for_masking : end_pos + cumulative_offset + seq_len[idx]] = False
                keep_mask[start_pos_for_masking:end_pos + cumulative_offset] = False
                # Generate positions for this sequence
                new_positions = torch.arange(
                    0, seq_len[idx].item(), # starting from zero since we have the start audio token
                    device=input_ids.device,
                    dtype=positions.dtype
                )
                # Update positions
                if end_pos + len(new_positions) <= len(positions):
                    positions[end_pos + cumulative_offset:end_pos + cumulative_offset + seq_len[idx]] = new_positions

            else:

                # Update masks
                keep_mask[start_pos_for_masking:end_pos + cumulative_offset + 1] = False

            cumulative_offset += (end_pos - start_pos + 1)

        # Apply masks to get final tensors
        # First we select tokens that are not used in the logit only mode
        # we have tre scenarios here:
        # 1. We are in a first pass where we have a sequence of 1s tokens terminated by a start audio token,
        # we completely remove this and we keep the index on where to insert since we have already precomputed the values
        # 2. We are in a "second pass" (autoregressive pass), using the default process of vllm with corrected positions ids
        # 3. We are in a logit only mode, since in xttsv2 we need to capture the hs,
        # and to do this we pass the conditioning alongside the generated tokens,
        # we need to remove the placeholder sequence at the beginning while adjusting
        # the positioning inside that condition
        non_logit_input_ids = input_ids[non_logit_mask & keep_mask]
        non_logit_positions = positions[non_logit_mask & keep_mask]

        correct_positions = self._maybe_correct_positions(
            # if we arrive here it means that we had mixed "second passes" and "logit only mode" in the batch,
            non_logit_input_ids,
            non_logit_positions,
            conditioning_inputs_list
        )
        if correct_positions is not None:
            # only happens if chunk prefill is enabled
            positions[non_logit_mask & keep_mask] = correct_positions

        modified_input_ids = input_ids[keep_mask]
        modified_positions = positions[keep_mask]
        assert (modified_positions < 608).all()
        assert (modified_positions >= 0).all()
        return starting_indexes, modified_input_ids, modified_positions


    # noinspection PyMethodOverriding
    def forward( # type: ignore
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            kv_caches: List[torch.Tensor],
            attn_metadata: AttentionMetadata,
            intermediate_tensors: Optional["IntermediateTensors"] = None,
            cond_latents: Optional[Union[torch.Tensor, List[torch.Tensor]]] = False, # so we can always have a list
            is_logits_only_mode: Union[torch.Tensor, bool] = False,
            sequence_length: Union[torch.Tensor,int] = -1,
            **kwargs,
    ) -> Union[torch.Tensor, "IntermediateTensors"]:
        """Forward pass following VLLM pattern."""

        is_profiling_run = False

        # we work with list conditioning so we convert them to list regardless of vllm batching
        if isinstance(cond_latents, torch.Tensor):
            if len(cond_latents.shape) > 4:
                is_profiling_run = True
            else:
                # if two equal tensors are passed, vllm aggregate them in a new (batched) tensor
                cond_latents = list(cond_latents)  # so we unbacth them :) (unless we are in the profiling run)

        is_logits_only_mode = self._check_is_logits_only_mode(is_logits_only_mode)

        starting_sequence_start_ids, input_ids, positions = self._apply_op_to_seq_in_batch(input_ids,
                                                                                           positions,
                                                                                           cond_latents,
                                                                                           is_logits_only_mode,
                                                                                           sequence_length,
                                                                                           is_profiling_run)


        hidden_states = self.gpt(
            input_ids=input_ids,
            position_ids=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            intermediate_tensors=intermediate_tensors,
            # this is the conditioning input ( voice conditioning + text_embeds )
            input_embeds=cond_latents,
            starting_sequence_start_ids=starting_sequence_start_ids,
            is_profiling_run= is_profiling_run,
            is_logit_only=is_logits_only_mode
        )

        return hidden_states

    # noinspection PyUnresolvedReferences
    def compute_logits(
            self,
            hidden_states: torch.Tensor,
            sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        # normalize the hidden states
        # we keep this, because in the xttsv2 code they have a nn.sequential with norm and then lm head
        hidden_states = self.final_norm(hidden_states)

        # we keep track of the last collected index to properly associate the hidden states with the correct request_id
        last_collected_idx = 0
        for seq in sampling_metadata.seq_groups:
            # Check if we need to collect hidden states
            sampling_params = seq.sampling_params
            if (hasattr(sampling_params, 'hidden_state_collector')
                    and sampling_params.hidden_state_collector is not None):
                self.positional_embeddings_correcter.clear_request(sampling_params.request_id)
                # Call the collector directly with the hidden states
                sampling_params.hidden_state_collector(hidden_states[last_collected_idx:last_collected_idx+seq.seq_len], sampling_params.request_id)  # The request_id is already bound

            last_collected_idx += seq.seq_len or 0

        # Compute logits using the mel_head
        logits = self.logits_processor(self.mel_head, hidden_states, sampling_metadata, self.mel_head.bias)
        return logits

    # noinspection PyUnresolvedReferences
    def sample(
            self,
            logits: torch.Tensor,
            sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        for seq_id, seq_groups in enumerate(sampling_metadata.seq_groups):
            if hasattr(seq_groups.sampling_params, 'request_id') and seq_groups.sampling_params.request_id is not None:
                idx = seq_groups.seq_ids[0]
                # Call the collector directly with the next tokens
                if not self.positional_embeddings_correcter.get_by_request_id(seq_groups.sampling_params.request_id):
                    self.positional_embeddings_correcter.init_request_id_prefill(
                        request_id = seq_groups.sampling_params.request_id,
                        prefill_len=len(seq_groups.seq_data[idx].prompt_token_ids),
                        nex_token=next_tokens.outputs[seq_id].samples[0].output_token # index out of error
                    )
                else:
                    self.positional_embeddings_correcter.associate_new_tokens(
                        request_id=seq_groups.sampling_params.request_id,
                        next_token_id=next_tokens.outputs[seq_id].samples[0].output_token)

        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load weights following VLLM pattern."""
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_names = set()
        for name, loaded_weight in weights:
            if name not in params_dict:
                continue

            param = params_dict[name]
            if "c_attn" in name or "c_proj" in name or "c_fc" in name:
                if name.endswith(".weight"):
                    loaded_weight = loaded_weight.t()

            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_names.add(name)
        # used to check if all weights were loaded
        assert set(params_dict.keys()) - loaded_names == set(), \
            (f"Missing weights: {set(params_dict.keys()) - loaded_names}, "
             f"this probably means you are using an incompatible model ")

class GPT2Model(nn.Module):

    def __init__(
            self,
            config: GPT2Config,
            cache_config: Optional[CacheConfig] = None,
            quant_config: Optional[QuantizationConfig] = None,
            prefix: str = "",
    ):
        super().__init__()
        self.config = config
        assert not config.add_cross_attention
        assert not config.scale_attn_by_inverse_layer_idx
        assert not config.reorder_and_upcast_attn
        self.audio_start_generation_token = None
        self.embed_dim = config.hidden_size
        self.wte = VocabParallelEmbedding(config.num_audio_tokens, self.embed_dim)
        self.wpe = (
            LearnedPositionEmbeddings(config.max_audio_tokens + 3, config.decoder_input_dim)
            if config.max_audio_tokens != -1
            else functools.partial(config.null_position_embeddings, dim=config.decoder_input_dim)
        )
        self.start_layer, self.end_layer, self.h = make_layers(
            config.num_hidden_layers,
            lambda prefix: GPT2Block(
                config, cache_config, quant_config, prefix=prefix),
            prefix=f"{prefix}.h")
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(["hidden_states"],
                                                    config.hidden_size))

    @staticmethod
    def _insert_conditioning_into_hidden_states(hidden_states: torch.Tensor,
                                                conditioning_inputs: Optional[List[torch.Tensor]],
                                                start_of_generation_embed: Optional[torch.Tensor],
                                                insertion_ids: List[int],
                                                is_logit_only: torch.Tensor) -> torch.Tensor:
        empty_tensor = torch.empty(
            (0,hidden_states.shape[-1]),
            device=hidden_states.device, dtype=hidden_states.dtype
        )
        for idx, (inserion_idx, conditioning_input) in enumerate(zip(insertion_ids, conditioning_inputs)):
                hidden_states = torch.cat([
                hidden_states[:inserion_idx],
                conditioning_input.squeeze(0),
                (start_of_generation_embed if ~is_logit_only[idx] else empty_tensor),
                hidden_states[inserion_idx:]], dim=0
            )

        return hidden_states

    def forward(
            self,
            input_ids: torch.Tensor,
            position_ids: torch.Tensor,
            kv_caches: List[torch.Tensor],
            attn_metadata: AttentionMetadata,
            intermediate_tensors: Optional[IntermediateTensors],
            input_embeds: Optional[torch.Tensor] = None,
            starting_sequence_start_ids: Optional[List[int]] = None,
            is_profiling_run: bool = False,
            is_logit_only: torch.Tensor = False
    ) -> Union[torch.Tensor, IntermediateTensors]:

        if get_pp_group().is_first_rank:
            starting_sequence_embed = None
            if isinstance(input_embeds, list) and len(input_embeds) > 0:
                # we could be either in start condition or in a final condition or both
                if len(starting_sequence_start_ids) > 0 and not (is_logit_only).all():
                    # we have starting sequences, so we just need to get one hs to insert later
                    starting_sequence_embed = self.wte(
                        torch.tensor(
                            self.audio_start_generation_token,
                            device=input_ids.device
                        ).unsqueeze(0)
                    )

                    starting_sequence_embed += self.wpe(starting_sequence_embed.reshape(-1, 1))

            audio_inputs_embeds = self.wte(input_ids).squeeze(0)

            if len(input_ids) == 0:
                # if we have just starting sequences audio_inputs_embeds is an empty tensor
                position_embeds = audio_inputs_embeds.clone()
            else:
                position_embeds = self.wpe.get_fixed_embedding(
                    position_ids, input_ids.device
                ) if not is_profiling_run else self.wpe(input_ids.reshape(-1, 1))

            hidden_states = (audio_inputs_embeds + position_embeds).view(-1, self.embed_dim)

            if isinstance(input_embeds, list) and len(input_embeds) > 0:
                hidden_states = self._insert_conditioning_into_hidden_states(
                    hidden_states,
                    input_embeds,
                    starting_sequence_embed,
                    starting_sequence_start_ids,
                    is_logit_only)

        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        for i in range(self.start_layer, self.end_layer):
            layer = self.h[i]
            hidden_states = layer(hidden_states,
                                  kv_caches[i - self.start_layer],
                                  attn_metadata)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        hidden_states = self.ln_f(hidden_states)
        return hidden_states

