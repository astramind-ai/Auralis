#  Copyright (c) 2024 Astramind. Licensed under Apache License, Version 2.0.
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
import torch

from auralis.common.definitions.batch.batchable_item import BatchableItem
from auralis.common.definitions.dto.requests import TTSRequest
from auralis.common.definitions.scheduler.context import GenerationContext


@dataclass
class BatchedItems:
    """Container for batched items with support for TTSRequest and GenerationContext"""
    stage: str
    items: List[BatchableItem] = field(default_factory=list)
    #_indexes: List[str] = field(default_factory=list)
    # all of this is commented since it will be useful when request are actually batched and not only generated in parallel

    # Batched data containers
    #texts: List[str] = field(default_factory=list)
    #speaker_files: Optional[torch.Tensor] = None
    #speaker_embeddings: Optional[torch.Tensor] = None
    #tokens: Optional[torch.Tensor] = None
    #decoding_modifiers: Optional[torch.Tensor] = None
    #spectrograms: Optional[torch.Tensor] = None

    # Metadata
    lengths: torch.Tensor = None
    #attention_mask: torch.Tensor = None

    def batch(self, item: BatchableItem):
        """Add an item to the batch and update tensors accordingly"""
        self.items.append(item)
        #self._indexes.append(item.request_id)

    @property
    def length(self) -> int:
        """Get the total length of all items in the batch"""
        return sum(item.length(self.stage) for item in self.items)

    # def _batch_tts_request(self, item: TTSRequest):
    #     """Handle batching for TTSRequest items"""
    #     if self.stage == 'conditioning_phase':
    #         self.texts.append(item.text)
    #         if isinstance(item.speaker_files, list):
    #             # Stack speaker files as tensors with padding
    #             files_tensor = self._pad_and_stack_audio(item.speaker_files)
    #             self.speaker_files = files_tensor if self.speaker_files is None else \
    #                 torch.cat([self.speaker_files, files_tensor])
    #         else:
    #             # Single speaker file
    #             audio_tensor = self._load_audio_as_tensor(item.speaker_files)
    #             self.speaker_files = audio_tensor if self.speaker_files is None else \
    #                 torch.cat([self.speaker_files, audio_tensor.unsqueeze(0)])
    #
    # def _batch_generation_context(self, item: GenerationContext):
    #     """Handle batching for GenerationContext items"""
    #     if item.tokens is not None:
    #         tokens_tensor = torch.tensor(item.tokens)
    #         self.tokens = tokens_tensor if self.tokens is None else \
    #             torch.cat([self.tokens, tokens_tensor.unsqueeze(0)])
    #
    #     if item.speaker_embeddings is not None:
    #         emb_tensor = torch.tensor(item.speaker_embeddings)
    #         self.speaker_embeddings = emb_tensor if self.speaker_embeddings is None else \
    #             torch.cat([self.speaker_embeddings, emb_tensor.unsqueeze(0)])
    #
    #     if item.decoding_embeddings_modifier is not None:
    #         mod_tensor = torch.tensor(item.decoding_embeddings_modifier)
    #         self.decoding_modifiers = mod_tensor if self.decoding_modifiers is None else \
    #             torch.cat([self.decoding_modifiers, mod_tensor.unsqueeze(0)])
    #
    #     if item.spectrogram is not None:
    #         spec_tensor = torch.tensor(item.spectrogram)
    #         self.spectrograms = spec_tensor if self.spectrograms is None else \
    #             torch.cat([self.spectrograms, spec_tensor.unsqueeze(0)])
    #
    # def _pad_and_stack_audio(self, audio_files: List[str]) -> torch.Tensor:
    #     """Load audio files and stack them with padding"""
    #     audio_tensors = [self._load_audio_as_tensor(f) for f in audio_files]
    #     max_length = max(t.size(-1) for t in audio_tensors)
    #
    #     padded_tensors = []
    #     for tensor in audio_tensors:
    #         padding = max_length - tensor.size(-1)
    #         padded = torch.nn.functional.pad(tensor, (0, padding))
    #         padded_tensors.append(padded)
    #
    #     return torch.stack(padded_tensors)
    #
    # def _load_audio_as_tensor(self, audio_file: str) -> torch.Tensor:
    #     """Load audio file and convert to tensor"""
    #     # Implementation depends on your audio loading library
    #     # This is a placeholder - implement actual audio loading
    #     return torch.zeros(1)  # placeholder
    #
    #
    # def _unbatch_tts_request(self, item: TTSRequest, idx: int) -> TTSRequest:
    #     """Unbatch a TTSRequest item"""
    #     new_item = item.copy()
    #     if self.speaker_files is not None:
    #         new_item.speaker_files = self.speaker_files[idx].numpy()
    #     return new_item
    #
    # def _unbatch_generation_context(self, item: GenerationContext, idx: int) -> GenerationContext:
    #     """Unbatch a GenerationContext item"""
    #     new_item = GenerationContext(
    #         request_id=item.request_id,
    #         tokens=self.tokens[idx].numpy() if self.tokens is not None else None,
    #         speaker_embeddings=self.speaker_embeddings[idx].numpy() if self.speaker_embeddings is not None else None,
    #         decoding_embeddings_modifier=self.decoding_modifiers[
    #             idx].numpy() if self.decoding_modifiers is not None else None,
    #         spectrogram=self.spectrograms[idx].numpy() if self.spectrograms is not None else None
    #     )
    #     return new_item

