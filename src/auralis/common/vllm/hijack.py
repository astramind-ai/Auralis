from typing import List, Optional

import torch
from vllm import SamplingParams

from auralis.models.xttsv2.components.vllm.hidden_state_collector import HiddenStatesCollector


class ExtendedSamplingParams(SamplingParams, kw_only=True):
    """Extended sampling parameters that allows additional fields while maintaining compatibility with SamplingParams.

    This class inherits from SamplingParams and allows adding new required fields
    without conflicting with the base class's optional fields ordering.
    """
    hidden_state_collector: Optional[HiddenStatesCollector] = None  # New required field
    request_id: Optional[str] = None  # New required field


class LogitsRepetitionPenalizer:
    """A logits processor that applies repetition penalty to prevent repetitive text generation."""

    def __init__(self, repetition_penalty: float):
        if repetition_penalty < 0:
            raise ValueError("Repetition penalty must be non-negative")
        self.repetition_penalty = repetition_penalty

    def __call__(self, prompt_token_ids:List[int], token_ids: List[int], logits: torch.Tensor) -> torch.Tensor:
        """Apply repetition penalty to the logits based on previous tokens."""
        # If no repetition penalty or no tokens to check, return original logits
        if self.repetition_penalty == 1.0 or (not token_ids and not prompt_token_ids):
            return logits

        # Create a mask for the repeated tokens
        repeated_tokens = torch.tensor(prompt_token_ids + token_ids,
                                       device=logits.device,
                                       dtype=torch.long)

        # Get logits of repeated tokens
        repeated_logits = logits[repeated_tokens]

        # Apply penalty: divide positive logits by penalty, multiply negative logits by penalty
        repeated_logits = torch.where(
            repeated_logits > 0,
            repeated_logits / self.repetition_penalty,
            repeated_logits * self.repetition_penalty
        )

        # Update only the logits for repeated tokens
        logits[repeated_tokens] = repeated_logits

        return logits

