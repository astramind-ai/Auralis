from typing import List, Optional

import torch
from vllm import SamplingParams

from auralis.models.xttsv2.components.vllm.hidden_state_collector import HiddenStatesCollector


class ExtendedSamplingParams(SamplingParams, kw_only=True):
    """Extended sampling parameters for VLLM with additional fields.
    
    This class extends VLLM's SamplingParams to include additional required fields
    for hidden state collection and request tracking, while maintaining compatibility
    with the base class's functionality.

    Attributes:
        hidden_state_collector (Optional[HiddenStatesCollector]): Collector for model's
            hidden states during generation.
        request_id (Optional[str]): Unique identifier for the generation request.
    """
    hidden_state_collector: Optional[HiddenStatesCollector] = None
    request_id: Optional[str] = None


class LogitsRepetitionPenalizer:
    """Logits processor for preventing repetitive text generation.
    
    This class implements a repetition penalty mechanism that modifies token logits
    based on their previous occurrences in the generated text. It helps prevent
    the model from getting stuck in repetitive patterns.
    """

    def __init__(self, repetition_penalty: float):
        """Initialize repetition penalizer.

        Args:
            repetition_penalty (float): Penalty factor for repeated tokens.
                Values > 1.0 decrease probability of repetition,
                Values < 1.0 increase probability of repetition,
                Value = 1.0 applies no penalty.

        Raises:
            ValueError: If repetition_penalty is negative.
        """
        if repetition_penalty < 0:
            raise ValueError("Repetition penalty must be non-negative")
        self.repetition_penalty = repetition_penalty

    def __call__(self, prompt_token_ids: List[int], token_ids: List[int], logits: torch.Tensor) -> torch.Tensor:
        """Apply repetition penalty to logits.

        This method modifies the logits of tokens that have appeared in either the
        prompt or the generated sequence. For repeated tokens:
        - Positive logits are divided by the penalty
        - Negative logits are multiplied by the penalty
        This effectively reduces the probability of generating repeated tokens.

        Args:
            prompt_token_ids (List[int]): Token IDs from the input prompt.
            token_ids (List[int]): Token IDs from the generated sequence.
            logits (torch.Tensor): Raw logits from the model.

        Returns:
            torch.Tensor: Modified logits with repetition penalty applied.
        """
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

