import os
from typing import TYPE_CHECKING, Any, List, Optional, Tuple
from mbfte import _logger
from mbfte.model_wrapper import (
    AbstractModelWrapper,
    get_predictions,
)
from transformers import GPT2Tokenizer, GPT2TokenizerFast, GPT2LMHeadModel, GPT2Config
from transformers.modeling_outputs import ModelOutput

import torch


class PyTorchModelWrapper(AbstractModelWrapper):
    NAME: str = "pytorch"

    def __init__(
        self,
        model_dir: str,
        temperature: float,
    ) -> None:
        # This is needed for cross-platform compatibility!
        torch.set_default_dtype(torch.float64)

        config = GPT2Config(
            scale_attn_by_inverse_layer_idx=True, reorder_and_upcast_attn=True
        )

        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

        if model_dir == "PRETRAINED":
            _logger.info(f"Loading pretrained GPT-2 model")
            self.model = GPT2LMHeadModel(config)
            self.model = self.model.from_pretrained("gpt2")
        else:
            _logger.info(f"Loading model from {model_dir}")
            self.model = torch.jit.load(os.path.join(model_dir, "gpt2.ptl"))

    def get_token(self, index: int) -> str:
        string = self.tokenizer.decode(index)
        if TYPE_CHECKING:
            assert isinstance(string, str)
        return string

    def tokenize(self, sentence: str) -> List[int]:
        indices = self.tokenizer(sentence)["input_ids"]
        if TYPE_CHECKING:
            assert isinstance(indices, List)
        return indices

    def prediction(
        self,
        sentence: str,
        past: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        first: bool = False,
        save_past: bool = True,
    ) -> Tuple[List[Any], Any]:
        inputs = self.tokenizer(sentence, return_tensors="pt")
        if not save_past:
            # If we're not saving the past, just set it to `None`.
            past = None
        elif first:
            past = _make_dummy_past()
        else:
            assert past is not None
            pass
        outputs: ModelOutput = self.model(inputs["input_ids"], past_key_values=past)
        # Note: We cannot use `outputs.logits` and `outputs.past_key_values`
        # here because those fields may be not be available for locally trained
        # models.
        logits = outputs[0].tolist()[0]
        new_past = outputs[1]

        predictions = get_predictions(logits, model="pytorch")
        return (predictions, new_past)


def _make_dummy_past() -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
    def make_tensor_pair() -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(1, 12, 0, 64, dtype=torch.float),
            torch.zeros(1, 12, 0, 64, dtype=torch.float),
        )

    return tuple(make_tensor_pair() for _ in range(12))
