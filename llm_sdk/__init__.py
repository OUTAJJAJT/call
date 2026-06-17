# ABOUTME: LLM SDK for local model inference using Hugging Face transformers.
# ABOUTME: Provides Small_LLM_Model class for loading and running causal language models.

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel, logging
from huggingface_hub import hf_hub_download
from typing import Tuple, List, Optional, Union


logging.set_verbosity_error()  # keep the console clean


class Small_LLM_Model:
    """
    Utility class wrapping a lightweight Hugging Face causal-LM for fast, low-memory experimentation.

    Attributes:
        model_name (str): Identifier of the model on the HF Hub.
        device (str): Computation device (mps, cuda, or cpu).
        dtype (torch.dtype): Numerical precision.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        *,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = True,
    ) -> None:
        """
        Initializes the model and tokenizer.

        Args:
            model_name (str): HF Hub repo ID.
            device (Optional[str]): Device to use. Auto-selected if None.
            dtype (Optional[torch.dtype]): Tensor precision. Auto-selected if None.
            trust_remote_code (bool): Whether to allow custom code from HF Hub.
        """
        self._model_name = model_name

        # Auto-select device with priority: mps > cuda > cpu
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self._device = device

        if dtype is None:
            dtype = torch.float16 if self._device in ["cuda", "mps"] else torch.float32
        self._dtype = dtype

        # --- load tokenizer & model -------------------------------------------------
        self._tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        if self._tokenizer.pad_token_id is None:
            # ensure we have a pad token to keep batch helpers happy
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        self._model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self._dtype,
            device_map="auto" if self._device == "cuda" else None,
            trust_remote_code=trust_remote_code,
        )
        self._model.to(self._device)
        self._model.eval()

        # switch to inference-only mode
        for p in self._model.parameters():
            p.requires_grad = False

    def encode(self, text: str) -> torch.Tensor:
        """
        Tokenizes text and returns a 2-D input_ids tensor.

        Args:
            text (str): The input string.

        Returns:
            torch.Tensor: Tensor of shape (1, sequence_length).
        """
        ids = self._tokenizer.encode(text, add_special_tokens=False)
        return torch.tensor([ids], device=self._device, dtype=torch.long)

    def decode(self, ids: Union[torch.Tensor, List[int]]) -> str:
        """
        Converts token IDs back into a string.

        Args:
            ids (Union[torch.Tensor, List[int]]): Sequence of token IDs.

        Returns:
            str: The decoded text.
        """
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
            if isinstance(ids[0], list): # Handle batch dim
                ids = ids[0]
        return self._tokenizer.decode(ids, skip_special_tokens=True)

    def get_logits_from_input_ids(self, input_ids: List[int]) -> List[float]:
        """
        Calculates logits for the next token given a sequence of input IDs.

        Args:
            input_ids (List[int]): The context sequence.

        Returns:
            List[float]: The logits for the entire vocabulary for the next position.
        """
        input_tensor = torch.tensor([input_ids], device=self._device, dtype=torch.long)
        with torch.no_grad():
            out = self._model(input_ids=input_tensor)
        # Get logits for the last token in the sequence for the batch (batch size 1)
        logits = out.logits[0, -1].tolist()
        return [float(x) for x in logits]

    def get_path_to_vocab_file(self) -> str:
        """
        Downloads and returns the local path to the vocabulary file.

        Returns:
            str: Path to vocab.json or equivalent.
        """
        vocab_file_name = self._tokenizer.vocab_files_names.get('vocab_file', "vocab.json")
        vocab_path = hf_hub_download(
            repo_id=self._model_name,
            filename=vocab_file_name
        )
        return vocab_path

    def get_path_to_merges_file(self) -> str:
        """
        Downloads and returns the local path to the merges file.

        Returns:
            str: Path to merges.txt or equivalent.
        """
        merges_file_name = self._tokenizer.vocab_files_names.get('merges_file', "merges.txt")
        merges_path = hf_hub_download(
            repo_id=self._model_name,
            filename=merges_file_name
        )
        return merges_path

    def get_path_to_tokenizer_file(self) -> str:
        """
        Downloads and returns the local path to the tokenizer.json file.

        Returns:
            str: Path to tokenizer.json.
        """
        tokenizer_file_name = self._tokenizer.vocab_files_names.get('tokenizer_file', "tokenizer.json")
        tokenizer_path = hf_hub_download(
            repo_id=self._model_name,
            filename=tokenizer_file_name
        )
        return tokenizer_path

