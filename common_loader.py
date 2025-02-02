from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional, List

from torch import float16, backends
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizer


@dataclass
class CustomGenerationConfig:
    """Configuration for text generation"""
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    do_sample: bool


class ModelError(Exception):
    """Custom exception for model errors"""
    pass


class Inference:
    """
    A class for handling model inference with proper error handling
    and optimized performance for Intel CPUs using Hugging Face transformers.
    """

    def __init__(
            self,
            model_name_or_path: Union[str, Path],
            tokenizer_name_or_path: Optional[Union[str, Path]] = None
    ):
        """Initialize the model and tokenizer."""
        self.model_name_or_path = str(model_name_or_path)
        self.tokenizer_name_or_path = str(tokenizer_name_or_path) if tokenizer_name_or_path else self.model_name_or_path
        self.device = 'mps' if backends.mps.is_available() else 'cpu'

        try:
            self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                torch_dtype=float16,
                low_cpu_mem_usage=True,
                device_map=self.device
            )
            self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name_or_path)
        except Exception as ex:
            raise ModelError(f"Failed to load model: {str(ex)}")

    @staticmethod
    def _prepare_input(text: str) -> str:
        """Prepare the input text (basic placeholder)"""
        return text.strip()

    def generate(self, prompt: str, config: CustomGenerationConfig) -> str:
        """Generate text based on the provided prompt and configuration."""
        try:
            prepared_input = self._prepare_input(prompt)
            inputs = self.tokenizer(prepared_input, return_tensors="pt")

            # Move inputs to the same device as the model
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            hf_config = GenerationConfig(
                max_new_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                do_sample=config.do_sample,
            )
            print(f"GenerationConfig: {hf_config}")
            print(hf_config.top_k)

            outputs = self.model.generate(**inputs, generation_config=hf_config)
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return result

        except Exception as ex:
            raise ModelError(f"Generation failed: {str(ex)}")

    def generate_batch(
            self,
            input_texts: List[str],
            config: Optional[CustomGenerationConfig] = None
    ) -> List[str]:
        """Generate text for multiple inputs in parallel."""
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(
                lambda text: self.generate(text, config or CustomGenerationConfig()),
                input_texts
            ))
        return results

    def get_model_info(self) -> dict:
        """Get basic model information"""
        return {
            "model_name_or_path": self.model_name_or_path,
            "tokenizer_name_or_path": self.tokenizer_name_or_path
        }


def read_file(file_path: str) -> str:
    """Read input text from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def write_file(file_path: str, text: str):
    """Write generated output to a file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text)
