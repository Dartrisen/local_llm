from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional, List, Tuple

from mlx_lm import load, generate
from mlx_lm.utils import load_config


@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    max_tokens: int = 150


class MLXModelError(Exception):
    """Custom exception for MLX model errors"""
    pass


class MLXInference:
    """
    A class for handling MLX model inference with proper error handling
    and optimized performance.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        tokenizer_path: Optional[Union[str, Path]] = None
    ):
        """Initialize the MLX model and tokenizer."""
        self.model_path = Path(model_path)
        self.tokenizer_path = Path(tokenizer_path) if tokenizer_path else self.model_path

        self._validate_paths()

        try:
            self.model_config = load_config(self.model_path)
            self.model, self.tokenizer = self._load_model()
        except Exception as ex:
            raise MLXModelError(f"Failed to load model: {str(ex)}")

    def _validate_paths(self):
        """Validate that model and tokenizer paths exist"""
        if not self.model_path.exists():
            raise MLXModelError(f"Model path does not exist: {self.model_path}")
        if not self.tokenizer_path.exists():
            raise MLXModelError(f"Tokenizer path does not exist: {self.tokenizer_path}")

    def _load_model(self) -> Tuple:
        """Load the model and tokenizer"""
        return load(str(self.model_path), self.model_config)

    @staticmethod
    def _prepare_input(text: str) -> str:
        """Prepare the input text (basic placeholder)"""
        return text.strip()

    def generate(self, prompt: str, config: GenerationConfig) -> str:
        """Generate text based on the provided prompt and configuration."""
        try:
            prepared_input = self._prepare_input(prompt)
            result = generate(
                self.model,
                self.tokenizer,
                prepared_input,
                max_tokens=config.max_tokens,
            )
            return result

        except Exception as ex:
            raise MLXModelError(f"Generation failed: {str(ex)}")

    def generate_batch(
            self,
            input_texts: List[str],
            config: Optional[GenerationConfig] = None
    ) -> List[str]:
        """Generate text for multiple inputs in parallel."""
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(
                lambda text: self.generate(text, config),
                input_texts
            ))
        return results

    def get_model_info(self) -> dict:
        """Get basic model information"""
        return {
            "model_path": self.model_path,
            "tokenizer_path": self.tokenizer_path,
            "has_config": self.model_config is not None
        }


def read_file(file_path: str) -> str:
    """Read input text from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def write_file(file_path: str, text: str):
    """Write generated output to a file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text)


if __name__ == "__main__":
    try:
        model_path = ""
        inference = MLXInference(model_path)

        config = GenerationConfig(
            max_tokens=250,
        )

        batch_inputs = [
            "What is MVP in ecology?"
        ]

        print("\nBatch Generation:")
        outputs = inference.generate_batch(batch_inputs, config)
        for input_text, output_text in zip(batch_inputs, outputs):
            print(f"\nInput: {input_text}")
            print(f"Output: {output_text}")

    except MLXModelError as e:
        print(f"Error: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
