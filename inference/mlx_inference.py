import contextlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional, List, Tuple, Generator

from mlx_lm import load, generate, stream_generate
from mlx_lm.utils import load_config


@dataclass
class CustomGenerationConfig:
    """Configuration for text generation"""
    max_tokens: int
    batch_size: int = 8
    num_draft_tokens: Optional[int] = None  # For draft model is None
    max_kv_size: Optional[int] = None  # For draft model is not None


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
        tokenizer_path: Optional[Union[str, Path]] = None,
        max_workers: int = 4
    ):
        """Initialize the MLX model and tokenizer."""
        self.model_path = Path(model_path)
        self.tokenizer_path = Path(tokenizer_path) if tokenizer_path else self.model_path
        self.max_workers = max_workers
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

    def _load_model(self, lazy=True) -> Tuple:
        """Load the model and tokenizer"""
        model = load(
            path_or_hf_repo=str(self.model_path),
            model_config=self.model_config,
            lazy=lazy
        )
        return model

    @staticmethod
    def _prepare_input(text: str) -> str:
        """Prepare the input text (basic placeholder)"""
        return text.strip()

    def generate(self, prompt: str, config: CustomGenerationConfig) -> str:
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
            config: Optional[CustomGenerationConfig] = None
    ) -> Generator[str, None, None]:
        """Generate text for multiple inputs in parallel with controlled memory usage."""
        if config is None:
            config = CustomGenerationConfig(max_tokens=100)

        def process_batch(batch: List[str]) -> Tuple[List[str], int]:
            """Process a single batch of texts using executor.map, returning the outputs and token count."""
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                outputs = list(executor.map(lambda text: self.generate(text, config), batch))
            joined_output = " ".join(outputs)
            token_ids = self.tokenizer.encode(joined_output)
            return outputs, len(token_ids)

        batch = []
        for text in input_texts:
            batch.append(text)
            if len(batch) >= config.batch_size:
                outputs, batch_token_count = process_batch(batch)
                yield (" ".join(outputs)), batch_token_count
                batch.clear()

        if batch:
            outputs, batch_token_count = process_batch(batch)
            yield (" ".join(outputs)), batch_token_count

    def generate_streaming(self, prompt: str, config: CustomGenerationConfig) -> Generator[str, None, None]:
        """Stream text generation output in real-time using MLX-LM."""
        try:
            prepared_input = self._prepare_input(prompt)
            kwargs = {"max_tokens": config.max_tokens}

            if config.num_draft_tokens is not None:
                kwargs["num_draft_tokens"] = config.num_draft_tokens
            elif config.max_kv_size is not None:
                kwargs["max_kv_size"] = config.max_kv_size

            with contextlib.closing(stream_generate(
                    self.model,
                    self.tokenizer,
                    prepared_input,
                    **kwargs
            )) as response_generator:
                yield from (response.text for response in response_generator)
        except Exception as ex:
            raise MLXModelError(f"Streaming generation failed: {str(ex)}")

    def get_model_info(self) -> dict:
        """Get basic model information"""
        return {
            "model_path": self.model_path,
            "tokenizer_path": self.tokenizer_path,
            "has_config": self.model_config is not None,
            "max_workers": self.max_workers
        }
