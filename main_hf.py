import os

from dotenv import load_dotenv

from inference.hf_inference import HFInference, CustomGenerationConfig, ModelError
from inference.utils import read_json, write_json

load_dotenv()

model_path = os.getenv("MODEL_PATH")
input_file_path = os.getenv("INPUT_FILE_PATH")
output_file_path = os.getenv("OUTPUT_FILE_PATH")

config = CustomGenerationConfig(
    max_new_tokens=200,  # Limits the maximum number of new tokens generated.
    temperature=0.7,  # Controls randomness. Lower values (0.1-0.5) make responses more deterministic; higher values (0.8-1.5) make them more creative.
    top_p=0.9,  # Enables nucleus sampling: only considers tokens that add up to 90% probability mass (lower values limit randomness).
    top_k=0,  # If >0, restricts sampling to the top-k highest probability tokens (smaller values make responses more predictable).
    do_sample=True,  # Enables sampling instead of greedy decoding. If False, always picks the highest probability token (deterministic output).
    repetition_penalty=1.2  # Discourages repeating words/phrases. Values >1 reduce repetition, but too high may cause incoherence.
)


def main() -> None:
    try:
        inference = HFInference(model_path)
        input_data = read_json(input_file_path)
        requests = input_data.get("requests", [])

        print("\nText Generation: ...")
        responses = inference.generate_batch(requests, config)
        write_json(output_file_path, {"responses": responses})
        print("Done!")

    except ModelError as e:
        print(f"Error: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")


if __name__ == "__main__":
    main()
