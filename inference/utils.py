import json


def read_json(file_path: str) -> dict:
    """Read input from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_json(file_path: str, data: dict):
    """Write output to a JSON file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
