"""Offline C++ vs Hugging Face Qwen3 numerical comparison.

pip install torch transformers safetensors
python tools/compare_logits.py models/qwen3-0.6b cpp_logits.txt 1 2 3
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_directory", type=Path)
    parser.add_argument("cpp_logits", type=Path)
    parser.add_argument("token_ids", type=int, nargs="+")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_directory), dtype=torch.float32, local_files_only=True
    ).eval()
    with torch.inference_mode():
        ids = torch.tensor([args.token_ids], dtype=torch.long)
        reference = model(input_ids=ids).logits[0, -1].float().cpu()

    cpp = []
    # PowerShell 5.1 redirects stdout as UTF-16LE. PowerShell 7 and other
    # shells normally write UTF-8. Accept either, including UTF-8 BOM.
    with args.cpp_logits.open("rb") as source:
        signature = source.read(4)
    file_encoding = "utf-16" if signature[:2] in (bytes((255, 254)), bytes((254, 255))) else "utf-8-sig"
    with args.cpp_logits.open(encoding=file_encoding) as infile:
        for expected_index, line in enumerate(infile):
            index, value = line.split()
            if int(index) != expected_index:
                raise ValueError("C++ logit indices are not sequential")
            cpp.append(float(value))
    if len(cpp) != reference.numel():
        raise ValueError(f"Expected {reference.numel()} logits, got {len(cpp)}")

    actual = torch.tensor(cpp, dtype=torch.float32)
    error = (actual - reference).abs()
    print(f"Vocabulary size: {reference.numel()}")
    print(f"Max absolute error: {error.max().item():.9f}")
    print(f"Mean absolute error: {error.mean().item():.9f}")
    print(f"Reference top token: {reference.argmax().item()}")
    print(f"C++ top token: {actual.argmax().item()}")
    print(f"Top-token agreement: {reference.argmax().item() == actual.argmax().item()}")


if __name__ == "__main__":
    main()
