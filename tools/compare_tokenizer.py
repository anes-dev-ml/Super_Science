"""Compare native Qwen3 tokenizer token IDs with the official Hugging Face tokenizer.

Install: python -m pip install transformers tokenizers
Run: python tools/compare_tokenizer.py models/qwen3-0.6b build-audit/Release/tokenizer_cli.exe
"""
import argparse
from pathlib import Path
import subprocess
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_directory", type=Path)
    parser.add_argument("tokenizer_executable", type=Path)
    args = parser.parse_args()
    exe = args.tokenizer_executable.resolve()
    model = args.model_directory.resolve()
    reference = AutoTokenizer.from_pretrained(str(model), local_files_only=True)

    samples = [
        "Hello, world!",
        "I'm learning C++ and AI!",
        "The quick brown fox jumps over the lazy dog.",
        "1234567890 + 42 = 1234567932",
        "def hello():\n    print('Hello!')",
        "  Multiple   spaces\tand\nnewlines",
        "Bonjour, comment allez-vous ?",
        "François et Montréal",
        "日本語 中文",
        "مرحبا بالعالم",
        "😀 🚀 ❤️",
        "What's the difference between A and B?",
        "a" * 200,
        "  ",
        "",
    ]
    passed = 0
    for index, sample in enumerate(samples, start=1):
        result = subprocess.run(
            [str(exe), str(model / "tokenizer.json"), sample],
            capture_output=True, check=True,
        )
        actual = [int(item) for item in result.stdout.decode("ascii").split()]
        expected = reference.encode(sample, add_special_tokens=False)
        correct = actual == expected
        print(f"Test {index:02d}: {'PASS' if correct else 'FAIL'}")
        if not correct:
            print(f"  Input: {sample!r}")
            print(f"  Reference: {expected}")
            print(f"  Native:    {actual}")
        passed += correct
    print(f"RESULT: {passed}/{len(samples)} passed")
    if passed != len(samples):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
