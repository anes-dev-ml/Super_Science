"""Offline development bridge from text to token IDs for the C++ engine."""
import argparse
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_directory")
    parser.add_argument("prompt")
    parser.add_argument("--chat", action="store_true", help="Use Qwen3 chat template")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_directory, local_files_only=True
    )
    if args.chat:
        tokens = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            add_generation_prompt=True,
            tokenize=True,
            enable_thinking=False,
        )
    else:
        tokens = tokenizer.encode(args.prompt, add_special_tokens=False)
    print(" ".join(str(token) for token in tokens))


if __name__ == "__main__":
    main()
