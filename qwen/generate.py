"""Example: python generate.py 'Explain gravity in one sentence.'"""
import argparse
import torch
from pretrained import DEFAULT_MODEL, load_pretrained


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prompt', nargs='?', default='Explain gravity in one sentence.')
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--revision', default='main')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--dtype', choices=['float32', 'bfloat16'], default='float32')
    parser.add_argument('--max-new-tokens', type=int, default=64)
    args = parser.parse_args()
    model, tokenizer = load_pretrained(args.model, revision=args.revision,
                                      dtype=getattr(torch, args.dtype), device=args.device)
    text = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': args.prompt}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors='pt', add_special_tokens=False).to(args.device)
    output = model.generate(inputs.input_ids, args.max_new_tokens, inputs.attention_mask)
    print(tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True))


if __name__ == '__main__':
    main()
