#!/usr/bin/env python

import os
from argparse import ArgumentParser
from os import PathLike
from time import monotonic
from typing import cast

import torch

from miniqwen.model import MiniQwen


def sample_response(model: MiniQwen, prompt: str, max_generate_len: int = 1000):
    num_tokens_generated = 0
    start_time = monotonic()
    first_token_time: float | None = None

    for token in model.generate(prompt, max_generate_len=max_generate_len):
        if first_token_time is None:
            first_token_time = monotonic()
        print(token, end="", flush=True)
        num_tokens_generated += 1

    end_time = monotonic()
    print()

    elapsed_sec = end_time - start_time
    tokens_per_sec = num_tokens_generated / elapsed_sec
    print(f"--- {num_tokens_generated} token(s) generated in {elapsed_sec:.2f} seconds")
    print(f"--- {tokens_per_sec:.2f} token(s)/sec")

    if first_token_time is not None:
        ttft = first_token_time - start_time
        tpot = (end_time - first_token_time) / (num_tokens_generated - 1)
        print(f"--- TTFT: {ttft} sec")
        print(f"--- TPOT: {tpot * 1000} ms")

    if cache := model.model.kv_cache:
        print(f"--- KV cache size: {cache.size} byte(s)")
        print(f"--- Cached length: {cache[0].cached_seq_len} token(s)")


def main():
    parser = ArgumentParser(description="MiniQwen REPL")
    parser.add_argument("-d", "--device", default="cpu", help="Device to use")
    parser.add_argument("-p", "--prompt", help="The prompt for non-interactive mode")
    parser.add_argument("model_dir", help="Path to the model directory")

    args = parser.parse_args()
    model_dir = cast(PathLike, args.model_dir or os.getcwd())
    device = torch.device(args.device)

    m = MiniQwen.from_pretrained(model_dir).to(device)
    if args.prompt is not None:
        sample_response(m, args.prompt)
        return

    while True:
        prompt = input("> ").strip()
        if prompt == ".exit" or prompt == ".quit":
            break
        sample_response(m, prompt)


if __name__ == "__main__":
    main()
