#!/usr/bin/env python3
import argparse
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List CUDA devices visible to torch")
    parser.add_argument(
        "--ids-only",
        action="store_true",
        help="Print only device indices (one per line) for shell scripting",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        import torch
    except Exception as exc:
        print(f"torch import failed: {exc}", file=sys.stderr)
        return 1

    if not torch.cuda.is_available():
        return 0

    device_count = torch.cuda.device_count()
    for index in range(device_count):
        if args.ids_only:
            print(index)
            continue

        try:
            name = torch.cuda.get_device_name(index)
        except Exception:
            name = "unknown"
        print(f"{index}\t{name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
