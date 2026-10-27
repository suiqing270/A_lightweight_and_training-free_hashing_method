from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_project_root_to_path

PROJECT_ROOT = add_project_root_to_path()

from src.attacks import resolve_tlh_blocks
from src.hashing import generate_tlh
from src.models import create_model
from src.utils.checkpoint import load_checkpoint_into_model
from src.utils.io import write_json, write_lines


def parse_args():
    parser = argparse.ArgumentParser(description="Generate TLH for a CNN checkpoint.")
    parser.add_argument("--model", default="resnet18", help="Model architecture name.")
    parser.add_argument("--checkpoint", required=True, help="Path to a .pth/.pt checkpoint.")
    parser.add_argument("--method", choices=["chaos", "baseline", "nochaos"], default="chaos")
    parser.add_argument("--blocks", default="auto", help="Number of fixed parameter blocks, or auto.")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mu", type=float, default=0.2)
    parser.add_argument("--k", type=float, default=2.0)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "artifacts" / "hashes" / "tlh" / "resnet18_tlh.json"),
        help="Output JSON path.",
    )
    parser.add_argument("--legacy-txt", help="Optional txt path containing one 17-bit code per line.")
    return parser.parse_args()


def main():
    args = parse_args()
    blocks = resolve_tlh_blocks(args.model) if args.blocks == "auto" else int(args.blocks)
    model = create_model(args.model, num_classes=args.num_classes)
    load_checkpoint_into_model(model, args.checkpoint, device=args.device)

    result = generate_tlh(
        model,
        method=args.method,
        num_blocks=blocks,
        mu=args.mu,
        k=args.k,
        iterations=args.iterations,
    )

    output = Path(args.output)
    write_json(output, result.to_dict())
    if args.legacy_txt:
        write_lines(args.legacy_txt, result.encoded)

    print(f"TLH saved to {output}")
    print(f"blocks: {len(result.encoded)}")


if __name__ == "__main__":
    main()
