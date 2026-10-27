from __future__ import annotations

import argparse
from pathlib import Path

import torch

from _bootstrap import add_project_root_to_path

PROJECT_ROOT = add_project_root_to_path()

from src.attacks import resolve_tlh_blocks, tamper_model_by_attack
from src.models import create_model
from src.utils.checkpoint import load_checkpoint_into_model
from src.utils.io import write_json


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a block-wise tampered checkpoint.")
    parser.add_argument("--model", required=True, help="Model architecture name.")
    parser.add_argument("--checkpoint", required=True, help="Path to the original .pth/.pt checkpoint.")
    parser.add_argument("--tamper-rate", type=float, required=True, help="Ratio of TLH blocks to tamper.")
    parser.add_argument(
        "--attack",
        choices=[
            "gaussian_noise",
            "sign_flip",
            "weight_scaling",
            "weight_zeroing",
            "random_reinit",
            "weight_permutation",
            "quantization",
        ],
        default="gaussian_noise",
        help="Block-wise tamper attack type.",
    )
    parser.add_argument("--blocks", default="auto", help="Number of fixed parameter blocks, or auto.")
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--scale-alpha", type=float, default=1.5, help="Alpha for weight_scaling.")
    parser.add_argument(
        "--reinit-mode",
        choices=["normal", "xavier", "kaiming"],
        default="normal",
        help="Initialization mode for random_reinit.",
    )
    parser.add_argument("--quant-bits", type=int, default=8, help="Symmetric quantization bits.")
    parser.add_argument(
        "--quant-decimals",
        type=int,
        help="Optional decimal rounding for quantization. Overrides --quant-bits when set.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for the tampered checkpoint.",
    )
    parser.add_argument(
        "--log-output",
        required=True,
        help="Output JSON path for the legacy-compatible tampered block log.",
    )
    parser.add_argument(
        "--metadata-output",
        help="Optional JSON path for metadata plus the tampered block log.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    blocks = resolve_tlh_blocks(args.model) if args.blocks == "auto" else int(args.blocks)

    model = create_model(args.model, num_classes=args.num_classes)
    load_checkpoint_into_model(model, args.checkpoint, device=args.device)

    result = tamper_model_by_attack(
        model,
        attack=args.attack,
        tamper_rate=args.tamper_rate,
        num_blocks=blocks,
        noise_std=args.noise_std,
        seed=args.seed,
        alpha=args.scale_alpha,
        reinit_mode=args.reinit_mode,
        quant_bits=args.quant_bits,
        quant_decimals=args.quant_decimals,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output)
    write_json(args.log_output, result.tampered_blocks)
    if args.metadata_output:
        write_json(args.metadata_output, result.to_dict())

    print(f"Tampered checkpoint saved to {output}")
    print(f"Tamper log saved to {args.log_output}")
    print(f"attack: {args.attack}")
    print(f"blocks: {blocks}")
    print(f"tampered_blocks: {len(result.tampered_blocks)}")


if __name__ == "__main__":
    main()
