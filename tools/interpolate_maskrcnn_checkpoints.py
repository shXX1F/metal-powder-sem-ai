from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interpolate two compatible Mask R-CNN model checkpoints."
    )
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--alphas",
        default="0.15,0.25,0.35,0.50,0.65,0.75",
        help="Comma-separated candidate-weight fractions in [0, 1].",
    )
    return parser.parse_args()


def load_checkpoint(path: Path) -> object:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def model_state(checkpoint: object) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), dict):
        return checkpoint["model"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError("Checkpoint is not a model state dictionary")


def parse_alphas(value: str) -> list[float]:
    alphas = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not alphas:
        raise ValueError("At least one alpha is required")
    if any(alpha < 0.0 or alpha > 1.0 for alpha in alphas):
        raise ValueError("All alphas must be in [0, 1]")
    return alphas


def alpha_tag(alpha: float) -> str:
    return f"a{int(round(alpha * 1000)):04d}"


def main() -> None:
    args = parse_args()
    base_checkpoint = load_checkpoint(args.base)
    candidate_checkpoint = load_checkpoint(args.candidate)
    base_state = model_state(base_checkpoint)
    candidate_state = model_state(candidate_checkpoint)

    if set(base_state) != set(candidate_state):
        missing = sorted(set(base_state) - set(candidate_state))
        extra = sorted(set(candidate_state) - set(base_state))
        raise ValueError(f"Incompatible state keys: missing={missing}, extra={extra}")
    for key in base_state:
        if base_state[key].shape != candidate_state[key].shape:
            raise ValueError(
                f"Incompatible tensor shape for {key}: "
                f"{tuple(base_state[key].shape)} != {tuple(candidate_state[key].shape)}"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for alpha in parse_alphas(args.alphas):
        blended: dict[str, torch.Tensor] = {}
        for key, base_tensor in base_state.items():
            candidate_tensor = candidate_state[key]
            if base_tensor.is_floating_point():
                blended[key] = torch.lerp(base_tensor, candidate_tensor, alpha)
            else:
                source = candidate_tensor if alpha >= 0.5 else base_tensor
                blended[key] = source.clone()

        output_path = args.output_dir / f"maskrcnn_particle_soup_{alpha_tag(alpha)}.pth"
        torch.save(
            {
                "model": blended,
                "model_soup": {
                    "base": str(args.base),
                    "candidate": str(args.candidate),
                    "candidate_fraction": alpha,
                },
            },
            output_path,
        )
        print(f"{alpha:.4f}\t{output_path}")


if __name__ == "__main__":
    main()
