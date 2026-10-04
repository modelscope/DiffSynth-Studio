"""Convert one Wan DiT safetensors adapter between DiffSynth and Diffusers."""

import argparse
from pathlib import Path

from safetensors.torch import load_file, save
from diffsynth.utils.lora.wan import WanLoRAConverter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--to", choices=["diffusers", "diffsynth"], required=True)
    parser.add_argument(
        "--scale", type=float, default=1.0, help="Additional input adapter scaling."
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new path.")
    convert = (
        WanLoRAConverter.align_to_opensource_format
        if args.to == "diffusers"
        else WanLoRAConverter.align_to_diffsynth_format
    )
    result = convert(load_file(str(args.input)), scale=args.scale)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = save(result)
    try:
        with args.output.open("xb") as output:
            output.write(payload)
    except FileExistsError:
        parser.error("Output already exists; choose a new path.")


if __name__ == "__main__":
    main()
