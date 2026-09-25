"""
Build a DiffSynth-Studio `metadata.csv` from a folder of image + caption pairs.

DiffSynth-Studio does not read one-.txt-per-image directly: datasets are described
by a metadata file (csv / json / jsonl) with an `image` column and a `prompt`
column (see docs/zh/Pipeline_Usage/Model_Training.md). This helper converts the
popular "same-name .txt next to each image" layout (kohya / civitai style) into
that metadata file.

Layout before:
    my_data/
    ├── 0001.jpg
    ├── 0001.txt
    ├── 0002.png
    └── 0002.txt

After running this script:
    my_data/
    ├── ...
    └── metadata.csv      # image,prompt

Usage:
    python examples/qwen_image_21/model_training/special/low_vram_training/prepare_dataset_from_txt.py \
        --image_dir my_data
"""
import argparse, csv, os

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def collect(image_dir, recursive, txt_ext):
    walker = os.walk(image_dir) if recursive else [(image_dir, [], os.listdir(image_dir))]
    rows, missing = [], []
    for root, _dirs, files in walker:
        for name in sorted(files):
            stem, ext = os.path.splitext(name)
            if ext.lower() not in IMAGE_EXTS:
                continue
            txt_path = os.path.join(root, stem + txt_ext)
            rel_image = os.path.relpath(os.path.join(root, name), image_dir).replace(os.sep, "/")
            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8") as f:
                    prompt = " ".join(f.read().split())
            else:
                missing.append(rel_image)
                prompt = None
            rows.append((rel_image, prompt))
    return rows, missing


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image_dir", required=True, help="Folder holding the image + .txt pairs.")
    parser.add_argument("--output", default=None, help="Metadata csv path. Default: <image_dir>/metadata.csv")
    parser.add_argument("--recursive", action="store_true", help="Also scan sub-folders (paths stay relative to --image_dir).")
    parser.add_argument("--txt_ext", default=".txt", help="Caption file extension, e.g. .txt or .caption")
    parser.add_argument("--default_prompt", default=None, help="Prompt for images without a caption file; they are skipped when omitted.")
    args = parser.parse_args()

    rows, missing = collect(args.image_dir, args.recursive, args.txt_ext)
    kept = [(i, p if p is not None else args.default_prompt) for i, p in rows if p is not None or args.default_prompt is not None]
    dropped = [i for i, p in rows if p is None and args.default_prompt is None]

    output = args.output or os.path.join(args.image_dir, "metadata.csv")
    with open(output, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "prompt"])
        writer.writerows(kept)

    print(f"{len(kept)} image/prompt pairs written to {output}")
    if missing and args.default_prompt is None:
        print(f"warning: {len(missing)} images had no caption file and were skipped, e.g. {missing[:5]}")
    if dropped:
        print(f"note: {len(dropped)} images used --default_prompt")


if __name__ == "__main__":
    main()