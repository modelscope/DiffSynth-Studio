"""Offline: rewrite a training metadata file's raw prompts into structured JSON captions.

Run this once before training if your dataset's prompt column holds free-form prose, then
train on the rewritten metadata. The rewriter is a separate VLM + stage-2 LoRA adapter
(not the DiT); provide it via --base/--adapter or REWRITER_BASE_MODEL/REWRITER_ADAPTER.

Usage:
    python rewrite_captions.py --metadata metadata.csv --output metadata_rewritten.csv \
        --base /path/to/rewriter-base --adapter /path/to/rewriter-step2-lora --duration 5

Supports .csv / .json / .jsonl. Rows whose stage-2 output fails to parse keep their
original prompt and are logged, so training never silently uses a broken row.
"""

import argparse
import json
import os

try:
    from .prompt_rewriter import Rewriter, make_backend
except ImportError:
    from prompt_rewriter import Rewriter, make_backend


def _load_rows(path):
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), "json"
    if path.endswith(".jsonl"):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows, "jsonl"
    import pandas
    df = pandas.read_csv(path)
    return [df.iloc[i].to_dict() for i in range(len(df))], "csv"


def _save_rows(rows, path, fmt):
    if fmt == "json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
    elif fmt == "jsonl":
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        import pandas
        pandas.DataFrame(rows).to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True, help="input metadata (.csv/.json/.jsonl)")
    ap.add_argument("--output", required=True, help="output metadata with rewritten captions")
    ap.add_argument("--prompt-column", default="prompt")
    ap.add_argument("--mode", default="t2v", choices=["t2v", "ti2v", "t2i"])
    ap.add_argument("--duration", type=float, default=5)
    ap.add_argument("--backend", default="transformers")
    ap.add_argument("--base", default=None, help="rewriter base model (or REWRITER_BASE_MODEL)")
    ap.add_argument("--adapter", default=None, help="rewriter stage-2 LoRA (or REWRITER_ADAPTER)")
    ap.add_argument("--first-frame-column", default=None,
                    help="for ti2v: column holding the first-frame image path/URL")
    args = ap.parse_args()

    rows, fmt = _load_rows(args.metadata)
    rewriter = Rewriter(make_backend(args.backend, args.base, args.adapter))

    ok, failed = 0, 0
    for i, row in enumerate(rows):
        raw = row.get(args.prompt_column, "")
        first_frame = row.get(args.first_frame_column) if args.first_frame_column else None
        result = rewriter.rewrite(raw, mode=args.mode, first_frame=first_frame, duration=args.duration)
        if result["json"] is not None:
            row[args.prompt_column] = json.dumps(result["json"], ensure_ascii=False, separators=(",", ":"))
            ok += 1
        else:
            failed += 1
            print(f"[warn] row {i}: stage-2 JSON did not parse; keeping original prompt.")
        if (i + 1) % 20 == 0:
            print(f"  ... {i + 1}/{len(rows)} rewritten")

    _save_rows(rows, args.output, fmt)
    print(f"[done] {ok} rewritten, {failed} kept-as-is -> {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
