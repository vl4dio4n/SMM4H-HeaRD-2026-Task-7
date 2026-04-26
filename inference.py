from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path

import pandas as pd

from evaluate import build_pipeline, load_experiment
from src.data.processor import load_split

DEFAULT_ID_PATTERN = r"^test_\d{1,3}$"

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True, type=str)
    ap.add_argument("--test-path", type=str, default="datasets/new_test_data.csv")
    ap.add_argument(
        "--id-pattern",
        type=str,
        default=DEFAULT_ID_PATTERN,
        help=(
            "Regex (Python re) that valid submission IDs must match. "
            "Pass '' to disable filtering."
        ),
    )
    args = ap.parse_args()

    exp_dir = Path(args.exp_dir)
    cfg = load_experiment(exp_dir)

    df = load_split(args.test_path, has_labels=False)

    if args.id_pattern:
        pat = re.compile(args.id_pattern)
        mask = df["ID"].astype(str).map(lambda s: bool(pat.match(s)))
        kept, dropped = int(mask.sum()), int((~mask).sum())
        if dropped:
            print(
                f"[filter] Keeping {kept} IDs matching {args.id_pattern!r}; "
                f"dropping {dropped} rows whose IDs don't match "
                f"(Codabench would reject them)."
            )
        df = df[mask].reset_index(drop=True)

    pipeline, _ = build_pipeline(cfg, exp_dir)

    print(f"Predicting {len(df)} test segments ...")
    preds = [pipeline.predict_tokens(list(tokens)) for tokens in df["tokens"]]

    sub_df = pd.DataFrame({"ID": df["ID"], "predicted_ner_tags": preds})
    sub_csv = exp_dir / "submission.csv"
    sub_df.to_csv(sub_csv, index=False)

    sub_zip = exp_dir / "submission.zip"
    with zipfile.ZipFile(sub_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(sub_csv, arcname="submission.csv")

    print(f"Wrote {sub_csv} ({len(sub_df)} rows) and {sub_zip}")

if __name__ == "__main__":
    main()
