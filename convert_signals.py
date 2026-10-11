#!/usr/bin/env python3
"""
convert_signals.py -- flatten freqtrade's exported signal pickles into CSV.gz files.

Input : a backtest result .zip (or a folder) that contains
          *_signals.pkl   indicator values at the entry/exit signal candles
          *_rejected.pkl  entry signals that were rejected (e.g. max open trades)
          *_exited.pkl    indicator values at the exit candles
Output: <outdir>/signals.csv.gz, rejected.csv.gz, exited.csv.gz
        (+ structure.txt describing what was found, so a new freqtrade layout is easy to adapt)

Usage : python convert_signals.py RESULT.zip OUTDIR
Needs : pandas (+ pyarrow if the pickles hold arrow-backed columns).
Nothing here modifies the backtest data; it only reads and re-saves it.
"""
import glob
import os
import pickle
import sys
import tempfile
import zipfile

import pandas as pd


def walk(obj, path, frames, tree):
    if isinstance(obj, pd.DataFrame):
        tree.append(f"{'/'.join(path)}: DataFrame {obj.shape}")
        df = obj.copy()
        if "date" not in df.columns and df.index.name in ("date", None) and \
                pd.api.types.is_datetime64_any_dtype(df.index):
            df = df.reset_index().rename(columns={df.index.name or "index": "date"})
        df.insert(0, "_path", "/".join(path))
        frames.append(df)
    elif isinstance(obj, dict):
        tree.append(f"{'/'.join(path)}: dict[{len(obj)}]")
        for k, v in obj.items():
            walk(v, path + [str(k)], frames, tree)
    elif isinstance(obj, (list, tuple)):
        tree.append(f"{'/'.join(path)}: {type(obj).__name__}[{len(obj)}]")
        for v in obj:
            walk(v, path, frames, tree)          # keep the same label for list items
    else:
        tree.append(f"{'/'.join(path)}: {type(obj).__name__} (skipped)")


def convert(pkl_path, out_csv, tree_lines):
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    frames, tree = [], []
    walk(obj, [os.path.basename(pkl_path).split("_")[-1].replace(".pkl", "")], frames, tree)
    tree_lines.append(f"== {os.path.basename(pkl_path)}")
    seen = set()
    for t in tree:                                   # collapse repeated lines
        label, _, desc = t.rpartition(": ")          # pair names contain ':' -> split from the right
        key = (label.count("/"), desc.split(" ")[0].split("[")[0])
        if len(seen) < 40 and key not in seen:
            tree_lines.append("   " + t)
            seen.add(key)
    if not frames:
        tree_lines.append("   (no DataFrame found)")
        return 0
    df = pd.concat(frames, ignore_index=True, sort=False)
    df.to_csv(out_csv, index=False, compression="gzip")
    tree_lines.append(f"   -> {out_csv}: {len(df)} rows x {df.shape[1]} columns")
    return len(df)


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    src, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    work = src
    if src.lower().endswith(".zip"):
        work = tempfile.mkdtemp()
        with zipfile.ZipFile(src) as z:
            z.extractall(work)
    tree_lines = []
    for kind in ("signals", "rejected", "exited"):
        files = glob.glob(os.path.join(work, "**", f"*_{kind}.pkl"), recursive=True)
        if not files:
            tree_lines.append(f"== no *_{kind}.pkl found")
            continue
        try:
            convert(files[0], os.path.join(outdir, f"{kind}.csv.gz"), tree_lines)
        except Exception as e:  # noqa: BLE001
            tree_lines.append(f"== {kind}: could not convert: {e!r}")
    text = "\n".join(tree_lines)
    print(text)
    with open(os.path.join(outdir, "structure.txt"), "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
