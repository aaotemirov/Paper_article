"""Download the NHANES 2015-2016 audiometry (AUX_I) and demographics (DEMO_I) files.

    python scripts/download_nhanes.py --dest data/nhanes

The files are public-domain data from the U.S. CDC/NCHS. The experiment runner also downloads
them automatically on first use.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cagfnet.data.nhanes import download_nhanes, load_nhanes_ears  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="data/nhanes")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    paths = download_nhanes(a.dest, force=a.force)
    for k, p in paths.items():
        print(f"{k}: {p} ({p.stat().st_size / 1e6:.1f} MB)")
    ears = load_nhanes_ears(a.dest, download=False)
    print(f"eligible ears: {len(ears)} from {ears['SEQN'].nunique()} participants")
