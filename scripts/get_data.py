"""Extract the Panasonic 18650PF dataset zip into data/raw/.

The dataset (P. Kollmeyer, Mendeley Data, doi:10.17632/wykht8y7tg.1, CC BY 4.0) is not redistributed in
this repository. Download it with "Download All" from https://data.mendeley.com/datasets/wykht8y7tg/1
and pass the zip's path:

    python scripts/get_data.py path/to/wykht8y7tg-1.zip

Every archive member is checked to resolve inside data/raw before extraction (no path traversal), and
the expected structure is verified afterwards.
"""
import argparse
import os
import zipfile

from socest.data import RAW, TEMP_DIRS, ROOT

EXPECTED_ENTRIES = 292


def main():
    """Extract and verify the dataset."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("zip_path")
    a = ap.parse_args()
    out = os.path.realpath(ROOT / "data" / "raw")
    os.makedirs(out, exist_ok=True)
    with zipfile.ZipFile(a.zip_path) as z:
        members = z.infolist()
        for m in members:
            target = os.path.realpath(os.path.join(out, m.filename))
            if not (target == out or target.startswith(out + os.sep)):
                raise ValueError(f"unsafe path in archive: {m.filename}")
        if len(members) != EXPECTED_ENTRIES:
            print(f"warning: {len(members)} entries, expected {EXPECTED_ENTRIES}; is this the right zip?")
        z.extractall(out)
    missing = [str(p) for p in TEMP_DIRS.values() if not p.exists()]
    if missing:
        raise SystemExit(f"extraction finished but expected folders are missing: {missing}")
    print(f"dataset ready in {RAW}")


if __name__ == "__main__":
    main()
