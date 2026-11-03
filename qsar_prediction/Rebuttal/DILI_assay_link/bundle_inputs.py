"""Copies the six input datasets into ./inputs/ so this folder is self-contained
and can be rsynced to Wahab on its own.

Run once from a full checkout of the repository, before copying the folder over.
Verifies each copy byte-for-byte (SHA-256) against the original, because these are
the same cleaned datasets every published result in the paper was built from --
a silently corrupted or re-derived copy would make the in vitro features
incomparable to those results.

Usage: python bundle_inputs.py [--force]
"""
import argparse
import hashlib
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paths import INPUTS, ORIGINALS, REB


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main(force):
    os.makedirs(INPUTS, exist_ok=True)
    total, failed = 0, []
    for name, rel in ORIGINALS.items():
        src = os.path.join(REB, rel)
        dst = os.path.join(INPUTS, name)
        if not os.path.exists(src):
            print(f'  MISSING SOURCE  {name}  ({src})')
            failed.append(name)
            continue
        if os.path.exists(dst) and not force and sha256(src) == sha256(dst):
            print(f'  ok (already bundled)  {name}')
            total += os.path.getsize(dst)
            continue
        shutil.copy2(src, dst)
        if sha256(src) != sha256(dst):
            print(f'  COPY MISMATCH  {name}')
            failed.append(name)
            continue
        size = os.path.getsize(dst)
        total += size
        print(f'  copied  {name:32s} {size / 1e6:6.1f} MB  (sha256 verified)')

    print(f'\ninputs/ total {total / 1e6:.1f} MB')
    if failed:
        print(f'FAILED: {failed}')
        sys.exit(1)
    print('All six inputs bundled and verified.')
    print('\nThis folder is now self-contained. Copy it with:')
    print('  rsync -avz --exclude "data_*" --exclude "results" --exclude "logs" \\')
    print('    --exclude "__pycache__" DILI_assay_link/ '
          'ychit001@wahab.hpc.odu.edu:~/DILI_assay_link/')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true', help='re-copy even if identical')
    main(ap.parse_args().force)
