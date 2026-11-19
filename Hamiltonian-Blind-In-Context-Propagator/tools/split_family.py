"""Copy the trajectories of one family out of a mixed dataset (data.generate --family mix stores it in "fam").
    python -m tools.split_family datasets/n10mix/train.h5 xxz datasets/n10fam/xxz/train.h5"""
import sys

import h5py
import numpy as np


def split_family(src: str, fam: str, dst: str) -> int:
    with h5py.File(src, "r") as f, h5py.File(dst, "w") as g:
        idx = np.flatnonzero(f["fam"][()] == fam.encode())
        assert len(idx), f"no trajectory of family {fam!r} in {src}"
        g.attrs.update(f.attrs)
        for k, d in f.items():
            if k in ("node", "edge"):  # big: copied one trajectory at a time, same chunking as data.generate
                out = g.create_dataset(k, (len(idx), *d.shape[1:]), d.dtype, chunks=(1, *d.shape[1:]))
                for j, i in enumerate(idx):
                    out[j] = d[i]
            else:
                g.create_dataset(k, data=d[()][idx])
    return len(idx)


if __name__ == "__main__":
    print(f"{sys.argv[3]}: {split_family(*sys.argv[1:4])} trajectories")
