"""Create synthetic features for a CPU-only CLI demonstration, not research results."""
import argparse
from pathlib import Path

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/demo/features"))
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Choose a new output directory; existing data is not overwritten")
    rng = np.random.default_rng(42)
    prototypes = rng.normal(size=(3, 16))
    task_direction = rng.normal(size=16)
    regions = np.repeat(np.arange(3), 40)
    coords = np.column_stack([np.arange(120) * 256, np.zeros(120)]).astype(np.int64)
    for i in range(8):
        label = i % 2
        for name, noise in [("synthetic_a", 0.15), ("synthetic_b", 1.0)]:
            folder = args.output_dir / name
            folder.mkdir(parents=True, exist_ok=True)
            features = prototypes[regions] + (2 * label - 1) * task_direction + noise * rng.normal(size=(120, 16))
            with h5py.File(folder / f"slide_{i:03d}.h5", "w") as stream:
                stream["features"] = features.astype(np.float32)
                stream["coords"] = coords
                stream["sam_region"] = regions
                stream["label"] = np.int64(label)
                stream.attrs["synthetic"] = True
    print(f"Synthetic demo only: {args.output_dir}")


if __name__ == "__main__":
    main()
