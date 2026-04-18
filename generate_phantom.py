"""
generate_phantom.py — Create a synthetic multi-organ phantom NIfTI.
Produces a single segmentation file with labelled ellipsoids for each organ.
Lets you test AnatomyAtlas immediately without downloading CT data.

Usage:
    python generate_phantom.py --out ./phantom
    # Load phantom/segmentation.nii.gz in Slicer as a Segmentation node
"""
import argparse
import json
import numpy as np
import SimpleITK as sitk
from pathlib import Path

# Organ phantoms: (key, label_id, centre_ijk, semi-axes_ijk)
# Grid is 256×256×200 voxels at 1.5mm isotropic
PHANTOMS = [
    ("liver",               1,  (148, 128, 100), (45, 55, 35)),
    ("spleen",              2,  ( 80, 100, 105), (22, 28, 20)),
    ("right_kidney",        3,  (170, 100,  90), (15, 20, 28)),
    ("left_kidney",         4,  ( 86, 100,  90), (15, 20, 28)),
    ("aorta",               5,  (128, 128,  70), ( 6,  6, 80)),
    ("inferior_vena_cava",  6,  (138, 128,  70), ( 7,  7, 80)),
    ("stomach",             7,  (105, 140, 110), (28, 35, 22)),
    ("pancreas",            8,  (120, 130,  95), (18, 10, 10)),
    ("gallbladder",         9,  (152, 148,  98), (10,  8, 12)),
    ("urinary_bladder",    10,  (128, 128,  20), (22, 22, 18)),
    ("right_lung",         11,  (168, 100, 130), (32, 40, 55)),
    ("left_lung",          12,  ( 88, 100, 130), (32, 40, 55)),
]

SPACING = (1.5, 1.5, 1.5)
SHAPE   = (256, 256, 200)   # x, y, z


def make_ellipsoid(shape, centre, axes) -> np.ndarray:
    cx, cy, cz = centre
    ax, ay, az = axes
    x = np.arange(shape[0])
    y = np.arange(shape[1])
    z = np.arange(shape[2])
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    return (((X - cx) / ax) ** 2 +
            ((Y - cy) / ay) ** 2 +
            ((Z - cz) / az) ** 2) <= 1.0


def generate(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    volume = np.zeros(SHAPE, dtype=np.uint8)

    mapping = {}
    for key, label_id, centre, axes in PHANTOMS:
        mask = make_ellipsoid(SHAPE, centre, axes)
        volume[mask] = label_id
        mapping[key] = label_id
        print(f"  {key:30s} label={label_id}  voxels={mask.sum():,}")

    img = sitk.GetImageFromArray(volume.transpose(2, 1, 0))  # sitk z,y,x
    img.SetSpacing(SPACING)
    out_path = out_dir / "segmentation.nii.gz"
    sitk.WriteImage(img, str(out_path))
    print(f"\nSaved: {out_path}")

    # Save label map JSON for reference
    with open(out_dir / "label_map.json", "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"Label map: {out_dir / 'label_map.json'}")
    print("\nLoad in Slicer:")
    print(f"  File → Add Data → {out_path.resolve()}")
    print("  Choose 'Segmentation' as file type")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="./phantom")
    args = parser.parse_args()
    generate(Path(args.out))
