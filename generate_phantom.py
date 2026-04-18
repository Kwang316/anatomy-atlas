"""
generate_phantom.py — Create a synthetic multi-organ phantom NIfTI.
Produces a segmentation file with all 15 labelled structures.
Organs placed later overwrite earlier ones, so background structures
(ribs, vertebrae) are written FIRST and soft organs on top.

Usage:
    python generate_phantom.py --out ./phantom
    # Load phantom/segmentation.nii.gz in Slicer as a Segmentation node
"""
import argparse
import json
import numpy as np
import SimpleITK as sitk
from pathlib import Path

# Grid: 256×256×200 voxels at 1.5 mm isotropic
# Physical size ≈ 384 × 384 × 300 mm  (roughly a human torso)
# Axes: x=left-right, y=anterior-posterior, z=superior-inferior
SPACING = (1.5, 1.5, 1.5)
SHAPE   = (256, 256, 200)   # x, y, z

# Voxel volume in mL
_VX_ML = (1.5 ** 3) / 1000.0   # = 0.003375 mL


# ---------------------------------------------------------------------------
# Primitive builders
# ---------------------------------------------------------------------------

def _grid():
    x = np.arange(SHAPE[0])
    y = np.arange(SHAPE[1])
    z = np.arange(SHAPE[2])
    return np.meshgrid(x, y, z, indexing="ij")


def ellipsoid(cx, cy, cz, ax, ay, az) -> np.ndarray:
    X, Y, Z = _grid()
    return (((X-cx)/ax)**2 + ((Y-cy)/ay)**2 + ((Z-cz)/az)**2) <= 1.0


def hollow_ellipsoid(cx, cy, cz, ax, ay, az, thickness=6) -> np.ndarray:
    """Shell = outer ellipsoid minus inner ellipsoid."""
    outer = ellipsoid(cx, cy, cz, ax,            ay,            az)
    inner = ellipsoid(cx, cy, cz, max(1,ax-thickness), max(1,ay-thickness), max(1,az-thickness))
    return outer & ~inner


def tube(cx, cy, cz_start, cz_end, rx, ry) -> np.ndarray:
    """Vertical elliptical cylinder between two z-planes."""
    X, Y, Z = _grid()
    in_circle = ((X-cx)/rx)**2 + ((Y-cy)/ry)**2 <= 1.0
    in_zrange  = (Z >= cz_start) & (Z <= cz_end)
    return in_circle & in_zrange


# ---------------------------------------------------------------------------
# Organ definitions  (written in this order — later entries overwrite earlier)
# Background skeleton written first, soft organs on top.
# ---------------------------------------------------------------------------

def build_organs() -> list[tuple]:
    """
    Returns list of (key, label_id, mask_fn) in paint order.
    Background structures first → soft organs last.
    """
    # Ribs: hollow shell around thorax (z=100–185)
    ribs_outer = ellipsoid(128, 118, 145, 80, 72, 44)
    ribs_inner = ellipsoid(128, 118, 145, 68, 60, 36)
    ribs_zband = (np.arange(SHAPE[2]) >= 100) & (np.arange(SHAPE[2]) <= 185)
    ribs_zband = np.broadcast_to(ribs_zband, SHAPE)
    ribs_mask  = (ribs_outer & ~ribs_inner) & ribs_zband

    return [
        # --- background skeleton (painted first) ---
        ("ribs",               14, ribs_mask),
        ("vertebrae",          13, tube(128, 168, 30, 185, 8, 10)),

        # --- vascular tubes ---
        ("aorta",               5, tube(122, 155,  25, 180,  5,  5)),
        ("inferior_vena_cava",  6, tube(133, 148,  25, 175,  5,  6)),

        # --- thoracic organs ---
        ("right_lung",         11, ellipsoid(175, 100, 145, 34, 42, 44)),
        ("left_lung",          12, ellipsoid( 81, 100, 145, 34, 42, 44)),

        # --- abdominal organs (largest first) ---
        ("liver",               1, ellipsoid(158, 112,  78, 42, 52, 30)),
        ("stomach",             7, ellipsoid(102, 130,  95, 26, 32, 20)),
        ("spleen",              2, ellipsoid( 78,  98, 100, 20, 25, 18)),
        ("right_kidney",        3, ellipsoid(172,  140,  82, 13, 16, 24)),
        ("left_kidney",         4, ellipsoid( 84,  140,  82, 13, 16, 24)),
        ("urinary_bladder",    10, ellipsoid(128, 120,  22, 20, 20, 16)),

        # --- small organs painted last so they're not overwritten ---
        ("pancreas",            8, ellipsoid(118, 138,  88, 16,  9,  9)),
        ("gallbladder",         9, ellipsoid(155, 142,  83,  9,  7, 10)),
        ("portal_vein",        15, tube(142, 138, 72, 98, 4, 4)),
    ]


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------

def generate(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    volume  = np.zeros(SHAPE, dtype=np.uint8)
    mapping = {}
    organs  = build_organs()

    print(f"{'Organ':<25} {'Label':>5}  {'Voxels':>10}  {'Volume (mL)':>11}")
    print("-" * 60)

    for key, label_id, mask in organs:
        volume[mask] = label_id
        mapping[key] = label_id

    # Count voxels AFTER all painting (respects overwrite order)
    for key, label_id, _ in organs:
        voxels = int((volume == label_id).sum())
        vol_ml = voxels * _VX_ML
        print(f"  {key:<23} {label_id:>5}  {voxels:>10,}  {vol_ml:>9.0f} mL")

    img = sitk.GetImageFromArray(volume.transpose(2, 1, 0))
    img.SetSpacing(SPACING)
    out_path = out_dir / "segmentation.nii.gz"
    sitk.WriteImage(img, str(out_path))

    with open(out_dir / "label_map.json", "w") as f:
        json.dump(mapping, f, indent=2)

    total_labelled = int((volume > 0).sum())
    print(f"\nTotal labelled: {total_labelled:,} voxels  ({total_labelled * _VX_ML:.0f} mL)")
    print(f"Saved: {out_path}")
    print("\nLoad in Slicer:")
    print(f"  File → Add Data → {out_path.resolve()}")
    print("  Choose 'Segmentation' as file type")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="./phantom")
    args = parser.parse_args()
    generate(Path(args.out))
