"""
download_totalsegmentator.py
Fetches a small public CT from TCIA and runs TotalSegmentator to produce
a ready-to-use multi-organ segmentation for the Anatomy Atlas.

Requirements:
    pip install TotalSegmentator SimpleITK

Usage:
    python download_totalsegmentator.py --out ./real_data
"""
import argparse
import os
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

SAMPLE_CT_URL = (
    "https://zenodo.org/record/6802614/files/s0864_ct.nii.gz"
)


def download_sample_ct(dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    ct_path = dest / "ct.nii.gz"
    if ct_path.exists():
        print(f"  CT already downloaded: {ct_path}")
        return ct_path
    print(f"  Downloading sample CT (~80 MB)...")
    urllib.request.urlretrieve(SAMPLE_CT_URL, ct_path)
    print(f"  Saved: {ct_path}")
    return ct_path


def run_totalsegmentator(ct_path: Path, out_dir: Path) -> Path:
    try:
        import totalsegmentator
    except ImportError:
        print("Installing TotalSegmentator...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "TotalSegmentator"])

    seg_dir = out_dir / "segmentations"
    seg_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Running TotalSegmentator (fast mode, ~2–5 min on CPU)...")
    subprocess.check_call([
        sys.executable, "-m", "totalsegmentator",
        "-i", str(ct_path),
        "-o", str(seg_dir),
        "--fast",          # lower resolution, much faster
        "--statistics",
    ])
    return seg_dir


def merge_labels(seg_dir: Path, out_dir: Path) -> Path:
    """
    TotalSegmentator outputs one NIfTI per structure.
    Merge into a single multi-label file matching anatomy_data.json label IDs.
    """
    import SimpleITK as sitk
    import numpy as np
    import json

    label_map = {
        "liver.nii.gz":            1,
        "spleen.nii.gz":           2,
        "kidney_right.nii.gz":     3,
        "kidney_left.nii.gz":      4,
        "aorta.nii.gz":            5,
        "inferior_vena_cava.nii.gz": 6,
        "stomach.nii.gz":          7,
        "pancreas.nii.gz":         8,
        "gallbladder.nii.gz":      9,
        "urinary_bladder.nii.gz":  10,
        "lung_right.nii.gz":       11,
        "lung_left.nii.gz":        12,
        "vertebrae_L1.nii.gz":     13,
        "rib_left_1.nii.gz":       14,
        "portal_vein_and_splenic_vein.nii.gz": 15,
    }

    ref_img  = None
    combined = None

    for fname, label_id in label_map.items():
        fpath = seg_dir / fname
        if not fpath.exists():
            print(f"  [skip] {fname} not found")
            continue
        img = sitk.ReadImage(str(fpath))
        arr = sitk.GetArrayFromImage(img).astype(np.uint8)
        if combined is None:
            combined = np.zeros_like(arr)
            ref_img  = img
        combined[arr > 0] = label_id
        print(f"  Merged {fname} → label {label_id}")

    if combined is None:
        raise RuntimeError("No segmentation files found in output directory.")

    out_img = sitk.GetImageFromArray(combined)
    out_img.CopyInformation(ref_img)
    out_path = out_dir / "segmentation.nii.gz"
    sitk.WriteImage(out_img, str(out_path))
    print(f"\nMerged segmentation saved: {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="./real_data",
                        help="Output directory (default: ./real_data)")
    args = parser.parse_args()
    out  = Path(args.out)

    ct_path = download_sample_ct(out)
    seg_dir = run_totalsegmentator(ct_path, out)
    seg_out = merge_labels(seg_dir, out)

    print("\nDone! Load in Slicer:")
    print(f"  File → Add Data → {seg_out.resolve()}")
    print("  Choose 'Segmentation' as the file type")
