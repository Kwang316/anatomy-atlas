"""
download_totalsegmentator.py
Downloads a real abdominal CT from TCIA (Creative Commons licence, no login
needed) and runs TotalSegmentator to produce a ready-to-use multi-label
segmentation for the Anatomy Atlas viewer.

Requirements:
    pip install TotalSegmentator SimpleITK

Usage:
    python download_totalsegmentator.py --out ./real_data
    python viewer.py --phantom ./real_data
"""
import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk

# ---------------------------------------------------------------------------
# TotalSegmentator v2 label ID → anatomy_data.json label ID
# ---------------------------------------------------------------------------
# Built from class_map['total'] in totalsegmentator.map_to_binary
_TS_TO_ANATOMY = {
    1:  2,   # spleen
    2:  3,   # kidney_right
    3:  4,   # kidney_left
    4:  9,   # gallbladder
    5:  1,   # liver
    6:  7,   # stomach
    7:  8,   # pancreas
    10: 12,  # lung_upper_lobe_left  → left lung (12)
    11: 12,  # lung_lower_lobe_left  → left lung
    12: 11,  # lung_upper_lobe_right → right lung (11)
    13: 11,  # lung_middle_lobe_right
    14: 11,  # lung_lower_lobe_right
    21: 10,  # urinary_bladder
    52:  5,  # aorta
    63:  6,  # inferior_vena_cava
    64: 15,  # portal_vein_and_splenic_vein
}
# vertebrae C1–T12–L5–S1 (labels 26–50) → vertebral column (13)
for _ts in range(26, 51):
    _TS_TO_ANATOMY[_ts] = 13
# left ribs 1–12 (92–103) + right ribs 1–12 (104–115) → ribs (14)
for _ts in range(92, 116):
    _TS_TO_ANATOMY[_ts] = 14


# ---------------------------------------------------------------------------
# CT download — smallest Pancreas-CT series from TCIA (CC BY 3.0, no login)
# ---------------------------------------------------------------------------
_TCIA_SERIES_UID = (
    "1.2.826.0.1.3680043.2.1125.1.41202274843063370955090296887703130"
)
_TCIA_API = "https://services.cancerimagingarchive.net/nbia-api/services/v1"


def download_ct(out_dir: Path) -> Path:
    ct_path = out_dir / "ct.nii.gz"
    if ct_path.exists():
        print(f"  CT already present: {ct_path}")
        return ct_path

    out_dir.mkdir(parents=True, exist_ok=True)
    dicom_dir = out_dir / "dicom"
    dicom_dir.mkdir(exist_ok=True)

    url = f"{_TCIA_API}/getImage?SeriesInstanceUID={_TCIA_SERIES_UID}"
    print("  Downloading PANCREAS_0080 from TCIA (CC BY 3.0) ~90 MB …")
    zip_bytes, _ = urllib.request.urlretrieve(url,
                                              reporthook=_progress_hook())
    print()

    print("  Extracting DICOM …")
    with zipfile.ZipFile(zip_bytes) as zf:
        zf.extractall(dicom_dir)

    print("  Converting DICOM → NIfTI …")
    reader = sitk.ImageSeriesReader()
    dcm_names = reader.GetGDCMSeriesFileNames(str(dicom_dir))
    reader.SetFileNames(dcm_names)
    img = reader.Execute()

    # Save in LPS orientation
    img = sitk.DICOMOrient(img, "LPS")
    sitk.WriteImage(img, str(ct_path))
    print(f"  CT saved: {ct_path}  shape={sitk.GetArrayFromImage(img).shape}")
    return ct_path


def _progress_hook():
    prev = [0]

    def hook(count, block_size, total_size):
        mb_done = count * block_size / 1024 / 1024
        if int(mb_done) != prev[0]:
            prev[0] = int(mb_done)
            suffix = f"/ {total_size/1024/1024:.0f} MB" if total_size > 0 else "MB"
            print(f"\r  {mb_done:.0f} {suffix}", end="", flush=True)

    return hook


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def segment_ct(ct_path: Path, out_dir: Path) -> Path:
    """Run TotalSegmentator and remap labels to anatomy_data.json IDs."""
    import nibabel as nib
    from totalsegmentator.python_api import totalsegmentator

    seg_path = out_dir / "segmentation.nii.gz"
    if seg_path.exists():
        print(f"  Segmentation already present: {seg_path}")
        return seg_path

    print("  Running TotalSegmentator (fast mode, CPU — ~5–15 min) …")
    print("  Model weights will be downloaded on first run (~1 GB).")
    ct_img = nib.load(str(ct_path))
    seg_img = totalsegmentator(ct_img, output=None, ml=True,
                               fast=True, device="cpu", quiet=False)

    print("  Remapping TotalSegmentator labels → anatomy_data labels …")
    ts_arr   = np.asarray(seg_img.dataobj).astype(np.uint16)
    ana_arr  = np.zeros_like(ts_arr, dtype=np.uint8)
    mapped   = {}
    skipped  = set()
    for ts_id, ana_id in _TS_TO_ANATOMY.items():
        mask = ts_arr == ts_id
        if mask.any():
            ana_arr[mask] = ana_id
            mapped[ts_id] = ana_id
        else:
            skipped.add(ts_id)

    if skipped:
        print(f"  [info] {len(skipped)} TS labels not found in scan (normal for partial FOV)")
    for ts_id, ana_id in sorted(mapped.items()):
        from totalsegmentator.map_to_binary import class_map
        name = class_map['total'].get(ts_id, f"label_{ts_id}")
        vox  = int((ana_arr == ana_id).sum())
        print(f"    TS {ts_id:3d} {name:<35} → anatomy {ana_id}  ({vox:,} voxels)")

    out_img = nib.Nifti1Image(ana_arr, seg_img.affine, seg_img.header)
    out_img.header.set_data_dtype(np.uint8)
    nib.save(out_img, str(seg_path))
    print(f"\n  Segmentation saved: {seg_path}")
    return seg_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="./real_data",
                        help="Output directory (default: ./real_data)")
    args = parser.parse_args()
    out  = Path(args.out)

    ct_path  = download_ct(out)
    seg_path = segment_ct(ct_path, out)

    print(f"\nDone! Launch the viewer:")
    print(f"  python viewer.py --phantom {out}")
