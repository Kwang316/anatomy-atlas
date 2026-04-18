# Anatomy Atlas — Interactive 3D Slicer Education Module

An interactive anatomy teaching tool built as a 3D Slicer scripted module.
Click any organ in the 3D view to see its name, volume, and clinical notes.
Toggle visibility by organ system. Quiz yourself with the built-in quiz mode.

## Features

- **15 organs** across 6 systems — digestive, vascular, respiratory, urinary, skeletal, lymphatic
- **Click-to-identify** — click any structure in the 3D view for instant info
- **System filters** — isolate organ groups with one click
- **Clinical notes** — concise radiology-relevant descriptions per organ
- **Fun facts** — memorable learning hooks
- **Quiz mode** — structures highlighted anonymously; type the name to score points
- **Volume display** — computed from the loaded segmentation

## Quick Start (synthetic phantom — no data download)

```bash
pip install SimpleITK numpy

# Generate a multi-organ phantom NIfTI
python generate_phantom.py --out ./phantom
```

Then in 3D Slicer:
1. File → Add Data → `phantom/segmentation.nii.gz` → load as **Segmentation**
2. Install module (see below) → open **Education → Anatomy Atlas**
3. Click organs, explore systems, start the quiz

## With Real CT (TotalSegmentator)

```bash
pip install TotalSegmentator SimpleITK

# Downloads a sample CT and runs segmentation (~5 min)
python download_totalsegmentator.py --out ./real_data
```

Load `real_data/segmentation.nii.gz` as a Segmentation node in Slicer.

## Install Module in Slicer

**Developer install (fastest — no build needed):**
1. Edit → Application Settings → Modules → Additional module paths
2. Add path: `/path/to/anatomy-atlas/AnatomyAtlas`
3. Restart Slicer
4. Find **"Anatomy Atlas"** under the **Education** category

**CMake build:**
```bash
mkdir build && cd build
cmake -DSlicer_DIR=/path/to/Slicer-build/Slicer-build ..
make
```

## Organ Systems Covered

| System | Organs |
|---|---|
| Digestive | Liver, Stomach, Pancreas, Gallbladder |
| Vascular | Aorta, IVC, Portal Vein |
| Urinary | Right Kidney, Left Kidney, Urinary Bladder |
| Respiratory | Right Lung, Left Lung |
| Skeletal | Vertebral Column, Ribs |
| Lymphatic | Spleen |
