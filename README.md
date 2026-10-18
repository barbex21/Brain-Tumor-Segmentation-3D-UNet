# Brain Tumor Segmentation with 3D U-Net

A small research implementation of binary whole-tumor segmentation from aligned FLAIR, T1, T1GD/T1ce, and T2 MRI volumes using PyTorch.

> This is research code, not a medical device. It has not been validated for clinical use.

## What is included

- A compact 3D U-Net and BCE-Dice training objective.
- A local manifest-based NIfTI dataset loader with modality and mask alignment checks.
- Reproducible train/validation/test splitting.
- Threshold selection on the validation split, checkpoint selection by validation Dice, and one final evaluation on the held-out test split.

## Data privacy

MRI scans, manifests, patient or case identifiers, clinical tables, and model checkpoints are intentionally excluded from this repository. Obtain the dataset through its official distribution channel and store it outside the repository. Do not commit private inputs or generated checkpoints.

The loader expects a local CSV with these columns (absolute file paths are accepted):

```csv
flair,t1,t1gd,t2,seg
/path/to/case/FLAIR.nii.gz,/path/to/case/T1.nii.gz,/path/to/case/T1GD.nii.gz,/path/to/case/T2.nii.gz,/path/to/case/seg.nii.gz
```

The manifest is a local input and is not supplied here. Each case's four modalities and segmentation must already share the same voxel grid and affine. Resample them to a common grid before training if needed. All non-zero segmentation labels are treated as tumor, so the target is whole tumor rather than a specific subregion.

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Train and evaluate

```bash
python brain_tumor_ai_3.py --manifest /path/to/local_manifest.csv --epochs 5
```

On Windows, quote paths containing spaces:

```powershell
python brain_tumor_ai_3.py --manifest "C:\data\local_manifest.csv" --output "outputs\best_model.pth"
```

The defaults use a fixed seed, a 70/15/15 case-level split, a 96x96x96 center crop/pad, and threshold candidates from 0.1 through 0.9. The selected threshold is chosen using validation Dice only and stored with the best validation checkpoint. The held-out test split is evaluated once after training. Results depend on the dataset, split, preprocessing, hardware, and training settings; no performance claim is made by this repository.

Checkpoints are written under `outputs/`, which is git-ignored. Do not upload them unless you have separately checked their provenance and the applicable dataset and model-sharing terms.

## License

No license is currently provided. All rights are reserved by default until the repository owner adds an appropriate license.
