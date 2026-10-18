# Brain Tumor Segmentation with 3D U-Net

A small research implementation of binary whole-tumor segmentation from aligned FLAIR, T1, T1GD/T1ce, and T2 MRI volumes using PyTorch.

> This is research code, not a medical device. It has not been validated for clinical use.

## What is included

- The thesis-matching 3D U-Net architecture (32/64/128 channels, BatchNorm3d) and Dice loss.
- A local manifest-based NIfTI dataset loader with modality and mask alignment checks.
- Seeded, patient-grouped train/validation/test splitting to prevent visits from the same subject crossing splits.
- Threshold selection on the validation split, checkpoint selection by validation Dice, and one final evaluation on the held-out test split.

## Data privacy

MRI scans, manifests, patient or case identifiers, clinical tables, and model checkpoints are intentionally excluded from this repository. Obtain the dataset through its official distribution channel and store it outside the repository. Do not commit private inputs or generated checkpoints.

The loader expects a local CSV with a `patient` column and these modality/label columns (absolute file paths are accepted):

```csv
patient,flair,t1,t1gd,t2,seg
synthetic-subject,/path/to/case/FLAIR.nii.gz,/path/to/case/T1.nii.gz,/path/to/case/T1GD.nii.gz,/path/to/case/T2.nii.gz,/path/to/case/seg.nii.gz
```

The manifest is a local input and is not supplied here. Its subject IDs are used only to keep all visits from one subject in the same split; they are not printed or written to outputs. Each case's four modalities and segmentation must already share the same voxel grid and affine. Resample them to a common grid before training if needed. All non-zero segmentation labels are treated as tumor, so the target is whole tumor rather than a specific subregion.

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
python brain_tumor_ai_3.py --manifest /path/to/local_manifest.csv
```

On Windows, quote paths containing spaces:

```powershell
python brain_tumor_ai_3.py --manifest "C:\data\local_manifest.csv" --output "outputs\best_model.pth"
```

The defaults follow the thesis protocol where specified: seed 42, a patient-level 70/15/15 split, 96x96x96 center crop/pad, Dice loss, Adam at 1e-4, batch size 2, up to 100 epochs, and early stopping on validation loss (patience 10, min-delta 0.001). Training-only augmentation applies sagittal/coronal flips and Gaussian noise (the default standard deviation of 0.01 is a repo setting because the thesis does not state its exact value). Threshold candidates from 0.1 through 0.9 are compared on validation Dice; the selected checkpoint is the one with the best validation loss. The held-out test split is evaluated once. A summary CSV records the selected threshold, validation/test metrics, split sizes, and seed without case identifiers. This is a documented-method implementation, not a bit-for-bit reproduction of the original Colab run; results depend on the data ordering, preprocessing, hardware, and settings.

Checkpoints and summary CSV files are written under `outputs/`, which is git-ignored. Do not upload checkpoints unless you have separately checked their provenance and the applicable dataset and model-sharing terms.

## Tests

The synthetic tests also check that the network's state-dict layer names and channel shapes match the thesis checkpoint architecture. Install development dependencies and run the tests:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The tests use generated toy volumes only; they do not download or require UPENN-GBM scans.

## Dataset citation and license

The UPENN-GBM collection is distributed by The Cancer Imaging Archive (TCIA) under CC BY 4.0, and TCIA requires a data citation. Cite the source dataset when using it:

> Bakas, S., Sako, C., Akbari, H., Bilello, M., Sotiras, A., Shukla, G., Rudie, J. D., Flores Santamaria, N., Fathi Kazerooni, A., Pati, S., Rathore, S., Mamourian, E., Ha, S. M., Parker, W., Doshi, J., Baid, U., Bergman, M., Binder, Z. A., Verma, R., … Davatzikos, C. (2021). *Multi-parametric magnetic resonance imaging (mpMRI) scans for de novo Glioblastoma (GBM) patients from the University of Pennsylvania Health System (UPENN-GBM)* (Version 2) [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/TCIA.709X-DN49

See the [official UPENN-GBM collection page](https://www.cancerimagingarchive.net/collection/upenn-gbm/) for the full attribution, current access instructions, and usage policy.

## License

No license is currently provided. All rights are reserved by default until the repository owner adds an appropriate license.
