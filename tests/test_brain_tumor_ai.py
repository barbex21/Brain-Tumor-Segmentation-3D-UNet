from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest
import torch

from brain_tumor_ai_3 import (
    UPennGBMDataset,
    UNet3D,
    case_dice,
    crop_or_pad,
    grouped_split_indices,
)


def test_crop_or_pad_centers_crop_and_padding():
    volume = np.arange(6**3, dtype=np.float32).reshape(6, 6, 6)
    cropped = crop_or_pad(volume, (4, 4, 4))
    assert cropped.shape == (4, 4, 4)
    np.testing.assert_array_equal(cropped, volume[1:5, 1:5, 1:5])

    padded = crop_or_pad(np.ones((2, 4, 6), dtype=np.float32), (4, 4, 8))
    assert padded.shape == (4, 4, 8)
    assert np.count_nonzero(padded) == 2 * 4 * 6
    assert np.count_nonzero(padded[[0, -1]]) == 0
    assert np.count_nonzero(padded[:, :, [0, -1]]) == 0


def test_dice_empty_masks_and_perfect_overlap():
    empty = torch.zeros((1, 4, 4, 4))
    full = torch.ones((1, 4, 4, 4))
    assert case_dice(empty, empty) == 1.0
    assert case_dice(full, full) == pytest.approx(1.0)


def test_grouped_split_keeps_repeated_visits_together():
    rows = [{"patient": f"subject-{i:02d}"} for i in range(12) for _ in range(2)]
    dataset = SimpleNamespace(rows=rows)

    splits = grouped_split_indices(dataset, seed=17)
    repeated = grouped_split_indices(dataset, seed=17)
    assert splits == repeated
    assert sorted(i for split in splits for i in split) == list(range(len(rows)))

    patient_sets = [
        {rows[i]["patient"] for i in split}
        for split in splits
    ]
    assert patient_sets[0].isdisjoint(patient_sets[1])
    assert patient_sets[0].isdisjoint(patient_sets[2])
    assert patient_sets[1].isdisjoint(patient_sets[2])


def test_dataset_reads_aligned_synthetic_nifti_case(tmp_path: Path):
    volume = np.arange(6**3, dtype=np.float32).reshape(6, 6, 6)
    mask = np.zeros((6, 6, 6), dtype=np.uint8)
    mask[2:4, 2:4, 2:4] = 1
    affine = np.eye(4)
    filenames = ("flair.nii.gz", "t1.nii.gz", "t1gd.nii.gz", "t2.nii.gz", "seg.nii.gz")
    for name, array in zip(filenames[:4], (volume, volume + 1, volume + 2, volume + 3)):
        nib.save(nib.Nifti1Image(array, affine), tmp_path / name)
    nib.save(nib.Nifti1Image(mask, affine), tmp_path / filenames[4])

    manifest = tmp_path / "synthetic_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("patient", "flair", "t1", "t1gd", "t2", "seg"))
        writer.writerow(("synthetic-subject", *(str(tmp_path / name) for name in filenames)))

    x, y = UPennGBMDataset(manifest, (4, 4, 4))[0]
    assert x.shape == (4, 4, 4, 4)
    assert y.shape == (1, 4, 4, 4)
    assert set(torch.unique(y).tolist()) == {0.0, 1.0}


def test_grouped_split_rejects_missing_subject_identifier():
    dataset = SimpleNamespace(rows=[{"patient": ""}, {"patient": "subject-2"}, {"patient": "subject-3"}])
    with pytest.raises(ValueError, match="Missing patient identifier"):
        grouped_split_indices(dataset, seed=1)

def test_unet3d_matches_thesis_checkpoint_layout():
    state = UNet3D().state_dict()
    assert state["enc1.conv.0.weight"].shape == (32, 4, 3, 3, 3)
    assert state["enc1.conv.1.running_mean"].shape == (32,)
    assert state["bottleneck.conv.0.weight"].shape == (128, 64, 3, 3, 3)
    assert "final.weight" in state
    assert not any(name.startswith("down1.") for name in state)
