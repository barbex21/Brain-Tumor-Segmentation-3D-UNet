"""Train a reproducible binary 3D U-Net on a local UPenn-GBM manifest.

The manifest and MRI volumes are local inputs and must not be committed.
The target mask treats every non-zero label as tumor (whole-tumor task).
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset


MODALITY_COLUMNS = ("flair", "t1", "t1gd", "t2")
REQUIRED_COLUMNS = ("patient", *MODALITY_COLUMNS, "seg")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def crop_or_pad(array: np.ndarray, target: tuple[int, int, int]) -> np.ndarray:
    """Center crop/pad a D,H,W volume to target size."""
    tensor = torch.as_tensor(np.ascontiguousarray(array), dtype=torch.float32)
    slices = tuple(slice(max((n - t) // 2, 0), max((n - t) // 2, 0) + min(n, t))
                   for n, t in zip(tensor.shape, target))
    tensor = tensor[slices]
    pads = []
    for n, t in reversed(list(zip(tensor.shape, target))):
        before = max((t - n) // 2, 0)
        after = max(t - n - before, 0)
        pads.extend((before, after))
    tensor = F.pad(tensor, pads)
    return tensor.numpy()


class UPennGBMDataset(Dataset):
    """Load aligned FLAIR/T1/T1GD/T2 volumes and a whole-tumor mask."""

    def __init__(self, manifest: Path, target_size: tuple[int, int, int]):
        self.manifest = manifest
        self.target_size = target_size
        with manifest.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames:
                raise ValueError(f"Manifest is empty or has no header: {manifest}")
            missing = sorted(set(REQUIRED_COLUMNS) - set(reader.fieldnames))
            if missing:
                raise ValueError(f"Manifest is missing columns: {', '.join(missing)}")
            self.rows = list(reader)
        if not self.rows:
            raise ValueError(f"Manifest contains no cases: {manifest}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        paths = [Path(row[name]).expanduser() for name in (*MODALITY_COLUMNS, "seg")]
        images = []
        reference_shape = None
        for column, path in zip(MODALITY_COLUMNS, paths[:4]):
            if not path.is_file():
                raise FileNotFoundError(
                    f"Input file missing for manifest row {index + 2}, column '{column}'."
                )
            image = nib.load(str(path))
            volume = np.asarray(image.dataobj, dtype=np.float32)
            if volume.ndim != 3:
                raise ValueError(f"Expected a 3D volume in manifest row {index + 2}, column '{column}'.")
            if reference_shape is None:
                reference_shape = volume.shape
                reference_affine = image.affine
            elif volume.shape != reference_shape or not np.allclose(image.affine, reference_affine, atol=1e-3):
                raise ValueError(
                    f"Modalities are not aligned in manifest row {index + 2}; "
                    "resample them to a common grid before training."
                )
            if not np.isfinite(volume).all():
                raise ValueError(f"Non-finite MRI values in manifest row {index + 2}, column '{column}'.")
            volume = crop_or_pad(volume, self.target_size)
            nonzero = volume != 0
            mean = volume.mean()
            std = volume.std()
            volume = (volume - mean) / max(float(std), 1e-8)
            images.append(volume)

        if not paths[4].is_file():
            raise FileNotFoundError(f"Segmentation file missing for manifest row {index + 2}.")
        mask_img = nib.load(str(paths[4]))
        mask_volume = np.asarray(mask_img.dataobj, dtype=np.float32)
        if (mask_volume.shape != reference_shape or
                not np.allclose(mask_img.affine, reference_affine, atol=1e-3)):
            raise ValueError(
                f"Segmentation is not aligned with MRI in manifest row {index + 2}; "
                "resample it to the MRI grid before training."
            )
        mask = crop_or_pad((mask_volume > 0).astype(np.float32), self.target_size)
        x = torch.from_numpy(np.stack(images)).float()
        y = torch.from_numpy(mask[None]).float()
        return x, y


class TrainingAugmentation(Dataset):
    """Random sagittal/coronal flips and Gaussian noise, applied to training cases only."""

    def __init__(self, dataset: Dataset, noise_std: float, flip_probability: float):
        self.dataset = dataset
        self.noise_std = noise_std
        self.flip_probability = flip_probability

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, mask = self.dataset[index]
        for axis in (1, 2):
            if random.random() < self.flip_probability:
                image = torch.flip(image, dims=(axis,))
                mask = torch.flip(mask, dims=(axis,))
        if self.noise_std > 0:
            image = image + torch.randn_like(image) * self.noise_std
        return image, mask


class ConvBlock3d(nn.Module):
    """Convolution block matching the thesis model's checkpoint layout."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet3D(nn.Module):
    """The thesis architecture: 32/64/128 channels and BatchNorm3d."""

    def __init__(self, in_channels: int = 4, out_channels: int = 1, init_features: int = 32):
        super().__init__()
        self.enc1 = ConvBlock3d(in_channels, init_features)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = ConvBlock3d(init_features, init_features * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.bottleneck = ConvBlock3d(init_features * 2, init_features * 4)
        self.up2 = nn.ConvTranspose3d(
            init_features * 4, init_features * 2, kernel_size=2, stride=2
        )
        self.dec2 = ConvBlock3d(init_features * 4, init_features * 2)
        self.up1 = nn.ConvTranspose3d(
            init_features * 2, init_features, kernel_size=2, stride=2
        )
        self.dec1 = ConvBlock3d(init_features * 2, init_features)
        self.final = nn.Conv3d(init_features, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        bottleneck = self.bottleneck(self.pool2(e2))
        d2 = self.dec2(torch.cat((self.up2(bottleneck), e2), dim=1))
        d1 = self.dec1(torch.cat((self.up1(d2), e1), dim=1))
        return self.final(d1)



def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits).flatten(1)
    target = target.flatten(1)
    dice = (2 * (probs * target).sum(1) + eps) / (probs.sum(1) + target.sum(1) + eps)
    return 1 - dice.mean()


def case_dice(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    intersection = (prediction * target).sum().item()
    denominator = prediction.sum().item() + target.sum().item()
    return 1.0 if denominator == 0 else (2 * intersection + eps) / (denominator + eps)


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: torch.device,
             thresholds: list[float]) -> tuple[float, float, float]:
    model.eval()
    scores = {threshold: [] for threshold in thresholds}
    losses = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        losses.append(dice_loss(logits, y).item())
        probs = torch.sigmoid(logits)
        for threshold in thresholds:
            pred = (probs >= threshold).float()
            scores[threshold].extend(case_dice(p, t) for p, t in zip(pred, y))
    best_threshold = max(thresholds, key=lambda t: float(np.mean(scores[t])))
    best_dice = float(np.mean(scores[best_threshold]))
    return float(np.mean(losses)), best_dice, best_threshold


@torch.no_grad()
def evaluate_test(model: nn.Module, loader: DataLoader, device: torch.device,
                  threshold: float) -> float:
    model.eval()
    scores = []
    for x, y in loader:
        probs = torch.sigmoid(model(x.to(device)))
        pred = (probs >= threshold).float()
        scores.extend(case_dice(p, t.to(device)) for p, t in zip(pred, y))
    return float(np.mean(scores))


def grouped_split_indices(dataset: UPennGBMDataset, seed: int) -> tuple[list[int], list[int], list[int]]:
    """Split whole subjects so repeated visits cannot cross split boundaries."""
    patients: dict[str, list[int]] = {}
    for index, row in enumerate(dataset.rows):
        subject = row.get("patient", "").strip()
        if not subject:
            raise ValueError(f"Missing patient identifier in manifest row {index + 2}.")
        patients.setdefault(subject, []).append(index)
    subjects = list(patients)
    if len(subjects) < 3:
        raise ValueError("At least 3 unique patients are required for grouped train/validation/test splits.")

    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    n_train = max(1, int(0.70 * len(subjects)))
    n_val = max(1, int(0.15 * len(subjects)))
    if n_train + n_val >= len(subjects):
        n_train, n_val = len(subjects) - 2, 1
    train_subjects = set(subjects[:n_train])
    val_subjects = set(subjects[n_train:n_train + n_val])
    test_subjects = set(subjects[n_train + n_val:])

    def indices_for(selected: set[str]) -> list[int]:
        return [i for i, row in enumerate(dataset.rows)
                if row["patient"].strip() in selected]

    return indices_for(train_subjects), indices_for(val_subjects), indices_for(test_subjects)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Local CSV with patient,flair,t1,t1gd,t2,seg columns; never commit it.")
    parser.add_argument("--output", type=Path, default=Path("outputs/best_model.pth"))
    parser.add_argument("--metrics-output", type=Path, default=None,
                        help="Summary CSV path; defaults beside the checkpoint.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument("--noise-std", type=float, default=0.01,
                        help="Training-only Gaussian noise standard deviation.")
    parser.add_argument("--flip-probability", type=float, default=0.5,
                        help="Training-only probability for each spatial flip.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=[i / 10 for i in range(1, 10)])
    parser.add_argument("--volume-size", type=int, nargs=3, default=(96, 96, 96),
                        metavar=("DEPTH", "HEIGHT", "WIDTH"))
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        parser.error("--epochs, --batch-size, and --patience must be positive")
    if args.min_delta < 0 or args.noise_std < 0:
        parser.error("--min-delta and --noise-std must be non-negative")
    if not 0 <= args.flip_probability <= 1:
        parser.error("--flip-probability must be between 0 and 1")
    if any(size < 4 or size % 4 != 0 for size in args.volume_size):
        parser.error("each --volume-size dimension must be a positive multiple of 4")
    if len(args.thresholds) == 0 or any(t <= 0 or t >= 1 for t in args.thresholds):
        parser.error("--thresholds must be between 0 and 1")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = UPennGBMDataset(args.manifest, tuple(args.volume_size))
    train_indices, val_indices, test_indices = grouped_split_indices(dataset, args.seed)
    train_loader = DataLoader(
        TrainingAugmentation(Subset(dataset, train_indices), args.noise_std, args.flip_probability),
        batch_size=args.batch_size, shuffle=True,
    )
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=1, shuffle=False)
    test_loader = DataLoader(Subset(dataset, test_indices), batch_size=1, shuffle=False)

    model = UNet3D().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_val_loss = float("inf")
    best_val_dice = -1.0
    best_threshold = 0.5
    best_epoch = 0
    epochs_without_improvement = 0
    best_state = None
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = dice_loss(model(x), y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        val_loss, val_dice, threshold = validate(model, val_loader, device, args.thresholds)
        print(f"Epoch {epoch + 1}/{args.epochs}: train_dice_loss={np.mean(losses):.4f}, "
              f"val_dice_loss={val_loss:.4f}, val_dice={val_dice:.4f}, threshold={threshold:.2f}")
        if val_loss < best_val_loss - args.min_delta:
            best_val_loss = val_loss
            best_val_dice, best_threshold = val_dice, threshold
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
            torch.save({
                "model_state_dict": best_state,
                "validation_loss": best_val_loss,
                "validation_dice": best_val_dice,
                "selected_threshold": best_threshold,
                "best_epoch": best_epoch,
                "seed": args.seed,
                "volume_size": tuple(args.volume_size),
            }, args.output)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after {epoch + 1} epochs.")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")
    model.load_state_dict(best_state)
    test_dice = evaluate_test(model, test_loader, device, best_threshold)
    metrics_output = args.metrics_output or args.output.with_name(f"{args.output.stem}_metrics.csv")
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    with metrics_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("metric", "value"))
        writer.writerows((
            ("best_epoch", best_epoch),
            ("selected_threshold", f"{best_threshold:.4f}"),
            ("validation_loss", f"{best_val_loss:.6f}"),
            ("validation_dice", f"{best_val_dice:.6f}"),
            ("test_dice", f"{test_dice:.6f}"),
            ("train_cases", len(train_indices)),
            ("validation_cases", len(val_indices)),
            ("test_cases", len(test_indices)),
            ("seed", args.seed),
        ))
    print(f"Test Dice (evaluated once): {test_dice:.4f}")
    print(f"Checkpoint: {args.output}")
    print(f"Metrics: {metrics_output}")
    print("Patient/case identifiers are not printed or included in outputs.")

if __name__ == "__main__":
    main()
