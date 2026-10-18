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
from torch.utils.data import DataLoader, Dataset, random_split


MODALITY_COLUMNS = ("flair", "t1", "t1gd", "t2")
REQUIRED_COLUMNS = (*MODALITY_COLUMNS, "seg")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"Input file not found (manifest row {index + 2}): {path}")

        images = []
        reference_shape = None
        for path in paths[:4]:
            image = nib.load(str(path))
            volume = np.asarray(image.dataobj, dtype=np.float32)
            if reference_shape is None:
                reference_shape = volume.shape
                reference_affine = image.affine
            elif volume.shape != reference_shape or not np.allclose(image.affine, reference_affine, atol=1e-3):
                raise ValueError(
                    f"Modalities are not aligned in manifest row {index + 2}; "
                    "resample them to a common grid before training."
                )
            if not np.isfinite(volume).all():
                raise ValueError(f"Non-finite MRI values in manifest row {index + 2}: {path.name}")
            volume = crop_or_pad(volume, self.target_size)
            nonzero = volume != 0
            if nonzero.any():
                mean = volume[nonzero].mean()
                std = volume[nonzero].std()
                volume[nonzero] = (volume[nonzero] - mean) / max(float(std), 1e-8)
            images.append(volume)

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


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SmallUNet3D(nn.Module):
    def __init__(self, in_channels: int = 4, base_channels: int = 8):
        super().__init__()
        self.down1 = DoubleConv(in_channels, base_channels)
        self.pool1 = nn.MaxPool3d(2)
        self.down2 = DoubleConv(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.bottom = DoubleConv(base_channels * 2, base_channels * 4)
        self.up2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.conv2 = DoubleConv(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose3d(base_channels * 2, base_channels, 2, stride=2)
        self.conv1 = DoubleConv(base_channels * 2, base_channels)
        self.out_conv = nn.Conv3d(base_channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1 = self.down1(x)
        c2 = self.down2(self.pool1(c1))
        bottom = self.bottom(self.pool2(c2))
        x = self.conv2(torch.cat((self.up2(bottom), c2), dim=1))
        x = self.conv1(torch.cat((self.up1(x), c1), dim=1))
        return self.out_conv(x)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits).flatten(1)
    target = target.flatten(1)
    dice = (2 * (probs * target).sum(1) + eps) / (probs.sum(1) + target.sum(1) + eps)
    return 1 - dice.mean()


def combined_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 0.5 * F.binary_cross_entropy_with_logits(logits, target) + 0.5 * dice_loss(logits, target)


def case_dice(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    intersection = (prediction * target).sum().item()
    denominator = prediction.sum().item() + target.sum().item()
    return 1.0 if denominator == 0 else (2 * intersection + eps) / (denominator + eps)


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: torch.device,
             thresholds: list[float]) -> tuple[float, float]:
    model.eval()
    scores = {threshold: [] for threshold in thresholds}
    losses = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        losses.append(combined_loss(logits, y).item())
        probs = torch.sigmoid(logits)
        for threshold in thresholds:
            pred = (probs >= threshold).float()
            scores[threshold].extend(case_dice(p, t) for p, t in zip(pred, y))
    best_threshold = max(thresholds, key=lambda t: float(np.mean(scores[t])))
    return float(np.mean(scores[best_threshold])), best_threshold


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Local CSV with flair,t1,t1gd,t2,seg path columns; never commit it.")
    parser.add_argument("--output", type=Path, default=Path("outputs/best_model.pth"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=[i / 10 for i in range(1, 10)])
    parser.add_argument("--volume-size", type=int, nargs=3, default=(96, 96, 96),
                        metavar=("DEPTH", "HEIGHT", "WIDTH"))
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    if len(args.thresholds) == 0 or any(t <= 0 or t >= 1 for t in args.thresholds):
        parser.error("--thresholds must be between 0 and 1")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = UPennGBMDataset(args.manifest, tuple(args.volume_size))
    if len(dataset) < 3:
        raise ValueError("At least 3 cases are required for train/validation/test splits.")
    n_train = max(1, int(0.70 * len(dataset)))
    n_val = max(1, int(0.15 * len(dataset)))
    n_test = len(dataset) - n_train - n_val
    if n_test < 1:
        n_train, n_val, n_test = len(dataset) - 2, 1, 1
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    model = SmallUNet3D().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_val_dice = -1.0
    best_threshold = 0.5
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = combined_loss(model(x), y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        val_dice, threshold = validate(model, val_loader, device, args.thresholds)
        print(f"Epoch {epoch + 1}/{args.epochs}: train_loss={np.mean(losses):.4f}, "
              f"val_dice={val_dice:.4f}, threshold={threshold:.2f}")
        if val_dice > best_val_dice:
            best_val_dice, best_threshold = val_dice, threshold
            torch.save({
                "model_state_dict": model.state_dict(),
                "validation_dice": best_val_dice,
                "selected_threshold": best_threshold,
                "seed": args.seed,
                "volume_size": tuple(args.volume_size),
            }, args.output)

    checkpoint = torch.load(args.output, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_dice = evaluate_test(model, test_loader, device, checkpoint["selected_threshold"])
    print(f"Test Dice (evaluated once): {test_dice:.4f}")
    print(f"Checkpoint: {args.output}")
    print("Patient/case identifiers are not printed or included in the checkpoint.")


if __name__ == "__main__":
    main()
