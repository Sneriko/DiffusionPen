from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from feature_extractor import ImageEncoder
from historical_manifest_dataset import StyleEncoderManifestDataset


class MixedEncoder(nn.Module):
    def __init__(self, model_name: str, num_classes: int, pretrained: bool = True, trainable: bool = True):
        super().__init__()
        self.backbone = ImageEncoder(model_name=model_name, num_classes=num_classes, pretrained=pretrained, trainable=trainable)

    def forward(self, x):
        out = self.backbone(x)
        if isinstance(out, tuple) and len(out) == 2:
            return out
        raise ValueError("ImageEncoder is expected to return (logits, features) for style training.")


def run_epoch(loader, model, optimizer, device, triplet_loss_fn, ce_loss_fn, train: bool):
    model.train(train)
    total_loss = 0.0
    total_triplet = 0.0
    total_ce = 0.0
    n_correct = 0
    n_samples = 0

    pbar = tqdm(loader, leave=False)
    for batch in pbar:
        image = batch["image"].to(device)
        positive = batch["positive"].to(device)
        negative = batch["negative"].to(device)
        writer_label = batch["writer_label"].to(device)

        with torch.set_grad_enabled(train):
            logits, features = model(image)
            _, pos_features = model(positive)
            _, neg_features = model(negative)

            ce = ce_loss_fn(logits, writer_label)
            tri = triplet_loss_fn(features, pos_features, neg_features)
            loss = ce + tri

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        preds = logits.argmax(dim=1)
        bs = image.size(0)
        total_loss += loss.item() * bs
        total_triplet += tri.item() * bs
        total_ce += ce.item() * bs
        n_correct += (preds == writer_label).sum().item()
        n_samples += bs
        pbar.set_postfix(loss=total_loss / max(1, n_samples), acc=n_correct / max(1, n_samples))

    return {
        "loss": total_loss / max(1, n_samples),
        "triplet": total_triplet / max(1, n_samples),
        "ce": total_ce / max(1, n_samples),
        "acc": n_correct / max(1, n_samples),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--save-dir", default="./style_models_manifest")
    ap.add_argument("--model-name", default="mobilenetv2_100")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--image-height", type=int, default=96)
    ap.add_argument("--image-width", type=int, default=768)
    ap.add_argument("--grayscale", action="store_true")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_ds = StyleEncoderManifestDataset(
        manifest_path=args.manifest,
        split="train",
        image_height=args.image_height,
        image_width=args.image_width,
        grayscale=args.grayscale,
    )
    val_ds = StyleEncoderManifestDataset(
        manifest_path=args.manifest,
        split="val",
        image_height=args.image_height,
        image_width=args.image_width,
        grayscale=args.grayscale,
        writer_to_label=train_ds.writer_to_label,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    num_classes = len(train_ds.writer_to_label)
    print(f"Train rows: {len(train_ds)}")
    print(f"Val rows:   {len(val_ds)}")
    print(f"Writers:    {num_classes}")

    model = MixedEncoder(model_name=args.model_name, num_classes=num_classes, pretrained=True, trainable=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.1)
    triplet_loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)
    ce_loss_fn = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_path = save_dir / "best_style_encoder.pt"

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(train_loader, model, optimizer, device, triplet_loss_fn, ce_loss_fn, train=True)
        val_metrics = run_epoch(val_loader, model, optimizer, device, triplet_loss_fn, ce_loss_fn, train=False)
        scheduler.step(val_metrics["loss"])

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train loss={train_metrics['loss']:.4f} acc={train_metrics['acc']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} acc={val_metrics['acc']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "writer_to_label": train_ds.writer_to_label,
                    "num_classes": num_classes,
                    "image_height": args.image_height,
                    "image_width": args.image_width,
                    "grayscale": args.grayscale,
                    "model_name": args.model_name,
                },
                best_path,
            )
            print(f"Saved best checkpoint to {best_path}")


if __name__ == "__main__":
    main()
