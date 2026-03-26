from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm
import torchvision
from torchvision import transforms
from PIL import Image, ImageDraw
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CanineTokenizer, CanineModel

from feature_extractor import ImageEncoder
from historical_manifest_dataset import Stage2ManifestDataset
from unet import UNetModel


class AvgMeter:
    def __init__(self, name="metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(1, self.count)


class EMA:
    def __init__(self, beta=0.995):
        self.beta = beta
        self.step = 0

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            ma_params.data = self.update_average(ma_params.data, current_params.data)

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
        else:
            self.update_model_average(ema_model, model)
        self.step += 1


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=(96, 768), device="cuda:0"):
        self.noise_steps = noise_steps
        self.beta = torch.linspace(beta_start, beta_end, noise_steps, device=device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.img_size = img_size
        self.device = device


def setup_logging(save_path: str):
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, "models"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "images"), exist_ok=True)


def resolve_model_source(model_path: str) -> str:
    candidate = Path(model_path).expanduser().resolve()
    if candidate.exists():
        return str(candidate)
    if model_path.startswith("./"):
        return model_path[2:]
    return model_path


def save_images(images, path):
    grid = torchvision.utils.make_grid(images, padding=2)
    im = transforms.ToPILImage()(grid.cpu())
    im.save(path)
    return im


def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    image_tensor = image_tensor.detach().cpu()
    image_tensor = (image_tensor * 0.5 + 0.5).clamp(0, 1)
    return transforms.ToPILImage()(image_tensor)


def save_eval_bundle(generated, batch, path: Path):
    """
    Save a side-by-side eval preview with:
    [style reference | target source line image | generated sample]
    and write captions with prompt text and style-image source path.
    """
    bsz = generated.size(0)
    style_ref = batch["style_images"][:, 0]  # first style reference per sample
    target_src = batch["image"]
    texts = list(batch["transcription"])
    style_paths_raw = batch.get("style_image_paths", [[] for _ in range(bsz)])
    target_paths = batch.get("image_path", [""] * bsz)

    # Default DataLoader collate transposes list-valued fields:
    # list[list[str]] (per-sample) -> list[tuple[str, ...]] (per-style-slot).
    # Normalize back to per-sample lists for robust caption rendering.
    style_paths = [[] for _ in range(bsz)]
    if isinstance(style_paths_raw, (list, tuple)):
        if len(style_paths_raw) == bsz:
            for i in range(bsz):
                cur = style_paths_raw[i]
                if isinstance(cur, (list, tuple)):
                    style_paths[i] = [str(x) for x in cur]
                elif cur:
                    style_paths[i] = [str(cur)]
        elif len(style_paths_raw) > 0:
            transposed_rows = []
            for slot in style_paths_raw:
                if isinstance(slot, (list, tuple)):
                    transposed_rows.append(list(slot))
                else:
                    transposed_rows.append([slot])
            for i, per_sample in enumerate(zip(*transposed_rows)):
                if i >= bsz:
                    break
                style_paths[i] = [str(x) for x in per_sample if x]

    rows = []
    row_h = 0
    row_w = 0
    for i in range(bsz):
        style_im = tensor_to_pil(style_ref[i])
        target_im = tensor_to_pil(target_src[i])
        gen_im = transforms.ToPILImage()(generated[i].clamp(0, 1))
        if gen_im.mode != "RGB":
            gen_im = gen_im.convert("RGB")

        w, h = style_im.size
        strip = Image.new("RGB", (w * 3 + 8, h), "white")
        strip.paste(style_im, (0, 0))
        strip.paste(target_im, (w + 4, 0))
        strip.paste(gen_im.resize((w, h), resample=Image.Resampling.BILINEAR), (2 * w + 8, 0))
        rows.append(strip)
        row_h = h
        row_w = strip.size[0]

    caption_h = 64
    canvas = Image.new("RGB", (row_w, (row_h + caption_h) * bsz), "white")
    draw = ImageDraw.Draw(canvas)
    for i, strip in enumerate(rows):
        y0 = i * (row_h + caption_h)
        canvas.paste(strip, (0, y0))
        style_path = style_paths[i][0] if style_paths[i] else ""
        caption = (
            f'text="{texts[i]}" | style_src="{style_path}" | target_src="{target_paths[i]}" '
            "| cols=[style,target,generated]"
        )
        draw.text((4, y0 + row_h + 4), caption, fill="black")

    canvas.save(path)

    meta_path = path.with_suffix(".txt")
    with meta_path.open("w", encoding="utf-8") as f:
        for i in range(bsz):
            style_path = style_paths[i][0] if style_paths[i] else ""
            f.write(f"[{i}] text={texts[i]}\n")
            f.write(f"    style_source={style_path}\n")
            f.write(f"    target_source={target_paths[i]}\n")


def load_style_extractor(style_checkpoint: str, device: torch.device):
    ckpt = torch.load(style_checkpoint, map_location=device)
    num_classes = ckpt["num_classes"]
    model_name = ckpt.get("model_name", "mobilenetv2_100")
    feature_extractor = ImageEncoder(model_name=model_name, num_classes=num_classes, pretrained=False, trainable=False)
    feature_extractor.load_state_dict(ckpt["model_state_dict"], strict=False)
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()
    feature_extractor.requires_grad_(False)
    writer_to_label = ckpt["writer_to_label"]
    return feature_extractor, writer_to_label, ckpt


def build_writer_diverse_preview_batch(dataset, max_samples: int):
    """Build an eval preview batch with broad writer/hand coverage."""
    if len(dataset) == 0:
        raise ValueError("Cannot build preview batch from an empty dataset.")
    if max_samples <= 0:
        raise ValueError(f"max_samples must be > 0, got {max_samples}.")

    max_samples = min(max_samples, len(dataset))
    selected_indices = []

    for writer_id in dataset.writer_ids:
        if len(selected_indices) >= max_samples:
            break
        selected_indices.append(dataset.writer_to_indices[writer_id][0])

    if len(selected_indices) < max_samples:
        selected_set = set(selected_indices)
        for idx in range(len(dataset)):
            if len(selected_indices) >= max_samples:
                break
            if idx in selected_set:
                continue
            selected_indices.append(idx)

    return default_collate([dataset[idx] for idx in selected_indices])


def run_train_epoch(loader, model, ema, ema_model, vae, optimizer, mse_loss, noise_scheduler, style_extractor, tokenizer, device, latent=True):
    model.train()
    meter = AvgMeter("mse")
    for batch in tqdm(loader, leave=False):
        images = batch["image"].to(device)
        transcr = list(batch["transcription"])
        s_id = batch["writer_label"].to(device)
        style_images = batch["style_images"].to(device)

        text_features = tokenizer(transcr, padding="max_length", truncation=True, return_tensors="pt", max_length=256).to(device)

        with torch.no_grad():
            bsz, k, c, h, w = style_images.shape
            style_flat = style_images.view(bsz * k, c, h, w)
            style_out = style_extractor(style_flat)
            if isinstance(style_out, tuple):
                _, style_feat = style_out
            else:
                style_feat = style_out
            style_features = style_feat.view(bsz, k, -1).mean(dim=1)

            if latent:
                images = vae.encode(images.to(torch.float32)).latent_dist.sample() * 0.18215

        noise = torch.randn_like(images)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (images.size(0),), device=device).long()
        noisy_images = noise_scheduler.add_noise(images, noise, timesteps)

        predicted_noise = model(noisy_images, timesteps=timesteps, context=text_features, y=s_id, style_extractor=style_features)
        loss = mse_loss(predicted_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema.step_ema(ema_model, model)

        meter.update(loss.item(), images.size(0))
    return meter.avg


@torch.no_grad()
def initialize_unet_lazy_layers(loader, model, style_extractor, device):
    """
    Materialize UNet lazy parameters with a lightweight pass.
    We only initialize lazy layers that depend on style feature width to avoid an expensive
    full UNet forward (which can be quadratic in spatial token count and OOM at large widths).
    """
    batch = next(iter(loader))
    style_images = batch["style_images"].to(device)

    bsz, k, c, h, w = style_images.shape
    style_flat = style_images.view(bsz * k, c, h, w)
    style_out = style_extractor(style_flat)
    if isinstance(style_out, tuple):
        _, style_feat = style_out
    else:
        style_feat = style_out
    style_features = style_feat.view(bsz, k, -1).mean(dim=1)
    # `style_lin` is currently the only lazy layer in UNet that needs feature-shape materialization.
    style_lin = getattr(model, "style_lin", None)
    if isinstance(style_lin, nn.modules.lazy.LazyModuleMixin) and style_lin.has_uninitialized_params():
        _ = style_lin(style_features)


@torch.no_grad()
def sample_preview(ema_model, vae, batch, style_extractor, tokenizer, scheduler, device, latent=True):
    ema_model.eval()
    style_images = batch["style_images"].to(device)
    transcr = list(batch["transcription"])
    bsz, k, c, h, w = style_images.shape

    style_flat = style_images.view(bsz * k, c, h, w)
    style_out = style_extractor(style_flat)
    if isinstance(style_out, tuple):
        _, style_feat = style_out
    else:
        style_feat = style_out
    style_features = style_feat.view(bsz, k, -1).mean(dim=1)

    text_features = tokenizer(transcr, padding="max_length", truncation=True, return_tensors="pt", max_length=256).to(device)
    labels = batch["writer_label"].to(device)

    if latent:
        x = torch.randn((bsz, 4, h // 8, w // 8), device=device)
    else:
        x = torch.randn((bsz, 3, h, w), device=device)

    scheduler.set_timesteps(50)
    for t in scheduler.timesteps:
        tt = torch.full((bsz,), int(t.item()), device=device, dtype=torch.long)
        eps = ema_model(x, timesteps=tt, context=text_features, y=labels, style_extractor=style_features)
        x = scheduler.step(eps, t, x).prev_sample

    if latent:
        latents = x / 0.18215
        image = vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        return image.cpu()
    image = (x.clamp(-1, 1) + 1) / 2
    return image.cpu()


@torch.no_grad()
def sample_preview_from_checkpoint(
    ema_model,
    checkpoint_path: Path,
    vae,
    batch,
    style_extractor,
    tokenizer,
    scheduler,
    device,
    latent=True,
):
    state = torch.load(checkpoint_path, map_location=device)
    ema_model.load_state_dict(state)
    return sample_preview(ema_model, vae, batch, style_extractor, tokenizer, scheduler, device, latent=latent)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--style-checkpoint", required=True)
    ap.add_argument("--save-path", default="./diffusionpen_manifest_run")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--image-height", type=int, default=96)
    ap.add_argument("--image-width", type=int, default=768)
    ap.add_argument("--grayscale", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--stable-dif-path", default="stable-diffusion-v1-5")
    ap.add_argument("--latent", action="store_true")
    ap.add_argument("--emb-dim", type=int, default=320)
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--num-res-blocks", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--interpolation", action="store_true", help="Enable style interpolation path in UNet")
    ap.add_argument("--mix-rate", type=float, default=None, help="Interpolation mix rate when --interpolation is enabled")
    ap.add_argument(
        "--resume-path",
        default=None,
        help=(
            "Path to a previous checkpoint file (ckpt.pt) or a models directory containing "
            "ckpt.pt/ema_ckpt.pt/optim.pt/train_state.pt."
        ),
    )
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    setup_logging(args.save_path)

    style_extractor, writer_to_label, style_meta = load_style_extractor(args.style_checkpoint, device)

    train_ds = Stage2ManifestDataset(
        manifest_path=args.manifest,
        split="train",
        image_height=args.image_height,
        image_width=args.image_width,
        grayscale=args.grayscale,
        writer_to_label=writer_to_label,
        style_refs=5,
    )
    val_ds = Stage2ManifestDataset(
        manifest_path=args.manifest,
        split="val",
        image_height=args.image_height,
        image_width=args.image_width,
        grayscale=args.grayscale,
        writer_to_label=writer_to_label,
        style_refs=5,
    )

    num_classes = len(writer_to_label)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
    text_encoder = CanineModel.from_pretrained("google/canine-c")

    unet = UNetModel(
        image_size=(args.image_height, args.image_width),
        in_channels=4 if args.latent else 3,
        model_channels=args.emb_dim,
        out_channels=4 if args.latent else 3,
        num_res_blocks=args.num_res_blocks,
        attention_resolutions=(1, 1),
        channel_mult=(1, 1),
        num_heads=args.num_heads,
        num_classes=num_classes,
        context_dim=args.emb_dim,
        vocab_size=None,
        text_encoder=text_encoder,
        args=args,
    ).to(device)

    stable_dif_source = resolve_model_source(args.stable_dif_path)

    if args.latent:
        vae = AutoencoderKL.from_pretrained(stable_dif_source, subfolder="vae").to(device)
        vae.eval().requires_grad_(False)
    else:
        vae = None

    ddim = DDIMScheduler.from_pretrained(stable_dif_source, subfolder="scheduler")
    initialize_unet_lazy_layers(train_loader, unet, style_extractor, device)

    optimizer = optim.AdamW(unet.parameters(), lr=args.lr)
    mse_loss = nn.MSELoss()
    diffusion = Diffusion(img_size=(args.image_height, args.image_width), device=device)
    ema = EMA(0.995)
    ema_model = copy.deepcopy(unet).eval().requires_grad_(False)

    best_path = Path(args.save_path) / "models" / "ckpt.pt"
    ema_path = Path(args.save_path) / "models" / "ema_ckpt.pt"
    optim_path = Path(args.save_path) / "models" / "optim.pt"
    state_path = Path(args.save_path) / "models" / "train_state.pt"
    preview_path = Path(args.save_path) / "images"

    meta_path = Path(args.save_path) / "models" / "meta.pt"
    torch.save(
        {
            "writer_to_label": writer_to_label,
            "image_height": args.image_height,
            "image_width": args.image_width,
            "grayscale": args.grayscale,
        },
        meta_path,
    )

    preview_batch_size = min(args.batch_size, 8)
    start_epoch = 1

    if args.resume_path:
        resume_path = Path(args.resume_path)
        if resume_path.is_dir():
            model_resume = resume_path / "ckpt.pt"
            ema_resume = resume_path / "ema_ckpt.pt"
            optim_resume = resume_path / "optim.pt"
            state_resume = resume_path / "train_state.pt"
        else:
            model_resume = resume_path
            ema_resume = resume_path.with_name("ema_ckpt.pt")
            optim_resume = resume_path.with_name("optim.pt")
            state_resume = resume_path.with_name("train_state.pt")

        if not model_resume.exists():
            raise FileNotFoundError(f"Could not find model checkpoint to resume from: {model_resume}")

        model_state = torch.load(model_resume, map_location=device)
        if isinstance(model_state, dict) and "model_state_dict" in model_state:
            model_state = model_state["model_state_dict"]
        unet.load_state_dict(model_state)
        print(f"Resumed model weights from {model_resume}")

        if ema_resume.exists():
            ema_state = torch.load(ema_resume, map_location=device)
            if isinstance(ema_state, dict) and "model_state_dict" in ema_state:
                ema_state = ema_state["model_state_dict"]
            ema_model.load_state_dict(ema_state)
            print(f"Resumed EMA weights from {ema_resume}")
        else:
            ema_model.load_state_dict(unet.state_dict())
            print("EMA checkpoint not found; initialized EMA weights from resumed model.")

        if optim_resume.exists():
            optimizer.load_state_dict(torch.load(optim_resume, map_location=device))
            print(f"Resumed optimizer state from {optim_resume}")
        else:
            print("Optimizer checkpoint not found; optimizer will start fresh.")

        if state_resume.exists():
            train_state = torch.load(state_resume, map_location="cpu")
            if isinstance(train_state, dict):
                last_epoch = int(train_state.get("epoch", 0))
                start_epoch = max(1, last_epoch + 1)
                ema_step = train_state.get("ema_step")
                if ema_step is not None:
                    ema.step = int(ema_step)
            print(f"Resuming training from epoch {start_epoch}.")

    if len(val_ds) > 0:
        preview_batch = build_writer_diverse_preview_batch(val_ds, max_samples=preview_batch_size)
        sampled = sample_preview(
            ema_model,
            vae,
            preview_batch,
            style_extractor,
            tokenizer,
            ddim,
            device,
            latent=args.latent,
        )
        save_images(sampled, preview_path / "epoch_0000.png")
        save_eval_bundle(sampled, preview_batch, preview_path / "epoch_0000_bundle.png")
        print("Pre-training eval complete | saved epoch_0000 previews")

    if start_epoch > args.epochs:
        print(f"Resume epoch {start_epoch} is greater than --epochs={args.epochs}; nothing to train.")
        return

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = run_train_epoch(
            train_loader, unet, ema, ema_model, vae, optimizer, mse_loss, ddim, style_extractor, tokenizer, device, latent=args.latent
        )
        print(f"Epoch {epoch}/{args.epochs} | train mse={train_loss:.6f}")

        torch.save(unet.state_dict(), best_path)
        torch.save(ema_model.state_dict(), ema_path)
        torch.save(optimizer.state_dict(), optim_path)
        torch.save({"epoch": epoch, "ema_step": ema.step}, state_path)

        if len(val_ds) > 0:
            preview_batch = build_writer_diverse_preview_batch(val_ds, max_samples=preview_batch_size)
            sampled = sample_preview_from_checkpoint(
                ema_model,
                ema_path,
                vae,
                preview_batch,
                style_extractor,
                tokenizer,
                ddim,
                device,
                latent=args.latent,
            )
            save_images(sampled, preview_path / f"epoch_{epoch:04d}.png")
            save_eval_bundle(sampled, preview_batch, preview_path / f"epoch_{epoch:04d}_bundle.png")


if __name__ == "__main__":
    main()
