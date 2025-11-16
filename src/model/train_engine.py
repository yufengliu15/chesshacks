"""Command-line script to train the transformer chess engine."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import (
    ActionValueIterableDataset,
    VOCAB_SIZE,
    SEQUENCE_LENGTH,
)
from .transformer_model import ActionValueTransformer, TransformerConfig

DATA_DIR = Path(__file__).resolve().parents[1] / "train"
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)
CHECKPOINT_FILENAME = "action_value_transformer.pt"


def build_dataloader(
    data_dir: Path,
    *,
    max_records: int | None,
    batch_size: int,
    shuffle_files: bool,
    seed: int | None,
) -> DataLoader:
    bag_paths = sorted(data_dir.glob("action_value-*_data.bag"))
    if not bag_paths:
        raise FileNotFoundError(
            f"No Bagz shards matching 'action_value-*_data.bag' found in {data_dir}"
        )
    dataset = ActionValueIterableDataset(
        bag_paths,
        max_records=max_records,
        shuffle_files=shuffle_files,
        seed=seed,
    )
    return DataLoader(dataset, batch_size=batch_size)


def train(args: argparse.Namespace) -> Path:
    device = torch.device(args.device)

    config = TransformerConfig(
        vocab_size=VOCAB_SIZE,
        max_seq_len=SEQUENCE_LENGTH,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    )
    model = ActionValueTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    steps_per_epoch = max(
        1, (args.max_records or args.batch_size) // args.batch_size
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * steps_per_epoch
    )
    criterion = torch.nn.BCEWithLogitsLoss()

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / CHECKPOINT_FILENAME
    print(f"[checkpoint] Directory: {checkpoint_dir.resolve()}")

    global_step = 0
    start_epoch = 0

    if args.resume_from:
        resume_path = Path(args.resume_from)
        if resume_path.exists():
            # Always load to CPU first to avoid device compatibility issues
            checkpoint = torch.load(resume_path, map_location='cpu')
            model.load_state_dict(checkpoint["model_state"])
            if "optimizer_state" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state"])
            if "scheduler_state" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state"])
            meta = checkpoint.get("meta", {})
            global_step = meta.get("last_step", 0)
            start_epoch = meta.get("epochs_completed", 0)
            print(
                f"Resumed from {resume_path} at epoch {start_epoch}, step {global_step}."
            )
        else:
            print(f"Resume checkpoint {resume_path} not found; starting fresh.")

    total_epochs = start_epoch + args.epochs
    model.train()

    def _save_checkpoint(epochs_completed: int, step: int, reason: str | None = None) -> Path:
        payload = {
            "config": config.to_dict(),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "meta": {
                "epochs_requested": total_epochs,
                "epochs_completed": epochs_completed,
                "trained_at": time.time(),
                "max_records": args.max_records,
                "batch_size": args.batch_size,
                "last_step": step,
            },
        }
        torch.save(payload, checkpoint_path)
        message = (
            f"[checkpoint] Saved at step {step} "
            f"(epoch {epochs_completed}/{total_epochs}) -> {checkpoint_path}"
        )
        if reason:
            message += f" | reason: {reason}"
        print(message)
        return checkpoint_path

    epochs_completed = start_epoch
    try:
        for epoch in range(start_epoch, total_epochs):
            dataloader = build_dataloader(
                args.data_dir,
                max_records=args.max_records,
                batch_size=args.batch_size,
                shuffle_files=args.shuffle_files,
                seed=args.seed + epoch if args.seed is not None else None,
            )
            epoch_index = epoch - start_epoch + 1
            total_remaining = max(1, total_epochs - start_epoch)
            progress = tqdm(dataloader, desc=f"epoch {epoch_index}/{total_remaining}")
            for sequences, targets in progress:
                sequences = sequences.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                logits = model(sequences)
                loss = criterion(logits, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()

                global_step += 1
                if args.log_every and global_step % args.log_every == 0:
                    with torch.no_grad():
                        probs = torch.sigmoid(logits)
                        mae = torch.mean(torch.abs(probs - targets)).item()
                    progress.set_postfix(loss=loss.item(), mae=mae)

                if (
                    args.checkpoint_interval
                    and global_step % args.checkpoint_interval == 0
                ):
                    _save_checkpoint(epoch + 1, global_step, "periodic interval")

            epochs_completed = epoch + 1
    finally:
        saved_path = _save_checkpoint(epochs_completed, global_step, "final")

    return saved_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-records", type=int, default=100_000)
    parser.add_argument("--shuffle-files", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dim-feedforward", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=50,
        help="Number of gradient steps between checkpoint saves (0 disables periodic saves).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=CHECKPOINT_DIR,
        help="Directory where checkpoints should be written.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Optional checkpoint path to resume training from.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    ckpt = train(args)
    print(f"Checkpoint saved to {ckpt}")


if __name__ == "__main__":
    main()

