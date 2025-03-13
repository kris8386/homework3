import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast  # Mixed precision training
from torch.optim.lr_scheduler import ReduceLROnPlateau

# --- Make sure to import ConfusionMatrix ---
from .models import Detector, load_model, save_model
from .datasets.road_dataset import load_data
from .metrics import DetectionMetric, ConfusionMatrix  # <--- HERE

def train(
    exp_dir: str = "logs",
    model_name: str = "detector",
    num_epoch: int = 30,
    lr: float = 2e-3,
    batch_size: int = 128,
    seed: int = 2024,
    transform_pipeline: str = "default",
    **kwargs,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    model = load_model(model_name, **kwargs).to(device)
    model.train()

    # Load train/val data
    train_data = load_data(
        "drive_data/train",
        shuffle=True,
        batch_size=batch_size,
        num_workers=2,
        transform_pipeline=transform_pipeline,
    )
    val_data = load_data("drive_data/val", shuffle=False)
    # Define loss weights
    seg_loss_weight = 3.0

    # Define losses
    segmentation_loss =  nn.CrossEntropyLoss()  # Or use cross-entropy, etc.
    depth_loss = nn.L1Loss(reduction="mean")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5, verbose=True)

    global_step = 0
    metrics = {key: [] for key in ["train_seg_loss", "train_depth_loss", "val_seg_loss", "val_depth_loss", "train_iou_loss", "val_iou_loss"]}

    for epoch in range(num_epoch):
        # ---------------------
        # Training Loop
        # ---------------------
        model.train()
        for key in metrics:
            metrics[key].clear()

        for batch in train_data:
            img = batch["image"].to(device)
            seg_target = batch["track"].to(device)
            depth_target = batch["depth"].to(device)
            optimizer.zero_grad()

            seg_output, depth_output = model(img)
            loss_seg = segmentation_loss(seg_output, seg_target) *seg_loss_weight
            # loss_seg = segmentation_loss(seg_output, seg_target)
            loss_depth = depth_loss(torch.clamp(depth_output, 0, 1), depth_target)

            total_loss = loss_seg + loss_depth

            total_loss.backward()
            optimizer.step()

            metrics["train_seg_loss"].append(loss_seg.item())
            metrics["train_depth_loss"].append(loss_depth.item())

            logger.add_scalar("Loss/train_segmentation", loss_seg.item(), global_step)
            logger.add_scalar("Loss/train_depth", loss_depth.item(), global_step)
            global_step += 1

        # ---------------------
        # Validation Loop
        # ---------------------
        with torch.no_grad():
            model.eval()
            confusion_matrix = ConfusionMatrix(num_classes=3)  # For mIoU & accuracy
            detection_metric = DetectionMetric(num_classes=3)  # If also tracking detection/road-depth metrics

            val_depth_errors = []
            for batch in val_data:
                img = batch["image"].to(device)
                seg_target = batch["track"].to(device)
                depth_target = batch["depth"].to(device)

                seg_output, depth_output = model(img)
                loss_seg = segmentation_loss(seg_output, seg_target)
                loss_depth = depth_loss(torch.clamp(depth_output, 0, 1), depth_target)

                metrics["val_seg_loss"].append(loss_seg.item())
                metrics["val_depth_loss"].append(loss_depth.item())

                # Argmax to get predicted class labels
                seg_preds = seg_output.argmax(dim=1)  # (B, H, W)

                # Update confusion matrix (for mIoU)
                confusion_matrix.add(seg_preds, seg_target)

                # Optionally use your detection metric (if needed)
                detection_metric.add(seg_preds, seg_target, depth_output, depth_target)

                # Example: track overall depth MAE
                val_depth_errors.append(torch.abs(depth_output - depth_target).mean().item())

            # Compute confusion matrix metrics (mean IoU, overall accuracy)
            confusion_matrix_metrics = confusion_matrix.compute()
            miou = confusion_matrix_metrics["iou"]  # Mean IoU
            accuracy = confusion_matrix_metrics["accuracy"]  # Pixel accuracy (if you want to track)

            # Compute detection metrics
            computed_det_metrics = detection_metric.compute()
            mean_depth_error = computed_det_metrics["abs_depth_error"]  # Example from your detection metric
            lane_boundary_error = computed_det_metrics["tp_depth_error"]

            print(
                f"Epoch {epoch} "
                f"| mIoU: {miou:.4f} "
                f"| Acc: {accuracy:.4f} "
                f"| Depth MAE: {mean_depth_error:.4f} "
                f"| Lane Depth MAE: {lane_boundary_error:.4f}"
            )

        # ---------------------
        # End of Epoch Logging & Scheduler
        # ---------------------
        epoch_train_seg_loss = torch.tensor(metrics["train_seg_loss"]).mean()
        epoch_train_depth_loss = torch.tensor(metrics["train_depth_loss"]).mean()
        epoch_val_seg_loss = torch.tensor(metrics["val_seg_loss"]).mean()
        epoch_val_depth_loss = torch.tensor(metrics["val_depth_loss"]).mean()

        # Log the new metrics
        logger.add_scalar("Loss/val_segmentation", epoch_val_seg_loss, epoch)
        logger.add_scalar("Loss/val_depth", epoch_val_depth_loss, epoch)
        logger.add_scalar("Loss/train_segmentation", epoch_train_seg_loss, epoch)
        logger.add_scalar("Loss/train_depth", epoch_train_depth_loss, epoch)
        logger.add_scalar("Metrics/mIoU", miou, epoch)
        logger.add_scalar("Metrics/Accuracy", accuracy, epoch)
        logger.add_scalar("Metrics/Depth_MAE", mean_depth_error, epoch)
        logger.add_scalar("Metrics/Lane_Depth_MAE", lane_boundary_error, epoch)

        # Use accuracy to drive the learning rate scheduler
        scheduler.step(accuracy)
        confusion_matrix.reset()
        detection_metric.reset()   
        
    # ---------------------
    # Saving the Model
    # ---------------------
    save_model(model)
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--transform_pipeline", type=str, default="default", help="Specify data augmentation pipeline")
    train(**vars(parser.parse_args()))
