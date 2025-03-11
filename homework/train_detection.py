import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb
import torch.nn as nn

from .models import Detector, load_model, save_model
from .datasets.road_dataset import load_data
from .metrics import DetectionMetric, ConfusionMatrix

def train(
    exp_dir: str = "logs",
    model_name: str = "detector",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
    transform_pipeline: str = "default",
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Set up logging directory
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # Load model and move to device
    model = load_model(model_name, **kwargs)  # Ensure the model uses the provided Detector structure
    model = model.to(device)
    model.train()

    # Load dataset with data augmentation for training
    train_data = load_data("drive_data/train", shuffle=True, batch_size=batch_size, num_workers=2, transform_pipeline=transform_pipeline)
    val_data = load_data("drive_data/val", shuffle=False)

    # Define loss functions and optimizer
    segmentation_loss = nn.CrossEntropyLoss()
    depth_loss = nn.L1Loss(reduction='mean')  # Switched to L1 loss for better depth prediction
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    global_step = 0
    metrics = {key: [] for key in ['train_seg_loss', 'train_depth_loss', 'val_seg_loss', 'val_depth_loss']}

     # Training loop
    for epoch in range(num_epoch):
        for key in metrics:
            metrics[key].clear()

        model.train()
        
        for batch in train_data:
            img, seg_target, depth_target = batch["image"].to(device), batch["track"].to(device), batch["depth"].to(device)
            
            optimizer.zero_grad()
            seg_output, depth_output = model(img)
            
            loss_seg = segmentation_loss(seg_output, seg_target)
            loss_depth = depth_loss(torch.clamp(depth_output, 0, 1), depth_target)  # Ensure depth output stays within 0-1
            total_loss = loss_seg + loss_depth
            
            total_loss.backward()
            optimizer.step()
            
            metrics["train_seg_loss"].append(loss_seg.item())
            metrics["train_depth_loss"].append(loss_depth.item())
            logger.add_scalar("Loss/train_segmentation", loss_seg.item(), global_step)
            logger.add_scalar("Loss/train_depth", loss_depth.item(), global_step)
            global_step += 1

        with torch.inference_mode():
            model.eval()
            confusion_matrix = ConfusionMatrix(num_classes=3)
            metric = DetectionMetric(num_classes=3)
            
            val_depth_errors = []
            lane_depth_errors = []
            
            for batch in val_data:
                img, seg_target, depth_target = batch["image"].to(device), batch["track"].to(device), batch["depth"].to(device)
                seg_output, depth_output = model(img)
                
                loss_seg = segmentation_loss(seg_output, seg_target)
                loss_depth = depth_loss(torch.clamp(depth_output, 0, 1), depth_target)
                
                metrics["val_seg_loss"].append(loss_seg.item())
                metrics["val_depth_loss"].append(loss_depth.item())
                
                confusion_matrix.add(seg_output.argmax(dim=1), seg_target)
                metric.add(seg_output.argmax(dim=1), seg_target, depth_output, depth_target)
                val_depth_errors.append(torch.abs(depth_output - depth_target).mean().item())
                lane_depth_errors.append(torch.abs(depth_output - depth_target).mean().item())
            
            computed_metrics = metric.compute()
            miou = computed_metrics['iou']
            mean_depth_error = computed_metrics['abs_depth_error']
            lane_boundary_error = computed_metrics['tp_depth_error']
            classwise_iou = computed_metrics.get('classwise_iou', {})
            
            print(f"Class-wise IoU: {classwise_iou}")
            print(f"Depth Prediction: Min={depth_output.min().item()}, Max={depth_output.max().item()}")
            print(f"Segmentation Output: Min={seg_output.min().item()}, Max={seg_output.max().item()}")
        
        epoch_train_seg_loss = torch.as_tensor(metrics["train_seg_loss"]).mean()
        epoch_train_depth_loss = torch.as_tensor(metrics["train_depth_loss"]).mean()
        epoch_val_seg_loss = torch.as_tensor(metrics["val_seg_loss"]).mean()
        epoch_val_depth_loss = torch.as_tensor(metrics["val_depth_loss"]).mean()

        logger.add_scalar("Loss/val_segmentation", epoch_val_seg_loss, epoch)
        logger.add_scalar("Loss/val_depth", epoch_val_depth_loss, epoch)
        logger.add_scalar("Metrics/mIoU", miou, epoch)
        logger.add_scalar("Metrics/Depth_MAE", mean_depth_error, epoch)
        logger.add_scalar("Metrics/Lane_Depth_MAE", lane_boundary_error, epoch)


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