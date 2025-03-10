import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb
import torch.nn as nn

from .models import Classifier, load_model, save_model
from datasets.classification_dataset import load_data

def train_classification(
    exp_dir: str = "logs",
    model_name: str = "classifier",
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
    model = load_model(model_name, **kwargs)  # Ensure the model uses the provided Classifier structure
    model = model.to(device)
    model.train()

    # Load dataset with data augmentation for training
    train_data = load_data("classification_data/train", shuffle=True, batch_size=batch_size, num_workers=2, transform_pipeline=transform_pipeline)
    val_data = load_data("classification_data/val", shuffle=False)

    # Define loss function and optimizer
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    global_step = 0
    metrics = {"train_acc": [], "val_acc": []}

    # Training loop
    for epoch in range(num_epoch):
        for key in metrics:
            metrics[key].clear()

        model.train()
        
        for img, label in train_data:
            img, label = img.to(device), label.to(device)
            
            optimizer.zero_grad()
            output = model(img)
            loss = loss_func(output, label)
            loss.backward()
            optimizer.step()
            
            _, preds = torch.max(output, 1)
            acc = (preds == label).float().mean()
            metrics["train_acc"].append(acc.item())
            logger.add_scalar("Loss/train", loss.item(), global_step)
            global_step += 1

        with torch.inference_mode():
            model.eval()
            
            for img, label in val_data:
                img, label = img.to(device), label.to(device)
                output = model(img)
                _, preds = torch.max(output, 1)
                acc = (preds == label).float().mean()
                metrics["val_acc"].append(acc.item())

        epoch_train_acc = torch.as_tensor(metrics["train_acc"]).mean()
        epoch_val_acc = torch.as_tensor(metrics["val_acc"]).mean()

        logger.add_scalar("Accuracy/train", epoch_train_acc, epoch)
        logger.add_scalar("Accuracy/val", epoch_val_acc, epoch)

        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_acc={epoch_train_acc:.4f} "
                f"val_acc={epoch_val_acc:.4f}"
            )

        # Check for early stopping if validation accuracy exceeds 0.80
        if epoch_val_acc >= 0.80:
            print("Early stopping: Validation accuracy threshold reached.")
            break

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
    train_classification(**vars(parser.parse_args()))
