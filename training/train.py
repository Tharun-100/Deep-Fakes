import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.fake_image_dataset import FakeImageDataset
from models.fake_image_model import build_model
from utils.augmentations import get_train_transforms, get_val_transforms
from utils.callbacks import EarlyStopping, SchedulerCallback

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Datasets and loaders
    train_ds = FakeImageDataset(args.train_dir, transform=get_train_transforms())
    val_ds   = FakeImageDataset(args.test_dir,  transform=get_val_transforms())
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

    # Model, optimizer, criterion
    model = build_model(unfreeze_at_epoch=args.unfreeze_epoch).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scheduler_cb = SchedulerCallback(scheduler)
    early_stopper = EarlyStopping(patience=args.patience, verbose=True)

    best_epoch = 0
    for epoch in range(args.epochs):
        model.train()
        if epoch == model.unfreeze_at:
            for p in model.base_model.features.parameters():
                p.requires_grad = True
            print(f"Unfroze backbone at epoch {epoch}")

        running_loss, correct, total = 0.0, 0, 0
        for images, labels in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        print(f"Epoch {epoch}: Train Loss {train_loss:.4f}, Train Acc {train_acc:.2%}, "
              f"Val Loss {val_loss:.4f}, Val Acc {val_acc:.2%}")

        # Callbacks
        scheduler_cb.step()
        early_stopper(val_loss, model, args.checkpoint_path)
        if early_stopper.early_stop:
            print("Early stopping triggered")
            break
        if val_acc > train_acc:  # example checkpoint criterion
            best_epoch = epoch

    print(f"Training complete. Best epoch: {best_epoch}")

