import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from datasets.fake_image_dataset import FakeImageDataset
from models.fake_image_model import build_model
from utils.augmentations import get_val_transforms

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = FakeImageDataset(args.test_dir, transform=get_val_transforms())
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    model = build_model().to(device)
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds),
        "recall": recall_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds),
        "roc_auc": roc_auc_score(all_labels, all_preds),
    }
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

