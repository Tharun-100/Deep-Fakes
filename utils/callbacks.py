import torch
import numpy as np

class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0.0):
        self.patience = patience
        self.counter = 0
        self.best_loss = np.Inf
        self.early_stop = False
        self.delta = delta
        self.verbose = verbose

    def __call__(self, val_loss, model, save_path):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), save_path)
            if self.verbose:
                print(f"Validation loss improved to {val_loss:.4f}. Model saved.")
        else:
            self.counter += 1
            if self.verbose:
                print(f"No improvement in val loss for {self.counter} epochs.")
            if self.counter >= self.patience:
                self.early_stop = True

class SchedulerCallback:
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def step(self):
        self.scheduler.step()

