import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import f1_score


class BCEOneHot(nn.Module):
    """Binary cross-entropy on one-hot targets: every class is its own yes/no question."""

    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, scores, labels):
        return self.loss(scores, F.one_hot(labels, scores.shape[1]).float())


def squared_error(scores, labels):
    """Sum over the batch of the squared gap between the probabilities and the one-hot truth."""
    probs = torch.softmax(scores, dim=1)
    one_hot = F.one_hot(labels, probs.shape[1]).float()
    return ((probs - one_hot) ** 2).sum().item()


def r2_from_sums(sse, label_counts, n):
    """R2 of the probabilities: 1 - (squared error) / (squared error of always guessing class frequencies)."""
    sst = float((label_counts * (1 - label_counts / n)).sum())
    return 1 - sse / sst


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    """One pass over the training images. Returns (average loss, accuracy, R2)."""
    model.train()
    total_loss, correct, seen, sse = 0.0, 0, 0, 0.0
    counts = None
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        scores = model(images)
        loss = loss_fn(scores, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        correct += (scores.argmax(dim=1) == labels).sum().item()
        seen += len(labels)
        sse += squared_error(scores.detach(), labels)
        batch_counts = torch.bincount(labels.cpu(), minlength=scores.shape[1]).double()
        counts = batch_counts if counts is None else counts + batch_counts
    return total_loss / seen, correct / seen, r2_from_sums(sse, counts, seen)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    """Measure the model without changing it. Returns a dictionary of results."""
    model.eval()
    total_loss, seen, sse = 0.0, 0, 0.0
    counts = None
    y_true, y_pred = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        scores = model(images)
        total_loss += loss_fn(scores, labels).item() * len(labels)
        seen += len(labels)
        sse += squared_error(scores, labels)
        batch_counts = torch.bincount(labels.cpu(), minlength=scores.shape[1]).double()
        counts = batch_counts if counts is None else counts + batch_counts
        y_true += labels.cpu().tolist()
        y_pred += scores.argmax(dim=1).cpu().tolist()

    correct = sum(t == p for t, p in zip(y_true, y_pred))
    return {
        "loss": total_loss / seen,
        "accuracy": correct / seen,
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "r2": r2_from_sums(sse, counts, seen),
        "y_true": y_true,
        "y_pred": y_pred,
    }