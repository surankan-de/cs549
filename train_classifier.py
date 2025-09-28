# train_classifier.py
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from models import Classifier
from utils import PTCTDataset
import numpy as np

def train_classifier(model, dataset, device='cpu', epochs=10, batch_size=256, lr=1e-4):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    for epoch in range(epochs):
        model.train()
        losses = []
        correct = 0
        total = 0
        for pt, ct, labels in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            pt = pt.to(device)
            ct = ct.to(device)
            labels = labels.to(device)
            logits = model(pt, ct)
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        acc = correct/total
        print(f"Epoch {epoch+1} loss {np.mean(losses):.4f} acc {acc:.4f}")
    return model

if __name__ == "__main__":
    # demo multiclass classifier on AES-ECB plaintext classes
    from data_gen import generate_plaintexts_classes, aes_ecb_encrypt_batch
    pts, labels = generate_plaintexts_classes(4096, length=16, num_classes=10)
    key, cts = aes_ecb_encrypt_batch(pts)
    ds = PTCTDataset(pts, cts, labels=labels, pt_len=16, ct_len=16)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Classifier(16, 16, hidden=256, num_classes=10)
    train_classifier(model, ds, device=device, epochs=10, batch_size=256)
