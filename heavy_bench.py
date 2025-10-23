#!/usr/bin/env python3
"""Benchmark larger/heavier models for IND-CPA classification.

This mirrors the style of `lightweight_bench.py` so outputs are directly
comparable. It iterates `systems_to_test` from `indcpa.py` and runs heavier
classifiers (large RF, HGB, RBF-SVM) plus PyTorch Deep MLP and 1D-CNN.
"""
import time
import pickle
import numpy as np
from indcpa import make_indcpa_dataset, train_test_split, systems_to_test
from systems import generate_keys

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


def flatten_X(X):
    return X.reshape(X.shape[0], -1)


def sizeof(obj):
    return len(pickle.dumps(obj))


class DeepMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.net(x)


class CNN1D(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        return self.fc(x)


def train_torch_model(model, X_tr, y_tr, X_val, epochs=15, batch_size=256, lr=1e-3, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    start = time.time()
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    train_time = time.time() - start

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_val, dtype=torch.float32).to(device))
        preds = logits.argmax(dim=1).cpu().numpy()
    return model, train_time, preds


def run_bench(cipher_name='Toy Substitution', n_samples=2000, msg_len=16):
    print(f"\nRunning heavy benchmark for {cipher_name} with {n_samples} samples per class")
    keys = generate_keys()
    X, y = make_indcpa_dataset(n_samples=n_samples, msg_len=msg_len, keys=keys, cipher_name=cipher_name)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)
    Xtr_flat = flatten_X(X_tr)
    Xte_flat = flatten_X(X_te)

    # Sklearn heavy models
    # RandomForest (larger)
    try:
        t0 = time.time()
        rf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
        rf.fit(Xtr_flat, y_tr)
        t_rf = time.time() - t0
        preds_rf = rf.predict(Xte_flat)
        acc_rf = accuracy_score(y_te, preds_rf)
        print(f"rf_large     | acc={acc_rf:.4f} | train_s={t_rf:.3f} | inf_s={0.0:.3f} | size_bytes={sizeof(rf)}")
    except Exception as e:
        print("rf_large failed:", e)

    # HistGradientBoosting
    try:
        t0 = time.time()
        hgb = HistGradientBoostingClassifier(max_iter=200)
        hgb.fit(Xtr_flat, y_tr)
        t_hgb = time.time() - t0
        preds_hgb = hgb.predict(Xte_flat)
        acc_hgb = accuracy_score(y_te, preds_hgb)
        print(f"hgb          | acc={acc_hgb:.4f} | train_s={t_hgb:.3f} | inf_s={0.0:.3f} | size_bytes={sizeof(hgb)}")
    except Exception as e:
        print("hgb failed:", e)

    # RBF SVM
    try:
        t0 = time.time()
        svc = SVC(kernel='rbf', gamma='scale')
        svc.fit(Xtr_flat, y_tr)
        t_svc = time.time() - t0
        preds_svc = svc.predict(Xte_flat)
        acc_svc = accuracy_score(y_te, preds_svc)
        print(f"svc_rbf      | acc={acc_svc:.4f} | train_s={t_svc:.3f} | inf_s={0.0:.3f} | size_bytes={sizeof(svc)}")
    except Exception as e:
        print("svc_rbf failed:", e)

    # PyTorch Deep MLP
    try:
        mlp = DeepMLP(input_dim=Xtr_flat.shape[1])
        model_mlp, t_mlp, preds_mlp = train_torch_model(mlp, Xtr_flat, y_tr, Xte_flat, epochs=15)
        acc_mlp = accuracy_score(y_te, preds_mlp)
        print(f"deep_mlp     | acc={acc_mlp:.4f} | train_s={t_mlp:.3f} | inf_s={0.0:.3f} | size_bytes={sizeof(model_mlp.state_dict())}")
    except Exception as e:
        print("deep_mlp failed:", e)

    # PyTorch CNN1D
    try:
        seq_len = Xtr_flat.shape[1]
        cnn = CNN1D(seq_len)
        model_cnn, t_cnn, preds_cnn = train_torch_model(cnn, Xtr_flat, y_tr, Xte_flat, epochs=15)
        acc_cnn = accuracy_score(y_te, preds_cnn)
        print(f"cnn1d        | acc={acc_cnn:.4f} | train_s={t_cnn:.3f} | inf_s={0.0:.3f} | size_bytes={sizeof(model_cnn.state_dict())}")
    except Exception as e:
        print("cnn1d failed:", e)


if __name__ == '__main__':
    for cipher in systems_to_test:
        run_bench(cipher_name=cipher, n_samples=2000, msg_len=16)
