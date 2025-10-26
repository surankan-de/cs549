#!/usr/bin/env python3
"""Visualization tool for heavy model benchmark results."""
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from indcpa import make_indcpa_dataset, train_test_split, systems_to_test
from systems import generate_keys

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


WEAK_SYSTEMS = [
    'No Encryption', 'Constant XOR', 'Toy Fixed XOR',
    'Toy Substitution', 'Toy Permutation', 'Toy 1-Round Feistel',
    'AES CTR Fixed Nonce', 'Toy Caesar', 'Toy Repeating XOR',
    'Toy Byte Rotate', 'Toy Mask HighNibble', 'Toy LFSR Stream',
    'Toy 2-Round Feistel',
]

SEMI_WEAK_SYSTEMS = [
    'Semi Reduced Feistel', 'Semi Partial Mask', 'Semi Truncated AES',
    'Semi Nonce Mix', 'Semi LFSR Long', 'Semi Key Rotation',
    'DES NonDet', 'AES CTR Reduced',
]

STRONG_SYSTEMS = [
    'DES', 'AES ECB', 'AES CTR', 'RSA Plain', 'RSA OAEP', 'RSA OAEP Reused','One-Time Pad'
]

HEAVY_MODELS = ['rf_large', 'hgb', 'svc_rbf', 'deep_mlp', 'cnn1d']


def flatten_X(X):
    return X.reshape(X.shape[0], -1)


def sizeof(obj):
    return len(pickle.dumps(obj))


class DeepMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.net(x)


class CNN1D(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 2)
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
            xb, yb = xb.to(device), yb.to(device)
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

    results = {}

    try:
        t0 = time.time()
        rf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
        rf.fit(Xtr_flat, y_tr)
        t_rf = time.time() - t0
        t_inf_start = time.time()
        preds_rf = rf.predict(Xte_flat)
        t_inf_rf = time.time() - t_inf_start
        acc_rf = accuracy_score(y_te, preds_rf)
        results['rf_large'] = (acc_rf, t_rf, t_inf_rf, sizeof(rf))
        print(f"rf_large     | acc={acc_rf:.4f} | train_s={t_rf:.3f} | inf_s={t_inf_rf:.3f} | size_bytes={sizeof(rf)}")
    except Exception as e:
        print("rf_large failed:", e)

    try:
        t0 = time.time()
        hgb = HistGradientBoostingClassifier(max_iter=200)
        hgb.fit(Xtr_flat, y_tr)
        t_hgb = time.time() - t0
        t_inf_start = time.time()
        preds_hgb = hgb.predict(Xte_flat)
        t_inf_hgb = time.time() - t_inf_start
        acc_hgb = accuracy_score(y_te, preds_hgb)
        results['hgb'] = (acc_hgb, t_hgb, t_inf_hgb, sizeof(hgb))
        print(f"hgb          | acc={acc_hgb:.4f} | train_s={t_hgb:.3f} | inf_s={t_inf_hgb:.3f} | size_bytes={sizeof(hgb)}")
    except Exception as e:
        print("hgb failed:", e)

    try:
        t0 = time.time()
        svc = SVC(kernel='rbf', gamma='scale')
        svc.fit(Xtr_flat, y_tr)
        t_svc = time.time() - t0
        t_inf_start = time.time()
        preds_svc = svc.predict(Xte_flat)
        t_inf_svc = time.time() - t_inf_start
        acc_svc = accuracy_score(y_te, preds_svc)
        results['svc_rbf'] = (acc_svc, t_svc, t_inf_svc, sizeof(svc))
        print(f"svc_rbf      | acc={acc_svc:.4f} | train_s={t_svc:.3f} | inf_s={t_inf_svc:.3f} | size_bytes={sizeof(svc)}")
    except Exception as e:
        print("svc_rbf failed:", e)

    try:
        mlp = DeepMLP(input_dim=Xtr_flat.shape[1])
        model_mlp, t_mlp, preds_mlp = train_torch_model(mlp, Xtr_flat, y_tr, Xte_flat, epochs=15)
        t_inf_start = time.time()
        with torch.no_grad():
            _ = model_mlp(torch.tensor(Xte_flat, dtype=torch.float32))
        t_inf_mlp = time.time() - t_inf_start
        acc_mlp = accuracy_score(y_te, preds_mlp)
        results['deep_mlp'] = (acc_mlp, t_mlp, t_inf_mlp, sizeof(model_mlp.state_dict()))
        print(f"deep_mlp     | acc={acc_mlp:.4f} | train_s={t_mlp:.3f} | inf_s={t_inf_mlp:.3f} | size_bytes={sizeof(model_mlp.state_dict())}")
    except Exception as e:
        print("deep_mlp failed:", e)

    try:
        seq_len = Xtr_flat.shape[1]
        cnn = CNN1D(seq_len)
        model_cnn, t_cnn, preds_cnn = train_torch_model(cnn, Xtr_flat, y_tr, Xte_flat, epochs=15)
        t_inf_start = time.time()
        with torch.no_grad():
            _ = model_cnn(torch.tensor(Xte_flat, dtype=torch.float32))
        t_inf_cnn = time.time() - t_inf_start
        acc_cnn = accuracy_score(y_te, preds_cnn)
        results['cnn1d'] = (acc_cnn, t_cnn, t_inf_cnn, sizeof(model_cnn.state_dict()))
        print(f"cnn1d        | acc={acc_cnn:.4f} | train_s={t_cnn:.3f} | inf_s={t_inf_cnn:.3f} | size_bytes={sizeof(model_cnn.state_dict())}")
    except Exception as e:
        print("cnn1d failed:", e)

    return results


def categorize_results(all_results):
    weak_results = {k: v for k, v in all_results.items() if k in WEAK_SYSTEMS}
    semi_results = {k: v for k, v in all_results.items() if k in SEMI_WEAK_SYSTEMS}
    strong_results = {k: v for k, v in all_results.items() if k in STRONG_SYSTEMS}
    return weak_results, semi_results, strong_results


def plot_accuracy_by_category(all_results):
    weak, semi, strong = categorize_results(all_results)
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 14))
    model_names = HEAVY_MODELS
    
    categories = [
        (weak, 'WEAK Systems', axes[0], 'Reds'),
        (semi, 'SEMI-WEAK Systems', axes[1], 'Oranges'),
        (strong, 'STRONG Systems', axes[2], 'Greens')
    ]
    
    for results, title, ax, cmap in categories:
        if not results:
            continue
            
        ciphers = list(results.keys())
        x = np.arange(len(model_names))
        width = 0.8 / max(len(ciphers), 1)
        
        colors = plt.cm.get_cmap(cmap)(np.linspace(0.3, 0.9, len(ciphers)))
        
        for i, cipher in enumerate(ciphers):
            accuracies = [results[cipher].get(model, (0,0,0,0))[0] for model in model_names]
            offset = width * (i - len(ciphers)/2 + 0.5)
            ax.bar(x + offset, accuracies, width, label=cipher, alpha=0.85, color=colors[i])
        
        ax.set_xlabel('Model', fontsize=11, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend(title='Cipher', ncol=2, fontsize=8, loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.05])
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    return fig


def plot_accuracy_diff_matrix(all_results):
    model_names = HEAVY_MODELS
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 16))
    
    weak, semi, strong = categorize_results(all_results)
    categories = [
        (weak, 'Accuracy Differences - WEAK Systems', axes[0]),
        (semi, 'Accuracy Differences - SEMI-WEAK Systems', axes[1]),
        (strong, 'Accuracy Differences - STRONG Systems', axes[2])
    ]
    
    for results, title, ax in categories:
        if not results:
            ax.set_visible(False)
            continue
        
        systems = list(results.keys())
        n_models = len(model_names)
        avg_diff_matrix = np.zeros((n_models, n_models))
        
        for system in systems:
            for i, model_i in enumerate(model_names):
                for j, model_j in enumerate(model_names):
                    acc_i = results[system].get(model_i, (0,0,0,0))[0]
                    acc_j = results[system].get(model_j, (0,0,0,0))[0]
                    avg_diff_matrix[i, j] += (acc_i - acc_j)
        
        avg_diff_matrix /= len(systems)
        
        im = ax.imshow(avg_diff_matrix, cmap='RdYlGn', aspect='auto', vmin=-0.3, vmax=0.3)
        ax.set_xticks(np.arange(n_models))
        ax.set_yticks(np.arange(n_models))
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_yticklabels(model_names)
        ax.set_title(title, fontweight='bold', fontsize=12, pad=10)
        ax.set_xlabel('Model (column)', fontweight='bold')
        ax.set_ylabel('Model (row)', fontweight='bold')
        
        for i in range(n_models):
            for j in range(n_models):
                color = 'white' if abs(avg_diff_matrix[i, j]) > 0.15 else 'black'
                text = ax.text(j, i, f'{avg_diff_matrix[i, j]:.3f}',
                             ha="center", va="center", color=color, fontsize=9)
        
        cbar = plt.colorbar(im, ax=ax, label='Avg Accuracy Difference (row - col)')
        cbar.set_label('Avg Accuracy Difference (row - col)', fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_category_performance_heatmap(all_results):
    weak, semi, strong = categorize_results(all_results)
    model_names = HEAVY_MODELS
    
    categories = ['Weak', 'Semi-Weak', 'Strong']
    acc_matrix = np.zeros((len(categories), len(model_names)))
    
    for cat_idx, results in enumerate([weak, semi, strong]):
        if results:
            for model_idx, model in enumerate(model_names):
                accs = [results[sys].get(model, (0,0,0,0))[0] for sys in results.keys()]
                acc_matrix[cat_idx, model_idx] = np.mean(accs) if accs else 0
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    im = ax.imshow(acc_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(model_names)))
    ax.set_yticks(np.arange(len(categories)))
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_yticklabels(categories)
    ax.set_title('Average Model Accuracy by Security Category', fontweight='bold', fontsize=14, pad=15)
    ax.set_xlabel('Model', fontweight='bold', fontsize=12)
    ax.set_ylabel('System Category', fontweight='bold', fontsize=12)
    
    for i in range(len(categories)):
        for j in range(len(model_names)):
            text = ax.text(j, i, f'{acc_matrix[i, j]:.3f}',
                         ha="center", va="center", 
                         color='white' if acc_matrix[i, j] < 0.6 else 'black',
                         fontsize=10, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, label='Average Accuracy')
    cbar.set_label('Average Accuracy', fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_model_effectiveness_across_categories(all_results):
    weak, semi, strong = categorize_results(all_results)
    model_names = HEAVY_MODELS
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(3)
    width = 0.8 / len(model_names)
    
    colors = plt.cm.tab10(np.arange(len(model_names)))
    
    for i, model in enumerate(model_names):
        weak_acc = np.mean([weak[sys].get(model, (0,0,0,0))[0] for sys in weak.keys()]) if weak else 0
        semi_acc = np.mean([semi[sys].get(model, (0,0,0,0))[0] for sys in semi.keys()]) if semi else 0
        strong_acc = np.mean([strong[sys].get(model, (0,0,0,0))[0] for sys in strong.keys()]) if strong else 0
        
        accs = [weak_acc, semi_acc, strong_acc]
        offset = width * (i - len(model_names)/2 + 0.5)
        ax.bar(x + offset, accs, width, label=model, alpha=0.85, color=colors[i])
    
    ax.set_xlabel('Security Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Model Effectiveness Across Security Categories', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(['Weak Systems', 'Semi-Weak Systems', 'Strong Systems'])
    ax.legend(title='Model', ncol=1, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    return fig


def plot_timing_analysis(all_results):
    weak, semi, strong = categorize_results(all_results)
    model_names = HEAVY_MODELS
    
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)
    
    categories_data = [weak, semi, strong]
    cat_names = ['Weak', 'Semi-Weak', 'Strong']
    colors_cat = ['#ff6b6b', '#ffa500', '#4caf50']
    
    x = np.arange(len(model_names))
    width = 0.25
    
    ax1 = fig.add_subplot(gs[0, 0])
    for idx, (results, cat_name, color) in enumerate(zip(categories_data, cat_names, colors_cat)):
        if not results:
            continue
        avg_times = []
        for model in model_names:
            times = [results[sys].get(model, (0,0,0,0))[1] for sys in results.keys()]
            avg_times.append(np.mean(times) if times else 0)
        ax1.bar(x + idx*width, avg_times, width, label=cat_name, alpha=0.8, color=color)
    ax1.set_xlabel('Model', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Avg Training Time (s)', fontsize=10, fontweight='bold')
    ax1.set_title('Training Time by Category', fontsize=11, fontweight='bold')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
    ax1.legend(fontsize=9)
    ax1.set_yscale('log')
    ax1.grid(axis='y', alpha=0.3)
    
    ax2 = fig.add_subplot(gs[0, 1])
    for idx, (results, cat_name, color) in enumerate(zip(categories_data, cat_names, colors_cat)):
        if not results:
            continue
        avg_times = []
        for model in model_names:
            times = [results[sys].get(model, (0,0,0,0))[2] for sys in results.keys()]
            avg_times.append(np.mean(times) if times else 0)
        ax2.bar(x + idx*width, avg_times, width, label=cat_name, alpha=0.8, color=color)
    ax2.set_xlabel('Model', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Avg Inference Time (s)', fontsize=10, fontweight='bold')
    ax2.set_title('Inference Time by Category', fontsize=11, fontweight='bold')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
    ax2.legend(fontsize=9)
    ax2.set_yscale('log')
    ax2.grid(axis='y', alpha=0.3)
    
    ax3 = fig.add_subplot(gs[0, 2])
    ratios = []
    for model in model_names:
        train_times = [all_results[sys].get(model, (0,0,0,0))[1] for sys in all_results.keys()]
        inf_times = [all_results[sys].get(model, (0,0,0,0))[2] for sys in all_results.keys()]
        train_times = [t for t in train_times if t > 0]
        inf_times = [t for t in inf_times if t > 0]
        ratio = np.mean(train_times) / np.mean(inf_times) if train_times and inf_times else 0
        ratios.append(ratio)
    ax3.bar(model_names, ratios, alpha=0.7, color='steelblue', edgecolor='black')
    ax3.set_xlabel('Model', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Train/Inference Ratio', fontsize=10, fontweight='bold')
    ax3.set_title('Training vs Inference Time Ratio', fontsize=11, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
    ax3.set_yscale('log')
    ax3.grid(axis='y', alpha=0.3)
    
    ax4 = fig.add_subplot(gs[1, 0])
    efficiency_weak = []
    efficiency_strong = []
    for model in model_names:
        if weak:
            accs = [weak[sys].get(model, (0,0,0,0))[0] for sys in weak.keys()]
            times = [weak[sys].get(model, (0,0,0,0))[1] for sys in weak.keys()]
            accs = [a for a in accs if a > 0]
            times = [t for t in times if t > 0]
            efficiency_weak.append(np.mean(accs) / np.mean(times) if accs and times else 0)
        if strong:
            accs = [strong[sys].get(model, (0,0,0,0))[0] for sys in strong.keys()]
            times = [strong[sys].get(model, (0,0,0,0))[1] for sys in strong.keys()]
            accs = [a for a in accs if a > 0]
            times = [t for t in times if t > 0]
            efficiency_strong.append(np.mean(accs) / np.mean(times) if accs and times else 0)
    x_pos = np.arange(len(model_names))
    width = 0.35
    if efficiency_weak:
        ax4.bar(x_pos - width/2, efficiency_weak, width, label='Weak Systems', alpha=0.8, color='#ff6b6b')
    if efficiency_strong:
        ax4.bar(x_pos + width/2, efficiency_strong, width, label='Strong Systems', alpha=0.8, color='#4caf50')
    ax4.set_xlabel('Model', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Accuracy / Train Time', fontsize=10, fontweight='bold')
    ax4.set_title('Training Efficiency (Acc/sec)', fontsize=11, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
    ax4.legend(fontsize=9)
    ax4.set_yscale('log')
    ax4.grid(axis='y', alpha=0.3)
    
    ax5 = fig.add_subplot(gs[1, 1])
    train_stds = []
    for model in model_names:
        times = [all_results[sys].get(model, (0,0,0,0))[1] for sys in all_results.keys()]
        times = [t for t in times if t > 0]
        train_stds.append(np.std(times) / np.mean(times) if times else 0)
    ax5.bar(model_names, train_stds, alpha=0.7, color='coral', edgecolor='black')
    ax5.set_xlabel('Model', fontsize=10, fontweight='bold')
    ax5.set_ylabel('CV (Std/Mean)', fontsize=10, fontweight='bold')
    ax5.set_title('Training Time Consistency', fontsize=11, fontweight='bold')
    ax5.tick_params(axis='x', rotation=45)
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
    ax5.grid(axis='y', alpha=0.3)
    
    ax6 = fig.add_subplot(gs[1, 2])
    colors_scatter = plt.cm.tab10(np.arange(len(model_names)))
    for idx, model in enumerate(model_names):
        accs = [all_results[sys].get(model, (0,0,0,0))[0] for sys in all_results.keys()]
        times = [all_results[sys].get(model, (0,0,0,0))[1] for sys in all_results.keys()]
        ax6.scatter(times, accs, alpha=0.6, s=30, color=colors_scatter[idx], label=model)
    ax6.set_xlabel('Training Time (s)', fontsize=10, fontweight='bold')
    ax6.set_ylabel('Accuracy', fontsize=10, fontweight='bold')
    ax6.set_title('Accuracy vs Training Time', fontsize=11, fontweight='bold')
    ax6.set_xscale('log')
    ax6.legend(fontsize=7, ncol=1)
    ax6.grid(alpha=0.3)
    ax6.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    
    ax7 = fig.add_subplot(gs[2, 0])
    time_matrix = np.zeros((len(cat_names), len(model_names)))
    for cat_idx, results in enumerate(categories_data):
        if results:
            for model_idx, model in enumerate(model_names):
                times = [results[sys].get(model, (0,0,0,0))[1] for sys in results.keys()]
                time_matrix[cat_idx, model_idx] = np.mean(times) if times else 0
    im = ax7.imshow(time_matrix, cmap='YlOrRd', aspect='auto')
    ax7.set_xticks(np.arange(len(model_names)))
    ax7.set_yticks(np.arange(len(cat_names)))
    ax7.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
    ax7.set_yticklabels(cat_names, fontsize=9)
    ax7.set_title('Avg Training Time Heatmap', fontsize=11, fontweight='bold')
    for i in range(len(cat_names)):
        for j in range(len(model_names)):
            if time_matrix[i, j] > 0:
                text = ax7.text(j, i, f'{time_matrix[i, j]:.2f}',
                              ha="center", va="center", color='black', fontsize=7)
    plt.colorbar(im, ax=ax7, label='Time (s)')
    
    ax8 = fig.add_subplot(gs[2, 1])
    speedups = []
    for model in model_names:
        train_times = [all_results[sys].get(model, (0,0,0,0))[1] for sys in all_results.keys()]
        inf_times = [all_results[sys].get(model, (0,0,0,0))[2] for sys in all_results.keys()]
        train_times = [t for t in train_times if t > 0]
        inf_times = [t for t in inf_times if t > 0]
        speedup = np.mean(train_times) / np.mean(inf_times) if train_times and inf_times else 0
        speedups.append(speedup)
    colors_speedup = ['green' if s > 100 else 'orange' if s > 10 else 'red' for s in speedups]
    bars = ax8.bar(model_names, speedups, alpha=0.7, color=colors_speedup, edgecolor='black')
    ax8.set_xlabel('Model', fontsize=10, fontweight='bold')
    ax8.set_ylabel('Speedup Factor', fontsize=10, fontweight='bold')
    ax8.set_title('Inference Speedup over Training', fontsize=11, fontweight='bold')
    ax8.tick_params(axis='x', rotation=45)
    plt.setp(ax8.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
    ax8.set_yscale('log')
    ax8.grid(axis='y', alpha=0.3)
    for bar, speedup in zip(bars, speedups):
        if speedup > 0:
            height = bar.get_height()
            ax8.text(bar.get_x() + bar.get_width()/2., height,
                    f'{speedup:.0f}x', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax9 = fig.add_subplot(gs[2, 2])
    for model in model_names:
        avg_acc = np.mean([all_results[sys].get(model, (0,0,0,0))[0] for sys in all_results.keys() if all_results[sys].get(model)])
        avg_time = np.mean([all_results[sys].get(model, (0,0,0,0))[1] for sys in all_results.keys() if all_results[sys].get(model)])
        avg_size = np.mean([all_results[sys].get(model, (0,0,0,0))[3] for sys in all_results.keys() if all_results[sys].get(model)]) / 1024
        if avg_time > 0:
            ax9.scatter(avg_time, avg_acc, s=avg_size/10, alpha=0.6)
            ax9.annotate(model, (avg_time, avg_acc), fontsize=7, alpha=0.7)
    ax9.set_xlabel('Avg Training Time (s)', fontsize=10, fontweight='bold')
    ax9.set_ylabel('Avg Accuracy', fontsize=10, fontweight='bold')
    ax9.set_title('Pareto Frontier (size = bubble size)', fontsize=11, fontweight='bold')
    ax9.set_xscale('log')
    ax9.grid(alpha=0.3)
    ax9.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    
    return fig


def plot_differential_analysis(all_results):
    weak, semi, strong = categorize_results(all_results)
    model_names = HEAVY_MODELS
    
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, :2])
    for cat_name, results, color in [('Weak', weak, 'red'), ('Semi-Weak', semi, 'orange'), ('Strong', strong, 'green')]:
        if not results:
            continue
        all_accs = []
        for system in results.keys():
            for model in model_names:
                acc = results[system].get(model, (0,0,0,0))[0]
                if acc > 0:
                    all_accs.append(acc)
        if all_accs:
            ax1.hist(all_accs, bins=20, alpha=0.5, label=f'{cat_name} (n={len(all_accs)})', color=color, edgecolor='black')
    ax1.set_xlabel('Accuracy', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax1.set_title('Accuracy Distribution by Security Category', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, linewidth=2)
    ax1.grid(alpha=0.3)
    
    ax2 = fig.add_subplot(gs[0, 2])
    best_models = []
    for system, results in all_results.items():
        best_acc = 0
        best_model = ''
        for model in model_names:
            acc = results.get(model, (0,0,0,0))[0]
            if acc > best_acc:
                best_acc = acc
                best_model = model
        if best_model:
            best_models.append(best_model)
    from collections import Counter
    model_counts = Counter(best_models)
    ax2.bar(range(len(model_counts)), list(model_counts.values()), tick_label=list(model_counts.keys()), alpha=0.7, color='steelblue', edgecolor='black')
    ax2.set_xlabel('Model', fontsize=10, fontweight='bold')
    ax2.set_ylabel('# Times Best', fontsize=10, fontweight='bold')
    ax2.set_title('Best Performing Model Count', fontsize=11, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    ax3 = fig.add_subplot(gs[1, 0])
    categories_data = [weak, semi, strong]
    cat_names = ['Weak', 'Semi-Weak', 'Strong']
    success_rates = []
    for results in categories_data:
        if not results:
            success_rates.append(0)
            continue
        total = 0
        success = 0
        for system in results.keys():
            for model in model_names:
                acc = results[system].get(model, (0,0,0,0))[0]
                if acc > 0:
                    total += 1
                    if acc > 0.6:
                        success += 1
        success_rates.append(success / total if total > 0 else 0)
    colors_bar = ['#ff6b6b', '#ffa500', '#4caf50']
    bars = ax3.bar(cat_names, success_rates, alpha=0.7, color=colors_bar, edgecolor='black', linewidth=2)
    ax3.set_ylabel('Success Rate', fontsize=11, fontweight='bold')
    ax3.set_title('Attack Success Rate (Acc > 60%)', fontsize=12, fontweight='bold')
    ax3.set_ylim([0, 1])
    ax3.grid(axis='y', alpha=0.3)
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height, f'{rate:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax4 = fig.add_subplot(gs[1, 1])
    model_stds = []
    for model in model_names:
        all_accs = [all_results[sys].get(model, (0,0,0,0))[0] for sys in all_results.keys()]
        all_accs = [a for a in all_accs if a > 0]
        model_stds.append(np.std(all_accs) if all_accs else 0)
    ax4.bar(model_names, model_stds, alpha=0.7, color='coral', edgecolor='black')
    ax4.set_xlabel('Model', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Std Dev of Accuracy', fontsize=10, fontweight='bold')
    ax4.set_title('Model Consistency Across All Systems', fontsize=11, fontweight='bold')
    ax4.tick_params(axis='x', rotation=45)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax4.grid(axis='y', alpha=0.3)
    
    ax5 = fig.add_subplot(gs[1, 2])
    gaps = []
    for model in model_names:
        weak_accs = [weak[sys].get(model, (0,0,0,0))[0] for sys in weak.keys()] if weak else []
        strong_accs = [strong[sys].get(model, (0,0,0,0))[0] for sys in strong.keys()] if strong else []
        weak_avg = np.mean([a for a in weak_accs if a > 0]) if weak_accs else 0
        strong_avg = np.mean([a for a in strong_accs if a > 0]) if strong_accs else 0
        gaps.append(weak_avg - strong_avg)
    colors_gap = ['green' if g > 0 else 'red' for g in gaps]
    ax5.bar(model_names, gaps, alpha=0.7, color=colors_gap, edgecolor='black')
    ax5.set_xlabel('Model', fontsize=10, fontweight='bold')
    ax5.set_ylabel('Accuracy Gap', fontsize=10, fontweight='bold')
    ax5.set_title('Weak vs Strong Systems Gap', fontsize=11, fontweight='bold')
    ax5.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax5.tick_params(axis='x', rotation=45)
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax5.grid(axis='y', alpha=0.3)
    
    return fig


def main():
    print("="*70)
    print("HEAVY MODEL DIFFERENTIAL ANALYSIS")
    print("="*70)
    
    all_results = {}
    for cipher in systems_to_test:
        results = run_bench(cipher_name=cipher, n_samples=2000, msg_len=16)
        all_results[cipher] = results
    
    print("\n" + "="*70)
    print("Generating differential plots...")
    print("="*70)
    
    fig1 = plot_accuracy_by_category(all_results)
    plt.savefig('heavy_diff_accuracy_by_category.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: heavy_diff_accuracy_by_category.png")
    
    fig2 = plot_accuracy_diff_matrix(all_results)
    plt.savefig('heavy_diff_model_comparison_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: heavy_diff_model_comparison_matrix.png")
    
    fig3 = plot_category_performance_heatmap(all_results)
    plt.savefig('heavy_diff_category_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: heavy_diff_category_heatmap.png")
    
    fig4 = plot_model_effectiveness_across_categories(all_results)
    plt.savefig('heavy_diff_model_effectiveness.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: heavy_diff_model_effectiveness.png")
    
    fig5 = plot_differential_analysis(all_results)
    plt.savefig('heavy_diff_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: heavy_diff_comprehensive_analysis.png")
    
    fig6 = plot_timing_analysis(all_results)
    plt.savefig('heavy_diff_timing_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: heavy_diff_timing_analysis.png")
    
    print("\n" + "="*70)
    print("All heavy model differential plots generated successfully!")
    print("="*70)
    
    weak, semi, strong = categorize_results(all_results)
    print(f"\nSystem Categories:")
    print(f"  Weak Systems:      {len(weak)} systems")
    print(f"  Semi-Weak Systems: {len(semi)} systems")
    print(f"  Strong Systems:    {len(strong)} systems")
    


if __name__ == '__main__':
    main()